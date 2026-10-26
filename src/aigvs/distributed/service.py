import asyncio
from contextlib import asynccontextmanager
import hmac
import logging
import time
import uuid

import httpx
from fastapi import FastAPI, Header, HTTPException, Path
from fastapi.responses import JSONResponse, Response
from prometheus_client import CollectorRegistry, Counter, Histogram, generate_latest, CONTENT_TYPE_LATEST

from .config import Settings
from .index import IndexCache
from .models import Collection, Mutation, Search, ShardSearch, worker_order
from .store import Conflict, Missing, Store

log = logging.getLogger("aigvs")


class Guard:
    """Bound body memory, reject excess work, authenticate before parsing JSON."""
    def __init__(self, app, settings, registry):
        self.app, self.settings, self.active = app, settings, 0
        self.requests = Counter("aigvs_http_requests_total", "HTTP responses", ["status"], registry=registry)
        self.latency = Histogram("aigvs_http_duration_seconds", "Request duration", registry=registry)

    async def __call__(self, scope, receive, send):
        if scope["type"] != "http":
            return await self.app(scope, receive, send)
        started = time.monotonic()
        request_id = uuid.uuid4().hex

        async def respond(message):
            if message["type"] == "http.response.start":
                self.requests.labels(str(message["status"])).inc()
                message.setdefault("headers", []).append((b"x-request-id", request_id.encode()))
            await send(message)

        async def reject(code, detail):
            await JSONResponse({"detail": detail}, status_code=code)(scope, receive, respond)

        path = scope["path"]
        if path not in ("/health/live", "/health/ready"):
            headers = dict(scope["headers"])
            internal = path.startswith("/internal/")
            token = self.settings.internal_key if internal else self.settings.api_key
            expected = ("Bearer " + token.get_secret_value()).encode()
            if not hmac.compare_digest(headers.get(b"authorization", b""), expected):
                return await reject(401, "invalid credentials")
        if self.active >= self.settings.max_concurrent_requests:
            return await reject(429, "node concurrency limit reached; retry with backoff")
        self.active += 1
        try:
            body = bytearray()
            async with asyncio.timeout(self.settings.request_timeout):
                while True:
                    message = await receive()
                    if message["type"] == "http.disconnect":
                        return
                    body.extend(message.get("body", b""))
                    if len(body) > self.settings.max_body_bytes:
                        return await reject(413, "request body too large")
                    if not message.get("more_body", False):
                        break
            delivered = False

            async def replay():
                nonlocal delivered
                if not delivered:
                    delivered = True
                    return {"type": "http.request", "body": bytes(body), "more_body": False}
                return await receive()

            await self.app(scope, replay, respond)
        except TimeoutError:
            await reject(408, "request body deadline exceeded")
        finally:
            self.active -= 1
            self.latency.observe(time.monotonic() - started)


async def distributed_search(client, settings, name, revision, shards, query):
    semaphore = asyncio.Semaphore(settings.max_parallel_shards)

    async def shard_search(shard):
        async with semaphore:
            for worker in worker_order(name, shard, settings.workers):
                try:
                    response = await client.post(
                        f"{worker.rstrip('/')}/internal/collections/{name}/shards/{shard}/search",
                        json={"revision": revision, "query": query.model_dump()},
                        headers={"Authorization": "Bearer " + settings.internal_key.get_secret_value()},
                        timeout=settings.request_timeout / len(settings.workers))
                    if response.status_code != 200:
                        continue
                    result = response.json()
                    if result["revision"] != revision or result["shard"] != shard:
                        continue
                    return result["hits"]
                except (httpx.HTTPError, ValueError, KeyError):
                    continue
            raise HTTPException(503, f"no worker could serve shard {shard}; no partial results returned")

    tasks = [asyncio.create_task(shard_search(s)) for s in range(shards)]
    try:
        async with asyncio.timeout(settings.request_timeout):
            results = await asyncio.gather(*tasks)
    except TimeoutError as exc:
        raise HTTPException(504, "distributed search deadline exceeded") from exc
    finally:
        for task in tasks:
            if not task.done():
                task.cancel()
        await asyncio.gather(*tasks, return_exceptions=True)
    hits = sorted((hit for part in results for hit in part), key=lambda h: (-h["score"], h["id"]))[:query.top_k]
    return {"revision": revision, "shards": shards, "exact": query.exact, "hits": hits}


def create_app(settings=None, store=None, cache=None, client=None):
    settings = settings or Settings()
    owned_store, owned_client = store is None, client is None
    store = store or Store(settings)
    client = client or httpx.AsyncClient(limits=httpx.Limits(max_connections=settings.max_parallel_shards * 2), trust_env=False)
    if settings.role == "worker":
        cache = cache or IndexCache(store, settings)
    registry = CollectorRegistry()

    @asynccontextmanager
    async def lifespan(app):
        yield
        if owned_client:
            await client.aclose()
        if owned_store:
            store.close()

    app = FastAPI(title="AIGVS Distributed", version="0.1.0", lifespan=lifespan,
                  docs_url=None, redoc_url=None)
    app.add_middleware(Guard, settings=settings, registry=registry)

    @app.exception_handler(Conflict)
    async def conflict(request, exc):
        return JSONResponse({"detail": str(exc)}, status_code=409)

    @app.exception_handler(Missing)
    async def missing(request, exc):
        return JSONResponse({"detail": str(exc)}, status_code=404)

    @app.exception_handler(ValueError)
    async def invalid(request, exc):
        return JSONResponse({"detail": str(exc)}, status_code=422)

    @app.exception_handler(Exception)
    async def failed(request, exc):
        log.error("request failed: %s", type(exc).__name__)
        return JSONResponse({"detail": "service unavailable"}, status_code=503)

    @app.get("/health/live")
    def live():
        return {"status": "alive", "role": settings.role}

    @app.get("/health/ready")
    def ready():
        try:
            if store.ready():
                return {"status": "ready"}
        except Exception:
            pass
        raise HTTPException(503, "database/schema unavailable")

    @app.get("/metrics")
    def metrics():
        return Response(generate_latest(registry), media_type=CONTENT_TYPE_LATEST)

    name_path = Path(pattern=r"^[a-zA-Z0-9_-]{1,128}$")
    if settings.role == "gateway":
        @app.put("/v1/collections/{name}")
        def create(config: Collection, name: str = name_path):
            return store.create(name, config)

        @app.get("/v1/collections/{name}")
        def collection(name: str = name_path):
            return store.collection(name)

        @app.post("/v1/collections/{name}/mutations")
        def mutate(mutation: Mutation, name: str = name_path,
                   idempotency_key: str = Header(min_length=1, max_length=128)):
            return store.mutate(name, idempotency_key, mutation)

        @app.post("/v1/collections/{name}/search")
        async def search(query: Search, name: str = name_path):
            row = await asyncio.to_thread(store.collection, name)
            if len(query.vector) != row["config"]["dimension"]:
                raise ValueError("query dimension does not match collection")
            return await distributed_search(client, settings, name, row["revision"], row["config"]["shards"], query)
    else:
        @app.post("/internal/collections/{name}/shards/{shard}/search")
        def shard_search(body: ShardSearch, shard: int = Path(ge=0), name: str = name_path):
            hits = cache.search(name, shard, body.revision, body.query)
            return {"revision": body.revision, "shard": shard, "hits": hits}
    return app
