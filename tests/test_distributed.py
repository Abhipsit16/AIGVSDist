import asyncio
from types import SimpleNamespace

import httpx
import numpy as np
import pytest
from fastapi import HTTPException
from fastapi.testclient import TestClient
from pydantic import ValidationError

from aigvs.distributed.config import Settings
from aigvs.distributed.index import IndexCache
from aigvs.distributed.models import Collection, Point, Search, shard_for, worker_order
from aigvs.distributed.service import create_app, distributed_search


def settings(**overrides):
    return Settings(database_url="postgresql://unused", api_key="a" * 32,
                    internal_key="b" * 32, **overrides)


@pytest.mark.parametrize("vector", [[0, 0], [float("nan"), 1], [float("inf")], [1e39], [1e-50]])
def test_invalid_vectors(vector):
    with pytest.raises(ValidationError):
        Point(id="a", embedding=vector)


def test_config_validation():
    with pytest.raises(ValidationError):
        Collection(dimension=3, hnsw_degree=64, ef_construction=32)
    with pytest.raises(ValidationError):
        settings(workers=[])


def test_stable_placement():
    assert shard_for("hello", 16) == 14
    workers = ["http://a", "http://b", "http://c"]
    order = worker_order("docs", 1, workers)
    assert order == worker_order("docs", 1, workers[::-1])
    assert worker_order("docs", 1, [w for w in workers if w != order[0]]) == order[1:]


def test_fanout_fallback_merge_and_revision():
    seen = []
    async def handle(request):
        import json
        body = json.loads(request.content)
        shard = int(request.url.path.split("/")[-2])
        seen.append(body["revision"])
        if request.url.host == "bad":
            return httpx.Response(503)
        return httpx.Response(200, json={"revision": body["revision"], "shard": shard,
            "hits": [{"id": str(shard), "score": shard / 10, "text": "", "metadata": {}}]})
    async def run():
        async with httpx.AsyncClient(transport=httpx.MockTransport(handle)) as client:
            return await distributed_search(client, settings(workers=["http://bad", "http://good"]),
                                            "docs", 7, 4, Search(vector=[1], top_k=2))
    result = asyncio.run(run())
    assert [h["id"] for h in result["hits"]] == ["3", "2"]
    assert set(seen) == {7}


def test_fail_closed_and_deadline():
    async def run(slow=False):
        async def handle(request):
            if slow:
                await asyncio.sleep(1)
            return httpx.Response(200, json={"revision": 999, "shard": 0, "hits": []})
        async with httpx.AsyncClient(transport=httpx.MockTransport(handle)) as client:
            await distributed_search(client, settings(request_timeout=0.02), "docs", 1, 1, Search(vector=[1]))
    with pytest.raises(HTTPException) as exc:
        asyncio.run(run())
    assert exc.value.status_code == 503
    with pytest.raises(HTTPException) as exc:
        asyncio.run(run(True))
    assert exc.value.status_code == 504


class FakeStore:
    def ready(self):
        return True

    def collection(self, name):
        return {"revision": 2, "config": Collection(dimension=2, shards=1).model_dump()}

    def snapshot(self, name, shard, revision, limit):
        return [{"id": "a", "embedding": [1, 0] if revision == 1 else [0, 1],
                 "payload": {"text": str(revision), "metadata": {}}}]


class OracleEngine:
    """Independent exact oracle to test cache invalidation, not native ANN."""
    def __init__(self, dim, **kwargs):
        self.ids, self.vectors = [], np.empty((0, dim))

    def native_batch_insert(self, ids, vectors):
        self.ids, self.vectors = ids, vectors / np.linalg.norm(vectors, axis=1, keepdims=True)

    def search(self, vector, top_k, **kwargs):
        scores = self.vectors @ (vector / np.linalg.norm(vector))
        return sorted(zip(self.ids, scores.tolist()), key=lambda h: -h[1])[:top_k]


def test_cache_is_versioned_and_old_snapshot_remains_readable():
    cache = IndexCache(FakeStore(), settings(cache_shards=1), OracleEngine)
    query = Search(vector=[1, 0])
    assert cache.search("docs", 0, 1, query)[0]["score"] == 1
    assert cache.search("docs", 0, 2, query)[0]["score"] == 0
    assert cache.search("docs", 0, 1, query)[0]["text"] == "1"
    assert len(cache.entries) == 1


def test_auth_roles_limits_and_health():
    app = create_app(settings(max_body_bytes=1024), store=FakeStore())
    with TestClient(app) as client:
        assert client.get("/health/live").status_code == 200
        assert client.get("/health/ready").status_code == 200
        assert client.get("/v1/collections/docs").status_code == 401
        auth = {"Authorization": "Bearer " + "a" * 32}
        assert client.get("/v1/collections/docs", headers=auth).status_code == 200
        assert client.get("/metrics", headers=auth).status_code == 200
        assert client.post("/v1/collections/docs/search", headers=auth, content="x" * 1025).status_code == 413
        assert client.get("/internal/anything", headers=auth).status_code == 401
        response = client.post("/v1/collections/docs/search", headers=auth, json={"vector": [1]})
        assert response.status_code == 422
        assert "x-request-id" in response.headers


def test_worker_rejects_public_key():
    s = settings(role="worker")
    app = create_app(s, store=FakeStore(), cache=IndexCache(FakeStore(), s, OracleEngine))
    with TestClient(app) as client:
        path = "/internal/collections/docs/shards/0/search"
        body = {"revision": 1, "query": {"vector": [1, 0]}}
        assert client.post(path, json=body, headers={"Authorization": "Bearer " + "a" * 32}).status_code == 401
        assert client.post(path, json=body, headers={"Authorization": "Bearer " + "b" * 32}).json()["hits"][0]["id"] == "a"
