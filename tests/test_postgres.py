"""Real database transactions. Set TEST_DATABASE_URL to a disposable database."""
import os
import uuid
import asyncio
from concurrent.futures import ThreadPoolExecutor

import pytest

from aigvs.distributed.models import Collection, Mutation
from aigvs.distributed.store import Store, Conflict
from test_distributed import settings

pytestmark = pytest.mark.integration


@pytest.fixture
def db():
    url = os.environ.get("TEST_DATABASE_URL")
    if not url:
        pytest.skip("TEST_DATABASE_URL is not set")
    s = settings()
    s.database_url = type(s.database_url)(url)
    store = Store(s)
    store.migrate()
    yield store
    store.close()


def test_atomic_replay_history_and_concurrent_writers(db):
    name = "test_" + uuid.uuid4().hex
    db.create(name, Collection(dimension=2, shards=1))
    m = Mutation(points=[{"id": "a", "embedding": [1, 0]}])
    with ThreadPoolExecutor(max_workers=4) as pool:
        results = list(pool.map(lambda _: db.mutate(name, "same", m), range(4)))
    assert all(r["revision"] == 1 for r in results)
    assert sum(not r["replayed"] for r in results) == 1
    with pytest.raises(Conflict):
        db.mutate(name, "same", Mutation(operation="delete", ids=["a"]))
    with pytest.raises(ValueError):
        db.mutate(name, "invalid", Mutation(points=[{"id": "b", "embedding": [1, 0]}, {"id": "c", "embedding": [1]}]))
    assert db.collection(name)["revision"] == 1
    assert len(db.snapshot(name, 0, 1, 100)) == 1
    # Force an actual DB adaptation error after the existing version is closed
    # and the first replacement insert is attempted. The whole transaction rolls back.
    from psycopg import DataError
    with pytest.raises(DataError):
        db.mutate(name, "db-failure", Mutation(points=[{"id": "a", "embedding": [0, 1]},
                                                       {"id": "bad\u0000id", "embedding": [1, 0]}]))
    assert db.collection(name)["revision"] == 1
    assert db.snapshot(name, 0, 1, 100)[0]["embedding"] == [1, 0]
    assert db.mutate(name, "update", Mutation(points=[{"id": "a", "embedding": [0, 1]}]))["revision"] == 2
    assert db.snapshot(name, 0, 1, 100)[0]["embedding"] == [1, 0]
    assert db.snapshot(name, 0, 2, 100)[0]["embedding"] == [0, 1]
    db.mutate(name, "delete", Mutation(operation="delete", ids=["a"]))
    assert db.snapshot(name, 0, 3, 100) == []
    assert db.snapshot(name, 0, 2, 100)[0]["id"] == "a"
    with ThreadPoolExecutor(max_workers=4) as pool:
        results = list(pool.map(lambda i: db.mutate(name, f"parallel-{i}", Mutation(points=[{"id": str(i), "embedding": [1, 1]}])), range(8)))
    assert sorted(r["revision"] for r in results) == list(range(4, 12))
    assert len(db.snapshot(name, 0, 11, 100)) == 8


def test_native_distributed_search_with_worker_failure(db):
    import httpx
    import numpy as np
    from aigvs.distributed.index import IndexCache
    from aigvs.distributed.models import Search
    from aigvs.distributed.service import create_app, distributed_search
    # This test requires the actual built extension; never substitute an oracle.
    from aigvs_rust import AIGVSEngine

    name = "native_" + uuid.uuid4().hex
    db.create(name, Collection(dimension=7, shards=4, leaf_capacity=8))
    rng = np.random.default_rng(123)
    vectors = rng.normal(size=(80, 7)).astype(np.float32)
    points = [{"id": str(i), "embedding": v.tolist(), "text": f"point {i}"} for i, v in enumerate(vectors)]
    db.mutate(name, "load", Mutation(points=points))
    s = settings(role="worker", workers=["http://dead", "http://one", "http://two"])
    apps = {host: create_app(s, store=db, cache=IndexCache(db, s, AIGVSEngine)) for host in ["one", "two"]}

    async def handle(request):
        if request.url.host == "dead":
            raise httpx.ConnectError("injected node failure", request=request)
        async with httpx.AsyncClient(transport=httpx.ASGITransport(app=apps[request.url.host]), base_url=str(request.url)) as client:
            return await client.post(str(request.url), content=request.content, headers=request.headers)

    query = Search(vector=rng.normal(size=7).tolist(), top_k=10, exact=True)
    async def run(revision):
        async with httpx.AsyncClient(transport=httpx.MockTransport(handle)) as client:
            return await distributed_search(client, s, name, revision, 4, query)

    result = asyncio.run(run(1))
    q = np.asarray(query.vector)
    scores = (vectors / np.linalg.norm(vectors, axis=1, keepdims=True)) @ (q / np.linalg.norm(q))
    expected = np.argsort(-scores)[:10]
    assert [h["id"] for h in result["hits"]] == [str(i) for i in expected]
    np.testing.assert_allclose([h["score"] for h in result["hits"]], scores[expected], atol=1e-6)
    deleted = result["hits"][0]["id"]
    db.mutate(name, "delete", Mutation(operation="delete", ids=[deleted]))
    assert deleted not in [h["id"] for h in asyncio.run(run(2))["hits"]]
    assert asyncio.run(run(1))["hits"] == result["hits"]
