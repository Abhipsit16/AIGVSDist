"""Real HTTP processes: native engine, PostgreSQL, worker death and fail-closed."""
import os
import json
import socket
import subprocess
import sys
import time
import uuid
from pathlib import Path

import httpx
import pytest

pytestmark = pytest.mark.integration


def test_http_cluster_survives_worker_loss(tmp_path):
    url = os.getenv("TEST_DATABASE_URL")
    if not url:
        pytest.skip("TEST_DATABASE_URL is not set")
    from aigvs_rust import AIGVSEngine  # fail explicitly if extension is missing
    from aigvs.distributed.store import Store
    from test_distributed import settings

    config = settings()
    config.database_url = type(config.database_url)(url)
    db = Store(config)
    db.migrate()
    db.close()

    ports = []
    reservations = []
    for _ in range(4):
        sock = socket.socket()
        sock.bind(("127.0.0.1", 0))
        ports.append(sock.getsockname()[1])
        reservations.append(sock)
    env = os.environ.copy()
    env.update(AIGVS_DATABASE_URL=url, AIGVS_API_KEY="a" * 32, AIGVS_INTERNAL_KEY="b" * 32,
               AIGVS_WORKERS=json.dumps([f"http://127.0.0.1:{p}" for p in ports[:3]]),
               PYTHONPATH=str(Path(__file__).resolve().parents[1] / "src"),
               AIGVS_REQUEST_TIMEOUT="3", RAYON_NUM_THREADS="1")
    processes, logs = [], []
    try:
        for i, port in enumerate(ports):
            reservations[i].close()
            log = open(tmp_path / f"node-{i}.log", "w")
            logs.append(log)
            process = subprocess.Popen(
                [sys.executable, "-m", "aigvs.distributed.cli", "serve", "--host", "127.0.0.1", "--port", str(port)],
                env={**env, "AIGVS_ROLE": "worker" if i < 3 else "gateway"}, stdout=log, stderr=log,
                creationflags=subprocess.CREATE_NO_WINDOW if os.name == "nt" else 0)
            processes.append(process)
        with httpx.Client(timeout=10, trust_env=False) as client:
            for port in ports:
                deadline = time.monotonic() + 20
                while True:
                    try:
                        if client.get(f"http://127.0.0.1:{port}/health/ready").status_code == 200:
                            break
                    except httpx.HTTPError:
                        pass
                    if time.monotonic() > deadline:
                        pytest.fail(f"node on {port} failed startup; logs: {tmp_path}")
                    time.sleep(0.1)
            base = f"http://127.0.0.1:{ports[3]}/v1/collections/http_{uuid.uuid4().hex}"
            headers = {"Authorization": "Bearer " + "a" * 32}
            assert client.put(base, headers=headers, json={"dimension": 2, "shards": 8}).status_code == 200
            response = client.post(base + "/mutations", headers={**headers, "Idempotency-Key": "load"},
                                   json={"points": [{"id": "a", "embedding": [1, 0]}, {"id": "b", "embedding": [0, 1]}]})
            assert response.status_code == 200, response.text
            body = {"vector": [1, 0], "exact": True}
            expected = client.post(base + "/search", headers=headers, json=body)
            assert expected.status_code == 200, expected.text
            processes[0].terminate()
            processes[0].wait(timeout=10)
            result = client.post(base + "/search", headers=headers, json=body)
            assert result.status_code == 200, result.text
            assert result.json() == expected.json()
            for p in processes[1:3]:
                p.terminate()
                p.wait(timeout=10)
            unavailable = client.post(base + "/search", headers=headers, json=body)
            assert unavailable.status_code in (503, 504)
            assert "hits" not in unavailable.json()
    finally:
        for sock in reservations:
            sock.close()
        for process in processes:
            if process.poll() is None:
                process.terminate()
                try:
                    process.wait(timeout=10)
                except subprocess.TimeoutExpired:
                    process.kill()
                    process.wait(timeout=5)
        for log in logs:
            log.close()
