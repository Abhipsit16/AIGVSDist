"""Run after docker compose up: python examples/quickstart.py (httpx required)."""
import os
import uuid
import httpx

client = httpx.Client(base_url=os.getenv("AIGVS_URL", "http://localhost:8000"),
                      headers={"Authorization": "Bearer " + os.environ["AIGVS_API_KEY"]}, timeout=60)

def call(method, path, **kwargs):
    response = client.request(method, path, **kwargs)
    response.raise_for_status()
    return response.json()

print(call("PUT", "/v1/collections/demo", json={"dimension": 3, "shards": 4}))
print(call("POST", "/v1/collections/demo/mutations", headers={"Idempotency-Key": str(uuid.uuid4())},
           json={"points": [{"id": "a", "embedding": [1, 0, 0], "text": "first"},
                            {"id": "b", "embedding": [0, 1, 0], "text": "second"}]}))
print(call("POST", "/v1/collections/demo/search", json={"vector": [1, 0, 0], "exact": True}))
