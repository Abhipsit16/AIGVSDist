# AIGVS Distributed

A distributed service around the existing AIGVS Rust cosine-search engine.
Python gateways handle transactional writes and fan-out; workers build and
search shard indexes; PostgreSQL stores vectors, payloads, collection settings,
idempotency records, and version history. The original local Python pipeline is
still available.

**Status:** implemented foundation with automated correctness tests, not a
production-certified database. Read [the limitations](docs/architecture.md#limits)
before adopting it for a large or write-heavy workload. In particular, writes
are centralized in PostgreSQL and changed indexes are rebuilt on demand.

## Run with Docker

Requires Docker Engine/Desktop with Compose v2.

1. Copy `.env.example` to `.env`.
2. Set `POSTGRES_PASSWORD` and two **different** random keys of at least 32
   characters. Generate each with `python -c "import secrets; print(secrets.token_hex(32))"`.
3. Run `docker compose up --build -d`.
4. Run `docker compose ps` and `docker compose logs migrate gateway`.

This starts a persistent PostgreSQL database, one gateway and three workers.
The API binds only to `127.0.0.1:8000`; workers/database have no published ports.
Database initialization runs before services start. Compose is a development
topology, not a replicated PostgreSQL production installation.

Set `AIGVS_API_KEY` in your terminal to the same key from `.env`, install `httpx`,
then run `python examples/quickstart.py`. Compose reads `.env`; Python does not.

```sh
curl -X PUT http://localhost:8000/v1/collections/docs \
  -H "Authorization: Bearer $AIGVS_API_KEY" -H 'Content-Type: application/json' \
  -d '{"dimension":3,"shards":16}'
curl -X POST http://localhost:8000/v1/collections/docs/mutations \
  -H "Authorization: Bearer $AIGVS_API_KEY" -H 'Content-Type: application/json' \
  -H 'Idempotency-Key: import-001' \
  -d '{"points":[{"id":"hello","embedding":[1,0,0],"text":"Hello","metadata":{"source":"demo"}}]}'
curl -X POST http://localhost:8000/v1/collections/docs/search \
  -H "Authorization: Bearer $AIGVS_API_KEY" -H 'Content-Type: application/json' \
  -d '{"vector":[1,0,0],"top_k":10,"exact":true}'
```

Use `curl.exe` in PowerShell, or the cross-platform Python example. A successful
mutation returns its committed `revision`. An identical retry with the same
idempotency key returns that revision without applying the mutation again.
Different content with the same key returns 409. Delete using the same endpoint
with `{"operation":"delete","ids":["hello"]}` and a fresh key.

## Kubernetes

Build and push your own image, then install the Helm chart with an existing
PostgreSQL primary endpoint and pre-created Secret. See the complete
[deployment and recovery runbook](docs/operations.md). Gateways are Deployments;
workers use StatefulSets for stable DNS, **not** for authoritative disk storage.
Workers can lose all local state without losing committed vectors.

## Documentation

- [Architecture, communication, placement, consistency and limits](docs/architecture.md)
- [API, parameters and sizing](docs/configuration.md)
- [Deployment, scaling, security, backup and recovery](docs/operations.md)
- [Tests performed and remaining validation](docs/validation.md)
- [Original Rust index design](src/ARCHITECTURE.md)

## Development and tests

Python 3.11+, Rust and a C/C++ linker are required. On Windows use a Visual
Studio Developer terminal. Keep the engine's NumPy 1.x ABI until its PyO3/NumPy
bindings are upgraded together.

```sh
python -m venv .venv
# Activate .venv using your shell's activation command.
pip install maturin pytest
maturin develop --release
cargo test --locked
pytest -v
# Set TEST_DATABASE_URL to a disposable PostgreSQL database, then:
pytest -v -m integration
```

The integration tests create uniquely named collections in the test database.
They test concurrent writes, crash-safe transaction boundaries through rollback,
idempotency, historical snapshots, real Rust search across four shards, and an
injected unavailable worker. They do not simulate power loss or certify HA.
CI additionally builds the image and lints/renders the Helm chart.

Existing local JSON data can be read with `LocalJSONStorage.iter_chunks()`
and ingested through mutation batches after translating `chunk_id` to `id`.
Do not point distributed workers at old JSON directories; PostgreSQL is their
source of truth. Collection dimension must match your embeddings.
For a resumable import, stop local writers and run
`python examples/import_local.py ./aigvs_data docs --dimension 768 --shards 16`.
The script uses the API key from your environment and content-derived retry keys.

License: the repository's existing GNU AGPL v3 license is preserved.
