# API and configuration reference

All application endpoints require `Authorization: Bearer <AIGVS_API_KEY>`.
Internal search requires the separate internal key. Health endpoints are public.
`GET /openapi.json` provides the authenticated machine-readable schema.
Unknown request fields are rejected. API paths use collection names matching
`[A-Za-z0-9_-]{1,128}`; IDs are nonempty strings up to 512 characters.

| Method / path | Body / behavior |
|---|---|
| PUT `/v1/collections/{name}` | Collection settings below; same settings are idempotent; different settings return 409 |
| GET `/v1/collections/{name}` | Settings and latest committed revision |
| POST `/v1/collections/{name}/mutations` | `operation=upsert` + `points`, or `operation=delete` + `ids`; required `Idempotency-Key` header, 1–128 characters |
| POST `/v1/collections/{name}/search` | Search options below; returns revision, shard count, exact flag and hits |
| GET `/metrics` | Authenticated Prometheus response counts by status and request-duration histogram |
| GET `/health/live` | Process responds |
| GET `/health/ready` | Database reachable and expected schema present |

Mutation batches contain 1–1000 unique IDs. Each point is `{id, embedding,
text?, metadata?}`. Embeddings must be finite, nonzero float32-representable
vectors of the collection dimension. Metadata is a JSON object. Text defaults
to empty; metadata defaults to `{}`. Empty batches and mixed upsert/delete
fields are rejected. Errors never intentionally include database credentials.

## Collection settings (immutable)

| Parameter | Default | Range / meaning |
|---|---|---|
| `dimension` | required | 1–65536; model embedding dimension |
| `shards` | 16 | 1–256 logical shards; choose before ingestion |
| `leaf_capacity` | 128 | 1–65536; target tree leaf size |
| `branching_factor` | 8 | 2–32 children |
| `beta` | 16 | >0 to 100; association sharpness |
| `max_tree_depth` | null | 0–64 or unlimited; 0 puts all rows in a terminal partition |
| `leaf_backend` | auto | auto, flat, hnsw |
| `hnsw_threshold` | 512 | >=2; graph build threshold in auto mode |
| `hnsw_degree` | 16 | 2–64 |
| `ef_construction` | 128 | 2–4096, at least twice degree |

## Per-query controls

| Parameter | Default | Range / meaning |
|---|---|---|
| `vector` | required | Same validation as point embeddings |
| `top_k` | 10 | 1–1000, global result limit |
| `candidate_budget` | 4096 | 1–1000000 **per shard**; target candidate count |
| `beam_width` | 64 | 1–65536 tree routes per shard |
| `leaf_ef` | 64 | 0–1000000; 0 scans selected leaves exactly |
| `exact` | false | True scans every live vector in every shard |
| `min_score` | -1 | -1 to 1; cosine-score cutoff |

Increase beam/candidate budget/leaf ef to explore recall vs latency. Only
`exact=true` guarantees exhaustive search. `leaf_ef=0` alone does not visit
pruned tree leaves. There is no universal recall guarantee for ANN settings.
Choose settings using held-out real queries, exact results and latency percentiles.

## Environment (prefix `AIGVS_`)

| Variable suffix | Default | Purpose |
|---|---|---|
| `ROLE` | gateway | gateway or worker |
| `DATABASE_URL` | required | PostgreSQL primary DSN; use sslmode=verify-full and trusted roots in production |
| `API_KEY` | required | Public API bearer key, >=32 characters |
| `INTERNAL_KEY` | required | Different peer key, >=32 characters |
| `WORKERS` | `["http://worker:8000"]` | JSON array of unique HTTP(S) origins; stable node URLs preferred |
| `DB_POOL_SIZE` | 8 | Maximum connections per process, 1–128 |
| `REQUEST_TIMEOUT` | 30 | Seconds, >0 to 300; body-read deadline, pool wait and overall fan-out deadline |
| `MAX_BODY_BYTES` | 16777216 | Request body cap; >=1024 |
| `MAX_CONCURRENT_REQUESTS` | 32 | Per-process admitted requests, 1–1024; overflow returns 429 |
| `MAX_PARALLEL_SHARDS` | 16 | Per-query fan-out concurrency, 1–256 |
| `CACHE_SHARDS` | 32 | Maximum cached shard entries per worker; LRU, >=1 |
| `MAX_SHARD_POINTS` | 1000000 | Maximum rows loaded for one shard |
| `MAX_SHARD_VECTOR_BYTES` | 536870912 | Maximum raw float32 bytes per shard; excludes payloads/graphs/build overhead |

`RAYON_NUM_THREADS` configures native search/build parallelism; deployments set
2. Use one service process per container. Each process owns its pool and cache;
starting many Uvicorn workers multiplies memory and database connections.

SQL statement timeout is 30 seconds, lock timeout 10 seconds, connect timeout
5 seconds. Native builds are not interruptible; the gateway deadline can expire
while a worker completes a build. Do not interpret `REQUEST_TIMEOUT` as a hard
CPU execution deadline. Peer per-attempt timeout is overall timeout divided by
worker count; overall deadline also includes waiting for the fan-out semaphore.

## Sizing

Raw vector RAM per shard is approximately `4 * N * D / S` bytes for N live
points, dimension D and S evenly populated shards. It is a lower bound: allow
headroom for native rebuild copies, Python database rows, payloads, graphs and
concurrent readers. Eight 64 MiB cached shards already require at least 512 MiB
of vector RAM before overhead. Helm defaults are intentionally smaller than
standalone cache/row limits. Load-test before raising them.

All gateway/worker connections go to PostgreSQL. Budget approximately
`(gateways + workers) * DB_POOL_SIZE` plus migration/admin connections.
Cold starts and membership changes cause a rebuild burst. Scale gradually and
warm representative queries before directing peak traffic.
