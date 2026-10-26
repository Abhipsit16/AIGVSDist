# Deployment and operations

## Production prerequisites

Provide a supported PostgreSQL primary with backups, monitored storage,
encrypted connections, and a tested HA failover procedure. Configure synchronous
replication if the required recovery point excludes loss of acknowledged writes
on primary loss. This application does not configure or guarantee that policy.
Compose's single database is for local evaluation.

Use a dedicated database and role. Migration requires DDL privileges; runtime
needs SELECT/INSERT/UPDATE on application tables. For stricter deployments run
`aigvs migrate` with an elevated migration DSN separately, then give runtime a
restricted DSN. The initial Helm chart uses one Secret/DSN for both.

Use a TLS ingress/load balancer for public requests, and mTLS service mesh or
HTTPS worker origins for inter-node encryption. The included chart exposes only
ClusterIP services; its internal URLs use HTTP and need mesh encryption for a
production encrypted topology. Do not expose worker services to the internet.

## Docker lifecycle

```sh
docker compose up --build -d
docker compose logs -f gateway
docker compose stop worker-1    # exercise fallback, then run a search
docker compose start worker-1
docker compose down            # preserves named database volume
```

`docker compose down -v` destroys database volumes: use only for disposable data.
Do not share a database volume between independent PostgreSQL processes.

## Helm installation

Build/push an image under your own registry; example commands use placeholders.
Pin your release image and base images to verified digests in your release
pipeline. The source chart accepts repository/tag and never pulls a hypothetical
prebuilt AIGVS image from an unverified registry.

```sh
docker build -t YOUR_REGISTRY/aigvs:0.1.0 .
docker push YOUR_REGISTRY/aigvs:0.1.0
kubectl create namespace aigvs
```

Create a Secret named `aigvs-secrets` in that namespace using your secret manager.
Required keys:

| Secret key | Content |
|---|---|
| `database-url` | `postgresql://USER:PASSWORD@PRIMARY:5432/aigvs?sslmode=verify-full` (URL-escape credentials) |
| `api-key` | Random public API key of at least 32 characters |
| `internal-key` | Different random peer key of at least 32 characters |

Mount the database CA and configure libpq's trusted root path when your provider
requires it. The chart is a starting point; private CA mounts and ingress are
infrastructure-specific and are not created automatically.

```sh
helm lint deploy/helm/aigvs
helm template aigvs deploy/helm/aigvs --namespace aigvs
helm upgrade --install aigvs deploy/helm/aigvs --namespace aigvs \
  --set image.repository=YOUR_REGISTRY/aigvs --set image.tag=0.1.0 \
  --wait --timeout 10m
kubectl -n aigvs port-forward service/aigvs-gateway 8000:8000
```

A pre-install/pre-upgrade Job runs the idempotent schema initialization under a
database advisory lock. It requires the Secret and database to exist already.
Failed migration prevents installation. Schema v1 is additive; future incompatible
migrations must have explicit upgrade/rollback tooling before deployment.

The chart provides two gateways, three worker DNS identities, resource requests
and limits, non-root/read-only containers, capability dropping, startup/readiness/
liveness probes, disruption budgets, and preferred node spreading. Worker ingress
NetworkPolicy allows gateway pods only; it requires an enforcing CNI. Egress is
not restricted by this chart: apply cluster policy for DNS, database and approved
destinations. Worker metrics scraping needs an additional permitted monitor rule.

## Scaling and rolling upgrades

Gateways can scale independently; keep each process's DB pool small. Change worker
count through Helm so gateway environment contains the new stable DNS list:

```sh
helm upgrade aigvs deploy/helm/aigvs -n aigvs --reuse-values \
  --set gateway.replicas=3 --set worker.replicas=5 --wait
```

No worker PVC is needed. Its StatefulSet exists for stable rendezvous identity.
Keep at least two healthy workers while upgrading. Preferred node spreading
does not enforce multi-zone redundancy; set placement policy for your cluster's
failure domains. PDB protects voluntary evictions, not every rollout/failure.
Rollouts may temporarily use different membership lists without changing data
correctness, but cold-cache retries can increase latency. Warm representative
queries before declaring the deployment healthy under load.

Changing shards/dimension/index-build settings requires a new collection. For
safe cutover: stop writes, export current live records, ingest into the new
collection, compare counts and exact queries, switch clients, then resume writes.
This release does not automate export/cutover or provide atomic collection aliases.

## Backups and recovery

Back up PostgreSQL, not worker caches. Use your provider's PITR/WAL archiving and
regular restore drills. A simple development logical backup is:

```sh
docker compose exec -T postgres pg_dump -U aigvs -d aigvs -Fc > aigvs.dump
```

Use a shell with binary-safe redirection (POSIX shell or PowerShell 7.4+); otherwise
write the dump inside the container and copy it out with `docker compose cp`.
Backups contain embeddings, text, metadata and history; encrypt and control access.

To restore: stop gateways/writers, restore the dump into a **fresh** PostgreSQL
database using `pg_restore`, point the deployment at its primary endpoint, restart
all workers to invalidate cached revisions, run readiness and representative exact
queries, then resume traffic. Never restore behind live worker caches: revision
numbers can be reused after a restore. Record RPO/RTO from a real drill.

## Monitoring and incident handling

Scrape authenticated `/metrics`: `aigvs_http_requests_total{status=...}` and
`aigvs_http_duration_seconds`. Credentials belong in Prometheus secret configuration,
not query strings. Use external PostgreSQL/container monitoring for disk/WAL growth,
connections, CPU, RSS, OOM kills and restarts. Metrics do not currently expose
per-shard lag/build time or distributed traces. Do not use collection/point IDs as
unbounded metric labels.

Alert on sustained 5xx/429, increased p95 search latency, unsuccessful synthetic
searches, database storage pressure, and lost backup freshness. Thresholds need
workload baselines; no universal alert thresholds are shipped.

| Symptom | Investigation / response |
|---|---|
| 401 | Verify API vs internal key and coordinated deployment |
| 409 on retry | Caller reused an idempotency key for changed content |
| 413 | Split batches below body/row limit |
| 422 | Inspect vector dimensions, zero/nonfinite inputs, immutable settings |
| 429 | Back off with jitter; examine CPU/DB pool and concurrency sizing |
| 503 | Check DB/schema, worker DNS, keys, readiness, capacity errors in worker responses |
| 504 | Cold builds or slow workers; measure shard size, reduce load, adjust deadline only with capacity evidence |
| Increasing database disk | Historical versions/idempotency records retain all changes; plan maintenance migration |
| OOM | Reduce cache/shard limits or increase resources/shards; account for build/payload overhead |

Logs use Uvicorn access records and exception class names without request bodies
or DSNs. Responses carry a generated `X-Request-ID`; full cross-node correlation
and tracing are future work. Single-key rotation requires a coordinated rollout;
dual-key zero-downtime rotation is not implemented.

## Release acceptance

Before production approval, run the full native/service/database test suites,
container smoke tests, Helm validation against your Kubernetes version, a load
test on realistic embeddings/payloads, worker-kill and database-failover drills,
and backup restoration. Measure recall against exact search, not against routing
weights. Validate TLS/mesh and resource policies in the target cluster.
