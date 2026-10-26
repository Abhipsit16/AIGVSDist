# Validation record

Validated locally on 2026-10-26, Windows x64, Python 3.12.4, Rust 1.96.0,
PostgreSQL 17.2 and Helm 3.17.3.

| Check | Result |
|---|---|
| `cargo test --locked` (Visual Studio developer environment) | 9 passed |
| `pytest -v` with disposable PostgreSQL and built native extension | 15 passed |
| Four real localhost HTTP processes, worker termination and all-workers-down | Included in passing suite |
| Transaction rollback after database adaptation failure | Included in passing suite |
| Exact distributed result comparison with independent NumPy cosine oracle | Included in passing suite |
| Release wheel build | Windows abi3 wheel built |
| `helm lint deploy/helm/aigvs` | Passed; optional icon suggestion only |
| Helm render and YAML parse | Eight Kubernetes resources rendered and parsed |
| Compose YAML parse | Six services parsed |

Native PyO3 0.20 emits a newer-Rust `non_local_definitions` macro warning.
The installed Starlette test client emits an HTTPX deprecation warning. Neither
failed the test suite; upgrade these dependency interfaces deliberately with tests.

Docker Engine and a Kubernetes cluster were not available in this workspace.
The image was **not** built/run locally, and Kubernetes manifests were **not**
applied to a live cluster. CI includes an image build and Helm validation, but
that workflow has not been executed by this local session.

Not validated: sustained load, recall on a real corpus, cross-zone failures,
database primary failover, power-loss durability, backup restoration, security
penetration testing, Linux container runtime behavior and production SLOs.
These are release acceptance work, not implied by the passing correctness tests.

The temporary database and HTTP processes used for local tests were stopped
after validation. Test artifacts and local tooling are ignored by Git.
