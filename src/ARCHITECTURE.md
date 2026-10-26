# AIGVS associative-memory tree

The production index is in `index.rs`; `hnsw.rs` supplies optional graph leaves;
`math.rs` contains normalized-dot-product kernels; `lib.rs` owns the Python boundary. Python handles payload storage and
result assembly. There is one tree implementation, in Rust. The disconnected
Python tree and unused Rust backup have been removed. `aigvs/engine/router.py`
is a small reference for local association weights, not a second index.

## Model and its limits

For a nonzero embedding x, store x / ||x||. All results use cosine similarity.
At an internal node, normalized child centroids are memories m_1,...,m_f. A
modern Hopfield energy for a continuous state z is, up to constants,

    E(z) = 0.5 ||z||² - log(sum_j exp(beta m_jᵀz)) / beta.

Its memory association weights are

    p_j(q) = exp(beta m_jᵀq) / sum_i exp(beta m_iᵀq).

The usual memory readout is sum_j p_j(q) m_j. Here we use the weights to select
child addresses, rather than replace the original query with that readout.
This is hierarchical routing using modern Hopfield associations, not a claim
that the tree inherits a convergence or retrieval guarantee for all its leaves.
See [Ramsauer et al., Hopfield Networks Is All You Need](https://arxiv.org/abs/2008.02217).

The log weight of a path is the sum of its local log association weights.
Stable log-sum-exp avoids overflow; f64 path arithmetic avoids underflow along
deep paths. These are routing weights, **not calibrated probabilities of
finding the true nearest neighbor**. There is no inverse-variance temperature
heuristic: beta is bounded and explicit (default 16).

At each level, expand all surviving routes and keep the B strongest complete
paths seen at that level. Ambiguous queries therefore retain multiple children.
A terminal leaf is carried forward unchanged if sibling paths are deeper.
After routing, search selected leaves using the original normalized query and
merge their top-k results. Small leaves are scanned exhaustively; large leaves
can use HNSW. Every returned score is an actual vector cosine, not a routing
weight. No binary hash discards candidates first.

In a binary node, softmax ordering equals centroid-dot-product ordering. The
Hopfield formulation alone is not a speed breakthrough. The speed mechanism
is bounded tree traversal and contiguous candidate scoring. Finite beams can
miss neighbors at partition boundaries. High-dimensional unstructured data
may require scanning a large fraction of the index for high recall. FAISS is
a collection of exact and approximate indexes, not a single complexity bound;
universal superiority over FAISS is neither proved nor asserted here.

## Tree construction

1. Validate and normalize every input row before mutating the index.
2. Seed two spherical clusters with a deterministic farthest-point heuristic.
3. Run four balanced refinement passes. Partition by the median of
   xᵀ(m_left - m_right), then recompute normalized means.
4. Split the largest group repeatedly until there are up to f children, then
   recurse on groups above leaf capacity L.
5. Pack vectors and IDs in leaf order. Leaves store contiguous row ranges;
   centroid vectors and tree nodes live in flat arenas.
6. Optionally build independent HNSW graphs over large terminal partitions.
   Graphs reference the existing vector arena; embeddings are not duplicated.

## Hybrid terminal partitions

`max_tree_depth` can stop division after a chosen number of tree levels (root
depth is zero). This allows an associative-memory tree to handle coarse
partitioning and HNSW to navigate the larger terminal partitions. It is unset
by default; the default 128-row leaves remain flat.

`leaf_backend="auto"` builds HNSW for leaves at or above `hnsw_threshold`
(default 512); smaller leaves use contiguous scans. The threshold is a starting
heuristic, not a measured universal crossover. `"flat"` disables graph creation;
`"hnsw"` requests graphs for all nontrivial leaves. `hnsw_degree` defaults to
16 (maximum base-layer degree 32); `ef_construction` defaults to 128.

The native implementation follows the multi-level graph approach of
[Malkov and Yashunin](https://arxiv.org/abs/1603.09320): deterministic geometric
level assignment, greedy upper-layer descent, bounded best-first base-layer
search, reciprocal edge insertion, and diversity-based neighbor pruning.
Graphs are immutable between rebuilds and can be built independently in parallel.
They use no external graph library. This new implementation needs production
workload evaluation; it does not inherit the performance maturity of hnswlib.

`leaf_ef` controls graph search breadth (default 64, widened to at least k).
`leaf_ef=0` forces exact scans **within selected leaves** without rebuilding.
If ef reaches the leaf size, a scan is used directly. A graph that cannot fill
local top-k falls back to a leaf scan. `exact=True` bypasses all tree and graph
approximation and scans the entire index. A full beam is exhaustive only with
flat leaf search or sufficient ef, not automatically when graph leaves exist.

Calibration compares `leaf_ef=0` with graph ef values on the same partitions.
It can choose flat scans even when a graph has been built. This makes the
flat/HNSW speed decision empirical while keeping the requested validation
quality target. Tree depth/leaf size are build-time choices: use the benchmark
to compare them; query calibration does not reconstruct trees automatically.

For example:

```python
hybrid = AIGVSPipeline(768, max_tree_depth=2, leaf_capacity=128,
                      leaf_backend="auto", hnsw_threshold=512)
hybrid.ingest_batches(records)
report = hybrid.calibrate(validation_queries, target_recall=0.99,
                          leaf_efs=(0, 32, 64, 128, 256))
```

Median partitions guarantee strict progress, including identical vectors and
antipodal data. Small groups stop early; no empty child is created. Centroid
accumulation and normalization use f64. Exact cancellation falls back to a
member vector. The split is balanced spherical two-means, not unconstrained
k-means: the balance constraint trades some local cluster quality for bounded
depth. Construction is deterministic on the same numerical backend and input
order. Equal-score result ties use packed row order, which may change on rebuild.

## Work and memory bounds

Let N be indexed rows, D dimension, f fanout, L actual maximum leaf size, H tree depth,
B effective beam width, R requested candidate budget, C actual tree candidates,
delta pending rows, and k returned neighbors.

- Tree construction: O(N D log(N/L)) for fixed refinement count and fanout.
  Graph construction adds separate work depending on degree, ef_construction,
  geometry and leaf size; its cost must be measured.
- Tree routing: O(B f D H), plus frontier selection and final beam ordering.
- Flat-leaf work: O((C + delta) D + (C + delta) log k).
  HNSW leaves replace their full scans with graph distance evaluations, queue
  operations, and query-local visited bitsets initialized in O(leaf_size/64).
  Graph search is approximate and has no universal logarithmic worst-case bound.
- A typical balanced depth is about log_f(N/L). The robust guarantee is
  logarithmic depth from repeated median splits, including non-power-of-two f.
- C <= min(N, B L, max(R,k) + L - 1): only whole leaves are scanned. The
  candidate budget is a target capped by surviving beam leaves, not a promise
  to inspect R rows. Increase both beam and budget for higher recall.
- The beam automatically widens to at least ceil(min(k,N+delta)/minimum_leaf_size)
  so retrieval can fill k even when configured leaves are tiny. It is capped
  at N (or one for an empty tree).
- Resident vector storage: 4(N+delta)D bytes, plus centroids O(ND/L), node
  metadata, IDs and their uniqueness set. No D-by-D weight matrices or
  1024-bit projections are stored.
- HNSW adds adjacency storage proportional to indexed graph rows times degree
  in expectation, plus per-node layer metadata. Depth caps can make terminal
  partitions larger than configured leaf_capacity; L above is their actual size.
- Rebuild temporarily holds the old and packed vector arrays and a row
  permutation. Python ingestion also owns the input batch and a boundary copy.

These are operation bounds, not latency predictions. A fixed beam makes search
sublinear on a packed tree, but offers no fixed-recall guarantee as N grows.
`exact=True` deliberately scans all N+delta rows and guarantees exhaustive
ranking subject to floating-point rounding. Full beam plus full candidate
budget visits all leaves; use leaf_ef=0 for exhaustive leaf scoring. Enlarging an approximate beam is not a formal
monotonic-recall guarantee, since the set of surviving paths can change.

## Execution and micro-optimizations

- Runtime-selected AVX2/FMA on x86-64; scalar fallback elsewhere. Two independent
  SIMD accumulators reduce dependency-chain stalls. Every remainder dimension
  is included; loads never cross a vector boundary.
- Normalize once on insertion and once per query. f64 normalization supports
  every finite nonzero f32 input, including tiny values and very large norms.
- Keep softmax scratch on the stack (fanout <= 32), reuse frontier buffers,
  and use partial selection instead of fully sorting each frontier.
- Contiguous leaf reads; bounded top-k heaps with the worst result at the root.
- Query batches use Rayon across queries. Large multi-leaf candidate scans
  (at least 4096 rows and 1,048,576 scalar coordinates) use Rayon across groups
  of leaves with local top-k heaps and an exact merge. Each task gets roughly
  at least 2048 rows. Small scans, single-worker pools, and searches already
  running inside batch workers stay serial to avoid scheduling overhead.
- The GIL is released for building and searching. Arrays are copied before
  release to prevent races with writable NumPy aliases; strided arrays work.
  Multiple read-only native searches may run concurrently. Do not mutate the
  same engine concurrently with a search. The Python payload cache and stats
  are not a transactional concurrent service.

## Ingestion and persistence

`native_batch_insert(..., rebuild=True)` automatically rebuilds the first tree
and subsequently when the delta reaches `rebuild_threshold` (default 4096).
Every query scans the delta, so new rows are immediately searchable. Inserting
with `rebuild=False` intentionally leaves the delta unbounded until `rebuild()`.
Before the first build, searches are flat scans.

`pipeline.ingest_batches(records)` defers construction across its batches and
rebuilds once at the end. For many streaming calls, use `rebuild=False` and
schedule explicit rebuilds. Rebuilding the entire tree after every small write
is costly; this version targets bulk ingestion/read-heavy workloads. Rebuilds
are synchronous. An online service needing continuous writes should add
immutable segment snapshots and background compaction rather than repeatedly
rebuilding one enormous tree.

IDs are insert-only and globally unique within an engine; duplicate IDs are
errors, not silent overwrites. There is no delete/upsert API. A native batch is
validated before index mutation. Pipeline batches persist records first;
completed batches remain committed if a later batch fails. A disk error can
leave some newly written records without a corresponding in-memory insertion.
Storage provides atomic replacement per record, not a multi-record transaction
or crash-durable WAL. Do not share a storage directory between independent
writers or ingest into an existing store without restoring its index first.

LocalJSONStorage uses SHA-256 filenames in `chunks/`, avoiding collisions from
filename sanitization. Cache misses call `read_chunk`, and the LRU cache budget
counts serialized payload bytes (not total Python RSS). Zero disables caching.
Embeddings are persisted with payloads. `pipeline.restore()` rebuilds an empty
index from these records; there is no native tree snapshot format yet.
Old top-level files from the previous sanitized-name format are not migrated
automatically. Re-ingest their original data into a new directory.

## Python API

```python
from aigvs.pipeline import AIGVSPipeline

index = AIGVSPipeline(embedding_dim=768, leaf_capacity=128,
                      branching_factor=8, beta=16, beam_width=64)
index.ingest_batches(records)  # id, text, embedding, optional metadata
hits = index.query(query, top_k=10, rerank_k=4096, beam_width=64)
truth = index.query(query, top_k=10, exact=True)
# Each result contains id, text, metadata and an actual cosine score.

# Calibrate after growth, using a representative validation set.
report = index.calibrate(validation_queries, target_recall=0.99, top_k=10)
# Optionally provide relevant_ids (one set of document IDs per query)
# and min_mrr=0.95 to constrain semantic MRR as well.
# Subsequent queries use report["selected"], unless explicitly overridden.
```

Defaults favor recall: a 64-route beam and a 4096-vector candidate target.
These are starting parameters, not a promised recall level. For MRR, evaluate
on query/document relevance judgments; nearest-neighbor recall alone cannot
measure semantic ranking quality. Exact cosine reranking removes routing-score
distortion among visited candidates, but it cannot recover a pruned neighbor or
repair a poor embedding model. A larger corpus does not intrinsically improve
MRR; track relevance metrics and search fidelity separately as N increases.

`calibrate()` sweeps beam/budget/leaf-ef combinations plus an exhaustive candidate,
including a full-index coverage budget for shallow hybrids, and measures serial
latency over repeated interleaved trials. It selects the
lowest measured p50 among profiles meeting the validation recall target and
optional relevance MRR target. Its report explicitly identifies an exhaustive
fallback; no feasible profile raises an error without changing defaults.
This is empirical calibration, not a per-query quality certificate. Use a
separate test set for the final accuracy report. Stats flag a profile as stale
after the corpus size changes; recalibrate after growth, rebuilding or query
distribution changes. A profile calibrated at one k need not meet the same
recall target at a different k. Approximate cosine ties can also change top-k
identity; the current calibration uses the exhaustive engine's tie ordering.

The native engine also exposes `batch_search`, `search_diagnostics` (hits,
vector distance evaluations, internal nodes expanded), `size`, `pending_count`,
`node_count`, and `graph_count`. Graph diagnostics count repeated evaluations
across layers, not unique rows. `unified_search` retains the old IDs-only interface.

`min_score` now defaults to -1 (all cosine scores) and is actually applied.
Legacy projection constructor arguments are accepted for source compatibility
but do not affect retrieval. Learned hash training now fails explicitly rather
than silently replacing a populated index with an untrained empty one.

## Validation and benchmarking

Rebuild the extension after editing Rust; an old installed `.pyd` will not expose
the new API. The repository's existing PyO3/numpy 0.20 and Rayon dependencies
are retained. Run from the repository root:

```text
cargo test
maturin develop --release
```

With `src` on PYTHONPATH, run from `src`:

```text
python -m unittest discover -s tests -v
python benchmarks/recall.py --vectors corpus.npy --queries heldout_queries.npy --faiss
python benchmarks/recall.py --vectors corpus.npy --queries heldout_queries.npy --max-tree-depth 2 --beams 8 32 64 --budgets 4096 16384 --leaf-efs 0 32 128
```

The tests check independent exact cosine ground truth, exhaustive beam equality,
ambiguous routes, atomic native validation, non-contiguous arrays, concurrency,
delta visibility, degenerate splits, SIMD tails, cache eviction and restore.
The benchmark sweeps beam and candidate budgets, reports recall@k, p50/p95/p99,
build time and actual search work, and can compare FAISS FlatIP and HNSW.
Pass `--qrels judgments.json` for relevance MRR@k and hit rate@k. The JSON maps
zero-based query-row strings to lists of relevant zero-based corpus-row integers.
Unjudged queries are excluded from relevance metrics; judged queries with no
relevant documents receive zero. Without qrels, NN-MRR@k measures the position
of the exact nearest neighbor and must not be reported as semantic MRR.
Use held-out queries, identical normalized vectors, the same k and thread
budget, release builds, and matched recall before comparing speed. Set
RAYON_NUM_THREADS=1 for a single-thread comparison with the script's FAISS
baselines. Synthetic defaults are smoke tests only. Measure payload latency
separately from engine latency and report memory/rebuild costs for deployment.

The parent Cargo manifest contains a `[build]` table which Cargo ignores;
target flags belong in `.cargo/config.toml` or RUSTFLAGS. No host-wide native
CPU flags are needed for this implementation's runtime SIMD dispatch.
