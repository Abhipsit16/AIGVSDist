"""Payload orchestration for the Rust associative-memory index."""
from collections import OrderedDict
import json
import math
import time
import warnings
from typing import List

import numpy as np
import aigvs_rust

from .core.models import TextChunk
from .storage import LocalJSONStorage


class AIGVSPipeline:
    """Cosine ANN with Hopfield routing and flat/HNSW terminal partitions.

    Concurrent read-only engine searches are supported. Serialize ingestion,
    rebuilds and payload mutations externally when sharing a pipeline.
    """

    def __init__(self, embedding_dim: int, storage_path: str = "./aigvs_data",
                 signature_bits: int = 1024, use_learned_projection: bool = False,
                 projection_seed: int = 42, cache_size_gb: float = 4.0, *,
                 leaf_capacity: int = 128, branching_factor: int = 8,
                 beta: float = 16.0, beam_width: int = 64,
                 rebuild_threshold: int = 4096, max_tree_depth=None,
                 leaf_backend: str = "auto", hnsw_threshold: int = 512,
                 hnsw_degree: int = 16, ef_construction: int = 128,
                 leaf_ef: int = 64):
        if not math.isfinite(cache_size_gb) or cache_size_gb < 0:
            raise ValueError("cache_size_gb must be finite and nonnegative")
        if beam_width < 1:
            raise ValueError("beam_width must be positive")
        if leaf_ef < 0:
            raise ValueError("leaf_ef must be nonnegative")
        if use_learned_projection:
            raise ValueError("learned hash projections are not used by the Hopfield tree")
        if signature_bits != 1024 or projection_seed != 42:
            warnings.warn("signature_bits and projection_seed are obsolete and ignored",
                          DeprecationWarning, stacklevel=2)
        self.engine = aigvs_rust.AIGVSEngine(
            embedding_dim, leaf_capacity=leaf_capacity,
            branching_factor=branching_factor, beta=beta,
            rebuild_threshold=rebuild_threshold, max_tree_depth=max_tree_depth,
            leaf_backend=leaf_backend, hnsw_threshold=hnsw_threshold,
            hnsw_degree=hnsw_degree, ef_construction=ef_construction)
        self.embedding_dim = embedding_dim
        self.beam_width = beam_width
        self.candidate_budget = 4096
        self.leaf_ef = leaf_ef
        self._default_exact = False
        self._calibrated_size = None
        self.storage_path = storage_path
        self.storage = LocalJSONStorage(storage_path)
        self.payload_cache = OrderedDict()
        self._cache_sizes = {}
        self._cache_bytes = 0
        # Budget is serialized payload bytes, not a process-RSS guarantee.
        self._cache_budget = int(cache_size_gb * 1024**3)
        self._ids = set()
        self.stats = {"total_vectors": 0, "avg_latency_ms": 0.0,
                      "query_count": 0, "last_batch_time": 0.0}

    def _cache_payload(self, chunk: TextChunk) -> dict:
        payload = {"id": chunk.chunk_id, "text": chunk.text, "metadata": chunk.metadata}
        if self._cache_budget == 0:
            return payload
        size = len(json.dumps(payload, ensure_ascii=False).encode("utf-8"))
        old_size = self._cache_sizes.pop(chunk.chunk_id, 0)
        self._cache_bytes -= old_size
        self.payload_cache.pop(chunk.chunk_id, None)
        if size > self._cache_budget:
            return payload
        while self.payload_cache and self._cache_bytes + size > self._cache_budget:
            cid, _ = self.payload_cache.popitem(last=False)
            self._cache_bytes -= self._cache_sizes.pop(cid)
        self.payload_cache[chunk.chunk_id] = payload
        self._cache_sizes[chunk.chunk_id] = size
        self._cache_bytes += size
        return payload

    def _embedding(self, value) -> np.ndarray:
        vector = np.asarray(value, dtype=np.float32)
        if vector.shape != (self.embedding_dim,):
            raise ValueError(f"embedding must have shape ({self.embedding_dim},)")
        if not np.isfinite(vector).all() or not np.any(vector):
            raise ValueError("embedding must be finite and nonzero")
        return np.ascontiguousarray(vector)

    def ingest_batches(self, json_data: List[dict], batch_size: int = 10000, *,
                       rebuild: bool = True) -> None:
        """Validate/persist/insert each batch; build once after bulk ingestion.

        Completed batches remain committed if a later batch fails. Deferred
        rebuilds keep all new rows searchable through an exact delta scan.
        IDs are insert-only: duplicates raise rather than silently overwrite.
        """
        if batch_size < 1:
            raise ValueError("batch_size must be positive")
        for start in range(0, len(json_data), batch_size):
            begin = time.perf_counter()
            chunks = []
            batch_ids = set()
            for entry in json_data[start:start + batch_size]:
                cid = entry["id"]
                if not isinstance(cid, str) or not cid or cid in self._ids or cid in batch_ids:
                    raise ValueError("IDs must be unique nonempty strings")
                if not isinstance(entry["text"], str):
                    raise ValueError("text must be a string")
                metadata = entry.get("metadata", {})
                if not isinstance(metadata, dict):
                    raise ValueError("metadata must be a dictionary")
                # Validate serialization before writing any record in this batch.
                metadata = json.loads(json.dumps(metadata, allow_nan=False))
                chunks.append(TextChunk(cid, entry["text"], self._embedding(entry["embedding"]), metadata))
                batch_ids.add(cid)
            if not chunks:
                continue
            vectors = np.stack([chunk.embedding for chunk in chunks])
            for chunk in chunks:
                self.storage.write_chunk(chunk)
            self.engine.native_batch_insert([c.chunk_id for c in chunks], vectors, rebuild=False)
            self._ids.update(batch_ids)
            for chunk in chunks:
                self._cache_payload(chunk)
            self.stats["last_batch_time"] = time.perf_counter() - begin
            self.stats["total_vectors"] = self.engine.size
        if rebuild:
            self.rebuild()

    def rebuild(self) -> None:
        """Pack pending vectors into the tree; query results remain visible."""
        if self.engine.pending_count:
            self._calibrated_size = None
        self.engine.rebuild()

    def restore(self, batch_size: int = 10000) -> None:
        """Reconstruct an empty index from LocalJSONStorage records."""
        if self.engine.size:
            raise ValueError("restore requires an empty index")
        if batch_size < 1:
            raise ValueError("batch_size must be positive")
        ids, vectors = [], []
        for chunk in self.storage.iter_chunks():
            ids.append(chunk.chunk_id)
            vectors.append(self._embedding(chunk.embedding))
            if len(ids) == batch_size:
                self.engine.native_batch_insert(ids, np.stack(vectors), rebuild=False)
                self._ids.update(ids)
                ids, vectors = [], []
        if ids:
            self.engine.native_batch_insert(ids, np.stack(vectors), rebuild=False)
            self._ids.update(ids)
        self.engine.rebuild()
        self.stats["total_vectors"] = self.engine.size

    def calibrate(self, queries, *, top_k: int = 10, target_recall: float = 0.99,
                  relevant_ids=None, min_mrr=None, beam_widths=(16, 64, 256),
                  candidate_budgets=(1024, 4096, 16384), leaf_efs=(0, 32, 64, 128),
                  repeats: int = 3) -> dict:
        """Select search defaults against a representative validation set.

        This is an empirical target on these queries, not a guarantee on future
        queries. Recalibrate after growth or distribution changes; assess final
        metrics on a separate test set. relevant_ids is one set of relevant
        string document IDs per query. Exhaustive search is an explicit fallback.
        """
        if not self.engine.size or top_k < 1 or repeats < 1:
            raise ValueError("calibration requires data, positive top_k and repeats")
        if not math.isfinite(target_recall) or not 0 < target_recall <= 1:
            raise ValueError("target_recall must be in (0, 1]")
        if min_mrr is not None and (not math.isfinite(min_mrr) or not 0 <= min_mrr <= 1):
            raise ValueError("min_mrr must be in [0, 1]")
        matrix = np.asarray(queries, dtype=np.float32)
        if matrix.ndim != 2 or len(matrix) == 0 or matrix.shape[1] != self.embedding_dim:
            raise ValueError("queries must be a nonempty matrix with embedding_dim columns")
        matrix = np.ascontiguousarray(matrix)
        if relevant_ids is None:
            if min_mrr is not None:
                raise ValueError("an MRR target requires relevance judgments")
            labels = None
        else:
            labels = [set(ids) for ids in relevant_ids]
            if len(labels) != len(matrix) or any(not isinstance(cid, str) for row in labels for cid in row):
                raise ValueError("relevant_ids must contain one set of string document IDs per query")
        beams = sorted(set(beam_widths))
        budgets = sorted(set(candidate_budgets))
        efs = sorted(set(leaf_efs)) if self.engine.graph_count else [0]
        if not beams or not budgets or min(beams) < 1 or min(budgets) < 1:
            raise ValueError("beam widths and candidate budgets must be positive")
        # Shallow hybrids may need every partition while HNSW still visits
        # only a fraction of the rows within each selected partition.
        budgets = sorted(set(budgets + [self.engine.size]))
        if not efs or min(efs) < 0:
            raise ValueError("leaf_efs must be nonnegative; zero selects flat scans")
        truth_rows = self.engine.batch_search(matrix, top_k=top_k, exact=True)
        truth = [{cid for cid, _ in row} for row in truth_rows]
        reports = []
        profiles = [(beam, budget, False, ef) for beam in beams for budget in budgets for ef in efs]
        profiles.append((1, self.engine.size, True, 0))
        # Interleave profiles to reduce systematic warm-cache/order bias.
        trials = [[] for _ in profiles]
        rankings = [None] * len(profiles)
        for profile in profiles:
            beam, budget, exact, ef = profile
            self.engine.search(matrix[0], top_k, budget, beam, exact, ef)
        for repetition in range(repeats):
            order = range(len(profiles)) if repetition % 2 == 0 else reversed(range(len(profiles)))
            for idx in order:
                beam, budget, exact, ef = profiles[idx]
                found = []
                for query in matrix:
                    start = time.perf_counter_ns()
                    row = self.engine.search(query, top_k, budget, beam, exact, ef)
                    trials[idx].append((time.perf_counter_ns() - start) / 1e6)
                    if repetition == 0:
                        found.append([cid for cid, _ in row])
                if repetition == 0:
                    rankings[idx] = found
        for idx, (beam, budget, exact, ef) in enumerate(profiles):
            found = rankings[idx]
            recall = float(np.mean([len(set(row) & expected) / len(expected)
                                    for row, expected in zip(found, truth)]))
            report = {"beam_width": beam, "candidate_budget": budget, "exact": exact, "leaf_ef": ef,
                      "recall_at_k": recall, "p50_ms": float(np.median(trials[idx])),
                      "p95_ms": float(np.percentile(trials[idx], 95))}
            if labels is not None:
                report["mrr_at_k"] = float(np.mean([
                    next((1 / rank for rank, cid in enumerate(row, 1) if cid in relevant), 0.0)
                    for row, relevant in zip(found, labels)]))
            reports.append(report)
        eligible = [r for r in reports if r["recall_at_k"] + 1e-12 >= target_recall
                    and (min_mrr is None or r["mrr_at_k"] + 1e-12 >= min_mrr)]
        if not eligible:
            best_mrr = max(r.get("mrr_at_k", 0.0) for r in reports)
            raise ValueError(f"no measured profile meets the targets; best observed MRR={best_mrr:.4f}")
        selected = min(eligible, key=lambda r: r["p50_ms"])
        self.beam_width = selected["beam_width"]
        self.candidate_budget = selected["candidate_budget"]
        self._default_exact = selected["exact"]
        self.leaf_ef = selected["leaf_ef"]
        self._calibrated_size = self.engine.size
        return {"selected": dict(selected), "profiles": reports, "top_k": top_k,
                "validation_queries": len(matrix), "index_size": self.engine.size,
                "exhaustive_fallback": selected["exact"]}

    def query(self, query_embedding: np.ndarray, top_k: int = 10,
              rerank_k=None, min_score: float = -1.0, *,
              beam_width=None, exact=None, leaf_ef=None) -> List[dict]:
        """Return payloads with actual cosine scores, in descending order.

        beam_width bounds surviving routes. rerank_k targets covered rows,
        rounded up to whole leaves and limited by surviving routes. HNSW
        leaf_ef controls work within those leaves; zero forces flat scans.
        exact=True scores every vector and is the recall reference.
        """
        if not math.isfinite(min_score) or not -1 <= min_score <= 1:
            raise ValueError("min_score must be finite and in [-1, 1]")
        vector = self._embedding(query_embedding)
        begin = time.perf_counter()
        hits = self.engine.search(vector, top_k=top_k,
                                  rerank_k=self.candidate_budget if rerank_k is None else rerank_k,
                                  beam_width=self.beam_width if beam_width is None else beam_width,
                                  exact=self._default_exact if exact is None else exact,
                                  leaf_ef=self.leaf_ef if leaf_ef is None else leaf_ef)
        elapsed_ms = (time.perf_counter() - begin) * 1000
        self.stats["query_count"] += 1
        self.stats["avg_latency_ms"] += ((elapsed_ms - self.stats["avg_latency_ms"])
                                          / self.stats["query_count"])
        results = []
        for cid, score in hits:
            if score < min_score:
                continue
            payload = self.payload_cache.get(cid)
            if payload is None:
                chunk = self.storage.read_chunk(cid)
                if chunk is None:
                    raise RuntimeError(f"indexed payload is missing from storage: {cid!r}")
                payload = self._cache_payload(chunk)
            else:
                self.payload_cache.move_to_end(cid)
            # Do not expose mutable cache metadata to callers.
            results.append({**payload, "metadata": json.loads(json.dumps(payload["metadata"])),
                            "score": score})
        return results

    def train_learned_projection(self, vectors: np.ndarray, epochs: int = 100):
        raise NotImplementedError("the Hopfield tree learns centroids during rebuild; hash training is obsolete")

    def get_stats(self) -> dict:
        return {**self.stats, "total_vectors": self.engine.size,
                "pending_vectors": self.engine.pending_count, "tree_nodes": self.engine.node_count,
                "cache_payload_bytes": self._cache_bytes,
                "calibrated_for_current_size": self._calibrated_size == self.engine.size,
                "default_beam_width": self.beam_width, "default_candidate_budget": self.candidate_budget,
                "default_exact": self._default_exact, "default_leaf_ef": self.leaf_ef,
                "hnsw_leaves": self.engine.graph_count}

    def memory_snapshot(self) -> dict:
        snapshot = {"cache_entries": len(self.payload_cache),
                    "cache_payload_bytes": self._cache_bytes, "total_vectors": self.engine.size}
        try:
            import psutil
            memory = psutil.Process().memory_info()
            snapshot.update(rss_mb=memory.rss / 1024**2, vms_mb=memory.vms / 1024**2)
        except ImportError:
            pass
        return snapshot
