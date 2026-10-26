from collections import OrderedDict
from threading import Lock

import numpy as np

from .models import Collection


class IndexCache:
    def __init__(self, store, settings, engine_factory=None):
        if engine_factory is None:
            from aigvs_rust import AIGVSEngine
            engine_factory = AIGVSEngine
        self.factory, self.store, self.settings = engine_factory, store, settings
        self.entries = OrderedDict()
        # Single-flight builds bound peak build memory per process. Searches use
        # immutable engine references and never hold this lock during scoring.
        self.lock = Lock()

    def search(self, name, shard, revision, query):
        key = (name, shard, revision)
        with self.lock:
            entry = self.entries.get(key)
            if entry is None:
                row = self.store.collection(name)
                config = Collection(**row["config"])
                if not 0 <= shard < config.shards or revision > row["revision"]:
                    raise ValueError("invalid shard or future revision")
                limit = min(self.settings.max_shard_points,
                            self.settings.max_shard_vector_bytes // (4 * config.dimension))
                rows = self.store.snapshot(name, shard, revision, limit)
                if len(rows) > limit:
                    raise ValueError("shard capacity exceeded; re-ingest into a collection with more shards")
                engine = self.factory(config.dimension, **config.engine_options())
                if rows:
                    engine.native_batch_insert([r["id"] for r in rows],
                                               np.asarray([r["embedding"] for r in rows], dtype=np.float32))
                payloads = {r["id"]: r["payload"] for r in rows}
                entry = (engine, payloads, config.dimension)
                # Keep at most one revision per shard, plus an LRU entry cap.
                for old in list(self.entries):
                    if old[:2] == key[:2]:
                        del self.entries[old]
                self.entries[key] = entry
                while len(self.entries) > self.settings.cache_shards:
                    self.entries.popitem(last=False)
            self.entries.move_to_end(key)
        engine, payloads, dimension = entry
        if len(query.vector) != dimension:
            raise ValueError("query dimension does not match collection")
        hits = engine.search(np.asarray(query.vector, dtype=np.float32), top_k=query.top_k,
                             rerank_k=query.candidate_budget, beam_width=query.beam_width,
                             exact=query.exact, leaf_ef=query.leaf_ef)
        return [{"id": id_, "score": score, **payloads[id_]} for id_, score in hits if score >= query.min_score]
