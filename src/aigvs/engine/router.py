"""Small NumPy reference router; production routing runs entirely in Rust.

Association weights are a local softmax, not calibrated recall probabilities.
"""
import numpy as np


class HopfieldRouter:
    def __init__(self, embedding_dim: int, beta_min: float = 16.0,
                 k_scale: float = 0.0, activation_threshold: float = -1.0):
        if embedding_dim < 1 or not np.isfinite(beta_min) or not 0 < beta_min <= 1000:
            raise ValueError("invalid dimension or beta")
        self.embedding_dim = embedding_dim
        self.beta_min = beta_min
        self.activation_threshold = activation_threshold
        self.id_lookup = []
        self.X_matrix = np.empty((0, embedding_dim), dtype=np.float32)
        self.node_capacities = {}

    def _normalize(self, value):
        vector = np.asarray(value, dtype=np.float64)
        if vector.shape != (self.embedding_dim,) or not np.isfinite(vector).all():
            raise ValueError("invalid embedding")
        norm = np.linalg.norm(vector)
        if norm == 0:
            raise ValueError("embedding must be nonzero")
        return (vector / norm).astype(np.float32)

    def add_or_update_attractor(self, node_id, centroid, current_count):
        vector = self._normalize(centroid)
        if node_id in self.id_lookup:
            self.X_matrix[self.id_lookup.index(node_id)] = vector
        else:
            self.X_matrix = np.vstack((self.X_matrix, vector))
            self.id_lookup.append(node_id)
        self.node_capacities[node_id] = current_count

    def remove_attractor(self, node_id):
        if node_id in self.id_lookup:
            idx = self.id_lookup.index(node_id)
            self.id_lookup.pop(idx)
            self.X_matrix = np.delete(self.X_matrix, idx, axis=0)
            self.node_capacities.pop(node_id)

    def calculate_dynamic_beta(self):
        # Fixed bounded temperature avoids the old inverse-variance explosion.
        return self.beta_min

    def route_query(self, query_vector, max_beam=15, target_confidence=0.95):
        if max_beam < 1 or not 0 < target_confidence <= 1:
            raise ValueError("invalid beam or cumulative association mass")
        query = self._normalize(query_vector)
        if not self.id_lookup:
            return []
        scores = self.X_matrix @ query
        logits = self.beta_min * scores.astype(np.float64)
        weights = np.exp(logits - logits.max())
        weights /= weights.sum()
        selected, mass = [], 0.0
        for idx in np.argsort(-weights, kind="stable")[:max_beam]:
            if scores[idx] < self.activation_threshold:
                continue
            selected.append(self.id_lookup[idx])
            mass += weights[idx]
            if mass >= target_confidence:
                break
        return selected
