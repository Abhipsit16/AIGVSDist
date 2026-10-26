import hashlib
import math
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator


class Model(BaseModel):
    model_config = ConfigDict(extra="forbid", allow_inf_nan=False)


class Collection(Model):
    dimension: int = Field(ge=1, le=65536)
    shards: int = Field(default=16, ge=1, le=256)
    leaf_capacity: int = Field(default=128, ge=1, le=65536)
    branching_factor: int = Field(default=8, ge=2, le=32)
    beta: float = Field(default=16, gt=0, le=100)
    max_tree_depth: int | None = Field(default=None, ge=0, le=64)
    leaf_backend: Literal["auto", "flat", "hnsw"] = "auto"
    hnsw_threshold: int = Field(default=512, ge=2)
    hnsw_degree: int = Field(default=16, ge=2, le=64)
    ef_construction: int = Field(default=128, ge=2, le=4096)

    @model_validator(mode="after")
    def graph_parameters(self):
        if self.ef_construction < 2 * self.hnsw_degree:
            raise ValueError("ef_construction must be at least twice hnsw_degree")
        return self

    def engine_options(self):
        return self.model_dump(exclude={"dimension", "shards"})


def valid_vector(value):
    # Validate before float32 conversion and database mutation.
    if not value or any(not math.isfinite(x) or abs(x) > 3.4028234e38 for x in value):
        raise ValueError("embedding must contain finite float32 values")
    if max(abs(x) for x in value) < 1.4012985e-45:
        raise ValueError("embedding must be nonzero in float32")
    return value


class Point(Model):
    id: str = Field(min_length=1, max_length=512)
    embedding: list[float] = Field(min_length=1, max_length=65536)
    text: str = Field(default="", max_length=1048576)
    metadata: dict = Field(default_factory=dict)
    _vector = field_validator("embedding")(valid_vector)


class Mutation(Model):
    operation: Literal["upsert", "delete"] = "upsert"
    points: list[Point] = Field(default_factory=list, max_length=1000)
    ids: list[str] = Field(default_factory=list, max_length=1000)


class Search(Model):
    vector: list[float] = Field(min_length=1, max_length=65536)
    top_k: int = Field(default=10, ge=1, le=1000)
    candidate_budget: int = Field(default=4096, ge=1, le=1000000)
    beam_width: int = Field(default=64, ge=1, le=65536)
    leaf_ef: int = Field(default=64, ge=0, le=1000000)
    exact: bool = False
    min_score: float = Field(default=-1, ge=-1, le=1)
    _vector = field_validator("vector")(valid_vector)


class ShardSearch(Model):
    revision: int = Field(ge=0)
    query: Search


def shard_for(point_id: str, count: int) -> int:
    return int.from_bytes(hashlib.sha256(point_id.encode()).digest()[:8], "big") % count


def worker_order(collection: str, shard: int, workers: list[str]) -> list[str]:
    """Rendezvous placement: stable preference order, no exclusive ownership."""
    key = f"{collection}/{shard}/"
    return sorted(workers, key=lambda w: hashlib.sha256((key + w).encode()).digest(), reverse=True)
