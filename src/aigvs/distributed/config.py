from typing import Literal
from pydantic import Field, SecretStr, model_validator
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_prefix="AIGVS_", extra="ignore")
    role: Literal["gateway", "worker"] = "gateway"
    database_url: SecretStr
    api_key: SecretStr
    internal_key: SecretStr
    workers: list[str] = Field(default_factory=lambda: ["http://worker:8000"])
    db_pool_size: int = Field(default=8, ge=1, le=128)
    request_timeout: float = Field(default=30, gt=0, le=300)
    max_body_bytes: int = Field(default=16777216, ge=1024)
    max_concurrent_requests: int = Field(default=32, ge=1, le=1024)
    max_parallel_shards: int = Field(default=16, ge=1, le=256)
    cache_shards: int = Field(default=32, ge=1)
    max_shard_points: int = Field(default=1000000, ge=1)
    max_shard_vector_bytes: int = Field(default=536870912, ge=1024)

    @model_validator(mode="after")
    def validate_settings(self):
        if any(len(k.get_secret_value()) < 32 for k in (self.api_key, self.internal_key)):
            raise ValueError("API and internal keys must be at least 32 characters")
        if self.api_key == self.internal_key:
            raise ValueError("API and internal keys must differ")
        if not self.workers or len(set(self.workers)) != len(self.workers):
            raise ValueError("workers must be a nonempty list of unique URLs")
        from urllib.parse import urlsplit
        for url in self.workers:
            parsed = urlsplit(url)
            if parsed.scheme not in ("http", "https") or not parsed.hostname or parsed.username or parsed.query or parsed.fragment or parsed.path not in ("", "/"):
                raise ValueError("workers must be HTTP(S) origin URLs")
        return self
