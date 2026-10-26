"""PostgreSQL is authoritative; native indexes are disposable read caches."""
import hashlib
import json

from psycopg.rows import dict_row
from psycopg.types.json import Jsonb
from psycopg_pool import ConnectionPool

from .models import Collection, shard_for


class Conflict(Exception):
    pass


class Missing(Exception):
    pass


SCHEMA = """
CREATE TABLE IF NOT EXISTS aigvs_schema(version integer PRIMARY KEY CHECK(version=1));
INSERT INTO aigvs_schema VALUES (1) ON CONFLICT DO NOTHING;
CREATE TABLE IF NOT EXISTS aigvs_collections (
    name text PRIMARY KEY, config jsonb NOT NULL, revision bigint NOT NULL DEFAULT 0
);
CREATE TABLE IF NOT EXISTS aigvs_points (
    collection text NOT NULL REFERENCES aigvs_collections(name),
    id text NOT NULL, shard integer NOT NULL,
    valid_from bigint NOT NULL, valid_to bigint,
    embedding real[] NOT NULL, payload jsonb NOT NULL,
    PRIMARY KEY(collection, id, valid_from),
    CHECK(valid_to IS NULL OR valid_to > valid_from)
);
CREATE UNIQUE INDEX IF NOT EXISTS aigvs_points_live ON aigvs_points(collection,id)
    WHERE valid_to IS NULL;
CREATE INDEX IF NOT EXISTS aigvs_points_snapshot ON aigvs_points(collection,shard,valid_from);
CREATE TABLE IF NOT EXISTS aigvs_requests (
    collection text NOT NULL REFERENCES aigvs_collections(name),
    request_id text NOT NULL, digest text NOT NULL, revision bigint NOT NULL,
    PRIMARY KEY(collection,request_id)
);
"""


class Store:
    def __init__(self, settings):
        self.pool = ConnectionPool(
            settings.database_url.get_secret_value(), min_size=1,
            max_size=settings.db_pool_size, timeout=settings.request_timeout, open=True,
            kwargs={"row_factory": dict_row, "connect_timeout": 5,
                    "options": "-c statement_timeout=30000 -c lock_timeout=10000"},
        )

    def close(self):
        self.pool.close()

    def migrate(self):
        with self.pool.connection() as conn:
            conn.execute("SELECT pg_advisory_xact_lock(14872133)")
            conn.execute(SCHEMA)

    def ready(self):
        with self.pool.connection() as conn:
            return conn.execute("SELECT version FROM aigvs_schema").fetchone() == {"version": 1}

    def create(self, name, config):
        with self.pool.connection() as conn:
            conn.execute("INSERT INTO aigvs_collections(name,config) VALUES (%s,%s) ON CONFLICT DO NOTHING",
                         (name, Jsonb(config.model_dump())))
            row = conn.execute("SELECT * FROM aigvs_collections WHERE name=%s", (name,)).fetchone()
            if row["config"] != config.model_dump():
                raise Conflict("collection exists with a different immutable configuration")
            return row

    def collection(self, name):
        with self.pool.connection() as conn:
            row = conn.execute("SELECT * FROM aigvs_collections WHERE name=%s", (name,)).fetchone()
            if not row:
                raise Missing("collection does not exist")
            return row

    def mutate(self, name, request_id, mutation):
        raw = mutation.model_dump()
        digest = hashlib.sha256(json.dumps(raw, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()).hexdigest()
        with self.pool.connection() as conn:
            row = conn.execute("SELECT * FROM aigvs_collections WHERE name=%s FOR UPDATE", (name,)).fetchone()
            if not row:
                raise Missing("collection does not exist")
            prior = conn.execute("SELECT * FROM aigvs_requests WHERE collection=%s AND request_id=%s", (name, request_id)).fetchone()
            if prior:
                if prior["digest"] != digest:
                    raise Conflict("idempotency key was already used with different content")
                return {"revision": prior["revision"], "replayed": True}
            config = Collection(**row["config"])
            ids = [p.id for p in mutation.points] if mutation.operation == "upsert" else mutation.ids
            if not ids or len(set(ids)) != len(ids) or any(not i or len(i) > 512 for i in ids):
                raise ValueError("provide 1..1000 unique nonempty IDs of at most 512 characters")
            if (mutation.operation == "upsert" and mutation.ids) or (mutation.operation == "delete" and mutation.points):
                raise ValueError("upsert accepts points only; delete accepts ids only")
            if any(len(p.embedding) != config.dimension for p in mutation.points):
                raise ValueError("embedding dimension does not match collection")
            revision = row["revision"] + 1
            conn.execute("UPDATE aigvs_points SET valid_to=%s WHERE collection=%s AND id=ANY(%s) AND valid_to IS NULL",
                         (revision, name, ids))
            if mutation.operation == "upsert":
                with conn.cursor() as cur:
                    cur.executemany("INSERT INTO aigvs_points(collection,id,shard,valid_from,embedding,payload) VALUES (%s,%s,%s,%s,%s,%s)",
                                    [(name, p.id, shard_for(p.id, config.shards), revision, p.embedding,
                                      Jsonb({"text": p.text, "metadata": p.metadata})) for p in mutation.points])
            conn.execute("UPDATE aigvs_collections SET revision=%s WHERE name=%s", (revision, name))
            conn.execute("INSERT INTO aigvs_requests VALUES (%s,%s,%s,%s)", (name, request_id, digest, revision))
            return {"revision": revision, "replayed": False}

    def snapshot(self, name, shard, revision, limit):
        # Logical version intervals preserve a snapshot even while writers commit.
        # Old versions are retained; never vacuum them with an unsafe TTL.
        with self.pool.connection() as conn:
            return conn.execute("""SELECT id,embedding,payload FROM aigvs_points
                WHERE collection=%s AND shard=%s AND valid_from<=%s
                AND (valid_to IS NULL OR valid_to>%s) ORDER BY id LIMIT %s""",
                (name, shard, revision, revision, limit + 1)).fetchall()
