"""Atomic payload records with collision-resistant, path-safe filenames."""
import hashlib
import json
import os
from pathlib import Path
import tempfile
from typing import Optional

from ..core.models import TextChunk


class LocalJSONStorage:
    def __init__(self, base_dir: str = "./aigvs_data"):
        self.base_dir = os.fspath(base_dir)
        self.chunks_dir = os.path.join(self.base_dir, "chunks")
        os.makedirs(self.chunks_dir, exist_ok=True)

    def _get_path(self, chunk_id: str) -> str:
        digest = hashlib.sha256(chunk_id.encode("utf-8")).hexdigest()
        return os.path.join(self.chunks_dir, digest + ".json")

    def write_chunk(self, chunk: TextChunk) -> None:
        # Serialize before touching the destination; replacement is atomic for
        # readers on the same filesystem, not a multi-record transaction.
        record = json.dumps(chunk.to_dict(), ensure_ascii=False, allow_nan=False)
        fd, temporary = tempfile.mkstemp(dir=self.chunks_dir, suffix=".tmp")
        try:
            with os.fdopen(fd, "w", encoding="utf-8") as output:
                output.write(record)
            os.replace(temporary, self._get_path(chunk.chunk_id))
        finally:
            if os.path.exists(temporary):
                os.unlink(temporary)

    def read_chunk(self, chunk_id: str) -> Optional[TextChunk]:
        try:
            with open(self._get_path(chunk_id), encoding="utf-8") as source:
                chunk = TextChunk.from_dict(json.load(source))
        except FileNotFoundError:
            return None
        if chunk.chunk_id != chunk_id:
            raise ValueError("stored record ID does not match its key")
        return chunk

    def iter_chunks(self):
        for path in sorted(Path(self.chunks_dir).glob("*.json")):
            with path.open(encoding="utf-8") as source:
                chunk = TextChunk.from_dict(json.load(source))
            if Path(self._get_path(chunk.chunk_id)).name != path.name:
                raise ValueError("stored record ID does not match its filename")
            yield chunk

    def delete_chunk(self, chunk_id: str) -> bool:
        try:
            os.remove(self._get_path(chunk_id))
            return True
        except FileNotFoundError:
            return False
