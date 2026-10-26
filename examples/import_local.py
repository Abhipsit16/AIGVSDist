"""Import existing LocalJSONStorage records with resumable content-based keys.

Stop local writers before importing. Example:
python examples/import_local.py ./aigvs_data docs --dimension 768 --shards 16
"""
import argparse
import hashlib
import json
import os
from pathlib import Path

import httpx


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("directory", type=Path)
    parser.add_argument("collection")
    parser.add_argument("--dimension", type=int, required=True)
    parser.add_argument("--shards", type=int, default=16)
    parser.add_argument("--batch-size", type=int, default=100)
    parser.add_argument("--url", default="http://localhost:8000")
    args = parser.parse_args()
    if not 1 <= args.batch_size <= 1000:
        parser.error("batch size must be 1..1000")
    import re
    if not re.fullmatch(r"[A-Za-z0-9_-]{1,128}", args.collection):
        parser.error("invalid collection name")
    files = sorted((args.directory / "chunks").glob("*.json"))
    if not files:
        parser.error("no chunks/*.json records found")
    with httpx.Client(base_url=args.url, timeout=60, headers={
        "Authorization": "Bearer " + os.environ["AIGVS_API_KEY"]}) as client:
        path = f"/v1/collections/{args.collection}"
        response = client.put(path, json={"dimension": args.dimension, "shards": args.shards})
        response.raise_for_status()
        for offset in range(0, len(files), args.batch_size):
            points = []
            for file in files[offset:offset + args.batch_size]:
                record = json.loads(file.read_text(encoding="utf-8"))
                points.append({"id": record["chunk_id"], "embedding": record["embedding"],
                               "text": record.get("text", ""), "metadata": record.get("metadata", {})})
            body = {"operation": "upsert", "points": points}
            key = "local-import-" + hashlib.sha256(json.dumps(body, sort_keys=True, allow_nan=False).encode()).hexdigest()
            response = client.post(path + "/mutations", json=body, headers={"Idempotency-Key": key})
            response.raise_for_status()
            print(f"{min(offset + args.batch_size, len(files))}/{len(files)} records; revision {response.json()['revision']}")


if __name__ == "__main__":
    main()
