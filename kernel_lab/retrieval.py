"""Verified training-only retrieval using OpenAI-compatible embeddings and pgvector."""
from __future__ import annotations

import argparse
import asyncio
import hashlib
import json
import math
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Any


def task_text(task: dict[str, Any]) -> str:
    return task["prompt"] + "\n" + task["reference_code"]


def training_record(record: dict[str, Any]) -> dict[str, Any]:
    task = record.get("task", record)
    if task.get("split") != "train" or record.get("validation", {}).get("status") != "passed":
        raise ValueError("Only sandbox-verified training records may enter retrieval")
    if not task.get("family") or not task.get("task_id") or not record.get("candidate_code"):
        raise ValueError("Record lacks task_id, family, or candidate_code")
    return task


@dataclass(frozen=True)
class EmbeddingConfig:
    base_url: str
    model: str
    dimensions: int
    api_key: str = "local"

    @classmethod
    def from_env(cls) -> "EmbeddingConfig":
        return cls(os.environ["EMBEDDING_BASE_URL"], os.environ["EMBEDDING_MODEL"],
                   int(os.environ["EMBEDDING_DIMENSIONS"]), os.getenv("EMBEDDING_API_KEY", "local"))

    def __post_init__(self) -> None:
        if type(self.dimensions) is not int or not 1 <= self.dimensions <= 16000:
            raise ValueError("Embedding dimensions must be 1..16000")


class EmbeddingClient:
    def __init__(self, config: EmbeddingConfig):
        self.config = config

    async def embed(self, text: str) -> list[float]:
        # Use the model's native output size; check it instead of silently truncating.
        import httpx
        async with httpx.AsyncClient(timeout=120) as client:
            result = await client.post(self.config.base_url.rstrip("/") + "/embeddings",
                                       headers={"Authorization": f"Bearer {self.config.api_key}"},
                                       json={"model": self.config.model, "input": text})
            result.raise_for_status()
            vector = result.json()["data"][0]["embedding"]
        if len(vector) != self.config.dimensions or not all(math.isfinite(x) for x in vector):
            raise ValueError("Embedding output differs from configured finite dimensions")
        if not any(vector):
            raise ValueError("Zero vectors have undefined cosine similarity")
        return vector


def vector_literal(vector: list[float], dimensions: int) -> str:
    if len(vector) != dimensions or not all(math.isfinite(x) for x in vector) or not any(vector):
        raise ValueError("Invalid embedding vector")
    return "[" + ",".join(str(float(x)) for x in vector) + "]"


class PgVectorStore:
    def __init__(self, dsn: str, config: EmbeddingConfig):
        self.dsn, self.config = dsn, config

    def _connect(self):
        import psycopg
        return psycopg.connect(self.dsn)

    def initialize(self) -> None:
        with self._connect() as conn:
            conn.execute("CREATE EXTENSION IF NOT EXISTS vector")
            conn.execute("CREATE TABLE IF NOT EXISTS kernel_embedding_config (singleton boolean PRIMARY KEY DEFAULT true CHECK(singleton), model text NOT NULL, base_url text NOT NULL, dimensions integer NOT NULL)")
            conn.execute("INSERT INTO kernel_embedding_config(singleton,model,base_url,dimensions) VALUES(true,%s,%s,%s) ON CONFLICT DO NOTHING",
                         (self.config.model, self.config.base_url, self.config.dimensions))
            current = conn.execute("SELECT model,base_url,dimensions FROM kernel_embedding_config WHERE singleton=true").fetchone()
            if current != (self.config.model, self.config.base_url, self.config.dimensions):
                raise ValueError("Database embedding configuration mismatch; use a separate database for another model")
            # dimensions is validated as integer in the frozen config, never user SQL.
            conn.execute(f"CREATE TABLE IF NOT EXISTS verified_kernels (task_id text PRIMARY KEY, family text NOT NULL, content_hash text NOT NULL, record jsonb NOT NULL, embedding vector({self.config.dimensions}) NOT NULL)")
            # Exact cosine search is sufficient for a few thousand rows and avoids approximate filtering recall bias.

    def upsert(self, record: dict[str, Any], vector: list[float]) -> None:
        task = training_record(record)
        record = dict(record, task=task)
        digest = hashlib.sha256(task_text(task).encode()).hexdigest()
        with self._connect() as conn:
            conn.execute("INSERT INTO verified_kernels(task_id,family,content_hash,record,embedding) VALUES(%s,%s,%s,%s::jsonb,%s::vector) ON CONFLICT(task_id) DO UPDATE SET family=excluded.family,content_hash=excluded.content_hash,record=excluded.record,embedding=excluded.embedding",
                         (task["task_id"], task["family"], digest, json.dumps(record), vector_literal(vector, self.config.dimensions)))

    def search(self, task: dict[str, Any], vector: list[float], k: int = 3,
               excluded_families: list[str] | None = None, excluded_task_ids: list[str] | None = None) -> list[dict[str, Any]]:
        if k < 1:
            raise ValueError("k must be positive")
        families = sorted(set((excluded_families or []) + [task["family"]]))
        ids = sorted(set((excluded_task_ids or []) + [task["task_id"]]))
        digest = hashlib.sha256(task_text(task).encode()).hexdigest()
        with self._connect() as conn:
            rows = conn.execute("SELECT record, embedding <=> %s::vector AS distance FROM verified_kernels WHERE NOT(family = ANY(%s::text[])) AND NOT(task_id = ANY(%s::text[])) AND content_hash <> %s AND record->'task'->>'split' = 'train' AND record->'validation'->>'status' = 'passed' ORDER BY embedding <=> %s::vector LIMIT %s",
                                (vector_literal(vector, self.config.dimensions), families, ids, digest, vector_literal(vector, self.config.dimensions), k)).fetchall()
        return [{"task": row[0]["task"], "task_id": row[0]["task"]["task_id"], "family": row[0]["task"]["family"],
                 "prompt": row[0]["task"]["prompt"], "reference_code": row[0]["task"]["reference_code"],
                 "candidate_code": row[0]["candidate_code"], "cosine_distance": float(row[1])} for row in rows]


async def index_file(path: str, store: PgVectorStore, embedder: EmbeddingClient) -> dict[str, int]:
    store.initialize()
    indexed = skipped = 0
    for line in Path(path).read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        record = json.loads(line)
        task = record.get("task", record)
        if task.get("split") != "train" or record.get("validation", {}).get("status") != "passed":
            skipped += 1
            continue
        training_record(record)
        store.upsert(record, await embedder.embed(task_text(task)))
        indexed += 1
    return {"indexed": indexed, "skipped_unverified_or_nontrain": skipped}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--verified", required=True)
    args = parser.parse_args()
    config = EmbeddingConfig.from_env()
    result = asyncio.run(index_file(args.verified, PgVectorStore(os.environ["PGVECTOR_DSN"], config), EmbeddingClient(config)))
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
