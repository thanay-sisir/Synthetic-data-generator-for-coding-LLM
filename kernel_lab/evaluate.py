"""Paired held-out execution evaluation. Every requested task stays in the denominator."""
from __future__ import annotations

import argparse
import asyncio
import hashlib
import json
import os
from pathlib import Path
import random
import time
from datetime import datetime, timezone
from typing import Any

from .metrics import summarize_attempts


async def evaluate_tasks(tasks: list[dict[str, Any]], teacher: Any, sandbox: Any,
                         output_dir: str | Path, *, store: Any = None, embedder: Any = None,
                         workers: int = 1, k: int = 3, seed: int = 42,
                         no_rag_only: bool = False) -> dict[str, Any]:
    if workers < 1 or k < 1 or not tasks:
        raise ValueError("Require tasks, positive workers, and positive k")
    if any(t.get("split") not in ("validation", "test") for t in tasks):
        raise ValueError("Evaluation accepts only held-out validation/test tasks")
    if len({t["task_id"] for t in tasks}) != len(tasks):
        raise ValueError("Evaluation task IDs must be unique")
    if not no_rag_only and (store is None or embedder is None):
        raise ValueError("Paired evaluation requires a retrieval store and embedder")
    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=True)
    ledger_path = output / "evaluation.jsonl"
    if ledger_path.exists():
        raise FileExistsError("Choose a new output directory to preserve the previous ledger")
    ledger_path.touch(exist_ok=False)
    (output / "tasks.jsonl").write_text("".join(json.dumps(task, sort_keys=True) + "\n" for task in tasks), encoding="utf-8")
    families = sorted({t["family"] for t in tasks})
    task_ids = sorted(t["task_id"] for t in tasks)
    semaphore = asyncio.Semaphore(workers)
    rows: list[dict[str, Any]] = []
    started = time.perf_counter()

    async def run_task(task: dict[str, Any], order: list[str]) -> None:
        for condition in order:
            async with semaphore:
                start = time.perf_counter()
                row: dict[str, Any] = {"task_id": task["task_id"], "family": task["family"], "split": task["split"],
                    "task_sha256": hashlib.sha256(json.dumps(task, sort_keys=True).encode()).hexdigest(),
                    "condition": condition, "teacher": {"cost_usd": 0, "prompt_tokens": 0, "completion_tokens": 0}, "validation": {"status": "failed"},
                    "retrieved_task_ids": [], "retrieval_seconds": 0.0, "sandbox_seconds": 0.0}
                stage = "retrieval" if condition == "rag" else "generation"
                try:
                    examples = []
                    if condition == "rag":
                        retrieval_start = time.perf_counter()
                        from .retrieval import task_text
                        try:
                            vector = await embedder.embed(task_text(task))
                            examples = await asyncio.to_thread(store.search, task, vector, k, families, task_ids)
                        finally:
                            row["retrieval_seconds"] = time.perf_counter() - retrieval_start
                        row["retrieved_task_ids"] = [e["task_id"] for e in examples]
                        # IDs alone are insufficient when a later ingestion replaces a row.
                        row["retrieved_examples"] = examples
                        row["retrieved_examples_sha256"] = hashlib.sha256(
                            json.dumps(examples, sort_keys=True).encode()).hexdigest()
                    stage = "generation"
                    row["teacher"] = {"cost_usd": None}
                    result = await teacher.complete(task, examples=examples)
                    row["teacher"] = {key: value for key, value in result.items() if key != "candidate_code"}
                    row["candidate_code"] = result["candidate_code"]
                    stage = "sandbox"
                    sandbox_start = time.perf_counter()
                    try:
                        row["validation"] = await sandbox.validate(task, result["candidate_code"])
                    finally:
                        row["sandbox_seconds"] = time.perf_counter() - sandbox_start
                except Exception as exc:
                    row["validation"] = {"status": "failed", "stage": stage, "error": f"{type(exc).__name__}: {exc}"}
                row["latency_seconds"] = time.perf_counter() - start
                # Writes occur on the event-loop thread with no await between rows.
                with ledger_path.open("a", encoding="utf-8") as handle:
                    handle.write(json.dumps(row, allow_nan=False) + "\n")
                    handle.flush()
                rows.append(row)

    rng = random.Random(seed)
    jobs = []
    for task in tasks:
        order = ["no_rag"] if no_rag_only else ["no_rag", "rag"]
        rng.shuffle(order)
        jobs.append(run_task(task, order))
    await asyncio.gather(*jobs)
    summaries = {condition: summarize_attempts([r for r in rows if r["condition"] == condition])
                 for condition in (["no_rag"] if no_rag_only else ["no_rag", "rag"])}
    paired = {}
    if not no_rag_only:
        results = {(r["task_id"], r["condition"]): r["validation"].get("status") == "passed" for r in rows}
        paired = {"both_pass": 0, "rag_only_pass": 0, "no_rag_only_pass": 0, "both_fail": 0}
        for task in tasks:
            a, b = results[(task["task_id"], "no_rag")], results[(task["task_id"], "rag")]
            paired["both_pass" if a and b else "rag_only_pass" if b else "no_rag_only_pass" if a else "both_fail"] += 1
    config = getattr(teacher, "config", None)
    generation_settings = {key: getattr(config, key, None) for key in ("model", "base_url", "max_tokens", "temperature", "retries", "timeout", "free_only", "request_interval")}
    retrieval_config = getattr(store, "config", None)
    report = {"schema_version": 1, "tasks": len(tasks), "seed": seed, "workers": workers, "retrieval_k": k,
              "generation_settings": generation_settings,
              "completed_at_utc": datetime.now(timezone.utc).isoformat(),
              "embedding_settings": {key: getattr(retrieval_config, key, None) for key in ("model", "base_url", "dimensions")},
              "sandbox": {key: getattr(sandbox, key, None) for key in ("image", "backend", "gpu_device", "timeout")},
              "conditions": summaries, "paired_outcomes": paired, "wall_seconds": time.perf_counter() - started,
              "task_set_sha256": hashlib.sha256(json.dumps(tasks, sort_keys=True).encode()).hexdigest(),
              "cost_scope": "teacher API only; embedding, database, GPU rental and idle time excluded; unknown prices remain null",
              "retrieval_failure_policy": "Failure retained as failed attempt; no fallback to no-RAG"}
    (output / "summary.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    return report


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--tasks", required=True)
    parser.add_argument("--split", choices=["validation", "test"], default="test")
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--workers", type=int, default=1)
    parser.add_argument("--k", type=int, default=3)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--no-rag-only", action="store_true")
    parser.add_argument("--baseline-checkpoint")
    parser.add_argument("--local-qwen", help="Score this Hugging Face model on the allocated GPU")
    parser.add_argument("--qwen-adapter", help="LoRA adapter directory to load on top of --local-qwen")
    parser.add_argument("--max-new-tokens", type=int, default=1024)
    parser.add_argument("--sandbox-backend", choices=["docker", "apptainer"], default="docker")
    parser.add_argument("--sandbox-image", default="kernel-lab-validator:latest")
    parser.add_argument("--sandbox-timeout", type=float, default=120)
    parser.add_argument("--gpu-device", default="0")
    parser.add_argument("--allow-weaker-isolation", action="store_true")
    args = parser.parse_args()
    from .teacher import TeacherClient, TeacherConfig
    from .sandbox import DockerSandbox
    if args.local_qwen and args.baseline_checkpoint:
        parser.error("Choose either --local-qwen or --baseline-checkpoint")
    if args.qwen_adapter and not args.local_qwen:
        parser.error("--qwen-adapter requires --local-qwen")
    if args.local_qwen:
        from .qwen_local import QwenLocalTeacher
        teacher = QwenLocalTeacher(args.local_qwen, adapter=args.qwen_adapter, max_new_tokens=args.max_new_tokens)
    elif args.baseline_checkpoint:
        from .baseline import BaselineTeacher
        teacher = BaselineTeacher(args.baseline_checkpoint, max_tokens=TeacherConfig.from_env().max_tokens)
    else:
        teacher = TeacherClient(TeacherConfig.from_env())
    store = embedder = None
    if not args.no_rag_only:
        from .retrieval import PgVectorStore, EmbeddingClient, EmbeddingConfig
        config = EmbeddingConfig.from_env()
        store = PgVectorStore(os.environ["PGVECTOR_DSN"], config)
        store.initialize()
        embedder = EmbeddingClient(config)
    tasks = [json.loads(line) for line in Path(args.tasks).read_text(encoding="utf-8").splitlines() if line.strip()]
    tasks = [task for task in tasks if task.get("split") == args.split]
    sandbox = DockerSandbox(image=args.sandbox_image, backend=args.sandbox_backend,
                            timeout=args.sandbox_timeout, gpu_device=args.gpu_device,
                            allow_weaker_isolation=args.allow_weaker_isolation)
    print(json.dumps(asyncio.run(evaluate_tasks(tasks, teacher, sandbox, args.output_dir,
          store=store, embedder=embedder, workers=args.workers, k=args.k, seed=args.seed,
          no_rag_only=args.no_rag_only)), indent=2))


if __name__ == "__main__":
    main()
