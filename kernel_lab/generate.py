"""Asynchronously generate candidates, validate in containers, persist every attempt."""
import argparse
import asyncio
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import time
from .schema import digest, read_jsonl, validate_task
from .teacher import TeacherClient, TeacherConfig
from .sandbox import DockerSandbox


async def run_generation(tasks, teacher, sandbox, output_dir, workers=4, validation_workers=1, attempts=1, retriever=None):
    if min(workers, validation_workers, attempts) <= 0:
        raise ValueError("worker and attempt counts must be positive")
    for task in tasks:
        validate_task(task)
    seen_ids, families, references = set(), {}, {}
    for task in tasks:
        if task["task_id"] in seen_ids:
            raise ValueError("Duplicate task_id in task input")
        seen_ids.add(task["task_id"])
        for mapping, key in [(families, task["family"]), (references, digest(task["reference_code"]))]:
            if key in mapping and mapping[key] != task["split"]:
                raise ValueError("Family/reference crosses data splits")
            mapping[key] = task["split"]
    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=True)
    # New output directory avoids mixing experiments or silently resuming with changed settings.
    ledger_path, verified_path = output / "attempts.jsonl", output / "verified.jsonl"
    if ledger_path.exists() or verified_path.exists():
        raise ValueError("Output already contains a run; choose a fresh --output-dir")
    ledger_path.touch()
    verified_path.touch()
    queue = asyncio.Queue(maxsize=workers * 2)
    validation_gate = asyncio.Semaphore(validation_workers)
    write_lock = asyncio.Lock()
    verified_hashes = set()
    counts = {"tasks": len(tasks), "attempts": 0, "verified": 0, "failed": 0, "error": 0, "timeout": 0}
    costs, unknown_costs = [], 0
    started = time.perf_counter()
    manifest = {"schema_version": 1, "started_at": datetime.now(timezone.utc).isoformat(), "task_manifest_sha256": digest(tasks),
                "workers": workers, "validation_workers": validation_workers, "attempts_per_task": attempts,
                "sandbox_backend": getattr(sandbox, "backend", "unknown"), "sandbox_image": getattr(sandbox, "image", "unknown"),
                "teacher_model": getattr(getattr(teacher, "config", None), "model", "unknown"),
                "cost_note": "Successful response token cost only; failed/ambiguous HTTP requests may also be billed. Embedding, validation and GPU costs excluded."}
    teacher_config = getattr(teacher, "config", None)
    if teacher_config:
        manifest["teacher_settings"] = {key: getattr(teacher_config, key, None) for key in (
            "model", "timeout", "retries", "max_tokens", "temperature", "input_price_per_million", "output_price_per_million", "free_only", "request_interval")}
    manifest["retrieval_enabled"] = retriever is not None
    (output / "selected_tasks.jsonl").write_text("".join(json.dumps(task) + "\n" for task in tasks), encoding="utf-8")
    (output / "manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")

    async def worker():
        nonlocal unknown_costs
        while True:
            task = await queue.get()
            if task is None:
                queue.task_done()
                return
            try:
                for attempt in range(attempts):
                    record = {"schema_version": 1, "task": task, **{key: task[key] for key in ("task_id", "family", "split", "prompt", "reference_code")},
                              "attempt_index": attempt, "candidate_id": digest([task["task_id"], attempt, manifest["started_at"]])[:24]}
                    try:
                        examples = await retriever(task) if retriever else []
                        response = await teacher.complete(task, examples=examples)
                        code = response.pop("candidate_code")
                        record.update(candidate_code=code, teacher=response,
                                      retrieved_task_ids=[ex.get("task_id") for ex in examples],
                                      retrieved_examples=examples, retrieved_examples_sha256=digest(examples))
                        async with validation_gate:
                            result = await sandbox.validate(task, code)
                        record["validation"] = result
                    except Exception as error:
                        record["validation"] = {"status": "error", "error": f"{type(error).__name__}: {str(error)[:1000]}"}
                    status = record["validation"]["status"]
                    async with write_lock:
                        counts["attempts"] += 1
                        teacher_usage = record.get("teacher", {})
                        cost = teacher_usage.get("known_successful_response_cost_usd", teacher_usage.get("cost_usd"))
                        if teacher_usage.get("cost_usd") is None:
                            unknown_costs += 1
                        if cost is not None:
                            costs.append(cost)
                        with ledger_path.open("a", encoding="utf-8") as handle:
                            handle.write(json.dumps(record) + "\n")
                            handle.flush()
                        if status == "passed":
                            content_hash = digest([task["reference_code"], record["candidate_code"]])
                            if content_hash not in verified_hashes:
                                verified_hashes.add(content_hash)
                                counts["verified"] += 1
                                with verified_path.open("a", encoding="utf-8") as handle:
                                    handle.write(json.dumps(record) + "\n")
                                    handle.flush()
                        else:
                            counts[status if status in counts else "error"] += 1
                    # One verified implementation per task; no paid repeated successes.
                    if status == "passed":
                        break
            finally:
                queue.task_done()

    async def produce():
        for task in tasks:
            await queue.put(task)
        for _ in range(workers):
            await queue.put(None)

    consumers = [asyncio.create_task(worker()) for _ in range(workers)]
    producer = asyncio.create_task(produce())
    try:
        await asyncio.gather(producer, *consumers)
    finally:
        for consumer in [producer, *consumers]:
            if not consumer.done():
                consumer.cancel()
        await asyncio.gather(producer, *consumers, return_exceptions=True)
    elapsed = time.perf_counter() - started
    known_cost = sum(costs)
    summary = {**counts, "elapsed_seconds": elapsed, "verified_pairs_per_hour": counts["verified"] * 3600 / elapsed,
               "known_response_token_cost_usd": known_cost if costs else None, "unknown_cost_attempts": unknown_costs,
               "response_token_cost_per_verified_pair_usd": known_cost / counts["verified"] if counts["verified"] and not unknown_costs else None,
               "total_cost_per_verified_pair_usd": None, "total_cost_note": "Not measured: requires provider invoice reconciliation plus embedding, GPU and validation allocation costs."}
    (output / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    return summary


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--tasks", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--split", choices=["train", "validation", "test"], default="train")
    parser.add_argument("--limit", type=int)
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--validation-workers", type=int, default=1)
    parser.add_argument("--attempts", type=int, default=1)
    parser.add_argument("--image", default="kernel-lab-validator:latest")
    parser.add_argument("--backend", choices=["docker", "apptainer"], default="docker")
    parser.add_argument("--allow-weaker-isolation", action="store_true")
    parser.add_argument("--timeout", type=float, default=120)
    parser.add_argument("--gpu-device", default="0")
    parser.add_argument("--retrieval-k", type=int, default=0)
    args = parser.parse_args()
    tasks = [task for task in read_jsonl(args.tasks) if task["split"] == args.split]
    if args.limit is not None:
        if args.limit <= 0:
            parser.error("--limit must be positive")
        tasks = tasks[:args.limit]
    if not tasks:
        parser.error("No tasks selected")
    teacher = TeacherClient(TeacherConfig.from_env())
    sandbox = DockerSandbox(image=args.image, timeout=args.timeout, gpu_device=args.gpu_device, backend=args.backend, allow_weaker_isolation=args.allow_weaker_isolation)
    if args.retrieval_k < 0:
        parser.error("--retrieval-k must be nonnegative")

    async def execute():
        retriever = None
        if args.retrieval_k:
            from .retrieval import EmbeddingConfig, EmbeddingClient, PgVectorStore, task_text
            config = EmbeddingConfig.from_env()
            store = PgVectorStore(os.environ["PGVECTOR_DSN"], config)
            await asyncio.to_thread(store.initialize)
            embedder = EmbeddingClient(config)

            async def retrieve(task):
                vector = await embedder.embed(task_text(task))
                return await asyncio.to_thread(store.search, task, vector, args.retrieval_k, [], [])

            retriever = retrieve
        return await run_generation(tasks, teacher, sandbox, args.output_dir, args.workers, args.validation_workers, args.attempts, retriever=retriever)

    summary = asyncio.run(execute())
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
