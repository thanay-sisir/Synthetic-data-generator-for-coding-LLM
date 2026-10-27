"""Non-streaming endpoint load benchmark; no fabricated TTFT or GPU costs."""
from __future__ import annotations

import argparse
import asyncio
import json
from pathlib import Path
import time
import hashlib
from datetime import datetime, timezone
from typing import Any

from .metrics import percentile


async def benchmark(tasks: list[dict[str, Any]], teacher: Any, output_dir: str | Path,
                    requests: int = 20, workers: int = 1, warmup: int = 1) -> dict[str, Any]:
    if not tasks or requests < 1 or workers < 1 or warmup < 0:
        raise ValueError("Require tasks, positive requests/workers, and nonnegative warmup")
    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=True)
    path = output / "benchmark.jsonl"
    if path.exists():
        raise FileExistsError("Choose a fresh benchmark output directory")
    path.touch(exist_ok=False)
    # Warmup errors abort the benchmark, so a dead endpoint cannot look fast.
    warmup_rows = []
    for index in range(warmup):
        result = await teacher.complete(tasks[index % len(tasks)], examples=[])
        warmup_rows.append({key: value for key, value in result.items() if key != "candidate_code"})
        (output / "warmup.jsonl").write_text("".join(json.dumps(row) + "\n" for row in warmup_rows), encoding="utf-8")
    semaphore = asyncio.Semaphore(workers)
    rows = []
    async def request(index: int) -> None:
        async with semaphore:
            start = time.perf_counter()
            row = {"request_id": index, "task_id": tasks[index % len(tasks)]["task_id"], "success": False}
            try:
                result = await teacher.complete(tasks[index % len(tasks)], examples=[])
                row.update({key: result.get(key) for key in ("prompt_tokens", "completion_tokens", "cost_usd", "model")})
                row["success"] = True
            except Exception as exc:
                row.update(error=f"{type(exc).__name__}: {exc}", cost_usd=None)
            row["latency_seconds"] = time.perf_counter() - start
            with path.open("a", encoding="utf-8") as handle:
                handle.write(json.dumps(row) + "\n")
                handle.flush()
            rows.append(row)
    start = time.perf_counter()
    await asyncio.gather(*(request(i) for i in range(requests)))
    duration = time.perf_counter() - start
    successful = [r for r in rows if r["success"]]
    tokens_known = all(r.get("completion_tokens") is not None for r in successful)
    tokens = sum(r.get("completion_tokens") or 0 for r in successful)
    costs_known = all(r.get("cost_usd") is not None for r in rows)
    config = getattr(teacher, "config", None)
    warmup_costs = [row.get("cost_usd") for row in warmup_rows]
    report = {"requests": requests, "successful_requests": len(successful), "failed_requests": requests - len(successful),
              "generation_settings": {key: getattr(config, key, None) for key in ("model", "base_url", "max_tokens", "temperature", "retries", "timeout")},
              "completed_at_utc": datetime.now(timezone.utc).isoformat(),
              "task_set_sha256": hashlib.sha256(json.dumps(tasks, sort_keys=True).encode()).hexdigest(),
              "workers": workers, "warmup_requests_excluded": warmup, "wall_seconds": duration,
              "successful_requests_per_second": len(successful) / duration,
              "aggregate_output_tokens_per_second": tokens / duration if successful and tokens_known else None,
              "p50_latency_seconds_all_requests": percentile([r["latency_seconds"] for r in rows], .5),
              "p95_latency_seconds_all_requests": percentile([r["latency_seconds"] for r in rows], .95),
              "p50_latency_seconds_successful_requests": percentile([r["latency_seconds"] for r in successful], .5),
              "p95_latency_seconds_successful_requests": percentile([r["latency_seconds"] for r in successful], .95),
              "p50_per_request_output_tokens_per_second": percentile([r["completion_tokens"] / r["latency_seconds"] for r in successful if r.get("completion_tokens") is not None], .5),
              "api_cost_usd": sum(r["cost_usd"] for r in rows) if costs_known else None,
              "warmup_api_cost_usd_excluded_from_measurement": sum(warmup_costs) if all(cost is not None for cost in warmup_costs) else None,
              "ttft_seconds": None, "measurement": "Non-streaming complete-response latency after worker admission; includes client HTTP and model queueing; excludes client worker queue and warmup; output throughput includes failures in elapsed wall time. Not isolated GPU decode speed. No kernel validation performed."}
    (output / "summary.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    return report


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--tasks", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--requests", type=int, default=20)
    parser.add_argument("--workers", type=int, default=1)
    parser.add_argument("--warmup", type=int, default=1)
    args = parser.parse_args()
    from .teacher import TeacherConfig, TeacherClient
    tasks = [json.loads(line) for line in Path(args.tasks).read_text(encoding="utf-8").splitlines() if line.strip()]
    print(json.dumps(asyncio.run(benchmark(tasks, TeacherClient(TeacherConfig.from_env()), args.output_dir,
                                         args.requests, args.workers, args.warmup)), indent=2))


if __name__ == "__main__":
    main()
