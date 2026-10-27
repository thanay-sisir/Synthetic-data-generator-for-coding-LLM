"""Metrics that preserve failures and distinguish unknown costs from zero."""
from __future__ import annotations

import math
from typing import Any, Iterable


def percentile(values: Iterable[float], q: float) -> float | None:
    values = sorted(values)
    if not values:
        return None
    if not 0 <= q <= 1:
        raise ValueError("q must lie between 0 and 1")
    rank = (len(values) - 1) * q
    low, high = math.floor(rank), math.ceil(rank)
    return values[low] + (values[high] - values[low]) * (rank - low)


def summarize_attempts(rows: list[dict[str, Any]]) -> dict[str, Any]:
    passed = sum(row.get("validation", {}).get("status") == "passed" for row in rows)
    costs = [row.get("teacher", {}).get("cost_usd") for row in rows]
    known_cost = sum(row.get("teacher", {}).get("cost_usd")
                     if row.get("teacher", {}).get("cost_usd") is not None
                     else row.get("teacher", {}).get("known_successful_response_cost_usd") or 0 for row in rows)
    total_cost = known_cost if all(cost is not None for cost in costs) else None
    latencies = [r["latency_seconds"] for r in rows if r.get("latency_seconds") is not None]
    prompt_counts = [r.get("teacher", {}).get("prompt_tokens") for r in rows]
    completion_counts = [r.get("teacher", {}).get("completion_tokens") for r in rows]
    usage_complete = all(x is not None for x in prompt_counts + completion_counts) and all(
        r.get("teacher", {}).get("api_attempts", 1) == 1 for r in rows)
    return {
        "attempts": len(rows), "verified": passed,
        "validation_pass_rate": passed / len(rows) if rows else None,
        "known_api_cost_usd": known_cost,
        "api_cost_usd": total_cost,
        "cost_per_verified_pair_usd": total_cost / passed if passed and total_cost is not None else None,
        "p50_latency_seconds": percentile(latencies, .5),
        "p95_latency_seconds": percentile(latencies, .95),
        "known_prompt_tokens": sum(x for x in prompt_counts if x is not None),
        "known_completion_tokens": sum(x for x in completion_counts if x is not None),
        "prompt_tokens": sum(prompt_counts) if usage_complete else None,
        "completion_tokens": sum(completion_counts) if usage_complete else None,
        "token_usage_complete": usage_complete,
    }


def markdown_reports(paths: list[str]) -> str:
    """Render measured summaries; an empty/missing measurement never becomes zero."""
    import json
    from pathlib import Path
    lines = ["| Run / condition | Attempts | Verified | Pass rate | p50 (s) | p95 (s) | Cost / verified pair (USD) | Aggregate output tokens/s |",
             "|---|---:|---:|---:|---:|---:|---:|---:|"]
    def fmt(value):
        return "not measured / unknown" if value is None else f"{value:.4g}" if isinstance(value, float) else str(value)
    for name in paths:
        path = Path(name)
        report = json.loads(path.read_text(encoding="utf-8"))
        entries = report.get("conditions", {"endpoint benchmark": report})
        for condition, values in entries.items():
            row = [f"{path.parent.name} / {condition}", values.get("attempts", values.get("requests")), values.get("verified"),
                   values.get("validation_pass_rate"), values.get("p50_latency_seconds", values.get("p50_latency_seconds_successful_requests")),
                   values.get("p95_latency_seconds", values.get("p95_latency_seconds_successful_requests")),
                   values.get("cost_per_verified_pair_usd"), values.get("aggregate_output_tokens_per_second")]
            lines.append("| " + " | ".join(fmt(value).replace("|", "\\|") for value in row) + " |")
    return "\n".join(lines) + "\n"


def main() -> None:
    import argparse
    parser = argparse.ArgumentParser(description="Render measured evaluation/benchmark summary JSON as a Markdown table")
    parser.add_argument("reports", nargs="+")
    args = parser.parse_args()
    print(markdown_reports(args.reports), end="")


if __name__ == "__main__":
    main()
