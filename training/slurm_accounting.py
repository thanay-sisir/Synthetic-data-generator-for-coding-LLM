"""Record completed Slurm job allocation accounting without inventing a dollar rate."""
import argparse
import json
import re
import subprocess
from pathlib import Path


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--job-id", required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if not re.fullmatch(r"[0-9]+", args.job_id):
        raise ValueError("Use a numeric job id")
    command = ["sacct", "-j", args.job_id, "-X", "-n", "-P", "--format=JobIDRaw,State,ElapsedRaw,AllocTRES"]
    raw = subprocess.check_output(command, text=True)
    rows = [line.split("|") for line in raw.splitlines() if line.strip()]
    matches = [r for r in rows if r[0] == args.job_id]
    if len(matches) != 1:
        raise ValueError("Expected one allocation row; wait for Slurm accounting to settle")
    job, state, elapsed, tres = matches[0][:4]
    counts = dict(item.split("=", 1) for item in tres.split(",") if "=" in item)
    # Prefer aggregate GPU count; typed GPU TRES may be present in addition.
    gpu_count = int(counts["gres/gpu"]) if "gres/gpu" in counts else sum(
        int(value) for key, value in counts.items() if key.startswith("gres/gpu:"))
    if not gpu_count:
        raise ValueError("GPU allocation count unavailable; do not infer GPU-hours")
    if state in {"RUNNING", "PENDING", "COMPLETING"}:
        raise ValueError("Job is still active; collect accounting after it finishes")
    report = {"job_id": job, "state": state, "elapsed_seconds": int(elapsed),
              "allocated_gpu_count": gpu_count, "allocated_gpu_hours": int(elapsed) * gpu_count / 3600,
              "alloc_tres": tres, "source": command, "raw_sacct": raw,
              "cost_usd": None, "note": "Allocated GPU wall time, not GPU utilization. SU charges are not USD."}
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
