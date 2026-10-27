#!/usr/bin/env bash
# DEPRECATED. Kept only as a pointer; do not submit.
# This script loaded python/3.11.4 (removed by the site on 2026-10-23), sourced
# the old TRAIN_ENV, and read artifacts/sft, which no longer matches the data.
# Use slurm/train_scratch.sbatch instead.
echo "slurm/train_baseline.sbatch is retired. Use slurm/train_scratch.sbatch." >&2
exit 2
