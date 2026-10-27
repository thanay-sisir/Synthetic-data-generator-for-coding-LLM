#!/usr/bin/env bash
# Remaining Grok 4.7 kernels: 973 unfinished train tasks, then 100 validation tasks.
# The October 18 client venv points at python/3.11.4, which Rivanna removed on
# 2026-10-23. This job uses the Python 3.13.11 client env instead.
#SBATCH --job-name=kernel-grok-rest
#SBATCH --account=orsdardencomputing
#SBATCH --partition=gpu
#SBATCH --gres=gpu:a100:1
#SBATCH --constraint=a100_80gb
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=14:00:00
#SBATCH --chdir=/project/orsDarden/thanay/personal_project/Synthetic-data-generator-for-coding-LLM
#SBATCH --output=/scratch/qdf7ea/logs/grok-rest-%j.out
set -euo pipefail

PROJECT_DIR=/project/orsDarden/thanay/personal_project/Synthetic-data-generator-for-coding-LLM
cd "$PROJECT_DIR"

module load gcc/14.2.0
module load python/3.13.11
module load apptainer/1.5.0

set -a
# shellcheck disable=SC1091
source .env
set +a

export TEACHER_BASE_URL="$XAI_BASE_URL"
export TEACHER_MODEL="$XAI_MODEL"
export TEACHER_API_KEY="$XAI_API_KEY"
export TEACHER_INPUT_PRICE_PER_MILLION="$XAI_INPUT_PRICE_PER_MILLION"
export TEACHER_OUTPUT_PRICE_PER_MILLION="$XAI_OUTPUT_PRICE_PER_MILLION"
export TEACHER_FREE_ONLY=0
export TEACHER_REASONING_EFFORT=low
export TEACHER_TIMEOUT=300

PY=/scratch/qdf7ea/venvs/kernel-client-py313/bin/python
IMAGE=/scratch/qdf7ea/containers/kernel-lab-validator

"$PY" -c 'import httpx, kernel_lab; print("client-ok", httpx.__version__)'
test -d "$IMAGE"
test -f "$IMAGE/opt/runner.py"
nvidia-smi

srun "$PY" -m kernel_lab.smoke --backend apptainer --allow-weaker-isolation \
  --image "$IMAGE" --timeout 180 \
  --output /scratch/qdf7ea/logs/grok-rest-smoke.json

srun "$PY" -m kernel_lab.generate \
  --tasks "$PROJECT_DIR/artifacts/tasks-grok-train-rest.jsonl" \
  --output-dir "$PROJECT_DIR/artifacts/grok-train-rest" \
  --split train --limit 973 \
  --workers 1 --validation-workers 1 --attempts 1 \
  --backend apptainer --allow-weaker-isolation \
  --image "$IMAGE" --timeout 120

srun "$PY" -m kernel_lab.generate \
  --tasks "$PROJECT_DIR/artifacts/tasks-grok-2100.jsonl" \
  --output-dir "$PROJECT_DIR/artifacts/grok-val-100" \
  --split validation --limit 100 \
  --workers 1 --validation-workers 1 --attempts 1 \
  --backend apptainer --allow-weaker-isolation \
  --image "$IMAGE" --timeout 120
