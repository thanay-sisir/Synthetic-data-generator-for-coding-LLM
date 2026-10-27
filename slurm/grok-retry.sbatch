#!/usr/bin/env bash
# Second pass on the training holes: tanh_mix, row_mean, row_square_sum, and SSL errors.
# Four array tasks, one A100 each. Three Grok calls stay in flight per GPU.
# The GPU check is shorter than the API call, so more than one GPU helps and
# more than about three calls per GPU does not.
#SBATCH --job-name=kernel-grok-retry
#SBATCH --account=orsdardencomputing
#SBATCH --partition=gpu
#SBATCH --gres=gpu:a100:1
#SBATCH --constraint=a100_80gb
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=03:00:00
#SBATCH --array=0-3
#SBATCH --chdir=/project/orsDarden/thanay/personal_project/Synthetic-data-generator-for-coding-LLM
#SBATCH --output=/scratch/qdf7ea/logs/grok-retry-%A_%a.out
set -euo pipefail

PROJECT_DIR=/project/orsDarden/thanay/personal_project/Synthetic-data-generator-for-coding-LLM
cd "$PROJECT_DIR"
SHARD="${SLURM_ARRAY_TASK_ID:?}"

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
TASKS="$PROJECT_DIR/artifacts/tasks-grok-retry-s${SHARD}.jsonl"
OUT="$PROJECT_DIR/artifacts/grok-retry-s${SHARD}"

"$PY" -c 'import httpx, kernel_lab; print("client-ok", httpx.__version__)'
test -d "$IMAGE"
test -f "$IMAGE/opt/runner.py"
test -s "$TASKS"
nvidia-smi

srun "$PY" -m kernel_lab.smoke --backend apptainer --allow-weaker-isolation \
  --image "$IMAGE" --timeout 180 \
  --output "/scratch/qdf7ea/logs/grok-retry-smoke-${SLURM_ARRAY_JOB_ID}_${SHARD}.json"

srun "$PY" -m kernel_lab.generate \
  --tasks "$TASKS" \
  --output-dir "$OUT" \
  --split train \
  --workers 3 --validation-workers 1 --attempts 1 \
  --backend apptainer --allow-weaker-isolation \
  --image "$IMAGE" --timeout 120
