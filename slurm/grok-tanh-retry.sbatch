#!/usr/bin/env bash
# Retry the 178 tanh_mix tasks whose Grok calls died when credits ran out.
# One A100, three API calls in flight. The prompt already forbids tl.tanh.
#SBATCH --job-name=kernel-grok-tanh
#SBATCH --account=orsdardencomputing
#SBATCH --partition=gpu
#SBATCH --gres=gpu:a100:1
#SBATCH --constraint=a100_80gb
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=02:00:00
#SBATCH --chdir=/project/orsDarden/thanay/personal_project/Synthetic-data-generator-for-coding-LLM
#SBATCH --output=/scratch/qdf7ea/logs/grok-tanh-%j.out
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

"$PY" -c 'import httpx, kernel_lab; from kernel_lab.prompts import SYSTEM_PROMPT; assert "tl.math.tanh" in SYSTEM_PROMPT; print("client-ok", httpx.__version__)'
test -d "$IMAGE"
test -f "$IMAGE/opt/runner.py"
nvidia-smi

srun "$PY" -m kernel_lab.smoke --backend apptainer --allow-weaker-isolation \
  --image "$IMAGE" --timeout 180 \
  --output /scratch/qdf7ea/logs/grok-tanh-smoke.json

srun "$PY" -m kernel_lab.generate \
  --tasks "$PROJECT_DIR/artifacts/tasks-grok-tanh-retry.jsonl" \
  --output-dir "$PROJECT_DIR/artifacts/grok-tanh-retry" \
  --split train \
  --workers 3 --validation-workers 1 --attempts 1 \
  --backend apptainer --allow-weaker-isolation \
  --image "$IMAGE" --timeout 120
