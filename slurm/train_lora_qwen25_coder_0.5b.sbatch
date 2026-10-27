#!/usr/bin/env bash
# LoRA of Qwen2.5-Coder-0.5B-Instruct on the same checked kernels as Qwen3.8-27B.
# One epoch, rank 16, and learning rate 1e-4 match that run, so the base model
# is the only change. One A100 80GB. This model is about 0.5B parameters.
# Do not use slurm/train_lora.sbatch: it still loads the removed python/3.11.4.
#SBATCH --job-name=kernel-lora-qwen25
#SBATCH --account=orsdardencomputing
#SBATCH --partition=gpu
#SBATCH --gres=gpu:a100:1
#SBATCH --constraint=a100_80gb
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=02:00:00
#SBATCH --chdir=/project/orsDarden/thanay/personal_project/Synthetic-data-generator-for-coding-LLM
#SBATCH --output=/scratch/qdf7ea/logs/lora-qwen25-0.5b-%j.out
set -euo pipefail

PROJECT_DIR=/project/orsDarden/thanay/personal_project/Synthetic-data-generator-for-coding-LLM
cd "$PROJECT_DIR"
PY=/scratch/qdf7ea/venvs/kernel-train-qwen38/bin/python
DATA=$PROJECT_DIR/artifacts/sft-qwen38
OUT=$PROJECT_DIR/artifacts/lora-qwen25-coder-0.5b

test -x "$PY"
test -s "$DATA/train.jsonl"
test -s "$DATA/validation.jsonl"
if [[ -d "$OUT" ]] && [[ -n "$(ls -A "$OUT" 2>/dev/null || true)" ]]; then
  echo "Output $OUT is not empty. Pick a new directory before resubmitting." >&2
  exit 2
fi

module load gcc/14.2.0
module load python/3.13.11

export HF_HOME=/scratch/qdf7ea/huggingface
export TOKENIZERS_PARALLELISM=false
export HF_HUB_ENABLE_HF_TRANSFER=0
mkdir -p "$HF_HOME"

nvidia-smi
srun "$PY" -m training.train_lora \
  --data "$DATA" \
  --output "$OUT" \
  --model Qwen/Qwen2.5-Coder-0.5B-Instruct \
  --epochs 1 \
  --max-length 4096 \
  --rank 16 \
  --batch-size 1 \
  --accumulation 16 \
  --learning-rate 1e-4
