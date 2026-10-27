#!/usr/bin/env bash
# LoRA of Qwen3.8-27B on one A100 80GB.
# The trainer refuses more than one visible GPU. Unsloth's published LoRA
# floor for this model is a little over 36 GB, so 80 GB is the Rivanna GPU
# that both fits the model and matches that one-GPU rule. B200 nodes were
# fully occupied; an A100 80GB is also what this account gets quickly.
# Do not submit this until slurm/setup_qwen38.sbatch has printed SETUP COMPLETE.
#SBATCH --job-name=kernel-lora-qwen38
#SBATCH --account=orsdardencomputing
#SBATCH --partition=gpu
#SBATCH --gres=gpu:a100:1
#SBATCH --constraint=a100_80gb
#SBATCH --cpus-per-task=8
#SBATCH --mem=256G
#SBATCH --time=08:00:00
#SBATCH --chdir=/project/orsDarden/thanay/personal_project/Synthetic-data-generator-for-coding-LLM
#SBATCH --output=/scratch/qdf7ea/logs/lora-qwen38-%j.out
set -euo pipefail

PROJECT_DIR=/project/orsDarden/thanay/personal_project/Synthetic-data-generator-for-coding-LLM
cd "$PROJECT_DIR"
PY=/scratch/qdf7ea/venvs/kernel-train-qwen38/bin/python
DATA=$PROJECT_DIR/artifacts/sft-qwen38
OUT=$PROJECT_DIR/artifacts/lora-qwen38-27b

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
  --model Qwen/Qwen3.8-27B \
  --epochs 1 \
  --max-length 4096 \
  --rank 16 \
  --batch-size 1 \
  --accumulation 16 \
  --learning-rate 1e-4
