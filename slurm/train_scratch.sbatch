#!/usr/bin/env bash
# Train the corrected from-scratch Transformer on the checked kernels.
# One A100 80GB. About 30M parameters, so the run itself is minutes; the
# 6-hour reservation is for queue time, not compute.
# Do not use slurm/train_baseline.sbatch: it still loads the removed python/3.11.4.
#SBATCH --job-name=kernel-scratch
#SBATCH --account=orsdardencomputing
#SBATCH --partition=gpu
#SBATCH --gres=gpu:a100:1
#SBATCH --constraint=a100_80gb
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=06:00:00
#SBATCH --chdir=/project/orsDarden/thanay/personal_project/Synthetic-data-generator-for-coding-LLM
#SBATCH --output=/scratch/qdf7ea/logs/scratch-%j.out
set -euo pipefail

module load gcc/14.2.0
module load python/3.13.11
export HF_HOME=/scratch/qdf7ea/huggingface
export TOKENIZERS_PARALLELISM=false

PY=/scratch/qdf7ea/venvs/kernel-train-qwen38/bin/python
PROJECT=/project/orsDarden/thanay/personal_project/Synthetic-data-generator-for-coding-LLM
DATA=$PROJECT/artifacts/sft-qwen38
OUT=${SCRATCH_OUT:-$PROJECT/artifacts/scratch-qwen38}

test -x "$PY"
test -s "$DATA/train.jsonl"
test -s "$DATA/validation.jsonl"
if [[ -d "$OUT" ]] && [[ -n "$(ls -A "$OUT" 2>/dev/null || true)" ]]; then
  echo "Output $OUT is not empty. Pick a new directory before resubmitting." >&2
  exit 2
fi

srun "$PY" -m kernel_lab.baseline \
  --train-file "$DATA/train.jsonl" \
  --val-file "$DATA/validation.jsonl" \
  --output-dir "$OUT" \
  --epochs "${EPOCHS:-3}" --max-length 4096 \
  --batch-size 1 --accumulation 8 --lr "${LR:-3e-4}"
