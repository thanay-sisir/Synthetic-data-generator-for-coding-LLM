#!/usr/bin/env bash
# Score the Qwen2.5-Coder-0.5B LoRA on the 1,200 held-out test tasks.
# Submit this only after slurm/train_lora_qwen25_coder_0.5b.sbatch has completed.
# One A100 is enough: the weights are about 1 GB, and the checker runs after
# each generation on that same GPU. Do not set KERNEL_LAB_SANDBOX_GPU=1 here.
#SBATCH --job-name=kernel-eval-qwen25
#SBATCH --account=orsdardencomputing
#SBATCH --partition=gpu
#SBATCH --gres=gpu:a100:1
#SBATCH --constraint=a100_80gb
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=08:00:00
#SBATCH --chdir=/project/orsDarden/thanay/personal_project/Synthetic-data-generator-for-coding-LLM
#SBATCH --output=/scratch/qdf7ea/logs/eval-qwen25-0.5b-%j.out
set -euo pipefail

module load gcc/14.2.0
module load python/3.13.11
module load apptainer/1.5.0
export HF_HOME=/scratch/qdf7ea/huggingface
export TOKENIZERS_PARALLELISM=false
unset KERNEL_LAB_SANDBOX_GPU

PY=/scratch/qdf7ea/venvs/kernel-train-qwen38/bin/python
PROJECT=/project/orsDarden/thanay/personal_project/Synthetic-data-generator-for-coding-LLM
ADAPTER=$PROJECT/artifacts/lora-qwen25-coder-0.5b/adapter
OUT=$PROJECT/artifacts/eval-qwen25-coder-0.5b-test

test -x "$PY"
test -f "$ADAPTER/adapter_config.json"
test -d /scratch/qdf7ea/containers/kernel-lab-validator
if [[ -e "$OUT/evaluation.jsonl" ]]; then
  echo "Output $OUT already has a ledger." >&2
  exit 2
fi

srun "$PY" -m kernel_lab.evaluate \
  --tasks "$PROJECT/artifacts/tasks-v1.jsonl" \
  --split test \
  --output-dir "$OUT" \
  --local-qwen Qwen/Qwen2.5-Coder-0.5B-Instruct \
  --qwen-adapter "$ADAPTER" \
  --max-new-tokens 1024 \
  --no-rag-only \
  --workers 1 \
  --sandbox-backend apptainer \
  --allow-weaker-isolation \
  --sandbox-image /scratch/qdf7ea/containers/kernel-lab-validator \
  --gpu-device 0 \
  --sandbox-timeout 120
