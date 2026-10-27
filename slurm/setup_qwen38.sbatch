#!/usr/bin/env bash
# CPU node. Login nodes are capped at 6 GB and cannot unpack this install.
# Builds the training environment that can load Qwen3.8-27B.
#SBATCH --job-name=kernel-qwen38-setup
#SBATCH --account=orsdardencomputing
#SBATCH --partition=standard
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=128G
#SBATCH --time=02:00:00
#SBATCH --chdir=/project/orsDarden/thanay/personal_project/Synthetic-data-generator-for-coding-LLM
#SBATCH --output=/scratch/qdf7ea/logs/qwen38-setup-%j.out
set -euo pipefail

PROJECT_DIR=/project/orsDarden/thanay/personal_project/Synthetic-data-generator-for-coding-LLM
ENV=/scratch/qdf7ea/venvs/kernel-train-qwen38
export PIP_CACHE_DIR=/scratch/qdf7ea/pip-cache/qwen38
export HF_HOME=/scratch/qdf7ea/huggingface
mkdir -p "$PIP_CACHE_DIR" "$HF_HOME" /scratch/qdf7ea/logs

module load gcc/14.2.0
module load python/3.13.11

if [[ ! -x "$ENV/bin/python" ]]; then
  python3 -m venv "$ENV"
fi
"$ENV/bin/python" -m pip install --upgrade pip
"$ENV/bin/python" -m pip install -r "$PROJECT_DIR/training/requirements-qwen38.txt"
"$ENV/bin/python" -m pip install -e "$PROJECT_DIR"
"$ENV/bin/python" - <<'PY'
import torch
from transformers import AutoModelForImageTextToText
import peft
print("setup-ok", torch.__version__, torch.version.cuda, peft.__version__)
print("qwen38-class", AutoModelForImageTextToText.__name__)
PY
echo "SETUP COMPLETE"
