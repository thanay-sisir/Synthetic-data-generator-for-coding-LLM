#!/usr/bin/env bash
# Build the vLLM serving environment on the standard partition, not a login node.
# vLLM pulls its own torch and CUDA libraries, so this venv is separate from
# /scratch/qdf7ea/venvs/kernel-train-qwen38 on purpose.
#SBATCH --job-name=kernel-setup-vllm
#SBATCH --account=orsdardencomputing
#SBATCH --partition=standard
#SBATCH --cpus-per-task=8
#SBATCH --mem=128G
#SBATCH --time=02:00:00
#SBATCH --chdir=/project/orsDarden/thanay/personal_project/Synthetic-data-generator-for-coding-LLM
#SBATCH --output=/scratch/qdf7ea/logs/setup-vllm-%j.out
set -euo pipefail

PROJECT_DIR=/project/orsDarden/thanay/personal_project/Synthetic-data-generator-for-coding-LLM
ENV_DIR=/scratch/qdf7ea/venvs/kernel-serve-py313
cd "$PROJECT_DIR"

module load gcc/14.2.0
module load python/3.13.11
module load cuda/12.8.0 2>/dev/null || true

if [[ ! -x "$ENV_DIR/bin/python" ]]; then
  python3 -m venv "$ENV_DIR"
fi
"$ENV_DIR/bin/python" -m pip install --upgrade pip setuptools wheel

# shellcheck disable=SC1091
source "$ENV_DIR/bin/activate"
python -m pip install -r training/requirements-vllm.txt
python -m pip install -e "$PROJECT_DIR"

python - <<'PY'
import importlib.metadata as md
import torch, vllm, transformers
print("torch", torch.__version__, "cuda", torch.version.cuda)
print("transformers", transformers.__version__)
print("vllm", vllm.__version__)
print("cuda_available", torch.cuda.is_available())
# peft belongs to the training env and is intentionally absent here.
try:
    print("peft_installed", md.version("peft"))
except md.PackageNotFoundError:
    print("peft_installed", "absent")
PY
echo "SETUP COMPLETE"
