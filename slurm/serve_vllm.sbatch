#!/usr/bin/env bash
# One A100 80 GB so vLLM answers faster during evaluation.
# Four hours is the reservation cap, not the expected runtime.
#SBATCH --job-name=kernel-serve
#SBATCH --account=orsdardencomputing
#SBATCH --partition=gpu
#SBATCH --gres=gpu:a100:1
#SBATCH --constraint=a100_80gb
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=04:00:00
#SBATCH --chdir=/project/orsDarden/thanay/personal_project/Synthetic-data-generator-for-coding-LLM
#SBATCH --output=/scratch/qdf7ea/logs/vllm-%j.out
set -euo pipefail
PROJECT_DIR=/project/orsDarden/thanay/personal_project/Synthetic-data-generator-for-coding-LLM
cd "$PROJECT_DIR"
if [[ -f .env ]]; then
  set -a
  # shellcheck disable=SC1091
  source .env
  set +a
fi
SERVE_ENV="${SERVE_ENV:-/scratch/qdf7ea/venvs/kernel-serve}"
if [[ -z "${VLLM_API_KEY:-}" ]]; then
  VLLM_API_KEY=$(python3 -c 'import secrets; print(secrets.token_urlsafe(32))')
  echo "Generated VLLM_API_KEY for this job; copy it from this log to call the server."
fi
export VLLM_API_KEY
module load gcc/11.4.0 openmpi/4.1.4 python/3.11.4
source "$SERVE_ENV/bin/activate"
export HF_HOME="${HF_HOME:-/scratch/$USER/huggingface}"
echo "Node: $(hostname -f); endpoint is loopback port ${PORT:-8000}; job $SLURM_JOB_ID"
nvidia-smi
srun bash scripts/serve_vllm.sh
