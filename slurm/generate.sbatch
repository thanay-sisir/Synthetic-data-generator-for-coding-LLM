#!/usr/bin/env bash
# One A100 80 GB to compile and check generated kernels quickly.
# The teacher API is still the slow part; a bigger GPU does not raise OpenRouter quota.
#SBATCH --job-name=kernel-generate
#SBATCH --account=orsdardencomputing
#SBATCH --partition=gpu
#SBATCH --gres=gpu:a100:1
#SBATCH --constraint=a100_80gb
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=02:00:00
#SBATCH --chdir=/project/orsDarden/thanay/personal_project/Synthetic-data-generator-for-coding-LLM
#SBATCH --output=/scratch/qdf7ea/logs/generate-%j.out
set -euo pipefail
# Run only on an allocated GPU compute node, after UVA approves this use of Apptainer.
# Apptainer shares the host kernel/network and GPU driver; it is weaker than Docker.
PROJECT_DIR=/project/orsDarden/thanay/personal_project/Synthetic-data-generator-for-coding-LLM
cd "$PROJECT_DIR"
if [[ -f .env ]]; then
  set -a
  # shellcheck disable=SC1091
  source .env
  set +a
fi
CLIENT_ENV="${CLIENT_ENV:-/scratch/qdf7ea/venvs/kernel-client}"
VALIDATOR_SIF="${VALIDATOR_SIF:-/scratch/qdf7ea/containers/kernel-lab-validator.sif}"
TASKS_FILE="${TASKS_FILE:-$PROJECT_DIR/artifacts/tasks-v1.jsonl}"
RUN_OUTPUT="${RUN_OUTPUT:-$PROJECT_DIR/artifacts/pilot-train}"
: "${TEACHER_BASE_URL:?Set the compatible provider URL, e.g. https://openrouter.ai/api/v1}"
: "${TEACHER_MODEL:?Set an exact model name available in your API account}"
: "${TEACHER_API_KEY:?Set your API key in the submission environment; never commit it}"
if [[ "${SITE_APPROVED_CONTAINER_VALIDATION:-0}" != "1" ]]; then
  echo 'Set SITE_APPROVED_CONTAINER_VALIDATION=1 only after site approval and review of generated-code risks.' >&2
  exit 2
fi
module load gcc/11.4.0 openmpi/4.1.4 python/3.11.4 apptainer/1.5.0
source "$CLIENT_ENV/bin/activate"
command -v apptainer >/dev/null || { echo 'Load the site Apptainer module first.' >&2; exit 2; }
test -f "$VALIDATOR_SIF"
nvidia-smi
# Real GPU smoke test is mandatory before API calls. Failure stops the job.
srun python -m kernel_lab.smoke --backend apptainer --allow-weaker-isolation \
  --image "$VALIDATOR_SIF" --timeout 180 --output "${RUN_OUTPUT}-smoke.json"
srun python -m kernel_lab.generate --tasks "$TASKS_FILE" --output-dir "$RUN_OUTPUT" \
  --backend apptainer --allow-weaker-isolation --image "$VALIDATOR_SIF" \
  --split "${TASK_SPLIT:-train}" --workers "${API_WORKERS:-4}" \
  --validation-workers 1 --attempts "${ATTEMPTS_PER_TASK:-1}" "$@"
