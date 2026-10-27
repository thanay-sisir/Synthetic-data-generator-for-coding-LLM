#!/usr/bin/env bash
# GPU smoke only. Submit after slurm/setup.sbatch has written the SIF.
# Does not call OpenRouter.
#SBATCH --job-name=kernel-smoke
#SBATCH --account=orsdardencomputing
#SBATCH --partition=gpu
#SBATCH --gres=gpu:a100:1
#SBATCH --constraint=a100_80gb
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=00:30:00
#SBATCH --chdir=/project/orsDarden/thanay/personal_project/Synthetic-data-generator-for-coding-LLM
#SBATCH --output=/scratch/qdf7ea/logs/smoke-%j.out
set -euo pipefail
PROJECT_DIR=/project/orsDarden/thanay/personal_project/Synthetic-data-generator-for-coding-LLM
CLIENT_ENV=/scratch/qdf7ea/venvs/kernel-client
VALIDATOR_SIF=/scratch/qdf7ea/containers/kernel-lab-validator.sif
cd "$PROJECT_DIR"
if [[ -f .env ]]; then
  set -a
  # shellcheck disable=SC1091
  source .env
  set +a
fi
test -f "$VALIDATOR_SIF" || { echo "Missing SIF. Submit slurm/setup.sbatch first." >&2; exit 2; }
module load gcc/11.4.0 openmpi/4.1.4 python/3.11.4 apptainer/1.5.0
source "$CLIENT_ENV/bin/activate"
nvidia-smi
srun python -m kernel_lab.smoke --backend apptainer --allow-weaker-isolation \
  --image "$VALIDATOR_SIF" --timeout 180 \
  --output /scratch/qdf7ea/logs/smoke-result.json
echo "SMOKE COMPLETE"
cat /scratch/qdf7ea/logs/smoke-result.json
