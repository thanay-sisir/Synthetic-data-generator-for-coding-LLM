#!/usr/bin/env bash
# CPU node with enough RAM. Login nodes are capped at 6 GB, which killed
# both the Apptainer build and the vLLM pip install.
#SBATCH --job-name=kernel-setup
#SBATCH --account=orsdardencomputing
#SBATCH --partition=standard
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=128G
#SBATCH --time=03:00:00
#SBATCH --chdir=/project/orsDarden/thanay/personal_project/Synthetic-data-generator-for-coding-LLM
#SBATCH --output=/scratch/qdf7ea/logs/setup-%j.out
set -euo pipefail

PROJECT_DIR=/project/orsDarden/thanay/personal_project/Synthetic-data-generator-for-coding-LLM
CLIENT_ENV=/scratch/qdf7ea/venvs/kernel-client
TRAIN_ENV=/scratch/qdf7ea/venvs/kernel-train
SERVE_ENV=/scratch/qdf7ea/venvs/kernel-serve
SIF=/scratch/qdf7ea/containers/kernel-lab-validator.sif
export HF_HOME=/scratch/qdf7ea/huggingface
export APPTAINER_CACHEDIR=/scratch/qdf7ea/apptainer-cache
export APPTAINER_TMPDIR=/scratch/qdf7ea/apptainer-tmp
export TMPDIR=/scratch/qdf7ea/apptainer-tmp
export PIP_CACHE_DIR=/scratch/qdf7ea/pip-cache/serve

mkdir -p "$HF_HOME" "$APPTAINER_CACHEDIR" "$APPTAINER_TMPDIR" \
  /scratch/qdf7ea/containers /scratch/qdf7ea/logs /scratch/qdf7ea/venvs \
  /scratch/qdf7ea/pip-cache/serve "$PROJECT_DIR/logs"

module load gcc/11.4.0 openmpi/4.1.4 python/3.11.4 apptainer/1.5.0

echo "host=$(hostname) mem=$(free -g | awk '/Mem:/{print $2}')G"
python3 --version
apptainer --version

if [[ ! -x "$CLIENT_ENV/bin/python" ]]; then
  python3 -m venv "$CLIENT_ENV"
  "$CLIENT_ENV/bin/python" -m pip install --upgrade pip
  "$CLIENT_ENV/bin/python" -m pip install -e "$PROJECT_DIR[retrieval,dev]"
fi
"$CLIENT_ENV/bin/python" -c 'import httpx, kernel_lab; print("client-ok", httpx.__version__)'

if [[ ! -x "$TRAIN_ENV/bin/python" ]]; then
  python3 -m venv "$TRAIN_ENV"
  "$TRAIN_ENV/bin/python" -m pip install --upgrade pip
fi
"$TRAIN_ENV/bin/python" -c 'import torch, transformers, peft' 2>/dev/null || {
  "$TRAIN_ENV/bin/python" -m pip install -r "$PROJECT_DIR/training/requirements.txt"
  "$TRAIN_ENV/bin/python" -m pip install -e "$PROJECT_DIR[baseline]"
}
"$TRAIN_ENV/bin/python" -c 'import torch, transformers, peft; print("train-ok", torch.__version__, transformers.__version__, peft.__version__)'

if [[ ! -x "$SERVE_ENV/bin/python" ]]; then
  python3 -m venv "$SERVE_ENV"
  "$SERVE_ENV/bin/python" -m pip install --upgrade pip
fi
"$SERVE_ENV/bin/python" -c 'import vllm; print("serve-ok", vllm.__version__)' 2>/dev/null || {
  "$SERVE_ENV/bin/python" -m pip install -r "$PROJECT_DIR/training/requirements-vllm.txt"
  "$SERVE_ENV/bin/python" -c 'import vllm; print("serve-ok", vllm.__version__)'
}
"$SERVE_ENV/bin/python" -m pip freeze > /scratch/qdf7ea/logs/vllm-pip-freeze.txt

if [[ ! -f "$SIF" ]]; then
  echo "Building validator SIF"
  apptainer build --fakeroot "$SIF" "$PROJECT_DIR/sandbox/kernel-lab-validator.def"
fi
test -f "$SIF"
apptainer exec "$SIF" python -c 'import torch, triton, pathlib; p=pathlib.Path("/opt/runner.py"); print("validator-ok", torch.__version__, triton.__version__, p.exists(), p.stat().st_size)'

"$TRAIN_ENV/bin/python" - <<'PY'
from huggingface_hub import snapshot_download
path = snapshot_download("Qwen/Qwen2.5-Coder-0.5B-Instruct")
print("qwen-ok", path)
print("MODEL_REVISION", path.rstrip("/").split("/")[-1] if "snapshots" in path else "see-path")
PY

echo "SETUP COMPLETE"
ls -lh "$SIF"
