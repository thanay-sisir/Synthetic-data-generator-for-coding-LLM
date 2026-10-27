#!/usr/bin/env bash
# Serve Qwen3.8-27B and its LoRA adapter with vLLM, then measure latency.
# Two GPUs: vLLM holds most of GPU 0; GPU 1 only needs to exist because the
# recipe notes that some kernels pick the first free device. Benchmarking does
# NOT run the sandbox. Kernel correctness comes from the separate eval jobs.
# Requires slurm/setup_vllm.sbatch to have printed SETUP COMPLETE.
#SBATCH --job-name=kernel-serve-bench
#SBATCH --account=orsdardencomputing
#SBATCH --partition=gpu
#SBATCH --gres=gpu:a100:1
#SBATCH --constraint=a100_80gb
#SBATCH --cpus-per-task=8
#SBATCH --mem=128G
#SBATCH --time=06:00:00
#SBATCH --chdir=/project/orsDarden/thanay/personal_project/Synthetic-data-generator-for-coding-LLM
#SBATCH --output=/scratch/qdf7ea/logs/serve-bench-%j.out
set -euo pipefail

PROJECT_DIR=/project/orsDarden/thanay/personal_project/Synthetic-data-generator-for-coding-LLM
cd "$PROJECT_DIR"
PY=/scratch/qdf7ea/venvs/kernel-serve-py313/bin/python
ADAPTER=$PROJECT_DIR/artifacts/lora-qwen38-27b/adapter
BASE_OUT=$PROJECT_DIR/artifacts/bench-qwen38-base
LORA_OUT=$PROJECT_DIR/artifacts/bench-qwen38-lora
PORT=8000

test -x "$PY"
test -f "$ADAPTER/adapter_config.json"
for dir in "$BASE_OUT" "$LORA_OUT"; do
  if [[ -e "$dir/summary.json" ]]; then
    echo "Output $dir already has a summary. Pick a new directory." >&2
    exit 2
  fi
done

module load gcc/14.2.0
module load python/3.13.11
# vllm is a console script in this venv; calling it by absolute path avoids
# depending on PATH activation inside the srun/batch shell.
export PATH=/scratch/qdf7ea/venvs/kernel-serve-py313/bin:$PATH
export HF_HOME=/scratch/qdf7ea/huggingface
export TOKENIZERS_PARALLELISM=false
export VLLM_API_KEY=$(python3 -c 'import secrets; print(secrets.token_urlsafe(32))')
mkdir -p "$HF_HOME"

# A small fixed slice of held-out tasks. Same tasks for both models.
"$PY" - <<'PY'
import json
from pathlib import Path
src = Path("artifacts/tasks-v1.jsonl")
rows = [json.loads(line) for line in src.read_text(encoding="utf-8").splitlines() if line.strip()]
test = [r for r in rows if r.get("split") == "test"][:24]
Path("artifacts/bench-tasks-24.jsonl").write_text(
    "".join(json.dumps(r, sort_keys=True) + "\n" for r in test), encoding="utf-8")
print("bench tasks", len(test))
PY

nvidia-smi
MODEL_ID=Qwen/Qwen3.8-27B \
SERVED_NAME=Qwen/Qwen3.8-27B \
THINKING_OFF=1 \
LORA_ADAPTER="$ADAPTER" \
LORA_NAME=kernel-lora \
PORT=$PORT \
MAX_MODEL_LEN=8192 \
GPU_MEMORY_UTILIZATION=0.90 \
  bash scripts/serve_vllm.sh > /scratch/qdf7ea/logs/vllm-serve-$SLURM_JOB_ID.log 2>&1 &
SERVER_PID=$!
trap 'kill $SERVER_PID 2>/dev/null || true' EXIT

echo "waiting for vLLM on 127.0.0.1:$PORT (pid $SERVER_PID)"
for i in $(seq 1 180); do
  if curl -sf "http://127.0.0.1:$PORT/health" >/dev/null 2>&1; then
    echo "server healthy after $((i*10))s"
    break
  fi
  if ! kill -0 "$SERVER_PID" 2>/dev/null; then
    echo "vLLM exited; see /scratch/qdf7ea/logs/vllm-serve-$SLURM_JOB_ID.log" >&2
    tail -n 60 /scratch/qdf7ea/logs/vllm-serve-$SLURM_JOB_ID.log >&2 || true
    exit 1
  fi
  sleep 10
done
curl -sf "http://127.0.0.1:$PORT/health" >/dev/null || { echo "server never became healthy" >&2; exit 1; }
echo "=== served models ==="
curl -s -H "Authorization: Bearer $VLLM_API_KEY" "http://127.0.0.1:$PORT/v1/models" | head -c 800
echo

export TEACHER_BASE_URL="http://127.0.0.1:$PORT/v1"
export TEACHER_API_KEY="$VLLM_API_KEY"
export TEACHER_FREE_ONLY=0
export TEACHER_REQUEST_INTERVAL=0
export TEACHER_RETRIES=1
export TEACHER_TIMEOUT=600
export TEACHER_MAX_TOKENS=1024
export TEACHER_TEMPERATURE=0
unset TEACHER_REASONING_EFFORT || true

echo "=== base model benchmark ==="
TEACHER_MODEL=Qwen/Qwen3.8-27B \
  "$PY" -m kernel_lab.benchmark \
    --tasks "$PROJECT_DIR/artifacts/bench-tasks-24.jsonl" \
    --output-dir "$BASE_OUT" \
    --requests 24 --workers 4 --warmup 2

echo "=== LoRA adapter benchmark ==="
TEACHER_MODEL=kernel-lora \
  "$PY" -m kernel_lab.benchmark \
    --tasks "$PROJECT_DIR/artifacts/bench-tasks-24.jsonl" \
    --output-dir "$LORA_OUT" \
    --requests 24 --workers 4 --warmup 2

echo "BENCHMARK COMPLETE"
