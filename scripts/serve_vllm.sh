#!/usr/bin/env bash
# Start one vLLM OpenAI-compatible server on loopback. Called from a Slurm job.
# Thinking is off for Qwen3.8 so message.content is the kernel source, not a
# reasoning trace. The scratch Transformer is not supported here.
set -euo pipefail
: "${VLLM_API_KEY:?Set a private VLLM_API_KEY; do not commit it}"

MODEL_ID="${MODEL_ID:-Qwen/Qwen3.8-27B}"
SERVED_NAME="${SERVED_NAME:-$MODEL_ID}"
# vLLM reads VLLM_API_KEY from the environment, so the key never appears in
# argv or in the process table.
args=(serve "$MODEL_ID" --host 127.0.0.1 --port "${PORT:-8000}"
      --served-model-name "$SERVED_NAME"
      --dtype "${DTYPE:-bfloat16}"
      --max-model-len "${MAX_MODEL_LEN:-8192}"
      --gpu-memory-utilization "${GPU_MEMORY_UTILIZATION:-0.90}")
if [[ -n "${MODEL_REVISION:-}" ]]; then args+=(--revision "$MODEL_REVISION"); fi
if [[ "${THINKING_OFF:-0}" == "1" ]]; then
  # Server-wide default; a request can still turn thinking back on.
  args+=(--default-chat-template-kwargs '{"enable_thinking": false}')
  args+=(--reasoning-parser qwen3)
fi
ADAPTER="${LORA_ADAPTER:-${ADAPTER_PATH:-}}"
if [[ -n "$ADAPTER" ]]; then
  [[ -f "$ADAPTER/adapter_config.json" ]] || { echo "Missing adapter config" >&2; exit 1; }
  args+=(--enable-lora --max-lora-rank "${LORA_RANK:-16}"
         --lora-modules "${LORA_NAME:-kernel-lora}=$ADAPTER")
fi
# Print the command without the key. It still ends up in the process table,
# but the log is the file people copy out of.
echo "vllm ${args[*]//$VLLM_API_KEY/***}"
exec vllm "${args[@]}"
