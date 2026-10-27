#!/usr/bin/env bash
# DEPRECATED. Kept only as a pointer; do not submit.
# This script loaded python/3.11.4, which Rivanna removed on 2026-10-23, and it
# defaulted to artifacts/sft (empty) with the broken kernel-train venv.
#
# Use one of these instead:
#   slurm/train_lora_qwen38.sbatch          Qwen3.8-27B LoRA on 1x A100 80GB
#   slurm/train_lora_qwen25_coder_0.5b.sbatch   Qwen2.5-Coder-0.5B LoRA
#
# Both read artifacts/sft-qwen38 and print run_summary.json in their output dir.
echo "slurm/train_lora.sbatch is retired. See the header of this file." >&2
exit 2
