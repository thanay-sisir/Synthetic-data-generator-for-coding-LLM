# Rivanna runbook: validation, training, held-out evaluation and vLLM

Everything in this file was executed on UVA Rivanna with account `orsdardencomputing` between 2026-10-18 and 2026-10-27. It records the commands and the exact hardware that produced the saved results. Where a step failed, the failure and the fix are kept, because those are the parts that cost time.

All GPU work ran inside Slurm allocations. Nothing was trained or served on a login node.

## 0. What the machine looks like now

Two facts shaped every command below.

**Python 3.11.4 was removed from the site on 2026-10-23.** Modulefiles were rewritten that afternoon. Any job that loads `python/3.11.4` now dies in four seconds with `module unknown` or `execve(): python: No such file or directory`. This is not a stale-cache problem and `--ignore_cache` cannot fix it. Two jobs were lost to this before it was understood.

What works:

```bash
module load gcc/14.2.0
module load python/3.13.11
```

`python/3.14.7` exists but only after `gcc/16.2.0`. The old virtualenvs (`kernel-client`, `kernel-train`, `kernel-serve`) are dangling symlinks to the deleted interpreter. They will never work again; use the two below.

**Docker is not available, and an Apptainer `.sif` cannot be built.** The account is not in `/etc/subuid`, so fakeroot and proot both fail, and `mksquashfs` dies with `ptrace(TRACEME): Operation not permitted`. The working checker is an Apptainer *sandbox directory*:

```
/scratch/qdf7ea/containers/kernel-lab-validator
```

with the runner at `/opt/runner.py` inside it. This shares the host GPU driver and is weaker isolation than Docker. Every job that uses it must pass `--backend apptainer --allow-weaker-isolation`. Do not trust `VALIDATOR_SIF` in `.env`; it names an image file that does not exist.

## 1. Storage, accounts and the two environments

Account: `orsdardencomputing`. The name `orsdarden` is a different account and will fail.

| Purpose | Path |
|---|---|
| Code | `/project/orsDarden/thanay/personal_project/Synthetic-data-generator-for-coding-LLM` |
| Scratch (90-day purge) | `/scratch/qdf7ea` |
| Hugging Face cache | `/scratch/qdf7ea/huggingface` |
| Logs | `/scratch/qdf7ea/logs` |
| Sandbox image | `/scratch/qdf7ea/containers/kernel-lab-validator` |
| Orchestration + teacher client | `/scratch/qdf7ea/venvs/kernel-client-py313` |
| Training and held-out generation | `/scratch/qdf7ea/venvs/kernel-train-qwen38` |
| vLLM serving | `/scratch/qdf7ea/venvs/kernel-serve-py313` |

The serving environment is deliberately separate: vLLM 0.32.0 pulls its own torch 2.13.0 and transformers 5.19.0, which would break the training stack that produced the adapters.

Back up checkpoints, adapters, `run_summary.json`, `evaluation.jsonl`, `summary.json` and the SFT manifest to durable storage. Scratch is purged after 90 days of no access. The 27B weights were already purged once during this project; the next job re-downloaded 51.75 GiB.

## 2. Verified data

Generation wrote two kinds of ledger per run: `attempts.jsonl` (everything) and `verified.jsonl` (passed only). Runs cannot be resumed in place, so a hole was filled by writing a new task manifest and a new output directory, then concatenating the verified files.

```bash
cat artifacts/grok-train-2000/verified.jsonl \
    artifacts/grok-train-rest/verified.jsonl \
    artifacts/grok-retry-s{0,1,2,3}/verified.jsonl \
    artifacts/grok-tanh-retry/verified.jsonl \
    artifacts/grok-val-100/verified.jsonl \
  > artifacts/verified-qwen38.jsonl

python -m training.prepare_data \
  --input artifacts/verified-qwen38.jsonl \
  --output artifacts/sft-qwen38
```

Result: 1,894 verified records became 1,798 train and 96 validation. The test file is empty on purpose; final evaluation uses the original task references, not teacher solutions.

Recorded hashes:

- source `d60f262919a316a8b002eac5385d6a7c76f8b8344cf7322eaf3cf6295ce40400`
- train `801f82a2f8106f4bc8c6ffa3a7213634c3dca22e6697c16ca1b3be9a68b7f86f`
- validation `dfd294b2cc03c7ef046e3b7a4b22a86766b7dda1747e993d49b5d504901f7c6a`

Per-family training counts: `tanh_mix` 198, `add_relu` 195, `abs_mix` 194, `cosine_mix` 194, `affine` 193, `row_square_sum` 190, `sqrt_positive` 186, `square_bias` 186, `row_sum` 161, `row_mean` 101. `row_mean` stayed weak; its remaining failures are numerical, and a third paid retry was not worth it.

## 3. The training environment

Built by `slurm/setup_qwen38.sbatch` on the `standard` partition (job 21910214). Not on a login node: the login session has a memory cap that kills large pip installs.

```bash
module load gcc/14.2.0
module load python/3.13.11
python3 -m venv /scratch/qdf7ea/venvs/kernel-train-qwen38
source /scratch/qdf7ea/venvs/kernel-train-qwen38/bin/activate
python -m pip install -r training/requirements-qwen38.txt
python -m pip install -e .
```

Resolved versions: torch 2.6.0+cu124, transformers 5.20.2, peft 0.21.2, accelerate 1.15.0.

A venv Python still needs the module's `libpython3.13.so.1.0` on the library path. In a fresh shell, load the modules first or the interpreter fails before any project code runs.

## 4. LoRA training on Qwen3.8-27B

```bash
sbatch slurm/train_lora_qwen38.sbatch
```

One A100 80GB, 8 CPUs, 256G RAM, 8-hour cap. The trainer refuses to start unless exactly one GPU is visible, so the GPU-hour number stays honest a 27B full fine-tune does not fit one card, LoRA does.

Two failures happened before the successful run, and both are worth knowing:

**Job 21910419, 19 seconds.** `ValueError: Chat template prefix mismatch`. Qwen3.8 with `enable_thinking=False` still closes an empty thinking block in the generation prompt. The rendered text of the prompt is a prefix of the full example, but the token ids are not, because byte-pair encoding merges across the boundary. `training/data.py` now renders both sides as text, tokenizes the full text with character offsets, and cuts at the first token that reaches the boundary, masking any token that straddles it. All 1,894 rows were re-checked with the real tokenizer: every label span starts at `import torch`, longest row 829 tokens against a 4,096 cap.

**Job 21910574, after loading all 1,184 weight shards.** `TypeError: TrainingArguments.__init__() got an unexpected keyword argument 'warmup_ratio'`. Transformers 5.20 folded the ratio into `warmup_steps`, where a float in `[0, 1)` means a fraction of optimizer steps. `warmup_ratio=0.03` became `warmup_steps=0.03`.

The trainer's `finally` block writes `run_summary.json` even on failure, and the script refuses a non-empty output directory. Both failed runs left a summary that had to be deleted before resubmitting. Keep that in mind for any rerun.

**Successful run, job 21910579.** 1 hour 15 minutes, exit 0.

| | |
|---|---|
| Adapter | `artifacts/lora-qwen38-27b/adapter` (431 MB) |
| Epochs / steps | 1 / 113 |
| Rank / alpha | 16 / 32 |
| Learning rate | 1e-4, cosine, 3% warmup |
| Train loss | 0.0533 |
| Validation loss | 0.0326 |
| Peak GPU memory | 59,679,263,232 bytes (~56 GB of 80) |
| Trainable / total parameters | 112,758,784 / 27,469,487,344 |
| LoRA target modules | 448 |

`causal_conv1d` and `flash-linear-attention` were not installed, so the DeltaNet layers used the PyTorch reference kernels. Correct, and slower. The run still finished inside the cap.

`resolved_model_revision` is null in the summary: the weights came from the local cache. Do not invent a commit hash for a report.

## 5. LoRA training on Qwen2.5-Coder-0.5B

```bash
sbatch slurm/train_lora_qwen25_coder_0.5b.sbatch
```

Same data, same one epoch, same rank 16 and learning rate 1e-4, so the base model is the only change. Job 22001802, nine minutes on one A100. Adapter at `artifacts/lora-qwen25-coder-0.5b/adapter`. Train loss 0.1554, validation loss 0.0483, 8.8M trainable of 503M total.

## 6. The scratch model

```bash
sbatch slurm/train_scratch.sbatch
```

Job 21913389, about two minutes on one A100. 29,496,192 parameters, AdamW, learning rate 3e-4, 3 epochs, batch 1, accumulation 8, context 4,096, tokenizer `HuggingFaceTB/SmolLM-135M`. Validation response loss 0.405, 0.353, 0.372. The best checkpoint is `artifacts/scratch-qwen38/model.pt`.

Before submission, `llm.py` imported pandas at module import time, and the training venv has no pandas. Job 21912780 died in 11 seconds on `ModuleNotFoundError`. Pandas is only used to read the legacy CSV, so that import now lives inside the CSV function. The model code was not changed.

## 7. Held-out evaluation

All four models were scored on the same 1,200 `test` tasks from `artifacts/tasks-v1.jsonl`: 400 each of `sigmoid_affine`, `row_max`, `row_relu_sum`. None of those families appears in training.

**Scratch model** (`slurm/train_scratch.sbatch` output checked by the same job family):

```bash
python -m kernel_lab.evaluate \
  --tasks artifacts/tasks-v1.jsonl --split test \
  --output-dir artifacts/eval-scratch-test \
  --baseline-checkpoint artifacts/scratch-qwen38 \
  --no-rag-only --workers 1 \
  --sandbox-backend apptainer --allow-weaker-isolation \
  --sandbox-image /scratch/qdf7ea/containers/kernel-lab-validator \
  --sandbox-timeout 120
```

**Qwen base and Qwen + LoRA** use `--local-qwen` and `--qwen-adapter`, which load the weights in the job and generate on one GPU while the sandbox runs on another:

```bash
KERNEL_LAB_SANDBOX_GPU=1 python -m kernel_lab.evaluate \
  --tasks artifacts/tasks-v1.jsonl --split test \
  --output-dir artifacts/eval-qwen38-lora-test \
  --local-qwen Qwen/Qwen3.8-27B \
  --qwen-adapter artifacts/lora-qwen38-27b/adapter \
  --max-new-tokens 1024 --no-rag-only --workers 1 \
  --sandbox-backend apptainer --allow-weaker-isolation \
  --sandbox-image /scratch/qdf7ea/containers/kernel-lab-validator \
  --gpu-device 1 --sandbox-timeout 120
```

`--gpu-device 1` plus `KERNEL_LAB_SANDBOX_GPU=1` pins the checker to the second card. The 27B model holds about 56 GB, so server plus checker cannot share one A100.

Results, and the jobs that produced them:

| Model | Passed | `sigmoid_affine` | `row_max` | `row_relu_sum` | Job | Wall |
|---|---|---|---|---|---|---|
| Qwen3.8-27B base | 537 | 400 | 117 | 20 | 21918029 | 10.1 h |
| Qwen3.8-27B + LoRA | 1,178 | 378 | 400 | 400 | 21918030 | 14.1 h |
| Qwen2.5-Coder-0.5B + LoRA | 147 | 147 | 0 | 0 | 22006999 | 4.1 h |
| Scratch 29.5M | 0 | 0 | 0 | 0 | 21913925 | 1.2 h |

One failed attempt is worth recording. The first scratch evaluation (job 21913390) reported 0 out of 1,200 in 16 seconds with 1-millisecond latencies. That was not a result: `apply_chat_template(..., return_tensors="pt")` returns a `BatchEncoding` in Transformers 5, and `BatchEncoding.__getattr__` raises a bare `AttributeError()` for a missing name such as `.shape`. The fix takes `encoded["input_ids"]` before calling `.to`. The ledger was deleted and the job rerun. A zero with sub-second latency is a broken job, not a measurement.

## 8. vLLM serving

Environment built by `slurm/setup_vllm.sbatch` (job 22062441, `standard` partition). vLLM 0.32.0, torch 2.13.0+cu130, transformers 5.19.0.

Two things broke here:

- The setup job's own verification step asked `importlib.metadata` for `peft`, which the serving environment intentionally lacks. The install had succeeded; the check failed the job. A verification step can fail a good build.
- The first serve attempt (job 22063970) died with `exec: vllm: not found`, because `scripts/serve_vllm.sh` assumed an activated venv and the batch shell never activated it. It also printed the generated API key into the log. Both are fixed: the job puts the venv `bin` on `PATH`, and the key goes through `VLLM_API_KEY` in the environment, which vLLM reads, so it never appears in argv.

Serve and benchmark:

```bash
sbatch slurm/serve_bench_vllm.sbatch
```

One A100 80GB, loopback only. Thinking is off server-side via `--default-chat-template-kwargs '{"enable_thinking": false}'`, the adapter is loaded as a second model name `kernel-lora`, and the benchmark client is the repo's own `kernel_lab.benchmark`.

Job 22064175, 13 minutes total:

| | Base 27B | + LoRA |
|---|---|---|
| Requests succeeded | 24 / 24 | 24 / 24 |
| p50 latency | 10.47 s | 10.25 s |
| p95 latency | 12.21 s | 11.44 s |
| Aggregate output tok/s, 4 workers | 101.9 | 94.5 |
| Median per-request output tok/s | 27.1 | 23.8 |

Startup dominates: 51.27 GiB of weights in about 18 seconds from WEKA, then roughly two minutes of `torch.compile` and FlashInfer JIT. The server reported a KV cache of 139,264 tokens, 17× concurrency at 8,192 tokens per request.

These are non-streaming complete-response latencies after worker admission, including HTTP and model queueing. No time-to-first-token, and no kernel validation. This measures how fast a served checkpoint answers, which is a different question from whether its kernel passes.

If you reach the server from your laptop rather than inside the allocation:

```bash
ssh -J YOUR_ID@UVA_LOGIN_HOST -N -L 8000:127.0.0.1:8000 YOUR_ID@ALLOCATED_COMPUTE_NODE
```

The hop must terminate at the compute node. A tunnel to the login host cannot reach a loopback port on the compute node.

## 9. Reproduction checklist

1. Load `gcc/14.2.0` then `python/3.13.11`. Never `python/3.11.4`.
2. Confirm the sandbox directory exists and `/opt/runner.py` is inside it.
3. Rebuild or reuse `artifacts/sft-qwen38`; check its manifest hashes against section 2.
4. Train LoRA or scratch with a fresh output directory.
5. Score with `kernel_lab.evaluate --split test --no-rag-only` and the same sandbox settings.
6. Serve with vLLM only for latency claims, never for correctness claims.
7. Copy numbers into `README.md` from `summary.json` / `evaluation.jsonl` only. Unknown costs stay null.

## 10. Limits

- The scratch Transformer is PyTorch-only and has no vLLM backend.
- Apptainer here is not a hardened boundary for hostile submissions. It is an approved weaker-isolation path for cooperative model output.
- The 24-request benchmark is a small sample.
- The pgvector retrieval comparison is implemented but was not part of the measured path.
- Coefficient variants inside one family are not independent operations. Report the three test families separately before averaging.
