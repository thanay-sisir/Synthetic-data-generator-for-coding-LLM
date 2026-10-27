# Project handoff: Triton kernel data, LoRA, scratch baseline and serving

Updated: 2026-10-27. Read this first, then [docs/RIVANNA.md](docs/RIVANNA.md) for the exact cluster commands and failure history. `JOURNEY.md` is the narrative record of what broke and why.

This project turns PyTorch reference functions into complete Triton kernels, accepts only the ones that pass a GPU numerical check, trains on those, and scores every model on operation families that were never trained on.

## 1. Present status

The pipeline has been run end to end on UVA Rivanna with one and two A100 80GB allocations.

| Stage | Status |
|---|---|
| Task manifests, 6,000 tasks / 15 families | Done, `artifacts/tasks-v1.jsonl` |
| Teacher generation | Done with xAI `grok-4.7`, 1,894 verified records |
| GPU numerical validation | Done in an Apptainer sandbox directory |
| SFT preparation | Done, 1,798 train / 96 validation / 0 test |
| Qwen3.8-27B LoRA | Done, job 21910579, adapter saved |
| Qwen2.5-Coder-0.5B LoRA | Done, job 22001802 |
| Scratch 29.5M baseline | Done, job 21913389 |
| Held-out evaluation, all four models | Done, 1,200 tasks each |
| vLLM serving and latency benchmark | Done, job 22064175 |
| pgvector retrieval comparison | Implemented, **not run** |

Held-out results, 1,200 tasks from `sigmoid_affine`, `row_max`, `row_relu_sum`, all absent from training:

| Model | Passed | Job |
|---|---|---|
| Qwen3.8-27B, no fine-tune | 537 | 21918029 |
| Qwen3.8-27B + LoRA r16 | 1,178 | 21918030 |
| Qwen2.5-Coder-0.5B + LoRA r16 | 147 | 22006999 |
| Scratch Transformer 29.5M | 0 | 21913925 |

Serving, one A100 80GB, vLLM 0.32.0, 24 requests, 4 workers: base p50 10.47 s / 101.9 tok/s aggregate; LoRA p50 10.25 s / 94.5 tok/s aggregate. No failed requests.

## 2. Original project and why its numbers were not reused

The original checkout had `llm.py`, `extract_training_data.py`, `run_all_tests.py`, 53 `data/combined_implementation*.py` files, and a 45-row CSV. The old README claimed 319 and 2,000+ pairs and a 50M model. Those counts do not describe the checked-in artifacts. The old text is preserved in [docs/LEGACY_README.md](docs/LEGACY_README.md), labelled superseded.

Specific defects in the old material:

- Extraction dropped wrappers, signatures, launch parameters and preprocessing, so some kernels implemented only part of the Python expression.
- Tests sometimes treated skipped or printed failures as zero-exit successes.
- The dataset split overlapping token windows after tokenization, which leaks validation text into training.
- No checkpoint, GPU result, or measured speed/cost evidence existed.

`llm.py` now fixes rotary-position tensor dimensions, partial gradient accumulation, and optimizer-step schedule counts. Its historical main routine requires `--legacy-experiment` so it cannot be run accidentally as a result. New baseline runs go through `kernel_lab.baseline`, which uses family-separated complete examples and response-only loss, AdamW, and about 29.5M parameters with the original tokenizer vocabulary.

## 3. Repository map

| Path | Responsibility |
|---|---|
| `kernel_lab/tasks.py` | Deterministic PyTorch reference tasks and family splits |
| `kernel_lab/schema.py` | Task validation, content hashes, conservative source screen |
| `kernel_lab/prompts.py` | Shared system prompt and message builder for teacher, training and eval |
| `kernel_lab/teacher.py` | Async OpenAI-compatible client, bounded retries, usage and cost metadata |
| `kernel_lab/sandbox.py` | Container invocation, timeouts, cleanup, validator result gate |
| `kernel_lab/generate.py` | Async orchestration, all-attempt ledger, deduplicated verified output |
| `kernel_lab/evaluate.py` | Held-out execution evaluation; supports `--local-qwen`, `--qwen-adapter`, `--baseline-checkpoint` |
| `kernel_lab/qwen_local.py` | Non-vLLM on-GPU generation used by the held-out checks |
| `kernel_lab/benchmark.py` | Non-streaming endpoint latency and throughput |
| `kernel_lab/metrics.py` | Aggregates that keep failures in the denominator and unknown costs null |
| `kernel_lab/baseline.py` | Trains the scratch model and exposes an evaluation adapter |
| `kernel_lab/retrieval.py`, `embed_server.py` | pgvector client and optional local CPU embedding service |
| `sandbox/runner.py` | Trusted reference generation, numerical/shape/dtype/mutation checks |
| `training/data.py` | Verification gates, split integrity, response-only label masking |
| `training/prepare_data.py` | Builds SFT splits and a content-hash manifest |
| `training/train_lora.py` | Single-GPU PEFT LoRA with run records |
| `training/slurm_accounting.py` | Allocated GPU-hours from `sacct` |
| `slurm/train_lora_qwen38.sbatch` | Qwen3.8-27B LoRA, 1× A100 80GB |
| `slurm/train_lora_qwen25_coder_0.5b.sbatch` | Qwen2.5-Coder-0.5B LoRA |
| `slurm/train_scratch.sbatch` | Scratch baseline training |
| `slurm/setup_qwen38.sbatch`, `slurm/setup_vllm.sbatch` | Environment builds on the `standard` partition |
| `slurm/serve_bench_vllm.sbatch`, `scripts/serve_vllm.sh` | Serve base + LoRA and benchmark |
| `slurm/grok-*.sbatch` | The generation and retry jobs that produced the dataset |
| `tests/` | CPU tests, 38 passing |

Retired on purpose, now exiting with a pointer: `slurm/train_lora.sbatch`, `slurm/train_baseline.sbatch`, `slurm/generate.sbatch`, `slurm/setup.sbatch`, `slurm/smoke.sbatch`. All loaded `python/3.11.4`, which the site removed on 2026-10-23.

## 4. Data and validation contracts

A task is a JSONL record with `schema_version`, `task_id`, `family`, `split`, `prompt`, `reference_code`, `input_specs`, `test_cases`, `atol`, `rtol`. The reference source defines `reference(*inputs)`. A candidate is a complete module defining `kernel(*inputs)` with the same positional inputs; it must allocate a fresh CUDA float32 output, launch Triton, preserve its inputs, and return the reference shape.

`torch` in a candidate is for allocation only. The static screen rejects host arithmetic, loops, indexing, `zero_`, `fill_`, `copy_`, private attributes, and imports other than `torch`, `triton`, `triton.language`.

The checker uses deterministic shapes and seeds, non-power-of-two sizes, finite-value checks, shape/dtype/device checks and input-mutation checks. Two extra cases are withheld from the model prompt. These are fixed cases, not a secret adversarial benchmark. Passing proves agreement on the tested cases at the stated tolerances, not mathematical equivalence or speed.

Outputs per generation run:

- `manifest.json` — task hash, worker settings, model and container metadata.
- `attempts.jsonl` — every attempt, including API errors, parse failures, static rejection, compile/runtime failure, timeout.
- `verified.jsonl` — passed candidates only, one per task, exact duplicate pairs removed.
- `summary.json` — outcomes, throughput, known/unknown cost fields.

Whole families are assigned to train/validation/test before tokenization, so no coefficient variant can straddle a split. `prepare_records` rejects a family or a reference-code hash that appears in two splits.

## 5. Splits

`artifacts/tasks-v1.jsonl` holds 6,000 tasks.

| Split | Families | Tasks |
|---|---|---|
| Train | `abs_mix`, `add_relu`, `affine`, `cosine_mix`, `row_mean`, `row_square_sum`, `row_sum`, `sqrt_positive`, `square_bias`, `tanh_mix` | 4,000 |
| Validation | `sine_affine`, `exp_decay` | 800 |
| Test | `sigmoid_affine`, `row_max`, `row_relu_sum` | 1,200 |

Realised supervised data is smaller than the manifest because many teacher attempts failed: 1,798 train, 96 validation, 0 test. The test file stays empty on purpose; final evaluation uses the original task specs and references, not teacher solutions.

Those 6,000 tasks are coefficient variants of 15 related families, not 6,000 independent operations.

## 6. Where things run

**Compute:** Slurm account `orsdardencomputing`. GPU jobs used `--partition=gpu --gres=gpu:a100:1 --constraint=a100_80gb` (two GPUs for the 27B held-out checks, because the model and the sandbox cannot share one card). The trainer refuses more than one visible GPU so GPU-hour accounting stays honest.

**Login nodes:** never for installs or model loads. The login session has a memory cap that kills large pip installs. Environment builds go to the `standard` partition with ~128G.

**Container:** Docker is unavailable and an Apptainer `.sif` cannot be built (no `/etc/subuid` entry; `mksquashfs` fails with `ptrace(TRACEME): Operation not permitted`). The working checker is an Apptainer sandbox directory at `/scratch/qdf7ea/containers/kernel-lab-validator` with `/opt/runner.py` inside. Every job using it passes `--backend apptainer --allow-weaker-isolation`. This is weaker than Docker: it shares the host GPU driver. It is adequate for cooperative model output and is not a hardened multi-tenant sandbox.

**Postgres and embeddings:** implemented in `infra/` and `kernel_lab/retrieval.py`. Not part of the measured path.

## 7. Teacher configuration

The teacher that produced the dataset is paid xAI `grok-4.7` at `https://api.x.ai/v1`, reasoning effort `low`, temperature 0.2, 4,096 max tokens, 300-second timeout. Low effort matters: high effort puts the module in a reasoning field, while the parser reads `message.content`.

The client stays OpenAI-compatible. Job scripts source `.env`, then override:

```
TEACHER_BASE_URL   <- XAI_BASE_URL
TEACHER_MODEL      <- XAI_MODEL  (grok-4.7)
TEACHER_API_KEY    <- XAI_API_KEY
TEACHER_INPUT_PRICE_PER_MILLION  <- XAI_INPUT_PRICE_PER_MILLION   ($2)
TEACHER_OUTPUT_PRICE_PER_MILLION <- XAI_OUTPUT_PRICE_PER_MILLION  ($6)
TEACHER_FREE_ONLY=0
TEACHER_REASONING_EFFORT=low
TEACHER_TIMEOUT=300
```

`TEACHER_FREE_ONLY=1` only restricts `openrouter.ai` to `:free` model names. The 3.4-second pacing gate is also OpenRouter-only; xAI is not paced.

Keys live in `.env`, mode 600, gitignored, sourced on the compute node. Never on an `sbatch` command line, never in a log. HTTP 429 and 5xx are retried; other 4xx fail on the first response. Stored errors are stripped of body and headers.

Known spend, from saved summaries: $7.18 in response tokens for 1,037 passes. One cancelled run (857 passes) wrote no summary, so its cost is unknown, not zero.

## 8. Generation cost and throughput

One attempt is about 24 seconds of teacher time plus about 8 seconds of GPU checking. One A100 saturates at roughly three in-flight teacher calls (`--workers 3 --validation-workers 1`). Four GPUs × three workers was the useful parallelism; more calls only queue on the GPU.

Generation cannot resume in place: `kernel_lab.generate` refuses an output directory that already contains a ledger. To fill holes, build a task manifest of unfinished IDs and use a new directory, then concatenate `verified.jsonl` files.

## 9. Model choice

`Qwen/Qwen3.8-27B` is a hybrid vision-language checkpoint (`Qwen3_5ForConditionalGeneration`, model type `qwen3_5`): Gated DeltaNet plus full attention every fourth layer, 64 layers, hidden 5120, vocab 248320. Thinking is on by default, so training and evaluation both pass `enable_thinking=False`; targets are kernel source only. Load it with `AutoModelForImageTextToText`, not `AutoModelForCausalLM`, and use Transformers 5.8 or newer.

LoRA targets text linear layers ending in `q_proj`, `k_proj`, `v_proj`, `o_proj`, `in_proj_qkv`, `in_proj_z`, `in_proj_b`, `out_proj`, `gate_proj`, `up_proj`, `down_proj`. Names containing `visual`, `vision_tower`, `vision_model` or `vision_encoder` stay frozen. The successful run found 448 modules.

`LoraConfig` does not accept `revision`. That argument was removed before the first real submission.

Training hyperparameters that produced the adapter: 1 epoch, rank 16, alpha 32, dropout 0.05, lr 1e-4 cosine with 3% warmup, batch 1, accumulation 16 (113 optimizer steps), max length 4,096, bf16, gradient checkpointing.

## 10. Known failures and their fixes

These are the ones that cost real time. Full narrative in `JOURNEY.md`.

| Symptom | Cause | Fix |
|---|---|---|
| `module unknown` / `execve(): python: No such file` | site removed `python/3.11.4` | load `gcc/14.2.0` + `python/3.13.11` |
| `.sif` build fails, `mksquashfs` ptrace error | no `/etc/subuid`, fakeroot/proot blocked | Apptainer sandbox directory |
| `tl.tanh` / `tl.math.tanh` AttributeError | not present in Triton 3.2 | prompt states tanh and sigmoid must come from `tl.exp`; retry passed 176/178 |
| HTTP 403, "used all available credits" | non-retryable provider error | added credits; retried into a new directory |
| `Chat template prefix mismatch` | BPE merges across the prompt boundary | character-offset mask in `training/data.py` |
| `unexpected keyword argument 'warmup_ratio'` | Transformers 5.20 renamed it | `warmup_steps=0.03` |
| `ModuleNotFoundError: pandas` | module-level import in `llm.py` | moved into the legacy CSV function |
| 0/1,200 at 1 ms per task | `BatchEncoding` has no `.shape` | take `encoded["input_ids"]` first |
| `exec: vllm: not found` | venv not activated in the batch shell | put venv `bin` on `PATH` |
| API key printed in a serve log | script echoed the full argv | key via `VLLM_API_KEY`, vLLM reads the environment |

A failed run leaves `run_summary.json` behind, and the training script refuses a non-empty output directory. Delete those files before resubmitting, never the verified data.

## 11. Results interpretation

- The 27B LoRA fixed both row-reduction families completely (400/400 each). All 22 of its failures are numerical and all sit in `sigmoid_affine`.
- The base 27B model mostly broke host-code rules: 249 host `for` loops, 147 `zero_` calls, 36 `fill_`, 28 syntax errors. The static screen, not the numerical check, caught most of its failures.
- The 0.5B coder learned part of one elementwise family and no row reduction. 419 failures returned a per-block value where one scalar per row was required.
- The scratch model never produced a legal Triton module from these 1,798 examples. That is the control: the gain comes from pretrained code ability, not from the data alone.

Training loss is next-token loss on kernel source. It is not a pass rate, and lowering it does not by itself raise held-out correctness.

## 12. Serving notes

vLLM 0.32.0 registers `Qwen3_5ForConditionalGeneration`, so it can serve this checkpoint directly. Thinking is disabled server-side with `--default-chat-template-kwargs '{"enable_thinking": false}'` and a `qwen3` reasoning parser. The adapter is registered as a second model name, `kernel-lora`, on the same server.

Do not also send `reasoning_effort=low` from the client: on this model that turns thinking back on.

The scratch Transformer has no vLLM backend and must be evaluated through `kernel_lab.baseline`.

Latency numbers come from non-streaming complete responses after worker admission. They include HTTP and model queueing, exclude time-to-first-token, and perform no kernel validation. Do not present them as correctness evidence.

## 13. Reproducibility

Record for each reported result: git revision, task manifest hash, prompt text, model and tokenizer revision, training summary, evaluation ledger, CUDA/driver, GPU name, container image path, and seeds.

Never tune on the test families. Prompt changes belong on `sine_affine` and `exp_decay`. The test score is reported once, on a frozen configuration.

Back up adapters, checkpoints, `run_summary.json`, `evaluation.jsonl`, `summary.json`, the SFT manifest and generated ledgers. Scratch is purged after 90 days.

## 14. Open work

1. pgvector retrieval comparison: build a train-only index, run the same held-out tasks with and without retrieved examples, tune on validation, score test once.
2. Optional: a repair loop that feeds checker error text back to the model. Not implemented; would be a new inference path, not a change to the checker or the existing adapters, and must stay behind an opt-in flag so the four current numbers stay reproducible.
3. Optional: kernel execution-time benchmarks. The runner explicitly does not measure speed, so no speedup claim exists today.

## 15. References

- [UVA Rivanna/Afton compute](https://rc.virginia.edu/services/compute-and-storage/aftonrivanna)
- [vLLM serving](https://docs.vllm.ai/en/latest/serving/online_serving/), [LoRA serving](https://docs.vllm.ai/en/stable/features/lora/)
- [vLLM Qwen3.8-27B recipe](https://github.com/vllm-project/recipes/blob/main/models/Qwen/Qwen3.8-27B.yaml)
- [Transformers training arguments](https://huggingface.co/docs/transformers/main_classes/trainer)
- [pgvector](https://github.com/pgvector/pgvector)
