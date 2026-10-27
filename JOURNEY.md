# What this project taught us

Written 24 October 2026, on Rivanna, while the scratch-model held-out check was still running. Every count below comes from a file that was actually saved. Where a run never wrote a summary, the file that was counted is named. Nothing in here is a pass rate for a check that had not finished.

The project lives at:

`/project/orsDarden/thanay/personal_project/Synthetic-data-generator-for-coding-LLM`

## What we are trying to show

A teacher model writes Triton kernels. A GPU checker runs each kernel against a trusted PyTorch reference. Only kernels that pass are kept. A coding model is then fine-tuned on those kernels, and a tiny model is trained from scratch on the same kernels. Both are judged later by whether the kernels they write pass the same checker, on operation families they never trained on.

The claim that is allowed, once the held-out checks exist, is narrow: GPU-checked kernels across 10 training families, then held-out family evaluation. It is not "6,000 independent operations," and it is not "the kernels are faster." The 6,000 tasks are coefficient variants of 15 related families.

Retrieval was built and then left out of the path we actually ran. Docker was not available. The working checker is an Apptainer sandbox directory, not an image file.

## The pipeline that actually ran

1. Build tasks whose train, validation, and test families do not overlap.
2. Ask the teacher for one Python module per task.
3. Reject the module if it fails a static screen or the GPU numerical check.
4. Keep every attempt in `attempts.jsonl`. Keep passes in `verified.jsonl`.
5. Turn passes into supervised records. The loss covers the kernel only, not the prompt.
6. LoRA-tune Qwen3.8-27B for one epoch.
7. Train a ~30M scratch Transformer on the same records.
8. Generate from the scratch model on the held-out test families and check those kernels on the GPU. This step was in progress when this note was written.
9. Still not done: serve the base Qwen model and the LoRA adapter, check them on the same test tasks, and copy measured numbers into the README.

Test families never received teacher kernels on purpose. The test file in the training manifest is empty. Final evaluation uses the original task specs and the PyTorch references, not teacher-written test solutions.

## How the data is split

Source tasks: `artifacts/tasks-v1.jsonl` (6,000 rows).

| Split | Families | Tasks |
|---|---|---|
| Train | `abs_mix`, `add_relu`, `affine`, `cosine_mix`, `row_mean`, `row_square_sum`, `row_sum`, `sqrt_positive`, `square_bias`, `tanh_mix` | 10 × 400 = 4,000 |
| Validation | `sine_affine`, `exp_decay` | 2 × 400 = 800 |
| Test, held out | `sigmoid_affine`, `row_max`, `row_relu_sum` | 3 × 400 = 1,200 |

The supervised set that was actually trained on is smaller, because many teacher attempts failed. After combining every passing file and dropping repeated `(task_id, code)` pairs:

| Split | Records | Families |
|---|---|---|
| Train | 1,798 | the 10 train families |
| Validation | 96 | `exp_decay` 47, `sine_affine` 49 |
| Test | 0 | none, on purpose |

Per-family training records in `artifacts/sft-qwen38/train.jsonl`:

| Family | Records |
|---|---|
| `tanh_mix` | 198 |
| `add_relu` | 195 |
| `abs_mix` | 194 |
| `cosine_mix` | 194 |
| `affine` | 193 |
| `row_square_sum` | 190 |
| `sqrt_positive` | 186 |
| `square_bias` | 186 |
| `row_sum` | 161 |
| `row_mean` | 101 |

Hashes, so a later run can prove it used this data:

- Combined verified source sha256: `d60f262919a316a8b002eac5385d6a7c76f8b8344cf7322eaf3cf6295ce40400`
- Train sha256: `801f82a2f8106f4bc8c6ffa3a7213634c3dca22e6697c16ca1b3be9a68b7f86f`
- Validation sha256: `dfd294b2cc03c7ef046e3b7a4b22a86766b7dda1747e993d49b5d504901f7c6a`

The old `training_data.csv` has 45 pairs. It is not trusted and was not used. An earlier extraction dropped wrappers, launch code, and preprocessing, and an old split leaked validation text by cutting overlapping token windows. `llm.py` still has that historical path. It now requires `--legacy-experiment` before anyone can treat it as a result. New scratch training goes through `kernel_lab.baseline`.

## Where things ran

Slurm account: `orsdardencomputing`. The account name `orsdarden` is the wrong one for these jobs.

GPU jobs that did real work used one A100 80GB:

`--partition=gpu --gres=gpu:a100:1 --constraint=a100_80gb`

The trainer refuses to start unless exactly one GPU is visible. That is deliberate, so the GPU-hour number is honest. A 27B full fine-tune does not fit on one 80GB card. LoRA did. Peak allocated memory in the successful LoRA run was 59,679,263,232 bytes, about 56GB of the 80GB card. B200 nodes existed but were full, and the installed PyTorch build (`2.6.0+cu124`) does not target Blackwell.

Login nodes are a bad place for heavy installs. The login session has historically been capped near 6GB. Builds, venv creation, and model loads belong on a `standard` or `gpu` allocation. Scratch space is `/scratch/qdf7ea` (about 90 days, then it can be purged). Hugging Face cache: `/scratch/qdf7ea/huggingface`. Logs: `/scratch/qdf7ea/logs`.

## Failure record

These are the breaks that cost time, in the order they showed up, and what actually fixed them.

### 1. The container image could not be packed

Apptainer `.sif` builds failed. The account is not in `/etc/subuid`. Fakeroot and proot both failed, and `mksquashfs` died with `ptrace(TRACEME): Operation not permitted`.

The working checker is a sandbox directory, not a `.sif`:

`/scratch/qdf7ea/containers/kernel-lab-validator`

The runner inside it is `/opt/runner.py`. A smoke test on an A100 80GB passed: Triton 3.2.0, torch 2.6.0+cu124, six cases. Jobs must pass `--backend apptainer --allow-weaker-isolation --image` that directory. `VALIDATOR_SIF` in `.env` still names an image file. Do not trust that variable.

Do not resubmit these scripts unchanged. They still point at a `.sif`, at `python/3.11.4`, or both:

- `slurm/setup.sbatch`
- `slurm/smoke.sbatch`
- `slurm/generate.sbatch`
- `slurm/train_lora.sbatch`
- `slurm/train_baseline.sbatch`
- `slurm/serve_vllm.sbatch`

### 2. Python 3.11.4 disappeared

On the afternoon of 23 October 2026 the site removed `python/3.11.4`. Modulefiles were rewritten. Two jobs died in 4–5 seconds: `module unknown`, then `execve(): python: No such file or directory`. That was not a stale cache. `--ignore_cache` cannot bring a removed module back.

What works now: `module load gcc/14.2.0` then `module load python/3.13.11`. `python/3.14.7` exists only after `gcc/16.2.0`.

The old virtualenvs are dangling symlinks to the deleted interpreter:

- `/scratch/qdf7ea/venvs/kernel-client`
- `/scratch/qdf7ea/venvs/kernel-train`
- `/scratch/qdf7ea/venvs/kernel-serve`

The environments that actually work:

| Env | Path | Used for |
|---|---|---|
| Client | `/scratch/qdf7ea/venvs/kernel-client-py313` | Teacher calls and the checker client |
| Train | `/scratch/qdf7ea/venvs/kernel-train-qwen38` | Qwen LoRA and the scratch model |

A venv Python still needs the module's `libpython3.13.so.1.0` on the library path. On a fresh login shell, `module load` first, or the interpreter fails before any project code runs.

### 3. Free teacher models were not good enough

The first teacher setting was an OpenRouter free model (`cohere/north-mini-code:free`), with a 3.4 second pause so one process stays under OpenRouter's free-tier rate. A small pilot did not produce kernels worth scaling. The teacher was switched to paid xAI `grok-4.7`.

The client is OpenAI-compatible (`kernel_lab/teacher.py`). For Grok the job must override the variables after sourcing `.env`, because `.env` still points `TEACHER_*` at the OpenRouter free model:

- `TEACHER_BASE_URL` from `XAI_BASE_URL` (`https://api.x.ai/v1`)
- `TEACHER_MODEL=grok-4.7`
- `TEACHER_API_KEY` from `XAI_API_KEY`
- prices from the `XAI_*_PRICE_PER_MILLION` variables ($2 / $6 per 1M input / output tokens)
- `TEACHER_FREE_ONLY=0`
- `TEACHER_REASONING_EFFORT=low`
- `TEACHER_TIMEOUT=300`

`TEACHER_FREE_ONLY=1` only rejects non-`:free` names on `openrouter.ai`. The 3.4 second pause also applies only to OpenRouter. Grok does not use it.

High reasoning effort puts the code in a reasoning field. `low` puts the module in `message.content`, which is what the parser reads. HTTP 429 and HTTP 5xx are retried. Any other 4xx is fatal on the first response. The stored error has the body and headers stripped so a log cannot keep the key.

Secrets stay in `.env`, mode 600, gitignored. They are sourced on the compute node. They are never put on the `sbatch` command line.

### 4. One GPU cannot keep up with too many teacher calls, and four GPUs hit a credit wall

Measured shape of one attempt: about 24 seconds of Grok, then about 8 seconds of GPU checking. One A100 stays busy at about 3 in-flight teacher calls (`--workers 3 --validation-workers 1`). More calls just wait on the GPU.

The useful hole-fill was four A100s, three calls each. That is 12 in-flight requests. It is not "use all 32 GPUs the account is allowed." The credit limit arrived before the GPU limit did.

### 5. `tanh` and `sigmoid` are not in this Triton build

The first large `tanh_mix` wave almost all failed. The kernels called `tl.tanh` or `tl.math.tanh`, and Triton 3.2 raised `AttributeError`. The one kernel that passed used the identity `tl.exp(2.0 * z)`.

The shared system prompt, used by the teacher, by training, and by evaluation, now says this explicitly: this Triton 3.2 build has no `tl.tanh`, `tl.math.tanh`, `tl.sigmoid`, or `tl.math.sigmoid`; implement both from `tl.exp`. That sentence is an environment fact, not a hint copied from a solution. The held-out family `sigmoid_affine` has the same missing builtin.

After the prompt change, a quiet retry of the 178 credit-failed `tanh_mix` tasks passed 176. The other two failed the static screen: a tensor `.view` outside the Triton JIT.

### 6. The teacher ran out of credits mid-retry

The four-GPU retry (job array 21908955) finished about 413 attempts: 125 passed, 109 failed, 179 errored. The errors were not rate limits. They were HTTP 403, non-retryable: the team had used its credits or its monthly spending limit. A later probe returned HTTP 200 after credits were added. The `tanh` retry above is that second spend.

Known response-token costs from the summaries that exist:

| Run | Job shape | Attempts | Passed | Failed | Errors | Known cost | Elapsed |
|---|---|---|---|---|---|---|---|
| `grok-train-2000` | cancelled, job 21830819 | 1,027 | 857 | not summarized | | no summary file | |
| `grok-train-rest` | 1 GPU, job 21898909 | 973 | 640 | 326 | 7 SSL | $4.62 | 5.6 h |
| `grok-val-100` | same job, second step | 100 | 96 | 4 | 0 | $0.46 | 21 min |
| `grok-retry-s0` through `s3` | 4 GPUs | 413 | 125 | 109 | 179 | $1.18 | ~9 min each |
| `grok-tanh-retry` | 1 GPU, job 21909887 | 178 | 176 | 2 | 0 | $0.92 | 16 min |

The seven errors in the rest run were `SSLV3_ALERT_BAD_RECORD_MAC`. They were retried by starting a new output directory, not by appending to the old one. `kernel_lab.generate` refuses an output directory that already contains `attempts.jsonl` or `verified.jsonl`.

`row_mean` stayed weak after the retry: 101 training records out of 200 tasks. The misses are numerical (`AssertionError: Tensor-likes are not close!`). The kernels do divide by N. A third paid pass was not worth it. `row_sum` ended at 161. Easy elementwise families landed around 185–195.

A small pilot was about $0.0045–$0.005 per task. That is a measured sample, not a promise for every family.

### 7. Choosing the model

The original training target was `Qwen2.5-Coder-0.5B-Instruct`. After the kernels existed, the base model was re-chosen.

There is no official Qwen3-Coder under 10B. The official Qwen3.8 sizes are 27B and a much larger mixture-of-experts model. Hugging Face names like "Qwen3.8-9B" are unofficial distillations, not Qwen releases. There is no official 34B. The model that was trained is **Qwen/Qwen3.8-27B**.

It is a hybrid vision-language model (`Qwen3_5ForConditionalGeneration`, model type `qwen3_5`): Gated DeltaNet plus full attention every fourth layer, 64 layers, hidden size 5120, vocab 248320. Thinking is on by default. Targets are kernel source only, so thinking has to be off. Load it with `AutoModelForImageTextToText`, not `AutoModelForCausalLM`. Transformers 5.8 or newer is required. The train env has 5.20.2.

LoRA targets the text linear layers whose names end in `q_proj`, `k_proj`, `v_proj`, `o_proj`, `in_proj_qkv`, `in_proj_z`, `in_proj_b`, `out_proj`, `gate_proj`, `up_proj`, or `down_proj`. Anything with `visual`, `vision_tower`, `vision_model`, or `vision_encoder` in the name stays frozen. The successful run found 448 target modules. Trainable parameters: 112,758,784 out of 27,469,487,344.

`LoraConfig` does not take `revision`. An earlier draft passed it and would have crashed. That argument was removed before the first real submission.

### 8. The first LoRA job died on the chat template

Job 21910419 failed in 19 seconds. The GPU was fine. The crash was:

`ValueError: Chat template prefix mismatch; cannot reliably mask the response`

Qwen3.8 with `enable_thinking=False` still closes an empty thinking block in the generation prompt. The rendered string of the prompt is a prefix of the full example. The token ids are not, because byte-pair encoding merges the boundary between the prompt and the kernel. A check of "are the prompt ids a prefix of the full ids?" is the wrong test for this template.

The mask now works in two steps (`training/data.py`):

1. If the token ids do line up, cut there. The unit tests use a fake tokenizer and still take this path.
2. Otherwise render both sides as text, require `full_text.startswith(prefix_text)`, tokenize the full text with character offsets, and mask through the last token that touches the prompt. A token that straddles the boundary is masked, so the prompt is never trained on. At most one response token is dropped.

All 1,894 train and validation rows were checked with the real Qwen3.8 tokenizer before the job was resubmitted. Every response label starts at the kernel (`import torch`). The longest row is 829 tokens. The cap is 4,096, so nothing was truncated.

The failed job's `finally` block had already written `run_summary.json` and `pip-freeze.txt`. The training script refuses a non-empty output directory. Those two files had to be removed before a resubmit. The verified kernels were not touched.

### 9. The second LoRA job died on a renamed argument

Job 21910574 loaded all 1,184 weight shards and then crashed:

`TypeError: TrainingArguments.__init__() got an unexpected keyword argument 'warmup_ratio'`

Transformers 5.20 folded the old ratio into `warmup_steps`. A float in `[0, 1)` means a fraction of the optimizer steps. `warmup_ratio=0.03` became `warmup_steps=0.03`. Same schedule. The output directory had to be emptied again.

### 10. The LoRA run that finished

Job **21910579**, one A100-SXM4-80GB, node time 1 hour 15 minutes, exit 0. Adapter: `artifacts/lora-qwen38-27b/adapter` (about 431MB).

| | |
|---|---|
| Epochs | 1 |
| Steps | 113 (1,798 rows, batch 1, accumulation 16) |
| Learning rate | 1e-4, cosine, 3% warmup |
| Rank / alpha | 16 / 32 |
| Train loss | 0.0533 |
| Validation loss | 0.0326 |
| Training GPU time | 1.24 hours inside the process |
| Full process GPU time | 1.26 hours, still excluding queue wait |

That loss is next-token loss on the kernel source. It is not a count of kernels that pass the checker. The LoRA model has not been scored on held-out tasks yet.

Two warnings showed up and did not stop the run:

- The host kernel is 4.18, below the 5.5 that some libraries recommend. Training finished anyway.
- `causal_conv1d` and `flash-linear-attention` are not installed, so the DeltaNet layers used the PyTorch reference kernels. Those are correct and slower. The run still finished inside the 8 hour limit. A later serving install should add them if the goal is speed.

`resolved_model_revision` in the summary is null. The weights came from the local Hugging Face cache. Do not invent a commit hash for the report.

Setup that made this possible is job 21910214 (`slurm/setup_qwen38.sbatch`): Python 3.13.11, torch 2.6.0+cu124, transformers 5.20.2, peft 0.21.2, accelerate 1.15.0, in `/scratch/qdf7ea/venvs/kernel-train-qwen38`.

### 11. The scratch model imported pandas before it trained

`llm.py` imported pandas at the top of the file. The training venv does not have pandas. Job 21912780 died in 11 seconds with `ModuleNotFoundError`. Pandas is only used to read the old `training_data.csv`. The import now sits inside that CSV function. The scratch architecture was not changed.

The held-out job that was waiting on 21912780 (21912781) was cancelled. Slurm had marked it `DependencyNeverSatisfied`. Leaving it queued would never have started it.

### 12. The scratch check scored 0/1,200 for a bad reason

Training itself succeeded. Job 21913389, about 2 minutes of GPU time, 29,496,192 parameters, AdamW, learning rate 3e-4, 3 epochs, batch 1, accumulation 8, max length 4096, tokenizer `HuggingFaceTB/SmolLM-135M`. Validation response loss: 0.405, then 0.353, then 0.372. Checkpoint: `artifacts/scratch-qwen38/model.pt`. This is not the old overlapping-window run.

Job 21913390 then "finished" in 16 seconds with 0 verified out of 1,200. Latency was about 1 millisecond. Every row was `AttributeError` with an empty message, stage `generation`. The GPU checker never ran.

`apply_chat_template(..., return_tensors="pt")` on Transformers 5 returns a `BatchEncoding`, not a tensor. `BatchEncoding.__getattr__` raises a bare `AttributeError()` when the name is missing, so `.shape` fails with no text. The fix in `kernel_lab/baseline.py` takes `encoded["input_ids"]` before calling `.to` or reading `.shape`.

A one-task check after the fix returned kernel text starting `import torch` / `import triton`. The bogus ledger was deleted. The check was submitted again as job **21913925**.

Prompt lengths under the SmolLM tokenizer, on all 1,200 test tasks, are short: minimum 504, median 523, maximum 527. The context is 4,096 and the generation budget is 2,048, so the "not enough room" refusal does not apply to this test set.

When this note was written, 21913925 was running on one A100 and the first completed rows were real checker failures, not the attribute error: `AssertionError: Tensor-likes are not close!` The generated text was actual Triton. That pass rate was not finished, so it is not reported here. Read `artifacts/eval-scratch-test/summary.json` after the job ends.

## Rules that kept the results clean

- One new output directory per generation run. Never resume in place.
- Response-only loss. Over-long examples are refused, not silently trimmed.
- `prepare_records` rejects a family or a reference-code hash that appears in two splits.
- Distinct kernels for the same task are kept. The exact same `(task_id, code)` pair is not.
- Evaluation refuses a directory that already has `evaluation.jsonl`, and it refuses any task that is not validation or test.
- Every requested task stays in the denominator. A generation crash is a failure, not a skipped row.
- Validation families can be used to adjust a prompt. The test families are the final score. They were not used to tune the tanh sentence beyond the environment fact that sigmoid is also missing.
- README numbers get copied from saved summaries only. `python -m kernel_lab.metrics` renders those summaries. Unknown money stays null.

## What is done, and what is not

Done:

- GPU smoke of the sandbox checker.
- Teacher data for 10 train families and 2 validation families.
- Supervised files `artifacts/sft-qwen38/`.
- Qwen3.8-27B LoRA, job 21910579, adapter on disk.
- Scratch checkpoint, job 21913389.

Running when this file was written:

- Scratch held-out check, job 21913925, output `artifacts/eval-scratch-test/`.

Not started:

- vLLM serving of base Qwen3.8-27B and of the LoRA adapter.
- The same 1,200-task checker run for those two models.
- README table filled with measured pass rates.

The serving scripts in the repo cannot do that last part yet. `training/requirements-vllm.txt` pins vLLM 0.11.0, and `kernel-serve` points at the removed Python 3.11. Qwen3.8 needs vLLM 0.17 or newer, plus Transformers 5.8 or newer. Thinking has to be off on the server (`--default-chat-template-kwargs '{"enable_thinking": false}'`). Do not also send `reasoning_effort=low` from the client: on this model that turns thinking back on. The checker and a 27B server should not share one 80GB card. The server holds most of the memory, and the checker needs the GPU too. Plan on two A100s, or generate every candidate first and check afterwards.

## Update, 27 October 2026: all four scores and the serving run

Everything above was written before the held-out checks finished. They have now finished, and the serving run is done too. These are the numbers a reader should see.

### Held-out correctness, 1,200 tasks from three families that were never trained on

| Model | Passed | `sigmoid_affine` | `row_max` | `row_relu_sum` | Job |
|---|---|---|---|---|---|
| Qwen3.8-27B, no fine-tune | 537 | 400 | 117 | 20 | 21918029 |
| Qwen3.8-27B + LoRA r16, 1 epoch | **1,178** | 378 | 400 | 400 | 21918030 |
| Qwen2.5-Coder-0.5B + LoRA r16, 1 epoch | 147 | 147 | 0 | 0 | 22006999 |
| Scratch Transformer, 29.5M, 3 epochs | 0 | 0 | 0 | 0 | 21913925 |

What each row actually says:

- The 27B LoRA fixed both row-reduction families completely. Its 22 failures are all numerical, all inside `sigmoid_affine`. It is not perfect, and the imperfection is in one family, not spread across the test set.
- The base 27B model mostly broke the checker's host-code rules: 249 kernels used a Python `for` loop outside the Triton kernel, 147 called `zero_`, 36 called `fill_`, and 28 had a syntax error. That is the static screen catching kernels that would have produced plausible numbers by PyTorch arithmetic instead of Triton.
- The 0.5B coder learned part of one elementwise family and none of the row reductions. 419 failures returned the wrong shape, a per-block value where one scalar per row was required. That is the specific thing a small model fails to infer: the reduction is across a block, but the output must be per row.
- The scratch model never produced a legal Triton module. That zero is the control. Without it, a reader could say the fine-tune only looks good because the test is easy.

So the useful claim is not "bigger is better." It is: a model that already writes code, adapted on 1,798 GPU-checked kernels, transfers to unseen operation families, including reductions it never saw; a from-scratch model of the same data cannot, and a 0.5B coder only partly can.

### Serving, measured with vLLM

Environment: `/scratch/qdf7ea/venvs/kernel-serve-py313`, built by job 22062441 on the `standard` partition. vLLM 0.32.0, torch 2.13.0+cu130, transformers 5.19.0. It is a separate venv because vLLM replaces torch and transformers.

The first setup attempt failed on a mistake in my own check: the script asked `importlib.metadata` for `peft`, which belongs to the training environment and is deliberately absent here. The install itself had succeeded; the check raised `PackageNotFoundError` and the job exited 1. That is worth remembering: a verification step can fail a good build.

The first serve attempt failed with `exec: vllm: not found`. `scripts/serve_vllm.sh` assumed the venv was activated, and the batch shell never activated it. Fix: put the venv's `bin` on `PATH` in the job. The same attempt also printed the API key into the log, because the script echoed the full argument list. The key now goes through `VLLM_API_KEY` in the environment, and vLLM reads it there, so it is not in argv at all.

Job 22064175, one A100 80GB, 13 minutes total:

| | Base 27B | + LoRA |
|---|---|---|
| Requests succeeded | 24 / 24 | 24 / 24 |
| p50 latency | 10.47 s | 10.25 s |
| p95 latency | 12.21 s | 11.44 s |
| Aggregate output tok/s, 4 workers | 101.9 | 94.5 |
| Median per-request output tok/s | 27.1 | 23.8 |

Startup dominates the job: 51.27 GiB of weights loaded in about 18 seconds from WEKA, then ~2 minutes of `torch.compile` and FlashInfer JIT. The server reported a KV cache of 139,264 tokens, 17× concurrency at 8,192 tokens per request. Twenty-four requests is a small sample, and this benchmark does no kernel validation. It answers "how fast does the served model reply," which is a different question from "does its kernel pass."

### What the full stack now covers

| Skill area | Evidence in this repo |
|---|---|
| Model training | PEFT LoRA on a 27B hybrid model, single-GPU accounting, response-only masking, a 0.5B comparison, a from-scratch control |
| Evaluation design | Family-separated splits, held-out families untouched until the end, failures retained in the denominator, four models on identical tasks |
| Sandbox engineering | Static screen plus GPU numerical check, Apptainer isolation with the weaker-isolation tradeoff written down, per-job output caps, fail-closed parsing |
| Serving | vLLM 0.32.0, thinking disabled server-side, LoRA hot-loaded as a second model name, latency measured through the repo's own benchmark client |
| Cluster operations | Slurm arrays, dependency chains, GPU pinning across two cards, reruns after a site removed an interpreter, saved summaries as the only source of numbers |

Still missing, and honestly labelled: the pgvector retrieval comparison. The code exists (`kernel_lab/retrieval.py`, `evaluate.py --k`), and it was deliberately left out of the measured path.

## How to read a future result

A lower LoRA loss does not mean more kernels passed. The number that matches the project goal is, for the same 1,200 test tasks:

- base Qwen, thinking off, no retrieval
- Qwen plus the LoRA adapter, thinking off, no retrieval
- the scratch checkpoint

Each report needs the pass count, the failure stage, the job id, the GPU, and the summary file. Coefficient variants inside one family are not 400 independent proofs. Report the three families separately before averaging them.
