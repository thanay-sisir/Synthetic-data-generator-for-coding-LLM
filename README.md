# Execution-validated Triton kernel lab

Generate complete Triton implementations with a teacher API, verify them against trusted PyTorch references in GPU containers, retrieve verified examples with PostgreSQL/pgvector, and compare a LoRA code model with a corrected small Transformer.

**Status:** measured end to end on UVA Rivanna with an A100 80GB allocation. Execution-validated kernels were generated, a LoRA and a scratch model were trained, all four models were scored on 1,200 held-out tasks with the same GPU checker, and the Qwen checkpoints were served with vLLM. PostgreSQL/pgvector retrieval is implemented but was not part of the executed path; the RAG comparison is still unmeasured. The historical CSV contains 45 pairs, not 319 or 2,000, and is not used by the new pipeline.

Implementation checks: **38 tests passed** on the CPU suite, which covers split isolation, prompt construction, candidate wrapper restrictions, ledger accounting and response-only labels. Those checks use mocks and a tiny model configuration, so they do not establish measured kernel correctness; the held-out evaluation below does.

This README carries the current context and execution order. **[docs/RIVANNA.md](docs/RIVANNA.md)** covers SSH-side setup, Slurm, downloads, LoRA, vLLM and tunnels. **[docs/EMBEDDINGS.md](docs/EMBEDDINGS.md)** covers local CPU embeddings.

## Pipeline

```text
Deterministic tasks (family-separated train/validation/test)
  -> xAI grok-4.7 teacher over the OpenAI-compatible client
  -> bounded async workers
  -> static screening + GPU Apptainer validation against the PyTorch reference
  -> attempts.jsonl (all outcomes) / verified.jsonl (passes only)
  -> response-only LoRA / corrected scratch-model training
  -> held-out execution check of every model on unseen families
  -> vLLM for Qwen base + LoRA -> endpoint benchmark

Optional, implemented but not part of the measured path:
  -> pgvector training index -> paired RAG/no-RAG execution evaluation
```

Initial scope: contiguous CUDA float32 elementwise operations and row reductions. The task builder supports up to 6,000 coefficient variants across **15 correlated operation families**. This does not mean 6,000 independent operations, formal correctness, or faster kernels. Accepted dataset size depends on execution outcomes.

## Development

Python 3.10+; the runs recorded here used Python 3.13.11 on Rivanna after `gcc/14.2.0`. Python 3.11.4 was removed from the site on 2026-10-23; the old `kernel-client`, `kernel-train` and `kernel-serve` venvs still point at it and do not work.

```bash
python -m venv .venv
source .venv/bin/activate
python -m pip install -e '.[retrieval,dev]'
python -m unittest discover -s tests -v
python -m kernel_lab.tasks --variants 20 --output artifacts/tasks.jsonl
```

Windows can run orchestration/unit tests. GPU containers need supported Linux/NVIDIA Docker or carefully configured WSL2. Docker is the primary validator; optional Apptainer requires explicit weaker-isolation opt-in and site approval. Neither is a hardened hostile-code service.

On Rivanna, Docker was unavailable and an Apptainer `.sif` could not be built (the account is not in `/etc/subuid`). The runs here used an Apptainer sandbox directory at `/scratch/qdf7ea/containers/kernel-lab-validator`, with `--backend apptainer --allow-weaker-isolation`. That shares the host GPU driver and is weaker than Docker.

Copy `.env.example` to private `.env`, edit values, then in Bash: `set -a; source .env; set +a`. CLIs do not auto-load `.env`. The default teacher is `cohere/north-mini-code:free` on OpenRouter; CPU embeddings can be hosted locally without a separate paid account.

## Results

Stage status, from the run ledger:

| Stage | Status |
|---|---|
| Task manifests, 6,000 tasks / 15 families | Done, `artifacts/tasks-v1.jsonl` |
| Teacher generation | Done with xAI `grok-4.7`, 1,894 verified records |
| GPU numerical validation | Done in an Apptainer sandbox directory |
| SFT preparation | Done, 1,798 train / 96 validation / 0 test |
| Qwen3.8-27B LoRA | Done, adapter saved |
| Qwen2.5-Coder-0.5B LoRA | Done |
| Scratch 29.5M baseline | Done |
| Held-out evaluation, all four models | Done, 1,200 tasks each |
| vLLM serving and latency benchmark | Done |
| pgvector retrieval comparison | Implemented, **not run** |

Every number below comes from a saved ledger or summary in `artifacts/`. Held-out means the 1,200 `test` tasks from `artifacts/tasks-v1.jsonl`: 400 each of `sigmoid_affine`, `row_max`, and `row_relu_sum`. None of those three families appears in training.

| Measurement | Current result | Evidence |
|---|---|---|
| Historical examples | 45 CSV pairs / 53 source files | Checked-in files; not new verification |
| New verified pairs | 1,894 passing records from 10 train families + 2 validation families | `artifacts/grok-*/summary.json`, `artifacts/verified-qwen38.jsonl` |
| Supervised records | 1,798 train / 96 validation / 0 test | `artifacts/sft-qwen38/manifest.json` |
| Held-out RAG / no-RAG pass rate | Not measured | Retrieval comparison was not run |
| p50 / p95 latency, output tokens/sec | Base: 10.47 / 12.21 s, 101.9 tok/s. LoRA: 10.25 / 11.44 s, 94.5 tok/s | `artifacts/bench-qwen38-*/summary.json` |
| API cost per verified pair | Known: $7.18 in response tokens for 1,037 passes across the runs that wrote a summary. The earlier cancelled run (`artifacts/grok-train-2000`, 857 passes) wrote no summary, so its cost is unknown, not zero. | `artifacts/grok-*/summary.json`; unknown stays null |

### Held-out kernel correctness, same 1,200 tasks, same GPU checker

| Model | Passed | `sigmoid_affine` | `row_max` | `row_relu_sum` | Job / ledger |
|---|---|---|---|---|---|
| Qwen3.8-27B, no fine-tune | 537 / 1,200 | 400 | 117 | 20 | 21918029, `artifacts/eval-qwen38-base-test` |
| Qwen3.8-27B + LoRA r16, 1 epoch | **1,178 / 1,200** | 378 | 400 | 400 | 21918030, `artifacts/eval-qwen38-lora-test` |
| Qwen2.5-Coder-0.5B + LoRA r16, 1 epoch | 147 / 1,200 | 147 | 0 | 0 | 22006999, `artifacts/eval-qwen25-coder-0.5b-test` |
| Scratch Transformer, 29.5M, 3 epochs | 0 / 1,200 | 0 | 0 | 0 | 21913925, `artifacts/eval-scratch-test` |

Reading the table:

- The 27B LoRA fixed both row-reduction families completely (400/400 and 400/400). All 22 of its failures are numerical, and all are in `sigmoid_affine`.
- The untouched 27B model mostly broke the sandbox's host-code rules: 249 kernels used a Python `for` loop on the host, 147 called `zero_`, and 28 had a syntax error. The checker is rejecting illegal host code, not only wrong numbers.
- The 0.5B coder learned part of the elementwise family and no row reduction. 419 of its failures returned the wrong shape (a per-block result instead of one scalar per row), 424 were numerically wrong, and 179 failed to compile.
- The from-scratch model never produced a legal Triton module from these 1,798 kernels. That is the control: the gain above comes from pretrained code ability, not from the data alone.

### Training resources

| Run | Hardware | Wall time | Loss |
|---|---|---|---|
| Qwen3.8-27B LoRA, job 21910579 | 1× A100 80GB | 1.24 h (process) | train 0.0533, validation 0.0326 |
| Qwen2.5-Coder-0.5B LoRA, job 22001802 | 1× A100 80GB | 0.15 h (process) | train 0.1554, validation 0.0483 |
| Scratch 29.5M, job 21913389 | 1× A100 80GB | 0.029 h | validation 0.405 → 0.353 → 0.372 |

These are next-token losses on kernel source. They are not pass rates, and lowering them further does not by itself raise the held-out score.

### Serving

| Measurement | Base Qwen3.8-27B | + LoRA r16 | Evidence |
|---|---|---|---|
| Requests succeeded | 24 / 24 | 24 / 24 | `artifacts/bench-qwen38-*/summary.json` |
| p50 complete-response latency | 10.47 s | 10.25 s | same |
| p95 complete-response latency | 12.21 s | 11.44 s | same |
| Aggregate output tokens/sec (4 workers) | 101.9 | 94.5 | same |
| Median per-request output tokens/sec | 27.1 | 23.8 | same |

Served by vLLM 0.32.0 on one A100 80GB, thinking off, 8,192-token context, 1,024-token cap, 24 held-out tasks, 2 warmup requests excluded. Job 22064175, 13 minutes total including a 51.27 GiB weight load and a ~2-minute `torch.compile`. The 24 requests are a small sample, and the LoRA row is slightly slower because the adapter adds a per-token matmul, not because its kernels are worse.

The held-out scores above were produced by loading each model in the evaluation job and generating one kernel at a time, then checking it in the sandbox. They were not produced by vLLM. This table measures a different thing: how fast the same checkpoints answer through an OpenAI-compatible server. No kernel validation happens in this benchmark.

Render completed reports with `python -m kernel_lab.metrics PATH_TO_SUMMARY_JSON ...`, then copy the measured table here with run IDs, hardware and methodology. Unknown costs remain null.

## Baseline

The custom model shares input/output embeddings: about **29.5M parameters** with the original 49,152-token vocabulary. RoPE now uses sequence positions, accumulation flushes partial groups, and scheduling counts optimizer updates. The old overlapping-window experiment requires `--legacy-experiment` because it leaks validation content. Use `python -m kernel_lab.baseline` for new family-separated comparisons.

The custom model uses PyTorch inference; it is not supported by vLLM. Qwen base/LoRA are served with vLLM 0.32.0 in `/scratch/qdf7ea/venvs/kernel-serve-py313`, separate from the training venv because vLLM replaces torch and transformers.
