# Implementation validation record

Date: 2026-10-18. Environment: the existing Windows project checkout and available Python installation. This records implementation checks, not model-quality or GPU-performance results.

## Completed checks

- `python -m unittest discover -s tests -v`: **38 tests passed** in the final run. PyTorch-backed position/causality tests executed; no tests were skipped in that run.
- `python -m compileall -q kernel_lab training sandbox tests llm.py`: completed successfully.
- `git diff --check`: no whitespace errors; Git emitted informational CRLF normalization warnings for existing root files.
- CLI help/startup checked for generation, evaluation, smoke validation, scratch training, LoRA training and the embedding server without downloading model weights.
- `pyproject.toml` parsed using `tomllib`; declared optional dependency groups include retrieval, embedding, baseline and development.
- `python -m kernel_lab.tasks --variants 400 --output artifacts/tasks-v1.jsonl`: generated **6,000 unverified tasks** across 15 families: 4,000 train, 800 validation, 1,200 test. This artifact is ignored by Git and can be regenerated deterministically; transfer it explicitly if using this exact manifest elsewhere.
- Root README/handoff local links checked and Slurm/shell files confirmed to use LF line endings.
- Independent code review identified PyTorch fallback acceptance, context-budget drift and retrieval provenance gaps. Wrapper screening, input/output storage checks, shared completion-budget instructions and retrieved-example snapshots were added to address them.

## What the tests cover

Split isolation, original-reference inclusion in prompts, corrected RoPE dimensions and causal forward behavior, tied embedding weights, candidate wrapper restrictions, subprocess invocation constraints, result/skip rejection, bounded output, API retry/malformed-output accounting, complete failure ledgers, verified-only output gating, retrieval filtering/parameter binding, unknown-cost handling, paired evaluation denominators, benchmark failure handling, response-only labels and local embedding HTTP behavior with a fake encoder.

Infrastructure tests use mocks. The model forward test uses a tiny configuration, not a trained 30M checkpoint. These checks do not establish production security, mathematical equivalence or broad model capability.

## Not run

- No paid Grok/provider calls; no API credits consumed by these implementation checks.
- No real CUDA/Triton candidate execution, Docker image build, Apptainer image conversion or cluster timeout/isolation smoke.
- No live PostgreSQL ingestion/search or downloaded embedding-model inference.
- No Rivanna SSH discovery, Slurm submissions, LoRA/scratch training, Qwen weight downloads or vLLM serving.
- No measured validation pass rate, accepted dataset expansion, latency/throughput benchmark, dollar cost per pair, GPU-hours or generated-kernel speedup.

The next milestone is the real hand-written GPU smoke test on the intended deployment, followed by a small OpenRouter pilot. Follow PROJECT_HANDOFF.md and docs/RIVANNA.md; replace placeholders with actual resource information and record new evidence as it becomes available.

## OpenRouter configuration update

The user subsequently selected `cohere/north-mini-code:free` through OpenRouter instead of Grok. Defaults, `.env.example`, the README and handoff now reflect that choice. A fixed `:free` model, zero-price provider filter, per-client pacing and bounded Retry-After handling were added. Three additional mocked tests verify paid/automatic model rejection, the zero-price request payload and pacing shared across workers; the current full suite is **38 passing tests**. No authenticated model calls or GPU validation were run during this update. Token price is configured as zero only for the selected free route; GPU/validation costs are separate.

## Update, 2026-10-27: the work above has since been executed

The sections above describe the state on 2026-10-18. Everything listed under "Not run" has now been run, except retrieval. This section supersedes those negative statements; the rest of the document is kept as the original record.

Now measured:

- Real teacher calls: paid xAI `grok-4.7`, 1,894 verified kernels across 10 train families and 2 validation families. Known spend $7.18 in response tokens for 1,037 passes in the runs that wrote a summary.
- Real CUDA/Triton execution: every accepted kernel and every held-out score ran in an Apptainer sandbox on an A100 80GB. Docker was unavailable and a `.sif` could not be built on this account, so the sandbox is a directory at `/scratch/qdf7ea/containers/kernel-lab-validator`.
- Real training: Qwen3.8-27B LoRA (job 21910579), Qwen2.5-Coder-0.5B LoRA (job 22001802), scratch Transformer 29.5M (job 21913389).
- Real held-out evaluation: four models, 1,200 tasks each, producing 537 / 1,178 / 147 / 0 passes.
- Real serving: vLLM 0.32.0 on one A100, base and LoRA adapters, 24 requests each, zero failures, p50 about 10.3 seconds.

Still not run: PostgreSQL ingestion, embedding-model inference, and the paired retrieval comparison. No generated-kernel speedup was measured; the runner deliberately reports none.

The CPU suite was re-run after the October code changes and still reports **38 tests passing**.
