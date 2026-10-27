# Local CPU embeddings for retrieval

Grok/API credits provide generation access, not necessarily an embeddings endpoint. This project includes a local CPU service with the OpenAI `/v1/embeddings` response shape, so you can build pgvector retrieval without a second paid account. It makes no paid API calls. The first start downloads open model files from Hugging Face; offline operation works after caching them. Running it does use your CPU, RAM, disk and electricity.

The fixed model is `sentence-transformers/all-MiniLM-L6-v2`, with 384-dimensional normalized vectors. It is an accessible retrieval baseline, not a code-specialized model. The [official model card](https://huggingface.co/sentence-transformers/all-MiniLM-L6-v2) explains that inputs beyond 256 word pieces are truncated by default. This server keeps the model's default limit and returns `truncated_inputs` indices in each response; consequently long kernels/references may lose their tail in the embedding. It does not chunk long code or claim full-code semantic understanding. Measure retrieval quality before changing the embedding strategy, and rebuild the entire index if it changes.

## Install and start

Use a separate local environment or the project's development environment. Python 3.11/3.12 is a practical starting point. There is no GPU requirement and no vLLM dependency. From the repository root:

```bash
python -m pip install -e '.[embedding,retrieval]'
python -m kernel_lab.embed_server --port 8001
```

On Windows PowerShell the same Python commands work; use double quotes around `".[embedding,retrieval]"` if necessary. PyTorch may download a GPU-capable wheel even though inference is forced to CPU; consult the official PyTorch CPU-wheel installation instructions if download/disk size matters. Retrieval was not part of the measured path; the setup here was validated only by the CPU mock tests in `tests/`.

Keep that terminal open. On startup the service prints its base URL, dimensions and **revision-qualified** model ID, such as `sentence-transformers/all-MiniLM-L6-v2@<resolved-commit-sha>`. The identity includes the actual cached snapshot commit, even when the requested revision is `main`. This prevents a moving model revision from silently sharing the same pgvector model identity.

The server always binds to `127.0.0.1`. It is a local research helper, with sequential CPU inference and no public deployment hardening. Do not expose it through a public port. Set `EMBEDDING_API_KEY` before starting it to use a private bearer key; otherwise it requires the development key `local`, matching the client's default. Health information is available without a key; input/authorization values are not logged.

For reproducibility, restart future sessions using the resolved commit:

```bash
python -m kernel_lab.embed_server --port 8001 --revision YOUR_RESOLVED_COMMIT_SHA
# Once cached, prevent all Hugging Face download attempts:
python -m kernel_lab.embed_server --port 8001 --revision YOUR_RESOLVED_COMMIT_SHA --offline
```

Use `HF_HOME` to place the cache somewhere with enough disk. On Rivanna choose scratch for disposable caches and follow UVA's resource policy for CPU work; back up datasets/results because scratch is purged. The model loader disables remote model code. The [Sentence Transformers API](https://www.sbert.net/docs/package_reference/sentence_transformer/model.html) documents CPU placement, revision selection and normalized encoding; this implementation also sets eval mode and uses PyTorch inference mode.

## Connect retrieval

In a second terminal:

```bash
curl http://127.0.0.1:8001/health
export EMBEDDING_BASE_URL=http://127.0.0.1:8001/v1
export EMBEDDING_MODEL='COPY_THE_EXACT_model_VALUE_FROM_HEALTH'
export EMBEDDING_DIMENSIONS=384
export EMBEDDING_API_KEY=local
```

PowerShell equivalent:

```powershell
$info = Invoke-RestMethod http://127.0.0.1:8001/health
$env:EMBEDDING_BASE_URL = 'http://127.0.0.1:8001/v1'
$env:EMBEDDING_MODEL = $info.model
$env:EMBEDDING_DIMENSIONS = '384'
$env:EMBEDDING_API_KEY = 'local'
```

Use the private key instead of `local` if you configured one in the server terminal. You can inspect one embedding with Python without shell JSON quoting:

```bash
python -c 'import os,httpx; r=httpx.post(os.environ["EMBEDDING_BASE_URL"]+"/embeddings",headers={"Authorization":"Bearer "+os.environ["EMBEDDING_API_KEY"]},json={"model":os.environ["EMBEDDING_MODEL"],"input":"elementwise addition of two CUDA float32 tensors"});r.raise_for_status();d=r.json();print(d["model"],len(d["data"][0]["embedding"]),d["usage"],d["truncated_inputs"])'
```

`EMBEDDING_MODEL` must exactly match the health value; the endpoint rejects unknown IDs. Requests support one text or a batch of 1..32 texts, float encoding and native dimensions=384. A batch preserves input ordering. Token usage counts the model's processed tokens, capped by its input limit; it is not a billable API token count.

Start PostgreSQL/pgvector, configure `PGVECTOR_DSN`, and build the index with `python -m kernel_lab.retrieval --verified artifacts/verified-qwen38.jsonl`. The paired run uses the retrieval flags on `python -m kernel_lab.evaluate` (see the CLI help); without `--no-rag-only` it requires a live store and embedder. The index records model ID, base URL and vector dimension and rejects inconsistent embedding configurations. Keep those settings stable for indexing and querying. A new revision, model, normalization method, truncation/chunking policy or dimension requires a new database/index; mixing vectors invalidates similarity comparisons. Do not bypass the mismatch check.

Index only verified training data. Reserve validation/test families outside retrieval. A working endpoint/database does not establish improved validation pass rate; the held-out with/without retrieval experiment must supply that evidence.

## Placement and cost

If generation runs locally against Grok, run this embedding service and PostgreSQL locally too. Grok never needs to reach your embedding endpoint. If the retrieval client runs on Rivanna, `127.0.0.1` means that cluster node: a service on your laptop is not reachable through that address. Move the CPU embedding service and its index to an approved reachable location, run the orchestration client locally, or establish an appropriate authenticated private tunnel. Do not assume Slurm nodes reach a laptop or a local Docker network.

There is no embedding API bill for this local service. Its cost is not literally zero: report CPU resource usage separately if cost accounting includes it. Keep `pip freeze`, the resolved model revision, retrieval configuration and index provenance with experiment artifacts. Stop the service with Ctrl-C when finished.

## Tests and limitations

`python -m unittest discover -s tests -p test_embeddings.py -v` covers request shape, dimensions, batching, revision identity, health, bearer authentication and the actual loopback HTTP route using a fake encoder. It does not download the real model or prove real embedding quality. Run the real-model health/one-vector smoke test after installation, then measure retrieval ranking/pass-rate effects. The service is deliberately CPU-only and sequential; scale it after measuring a real bottleneck.
