"""Loopback-only CPU embeddings; optional sentence-transformers dependency."""
import argparse
from http.server import BaseHTTPRequestHandler, HTTPServer
import hmac
import json
import math
import os
from pathlib import Path

MODEL_ID = "sentence-transformers/all-MiniLM-L6-v2"
DIMENSIONS = 384
MAX_BODY = 2 * 1024 * 1024


class LocalEncoder:
    def __init__(self, revision="main", offline=False):
        from huggingface_hub import snapshot_download
        from sentence_transformers import SentenceTransformer
        snapshot = snapshot_download(MODEL_ID, revision=revision, local_files_only=offline,
            allow_patterns=["*.json", "*.safetensors", "*.txt", "*.model", "1_Pooling/*"])
        self.revision = Path(snapshot).name
        self.model_id = f"{MODEL_ID}@{self.revision}"
        self.model = SentenceTransformer(snapshot, device="cpu", trust_remote_code=False)
        self.model.eval()
        if self.model.get_sentence_embedding_dimension() != DIMENSIONS:
            raise ValueError("Unexpected embedding dimensions")

    def encode(self, texts):
        import torch
        with torch.inference_mode():
            vectors = self.model.encode(texts, device="cpu", normalize_embeddings=True,
                convert_to_numpy=True, show_progress_bar=False, batch_size=16).tolist()
        lengths = [len(self.model.tokenizer.encode(text, truncation=False)) for text in texts]
        limit = self.model.max_seq_length
        return vectors, sum(min(length, limit) for length in lengths), [i for i, n in enumerate(lengths) if n > limit]


def embed_request(encoder, body):
    if not isinstance(body, dict) or body.get("model") != encoder.model_id:
        raise ValueError("Unknown model; read /health for the exact revision-qualified model id")
    if body.get("encoding_format", "float") != "float":
        raise ValueError("Only float encoding_format is supported")
    if body.get("dimensions", DIMENSIONS) != DIMENSIONS:
        raise ValueError("This model has native dimensions=384")
    texts = body.get("input")
    if isinstance(texts, str):
        texts = [texts]
    if not isinstance(texts, list) or not 1 <= len(texts) <= 32:
        raise ValueError("input must be one string or 1..32 strings")
    if any(not isinstance(t, str) or not t.strip() or len(t) > 100_000 for t in texts):
        raise ValueError("Each input must be a nonempty string of at most 100000 characters")
    vectors, tokens, truncated = encoder.encode(texts)
    if len(vectors) != len(texts) or any(len(v) != DIMENSIONS or any(not math.isfinite(x) for x in v) for v in vectors):
        raise RuntimeError("Encoder returned invalid vectors")
    return {"object": "list", "model": encoder.model_id,
            "data": [{"object": "embedding", "index": i, "embedding": v} for i, v in enumerate(vectors)],
            "usage": {"prompt_tokens": tokens, "total_tokens": tokens},
            "truncated_inputs": truncated}


def make_handler(encoder, api_key="local"):
    class Handler(BaseHTTPRequestHandler):
        def log_message(self, *args):
            pass  # No request bodies, URL queries, or authorization headers in logs.

        def reply(self, status, body):
            data = json.dumps(body, allow_nan=False).encode()
            self.send_response(status)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(data)))
            self.end_headers()
            self.wfile.write(data)

        def do_GET(self):
            if self.path != "/health":
                return self.reply(404, {"error": {"message": "Not found"}})
            self.reply(200, {"status": "ready", "model": encoder.model_id, "dimensions": DIMENSIONS,
                             "device": "cpu", "normalized": True})

        def do_POST(self):
            if self.path != "/v1/embeddings":
                return self.reply(404, {"error": {"message": "Not found"}})
            if not hmac.compare_digest(self.headers.get("Authorization", ""), f"Bearer {api_key}"):
                return self.reply(401, {"error": {"message": "Invalid API key"}})
            try:
                length = int(self.headers.get("Content-Length", "0"))
                if not 0 < length <= MAX_BODY:
                    raise ValueError("Body must be 1..2097152 bytes")
                body = json.loads(self.rfile.read(length))
                result = embed_request(encoder, body)
            except (ValueError, UnicodeDecodeError) as exc:
                return self.reply(400, {"error": {"message": str(exc)}})
            except Exception:
                return self.reply(500, {"error": {"message": "Embedding inference failed"}})
            self.reply(200, result)
    return Handler


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--port", type=int, default=8001)
    parser.add_argument("--revision", default="main", help="Pin a Hugging Face commit SHA for final experiments")
    parser.add_argument("--offline", action="store_true", help="Use only an already cached model snapshot")
    args = parser.parse_args()
    encoder = LocalEncoder(args.revision, args.offline)
    server = HTTPServer(("127.0.0.1", args.port), make_handler(encoder, os.getenv("EMBEDDING_API_KEY", "local")))
    print(json.dumps({"url": f"http://127.0.0.1:{args.port}/v1", "model": encoder.model_id,
                      "dimensions": DIMENSIONS, "device": "cpu"}), flush=True)
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        pass
    finally:
        server.server_close()


if __name__ == "__main__":
    main()
