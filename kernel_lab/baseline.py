"""Corrected from-scratch architecture on the same supervised examples as LoRA.

Run `python -m kernel_lab.baseline --help`. No legacy CSV/cache is consumed.
"""
import argparse
import asyncio
import hashlib
import importlib.metadata
import json
import os
from pathlib import Path
import time


def check_splits(train, validation):
    if not train or not validation:
        raise ValueError("Both train and validation must be nonempty")
    for field in ("family", "reference_code"):
        def values(rows):
            return {row.get("task", row).get(field, row.get(field)) for row in rows}
        left, right = values(train), values(validation)
        if None in left or None in right or left & right:
            raise ValueError(f"Missing or overlapping {field} across splits")


def train(args):
    import torch
    from transformers import AutoTokenizer
    from llm import MinimalLLM, ModelConfig, set_seed
    from training.data import read_jsonl, response_only_example, prepare_records
    from kernel_lab.prompts import build_messages

    train_rows, val_rows = read_jsonl(args.train_file), read_jsonl(args.val_file)
    check_splits(train_rows, val_rows)
    checked = prepare_records(train_rows + val_rows)
    if any(row.get("split") != "train" for row in train_rows) or any(row.get("split") not in {"validation", "val"} for row in val_rows):
        raise ValueError("Input files do not match declared splits")
    train_rows, val_rows = checked["train"], checked["validation"]
    if len(train_rows) + len(val_rows) == 0:
        raise ValueError("No verified supervised examples")
    if not torch.cuda.is_available() and not args.allow_cpu:
        raise RuntimeError("GPU required for measured runs; --allow-cpu is only a smoke-test option")
    set_seed(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if device.type == "cuda" and torch.cuda.device_count() != 1:
        raise RuntimeError("Allocate exactly one visible GPU for this single-GPU baseline")
    output = Path(args.output_dir)
    if output.exists() and any(output.iterdir()):
        raise ValueError("Use a new empty output directory")
    tokenizer = AutoTokenizer.from_pretrained(args.tokenizer, revision=args.tokenizer_revision)
    if not tokenizer.chat_template:
        tokenizer.chat_template = "{% for message in messages %}{{ message['role'] + ':\\n' + message['content'] }}{% if message['role'] == 'assistant' %}{{ eos_token }}{% endif %}{{ '\\n' }}{% endfor %}{% if add_generation_prompt %}{{ 'assistant:\\n' }}{% endif %}"
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token
    def encode(rows):
        return [response_only_example(tokenizer, build_messages(row.get("task", row)),
                                      row["completion"], args.max_length) for row in rows]
    train_data, val_data = encode(train_rows), encode(val_rows)
    config = ModelConfig(vocab_size=len(tokenizer), max_seq_len=args.max_length,
                         num_epochs=args.epochs, use_amp=False)
    model = MinimalLLM(config).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=0.1)
    def collate(rows):
        length = max(len(row["input_ids"]) for row in rows)
        x = torch.full((len(rows), length), tokenizer.pad_token_id, dtype=torch.long)
        y = torch.full_like(x, -100)
        for index, row in enumerate(rows):
            size = len(row["input_ids"])
            x[index, :size] = torch.tensor(row["input_ids"])
            y[index, :size] = torch.tensor(row["labels"])
        return x.to(device), y.to(device)
    loader = torch.utils.data.DataLoader(train_data, batch_size=args.batch_size,
                                         shuffle=True, collate_fn=collate)
    val_loader = torch.utils.data.DataLoader(val_data, batch_size=args.batch_size,
                                             collate_fn=collate)
    output.mkdir(parents=True, exist_ok=True)
    start = time.perf_counter()
    losses = []
    best = float("inf")
    optimizer.zero_grad(set_to_none=True)
    for epoch in range(args.epochs):
        model.train()
        for index, (x, labels) in enumerate(loader):
            group_start = index // args.accumulation * args.accumulation
            divisor = min(args.accumulation, len(loader) - group_start)
            logits = model(x)
            loss = torch.nn.functional.cross_entropy(logits[:, :-1].reshape(-1, len(tokenizer)),
                                                       labels[:, 1:].reshape(-1), ignore_index=-100)
            (loss / divisor).backward()
            if (index + 1) % args.accumulation == 0 or index + 1 == len(loader):
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                optimizer.step()
                optimizer.zero_grad(set_to_none=True)
        model.eval()
        total_loss = total_tokens = 0
        with torch.no_grad():
            for x, labels in val_loader:
                logits = model(x)
                targets = labels[:, 1:]
                total_loss += torch.nn.functional.cross_entropy(
                    logits[:, :-1].reshape(-1, len(tokenizer)), targets.reshape(-1),
                    ignore_index=-100, reduction="sum").item()
                total_tokens += (targets != -100).sum().item()
        val_loss = total_loss / total_tokens
        losses.append({"epoch": epoch + 1, "validation_response_loss": val_loss})
        print(json.dumps(losses[-1]), flush=True)
        if val_loss < best:
            best = val_loss
            torch.save({"config_dict": vars(config), "model_state_dict": model.state_dict(),
                        "tokenizer": args.tokenizer}, output / "model.pt")
    if device.type == "cuda":
        torch.cuda.synchronize()
    elapsed = time.perf_counter() - start
    tokenizer.save_pretrained(output / "tokenizer")
    allocated = torch.cuda.device_count() if device.type == "cuda" else 0
    report = {"model": "scratch-transformer", "parameters": sum(p.numel() for p in model.parameters()),
              "optimizer": "AdamW", "seed": args.seed, "device": str(device),
              "training_elapsed_seconds": elapsed, "allocated_gpus": allocated,
              "training_gpu_hours": allocated * elapsed / 3600,
              "gpu_hours_scope": "training+epoch validation+checkpoint saves; excludes queue/download/setup",
              "gpu_name": torch.cuda.get_device_name() if allocated else None,
              "tokenizer_revision_requested": args.tokenizer_revision,
              "tokenizer_revision_resolved": tokenizer.init_kwargs.get("_commit_hash"),
              "versions": {p: importlib.metadata.version(p) for p in ("torch", "transformers", "numpy")},
              "train_sha256": hashlib.sha256(Path(args.train_file).read_bytes()).hexdigest(),
              "validation_sha256": hashlib.sha256(Path(args.val_file).read_bytes()).hexdigest(),
              "history": losses, "settings": vars(args),
              "note": "Parameter count depends on tokenizer vocabulary; this is not the uncorrected historical run."}
    (output / "metrics.json").write_text(json.dumps(report, indent=2), encoding="utf-8")


class BaselineTeacher:
    """Teacher-compatible inference adapter; the custom model is not supported by vLLM."""
    def __init__(self, checkpoint, max_tokens=2048, **kwargs):
        import torch
        from transformers import AutoTokenizer
        from llm import MinimalLLM, ModelConfig
        self.torch = torch
        folder = Path(checkpoint)
        # Only load a locally produced, trusted checkpoint (pickle-based format).
        saved = torch.load(folder / "model.pt", map_location="cpu", weights_only=True)
        config = ModelConfig(**{key: value for key, value in saved["config_dict"].items() if key != "d_k"})
        self.model = MinimalLLM(config)
        self.model.load_state_dict(saved["model_state_dict"])
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.model.to(self.device).eval()
        self.tokenizer = AutoTokenizer.from_pretrained(folder / "tokenizer")
        self.max_tokens = max_tokens
        self.lock = asyncio.Lock()

    async def complete(self, task, examples=None):
        async with self.lock:
            return await asyncio.to_thread(self._complete, task, examples or [])

    def _complete(self, task, examples):
        from .prompts import build_messages
        start = time.perf_counter()
        encoded = self.tokenizer.apply_chat_template(build_messages(task, examples), tokenize=True,
                                                     add_generation_prompt=True, return_tensors="pt")
        # Transformers 5 returns a BatchEncoding here. Its missing-key lookup
        # raises a bare AttributeError, so the prompt tensor has to be taken
        # out explicitly before .shape or .to is used.
        ids = encoded["input_ids"] if hasattr(encoded, "keys") else encoded
        ids = ids.to(self.device)
        available = self.model.config.max_seq_len - ids.shape[1]
        if available < self.max_tokens:
            raise ValueError("Insufficient baseline context for the full requested completion budget; increase training context or reduce the shared API/baseline budget")
        generated = self.model.generate(ids, max_length=ids.shape[1] + self.max_tokens,
                                        do_sample=False, top_k=0, top_p=1.0,
                                        eos_token_id=self.tokenizer.eos_token_id)
        completion = self.tokenizer.decode(generated[0, ids.shape[1]:], skip_special_tokens=True)
        return {"candidate_code": completion, "model": "scratch-transformer", "prompt_tokens": ids.shape[1],
                "completion_tokens": generated.shape[1] - ids.shape[1], "cost_usd": None,
                "latency_seconds": time.perf_counter() - start, "request_attempts": 1}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--train-file", required=True)
    parser.add_argument("--val-file", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--tokenizer", default="HuggingFaceTB/SmolLM-135M")
    parser.add_argument("--tokenizer-revision", default="main")
    parser.add_argument("--max-length", type=int, default=4096)
    parser.add_argument("--epochs", type=int, default=3)
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--accumulation", type=int, default=8)
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--allow-cpu", action="store_true")
    args = parser.parse_args()
    if min(args.epochs, args.batch_size, args.accumulation, args.max_length) < 1:
        parser.error("Training sizes must be positive")
    train(args)


if __name__ == "__main__":
    main()
