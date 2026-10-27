"""Single-GPU LoRA training; run only inside an allocated GPU job."""
import argparse
import hashlib
import importlib.metadata
import json
import os
import platform
import subprocess
import time
from pathlib import Path

from training.data import prepare_records, read_jsonl, response_only_example

# Language-model projections only. The vision tower stays frozen.
LORA_SUFFIXES = ("q_proj", "k_proj", "v_proj", "o_proj", "in_proj_qkv", "in_proj_z", "in_proj_b",
                 "out_proj", "gate_proj", "up_proj", "down_proj")
VISION_PARTS = ("visual", "vision_tower", "vision_model", "vision_encoder")


def load_language_model(model_id, revision, dtype):
    """Qwen3.8-27B is a vision-language checkpoint. Text-only classes fail to load it."""
    from transformers import AutoModelForCausalLM, AutoModelForImageTextToText
    kwargs = {"revision": revision, "torch_dtype": dtype}
    if model_id == "Qwen/Qwen3.8-27B":
        return AutoModelForImageTextToText.from_pretrained(model_id, **kwargs)
    try:
        return AutoModelForCausalLM.from_pretrained(model_id, **kwargs, attn_implementation="sdpa")
    except TypeError:
        return AutoModelForCausalLM.from_pretrained(model_id, **kwargs)


def lora_target_modules(model):
    import torch
    names = []
    for name, module in model.named_modules():
        if not isinstance(module, torch.nn.Linear):
            continue
        if any(part in name for part in VISION_PARTS):
            continue
        if name.endswith(LORA_SUFFIXES):
            names.append(name)
    if not names:
        raise RuntimeError("No language-model Linear layers matched the LoRA target list")
    return names


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--model", default="Qwen/Qwen2.5-Coder-0.5B-Instruct",
                        choices=["Qwen/Qwen2.5-Coder-0.5B-Instruct",
                                 "Qwen/Qwen2.5-Coder-1.5B-Instruct",
                                 "Qwen/Qwen3.8-27B"])
    parser.add_argument("--revision", default="main", help="Use a model commit SHA for final comparisons")
    parser.add_argument("--epochs", type=float, default=3)
    parser.add_argument("--max-length", type=int, default=4096)
    parser.add_argument("--rank", type=int, default=16)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--accumulation", type=int, default=16)
    parser.add_argument("--learning-rate", type=float, default=2e-4)
    args = parser.parse_args()
    import torch
    from peft import LoraConfig, get_peft_model
    from transformers import AutoTokenizer, Trainer, TrainingArguments, set_seed
    from kernel_lab.prompts import build_messages
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA required: submit slurm/train_lora.sbatch")
    if int(os.environ.get("WORLD_SIZE", "1")) != 1 or torch.cuda.device_count() != 1:
        raise RuntimeError("This entrypoint requires exactly one visible GPU for honest resource accounting")
    if args.output.exists() and any(args.output.iterdir()):
        raise ValueError("Output directory must be empty; use a unique run directory")
    args.output.mkdir(parents=True, exist_ok=True)
    started = time.perf_counter()
    set_seed(args.seed)
    summary = {"status": "running", "args": vars(args).copy(), "gpu_count": 1,
               "gpu_name": torch.cuda.get_device_name(0), "slurm_job_id": os.getenv("SLURM_JOB_ID"),
               "python": platform.python_version(), "input_sha256": {},
               "versions": {p: importlib.metadata.version(p) for p in ("torch", "transformers", "peft", "accelerate")}}
    summary["args"] = {k: str(v) if isinstance(v, Path) else v for k, v in summary["args"].items()}
    try:
        tokenizer = AutoTokenizer.from_pretrained(args.model, revision=args.revision)
        tokenizer.pad_token = tokenizer.eos_token
        datasets = {}
        families = {}
        raw_splits = {split: read_jsonl(args.data / f"{split}.jsonl") for split in ("train", "validation")}
        # Recheck verification and split integrity even if callers bypassed preparation.
        prepare_records(raw_splits["train"] + raw_splits["validation"])
        for split in ("train", "validation"):
            path = args.data / f"{split}.jsonl"
            summary["input_sha256"][split] = hashlib.sha256(path.read_bytes()).hexdigest()
            rows = raw_splits[split]
            if not rows:
                raise ValueError(f"Empty {split} data")
            families[split] = {row["family"] for row in rows}
            template_kwargs = {}
            if args.model == "Qwen/Qwen3.8-27B":
                # The checkpoint thinks before answering unless this is set.
                # Training targets are kernel source, so the loss must not cover a thinking trace.
                template_kwargs = {"enable_thinking": False, "preserve_thinking": False}
            datasets[split] = [response_only_example(tokenizer, build_messages(row.get("task", row)),
                                                      row["completion"], args.max_length, template_kwargs) for row in rows]
        if families["train"] & families["validation"]:
            raise ValueError("Training and validation families overlap")
        dtype = torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16
        model = load_language_model(args.model, args.revision, dtype)
        summary["resolved_model_revision"] = getattr(model.config, "_commit_hash", None)
        model.config.use_cache = False
        text_config = getattr(model.config, "text_config", None)
        if text_config is not None:
            text_config.use_cache = False
        targets = lora_target_modules(model)
        summary["lora_target_count"] = len(targets)
        model = get_peft_model(model, LoraConfig(r=args.rank, lora_alpha=2 * args.rank,
                lora_dropout=0.05, target_modules=targets, task_type="CAUSAL_LM"))
        model.enable_input_require_grads()
        summary["parameters"] = {"trainable": sum(p.numel() for p in model.parameters() if p.requires_grad),
                                  "total": sum(p.numel() for p in model.parameters())}
        def collate(rows):
            size = max(len(r["input_ids"]) for r in rows)
            result = {}
            for name, padding in (("input_ids", tokenizer.pad_token_id), ("attention_mask", 0), ("labels", -100)):
                result[name] = torch.tensor([r[name] + [padding] * (size - len(r[name])) for r in rows])
            return result
        training_args = TrainingArguments(output_dir=str(args.output / "checkpoints"),
            num_train_epochs=args.epochs, per_device_train_batch_size=args.batch_size,
            per_device_eval_batch_size=1, gradient_accumulation_steps=args.accumulation,
            learning_rate=args.learning_rate, warmup_steps=0.03, lr_scheduler_type="cosine",
            bf16=dtype == torch.bfloat16, fp16=dtype == torch.float16,
            gradient_checkpointing=True, eval_strategy="epoch", save_strategy="epoch",
            save_total_limit=2, load_best_model_at_end=True, metric_for_best_model="eval_loss",
            logging_steps=1, report_to="none", seed=args.seed, data_seed=args.seed)
        trainer = Trainer(model=model, args=training_args, train_dataset=datasets["train"],
                          eval_dataset=datasets["validation"], data_collator=collate)
        train_started = time.perf_counter()
        result = trainer.train()
        torch.cuda.synchronize()
        summary["training_elapsed_seconds"] = time.perf_counter() - train_started
        summary["training_gpu_hours"] = summary["training_elapsed_seconds"] / 3600
        summary["training_metrics"] = result.metrics
        summary["validation_metrics"] = trainer.evaluate()
        trainer.save_model(str(args.output / "adapter"))
        tokenizer.save_pretrained(args.output / "adapter")
        summary["status"] = "completed"
        summary["peak_allocated_gpu_bytes"] = torch.cuda.max_memory_allocated()
    except BaseException as exc:
        summary["status"] = "failed"
        summary["error"] = f"{type(exc).__name__}: {exc}"
        raise
    finally:
        summary["process_elapsed_seconds"] = time.perf_counter() - started
        summary["process_gpu_hours"] = summary["process_elapsed_seconds"] / 3600
        summary["resource_note"] = "One visible GPU. Process wall time excludes queue wait and job setup; sacct provides full allocated GPU-hours. Not GPU utilization or pretraining cost."
        (args.output / "run_summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
        with (args.output / "pip-freeze.txt").open("w", encoding="utf-8") as stream:
            subprocess.run([os.sys.executable, "-m", "pip", "freeze"], stdout=stream, check=False)


if __name__ == "__main__":
    main()
