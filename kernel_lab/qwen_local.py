"""On-GPU Qwen generation for held-out checks. Not a vLLM server.

Weights stay on cuda:0. A 27B check needs a second GPU for the checker.
Qwen2.5-Coder is small enough to share one A100 with the checker.
Qwen3.8 thinks by default, so that model is the only one that turns thinking off.
"""
import asyncio
import time


class QwenLocalTeacher:
    def __init__(self, model_id, adapter=None, max_new_tokens=1024):
        import torch
        from transformers import AutoModelForCausalLM, AutoModelForImageTextToText, AutoTokenizer

        if max_new_tokens < 1:
            raise ValueError("max_new_tokens must be positive")
        if not torch.cuda.is_available():
            raise RuntimeError("Qwen held-out generation needs a visible GPU")
        self.torch = torch
        self.model_id = model_id
        self.adapter = adapter or None
        self.max_new_tokens = max_new_tokens
        self.lock = asyncio.Lock()
        self.config = None
        # Qwen2.5-Coder has no thinking switch. Passing those kwargs makes its template fail.
        self.template_kwargs = {}
        if model_id == "Qwen/Qwen3.8-27B":
            self.template_kwargs = {"enable_thinking": False, "preserve_thinking": False}
        dtype = torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16
        self.tokenizer = AutoTokenizer.from_pretrained(model_id)
        if self.tokenizer.pad_token_id is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
        loader = AutoModelForImageTextToText if model_id == "Qwen/Qwen3.8-27B" else AutoModelForCausalLM
        model = loader.from_pretrained(
            model_id, torch_dtype=dtype, device_map={"": 0}, low_cpu_mem_usage=True)
        if self.adapter:
            from peft import PeftModel
            model = PeftModel.from_pretrained(model, self.adapter)
        model.eval()
        model.config.use_cache = True
        text_config = getattr(model.config, "text_config", None)
        if text_config is not None:
            text_config.use_cache = True
        self.model = model

    async def complete(self, task, examples=None):
        async with self.lock:
            return await asyncio.to_thread(self._complete, task, examples or [])

    def _complete(self, task, examples):
        from .prompts import build_messages
        from .teacher import extract_code

        started = time.perf_counter()
        encoded = self.tokenizer.apply_chat_template(
            build_messages(task, examples), tokenize=True, add_generation_prompt=True,
            return_tensors="pt", **self.template_kwargs)
        if hasattr(encoded, "keys"):
            batch = encoded
        else:
            batch = {"input_ids": encoded}
        inputs = {"input_ids": batch["input_ids"].to(self.model.device)}
        mask = batch.get("attention_mask") if hasattr(batch, "get") else None
        if hasattr(mask, "to"):
            inputs["attention_mask"] = mask.to(self.model.device)
        prompt_len = inputs["input_ids"].shape[1]
        with self.torch.inference_mode():
            output = self.model.generate(
                **inputs, max_new_tokens=self.max_new_tokens, do_sample=False,
                pad_token_id=self.tokenizer.pad_token_id,
                eos_token_id=self.tokenizer.eos_token_id)
        new_ids = output[0, prompt_len:]
        text = self.tokenizer.decode(new_ids, skip_special_tokens=True)
        try:
            code = extract_code(text)
            parse_error = None
        except (ValueError, KeyError, IndexError, TypeError) as error:
            code, parse_error = "", f"Malformed candidate response ({type(error).__name__})"
        return {"candidate_code": code, "model": self.model_id + ("+lora" if self.adapter else ""),
                "prompt_tokens": prompt_len, "completion_tokens": int(new_ids.shape[0]),
                "cost_usd": None, "parse_error": parse_error,
                "latency_seconds": time.perf_counter() - started, "api_attempts": 1}
