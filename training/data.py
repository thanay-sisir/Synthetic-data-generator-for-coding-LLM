"""Dependency-free verified-record checks and response-only tokenization."""
import hashlib
import json
from pathlib import Path


def read_jsonl(path):
    with Path(path).open(encoding="utf-8") as stream:
        return [json.loads(line) for line in stream if line.strip()]


def prepare_records(records):
    groups = {"train": [], "validation": [], "test": []}
    family_splits, reference_splits, seen = {}, {}, set()
    for row in records:
        task = row.get("task", row)
        if row.get("validation", {}).get("status") != "passed":
            raise ValueError("Only records with validation.status=passed can enter training")
        split = {"val": "validation"}.get(task["split"], task["split"])
        if split not in groups:
            raise ValueError(f"Unknown split: {split}")
        family = task["family"]
        if family in family_splits and family_splits[family] != split:
            raise ValueError(f"Family leakage: {family}")
        family_splits[family] = split
        reference = hashlib.sha256(task["reference_code"].strip().encode()).hexdigest()
        if reference in reference_splits and reference_splits[reference] != split:
            raise ValueError("Identical reference code appears across splits")
        reference_splits[reference] = split
        code = row["candidate_code"].strip()
        if not code:
            raise ValueError("Empty candidate")
        # Keep distinct candidates for the same task, but never repeated pairs.
        key = (task["task_id"], code)
        if key in seen:
            continue
        seen.add(key)
        groups[split].append({**row, "id": row.get("candidate_id", task["task_id"]),
                              "family": family, "split": split,
                              "prompt": task["prompt"], "completion": code})
    return groups


def _token_ids(value):
    if hasattr(value, "input_ids"):
        value = value.input_ids
    if hasattr(value, "tolist"):
        value = value.tolist()
    if value and isinstance(value[0], list):
        value = value[0]
    return list(value)


def _response_start(tokenizer, messages, completion, extra):
    """Cut where the rendered prompt ends, using character offsets.

    Qwen3.8 closes an empty thinking block in the generation prompt. Those
    characters are a string prefix of the full example, but byte-pair encoding
    can merge the boundary, so the token ids of the prompt alone are not a
    prefix of the token ids of the whole example.
    """
    prefix_text = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True, **extra)
    full_text = tokenizer.apply_chat_template(messages + [{"role": "assistant", "content": completion}],
                                              tokenize=False, add_generation_prompt=False, **extra)
    if not full_text.startswith(prefix_text):
        raise ValueError("Chat template prefix mismatch; cannot reliably mask the response")
    encoded = tokenizer(full_text, add_special_tokens=False, return_offsets_mapping=True)
    ids = _token_ids(encoded["input_ids"])
    offsets = encoded["offset_mapping"]
    if hasattr(offsets, "tolist"):
        offsets = offsets.tolist()
    if offsets and isinstance(offsets[0], list) and offsets[0] and isinstance(offsets[0][0], list):
        offsets = offsets[0]
    boundary = len(prefix_text)
    cut = None
    for index, (start, end) in enumerate(offsets):
        if start >= boundary:
            cut = index
            break
        if start < boundary < end:
            # This token contains both prompt and response characters. Keep it
            # out of the loss so the prompt is never trained on.
            cut = index + 1
            break
    if cut is None or cut >= len(ids):
        raise ValueError("No response tokens")
    return ids, cut


def response_only_example(tokenizer, messages, completion, max_length, template_kwargs=None):
    """Tokenize chat prefix separately; never train on user/system/padding tokens.

    Refuse overlong examples instead of silently discarding the target tail.
    The boundary prefix check catches chat templates that cannot use this scheme.
    template_kwargs is how Qwen3.8 turns thinking off. An empty mapping keeps
    older tokenizers, whose apply_chat_template does not accept those options.
    """
    extra = template_kwargs or {}
    prefix = _token_ids(tokenizer.apply_chat_template(messages, tokenize=True, add_generation_prompt=True, **extra))
    full = _token_ids(tokenizer.apply_chat_template(
        messages + [{"role": "assistant", "content": completion}],
        tokenize=True, add_generation_prompt=False, **extra))
    if full[:len(prefix)] == prefix:
        ids, cut = full, len(prefix)
    else:
        ids, cut = _response_start(tokenizer, messages, completion, extra)
    if len(ids) > max_length:
        raise ValueError(f"Example has {len(ids)} tokens, exceeds max_length={max_length}")
    if cut >= len(ids):
        raise ValueError("No response tokens")
    return {"input_ids": ids, "attention_mask": [1] * len(ids),
            "labels": [-100] * cut + ids[cut:]}
