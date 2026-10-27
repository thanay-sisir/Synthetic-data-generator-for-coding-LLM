"""One prompt contract for teacher, training, and model evaluation."""

SYSTEM_PROMPT = """You write complete, numerically correct Triton implementations.
Return ONLY a complete Python module, without Markdown or explanations.
Use import torch, import triton, import triton.language as tl.
This Triton 3.2 build has no tl.tanh, tl.math.tanh, tl.sigmoid, or tl.math.sigmoid. Implement tanh and sigmoid from tl.exp.
Define one or more @triton.jit device kernels and def kernel(...), with the same positional inputs as reference(...).
The wrapper must launch Triton and return a newly allocated CUDA float32 tensor of the reference shape.
Use torch ONLY to allocate outputs (torch.empty/empty_like/zeros/zeros_like) and for torch.float32 annotations.
Outside @triton.jit, use only allocations, tensor metadata (shape/ndim/device/dtype/numel/size/stride/is_contiguous), scalar shape arithmetic, triton.cdiv/next_power_of_2, and Triton launches.
No host tensor arithmetic, indexing, reshape/view, activation/reduction methods, loops, or helper functions. A scalar metadata lambda may define a launch grid.
The kernel wrapper must have plain positional tensor arguments without defaults and return fresh output storage, never an input or view of one.
No torch arithmetic, CPU transfers, file access, network, print, reflection, mutation of inputs, environment inspection, or test code.
Do not assume a fixed tensor size. Mask out-of-bounds accesses. Handle non-power-of-two dimensions.
Only contiguous float32 tensors and the ranks specified in the task are required."""


def build_messages(task, examples=None):
    messages = [{"role": "system", "content": SYSTEM_PROMPT}]
    for example in examples or []:
        ex_task = example.get("task", example)
        messages += [{"role": "user", "content": task_text(ex_task)},
                     {"role": "assistant", "content": example["candidate_code"]}]
    messages.append({"role": "user", "content": task_text(task)})
    return messages


def task_text(task):
    specs = task.get("input_specs", [])
    shapes = [case["shape"] for case in task.get("test_cases", [])]
    return f"{task.get('prompt', '')}\nInput contract: {specs}\nExample shapes: {shapes}\nPyTorch reference:\n{task.get('reference_code', '')}"
