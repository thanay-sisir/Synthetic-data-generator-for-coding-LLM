"""Trusted numerical checker. Inputs/code are mounted read-only; CUDA is mandatory.

Intended for cooperative generated code after static screening, NOT adversarial
programs. Candidate shares a Python runtime and GPU driver with this checker.
"""
import contextlib
import io
import json
import sys
import traceback


def run(task, code):
    import torch
    import triton
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is unavailable; this is a failure, never a pass or skip")
    reference_namespace = {}
    exec(task["reference_code"], reference_namespace)
    # Restricted builtins are defense in depth; this is not a Python security boundary.
    safe_names = ("__build_class__", "__import__", "range", "len", "int", "float", "bool", "tuple", "list", "dict", "min", "max", "abs", "sum", "enumerate", "zip", "isinstance", "ValueError", "RuntimeError", "Exception")
    import builtins
    namespace = {"__builtins__": {name: getattr(builtins, name) for name in safe_names}, "__name__": "candidate"}
    # A real module file lets Triton inspect source, essential for @triton.jit.
    # Source is screened again by the host; the file is a read-only mount.
    import importlib.util
    spec = importlib.util.spec_from_file_location("candidate", sys.argv[2])
    module = importlib.util.module_from_spec(spec)
    module.__dict__.update(namespace)
    sys.modules["candidate"] = module
    spec.loader.exec_module(module)
    cases = list(task["test_cases"])
    rank = len(cases[0]["shape"])
    # Additional checker cases are omitted from the teacher prompt.
    cases += [{"shape": [509] if rank == 1 else [5, 509], "seed": 101, "distribution": "uniform"},
              {"shape": [1023] if rank == 1 else [4, 63], "seed": 131, "distribution": "normal"}]
    passed = 0
    for case in cases:
        torch.manual_seed(case["seed"])
        inputs = []
        for _ in task["input_specs"]:
            shape = case["shape"]
            if case["distribution"] == "zeros":
                value = torch.zeros(shape, device="cuda", dtype=torch.float32)
            elif case["distribution"] == "uniform":
                value = torch.rand(shape, device="cuda", dtype=torch.float32) * 8 - 4
            else:
                value = torch.randn(shape, device="cuda", dtype=torch.float32)
            inputs.append(value)
        originals = [value.clone() for value in inputs]
        expected = reference_namespace["reference"](*originals)
        torch.cuda.synchronize()
        actual = module.kernel(*inputs)
        torch.cuda.synchronize()
        if not isinstance(actual, torch.Tensor) or not actual.is_cuda or actual.dtype != torch.float32:
            raise AssertionError("Output must be a CUDA float32 Tensor")
        if actual.shape != expected.shape:
            raise AssertionError(f"Wrong shape: {actual.shape} != {expected.shape}")
        if any(actual.untyped_storage().data_ptr() == value.untyped_storage().data_ptr() for value in inputs):
            raise AssertionError("Output must have fresh storage and cannot alias any input")
        if not torch.isfinite(expected).all() or not torch.isfinite(actual).all():
            raise AssertionError("Nonfinite output")
        torch.testing.assert_close(actual, expected, atol=task["atol"], rtol=task["rtol"], equal_nan=False)
        for before, after in zip(originals, inputs):
            torch.testing.assert_close(after, before, atol=0, rtol=0)
        passed += 1
    return {"status": "passed", "cases_passed": passed, "gpu": torch.cuda.get_device_name(), "torch_version": torch.__version__, "triton_version": triton.__version__, "cuda_version": torch.version.cuda,
            "scope": "numerical agreement on finite deterministic float32 cases; not formal correctness or a speed benchmark"}


if __name__ == "__main__":
    try:
        with open(sys.argv[1], encoding="utf-8") as handle:
            task = json.load(handle)
        with open(sys.argv[2], encoding="utf-8") as handle:
            code = handle.read()
        with contextlib.redirect_stdout(io.StringIO()):
            result = run(task, code)
        print(json.dumps(result))
    except BaseException as error:
        print(json.dumps({"status": "failed", "error": f"{type(error).__name__}: {str(error)[:2000]}"}))
        traceback.print_exc(file=sys.stderr)
        sys.exit(1)
