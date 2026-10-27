"""Validate a known affine Triton program before spending money on a teacher API."""
import argparse
import asyncio
import json
from pathlib import Path
from .sandbox import DockerSandbox
from .tasks import build_tasks

SMOKE_CODE = '''import torch
import triton
import triton.language as tl

@triton.jit
def affine_kernel(X, Y, N: tl.constexpr, BLOCK: tl.constexpr):
    offsets = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
    values = tl.load(X + offsets, mask=offsets < N, other=0.0)
    tl.store(Y + offsets, values * 0.125 - 1.25, mask=offsets < N)

def kernel(x):
    out = torch.empty_like(x)
    n = x.numel()
    affine_kernel[(triton.cdiv(n, 256),)](x, out, n, BLOCK=256)
    return out
'''


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--image", default="kernel-lab-validator:latest")
    parser.add_argument("--backend", choices=["docker", "apptainer"], default="docker")
    parser.add_argument("--allow-weaker-isolation", action="store_true")
    parser.add_argument("--timeout", type=float, default=120)
    parser.add_argument("--gpu-device", default="0")
    parser.add_argument("--output")
    args = parser.parse_args()
    task = build_tasks(1)[0]
    sandbox = DockerSandbox(image=args.image, timeout=args.timeout, gpu_device=args.gpu_device, backend=args.backend, allow_weaker_isolation=args.allow_weaker_isolation)
    result = asyncio.run(sandbox.validate(task, SMOKE_CODE))
    result["smoke_test"] = "affine_float32"
    rendered = json.dumps(result, indent=2)
    if args.output:
        path = Path(args.output)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(rendered, encoding="utf-8")
    print(rendered)
    if result["status"] != "passed":
        raise SystemExit(1)


if __name__ == "__main__":
    main()
