"""Deterministic expression workload. Variants are not independent operator families."""
import argparse
import json
from pathlib import Path
from .schema import digest

FAMILIES = {
    "affine": "{a} * x + {b}",
    "add_relu": "torch.relu(x + {a} * y + {b})",
    "square_bias": "(x + {a}) ** 2 + {b}",
    "sine_affine": "torch.sin({a} * x) + {b}",
    "cosine_mix": "torch.cos(x) + {a} * y + {b}",
    "sigmoid_affine": "torch.sigmoid({a} * x + {b})",
    "tanh_mix": "torch.tanh(x + {a} * y) + {b}",
    "abs_mix": "torch.abs(x + {a} * y) + {b}",
    "exp_decay": "torch.exp(-torch.abs({a} * x)) + {b}",
    "sqrt_positive": "torch.sqrt(torch.abs(x) + {a}) + {b}",
    "row_sum": "(x * {a} + {b}).sum(dim=-1)",
    "row_mean": "(x * {a} + {b}).mean(dim=-1)",
    "row_max": "(x * {a} + {b}).amax(dim=-1)",
    "row_square_sum": "((x + {a}) ** 2 + {b}).sum(dim=-1)",
    "row_relu_sum": "torch.relu(x * {a} + {b}).sum(dim=-1)",
}


def build_tasks(variants=20):
    if not 1 <= variants <= 400:
        raise ValueError("variants must be 1..400 (finite coefficient grid)")
    tasks = []
    # Whole operation families held out. Coefficient variants never cross splits.
    for index, (family, template) in enumerate(FAMILIES.items()):
        split = "test" if index in {5, 12, 14} else "validation" if index in {3, 8} else "train"
        for variant in range(variants):
            a, b = (variant % 20 + 1) / 8, (variant // 20 - 10) / 8
            expression = template.format(a=a, b=b)
            names = ["x", "y"] if "y" in expression else ["x"]
            reference = f"import torch\n\ndef reference({', '.join(names)}):\n    return {expression}\n"
            shapes = [[1], [31], [257], [4097]] if not family.startswith("row_") else [[1, 1], [3, 31], [7, 257], [2, 1025]]
            cases = [{"shape": shape, "seed": seed, "distribution": distribution} for shape, seed, distribution in zip(shapes, [17, 29, 43, 71], ["normal", "zeros", "uniform", "normal"])]
            task = {"schema_version": 1, "task_id": digest(reference)[:20], "family": family, "split": split,
                    "prompt": "Implement this complete PyTorch reference using Triton for contiguous CUDA float32 tensors. Inputs have identical shapes. Support arbitrary positive dimensions in the tested rank; return a CUDA float32 tensor with exactly the reference shape.",
                    "reference_code": reference, "input_specs": [{"name": name, "dtype": "float32"} for name in names], "test_cases": cases, "atol": 1e-4, "rtol": 1e-4}
            tasks.append(task)
    return tasks


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", required=True)
    parser.add_argument("--variants", type=int, default=20)
    args = parser.parse_args()
    tasks = build_tasks(args.variants)
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text("".join(json.dumps(t) + "\n" for t in tasks), encoding="utf-8")
    print(json.dumps({"tasks": len(tasks), "families": len(FAMILIES), "warning": "Coefficient variants are correlated; these are NOT thousands of unique operations."}))


if __name__ == "__main__":
    main()
