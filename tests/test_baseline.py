import ast
from pathlib import Path
import unittest
import math
from types import SimpleNamespace
from kernel_lab.baseline import check_splits


class SplitTests(unittest.TestCase):
    def test_rejects_family_and_reference_leakage(self):
        train = [{"family": "a", "reference_code": "return x"}]
        with self.assertRaises(ValueError):
            check_splits(train, [{"family": "a", "reference_code": "return y"}])
        with self.assertRaises(ValueError):
            check_splits(train, [{"family": "b", "reference_code": "return x"}])
        check_splits(train, [{"family": "b", "reference_code": "return y"}])

    def test_rotary_varies_by_position_not_head(self):
        try:
            import torch
            import torch.nn as nn
        except ImportError:
            self.skipTest("PyTorch optional")
        # Test the real class without importing tokenizer/training dependencies.
        tree = ast.parse(Path("llm.py").read_text(encoding="utf-8"))
        rotary = next(node for node in tree.body if isinstance(node, ast.ClassDef) and node.name == "Rotary")
        scope = {"torch": torch, "nn": nn}
        exec(compile(ast.Module(body=[rotary], type_ignores=[]), "llm.py", "exec"), scope)
        module = scope["Rotary"](8, 32)
        x = torch.ones(1, 2, 5, 8)
        result = module(x)
        self.assertTrue(torch.equal(result[:, 0], result[:, 1]))
        self.assertFalse(torch.equal(result[:, :, 0], result[:, :, 1]))
        with self.assertRaises(ValueError):
            module(torch.ones(1, 2, 33, 8))

    def test_corrected_model_forward_is_causal(self):
        try:
            import torch
            import torch.nn as nn
            import torch.nn.functional as functional
        except ImportError:
            self.skipTest("PyTorch optional")
        tree = ast.parse(Path("llm.py").read_text(encoding="utf-8"))
        names = {"Rotary", "MultiHeadAttention", "FeedForward", "TransformerBlock", "MinimalLLM"}
        classes = [node for node in tree.body if isinstance(node, ast.ClassDef) and node.name in names]
        scope = {"torch": torch, "nn": nn, "F": functional, "math": math, "ModelConfig": SimpleNamespace}
        exec(compile(ast.Module(body=classes, type_ignores=[]), "llm.py", "exec"), scope)
        config = SimpleNamespace(d_model=16, n_heads=2, d_ff=32, max_seq_len=32,
                                 dropout=0.0, vocab_size=31, n_layers=2)
        model = scope["MinimalLLM"](config).eval()
        tokens = torch.tensor([[1, 2, 3, 4, 5]])
        changed = torch.tensor([[1, 2, 3, 4, 6]])
        with torch.no_grad():
            original, altered = model(tokens), model(changed)
        self.assertEqual(tuple(original.shape), (1, 5, 31))
        self.assertTrue(torch.isfinite(original).all())
        self.assertTrue(torch.allclose(original[:, :-1], altered[:, :-1], atol=1e-6))
        self.assertIs(model.lm_head.weight, model.token_embedding.weight)


if __name__ == "__main__":
    unittest.main()
