import unittest
from training.data import prepare_records, response_only_example


def record(family="elementwise", split="train", code="def kernel(x): return x", reference="return x"):
    return {"task": {"task_id": family, "family": family, "split": split,
                     "prompt": "Implement identity", "reference_code": reference},
            "candidate_code": code, "validation": {"status": "passed"}}


class Tokenizer:
    def apply_chat_template(self, messages, tokenize, add_generation_prompt):
        prefix = [1, 2, 3]
        return prefix if add_generation_prompt else prefix + [4, 5, 6]


class TrainingDataTests(unittest.TestCase):
    def test_unverified_rejected(self):
        row = record()
        row["validation"]["status"] = "failed"
        with self.assertRaises(ValueError):
            prepare_records([row])

    def test_family_leakage_rejected(self):
        with self.assertRaisesRegex(ValueError, "Family leakage"):
            prepare_records([record(), record(split="test")])

    def test_reference_leakage_rejected(self):
        with self.assertRaisesRegex(ValueError, "reference code"):
            prepare_records([record(), record(family="other", split="test")])

    def test_duplicates_removed(self):
        self.assertEqual(len(prepare_records([record(), record()])["train"]), 1)

    def test_response_only_mask(self):
        example = response_only_example(Tokenizer(), [{"role": "user", "content": "test"}], "code", 8)
        self.assertEqual(example["labels"], [-100, -100, -100, 4, 5, 6])

    def test_overlong_rejected_instead_of_truncated(self):
        with self.assertRaisesRegex(ValueError, "exceeds"):
            response_only_example(Tokenizer(), [], "code", 3)


if __name__ == "__main__":
    unittest.main()
