import asyncio
import json
from pathlib import Path
import tempfile
import unittest

from kernel_lab.benchmark import benchmark
from kernel_lab.evaluate import evaluate_tasks
from kernel_lab.metrics import percentile, summarize_attempts
from kernel_lab.retrieval import EmbeddingConfig, PgVectorStore, training_record, vector_literal


TASK = {"task_id": "heldout", "family": "heldout_family", "split": "test", "prompt": "add one", "reference_code": "def reference(x): return x+1"}


class FakeTeacher:
    async def complete(self, task, examples=None):
        if task["task_id"] == "api_error":
            raise RuntimeError("endpoint down")
        return {"candidate_code": "pass" if examples else "fail", "prompt_tokens": 10,
                "completion_tokens": 20, "cost_usd": .01, "model": "mock"}


class FakeSandbox:
    async def validate(self, task, code):
        return {"status": "passed" if code == "pass" else "failed"}


class FakeEmbedder:
    async def embed(self, text):
        return [1., 2.]


class FakeStore:
    def __init__(self):
        self.calls = []

    def search(self, task, vector, k, families, ids):
        self.calls.append((families, ids))
        return [{"task_id": "training", "candidate_code": "valid"}]


class RetrievalEvaluationTests(unittest.TestCase):
    def test_percentiles_and_unknown_costs(self):
        self.assertEqual(percentile([1, 3], .5), 2)
        report = summarize_attempts([{"validation": {"status": "passed"}, "teacher": {"cost_usd": 2}},
                                     {"validation": {"status": "failed"}, "teacher": {"cost_usd": None}}])
        self.assertEqual(report["validation_pass_rate"], .5)
        self.assertIsNone(report["api_cost_usd"])
        self.assertIsNone(report["cost_per_verified_pair_usd"])
        self.assertEqual(report["known_api_cost_usd"], 2)
        self.assertIsNone(report["prompt_tokens"])
        self.assertFalse(report["token_usage_complete"])

    def test_cost_per_pair_includes_failed_attempts(self):
        rows = [{"validation": {"status": status}, "teacher": {"cost_usd": 2}} for status in ["passed", "failed"]]
        self.assertEqual(summarize_attempts(rows)["cost_per_verified_pair_usd"], 4)

    def test_retried_calls_keep_known_subtotal_without_claiming_complete_usage(self):
        result = summarize_attempts([{"validation": {"status": "passed"}, "teacher": {
            "cost_usd": None, "known_successful_response_cost_usd": .02,
            "prompt_tokens": 10, "completion_tokens": 20, "api_attempts": 2}}])
        self.assertEqual(result["known_api_cost_usd"], .02)
        self.assertIsNone(result["api_cost_usd"])
        self.assertIsNone(result["prompt_tokens"])
        self.assertEqual(result["known_prompt_tokens"], 10)

    def test_train_verification_gate(self):
        for split, status in [("test", "passed"), ("train", "failed"), ("train", "skipped")]:
            with self.assertRaises(ValueError):
                training_record({"task": dict(TASK, split=split), "candidate_code": "x", "validation": {"status": status}})
        self.assertEqual(training_record({"task": dict(TASK, split="train"), "candidate_code": "x", "validation": {"status": "passed"}})["split"], "train")

    def test_embedding_validation(self):
        for vector in ([0, 0], [float("nan"), 1], [1]):
            with self.assertRaises(ValueError):
                vector_literal(vector, 2)
        self.assertEqual(vector_literal([1, 2], 2), "[1.0,2.0]")

    def test_parameterized_search_exclusions(self):
        class Cursor:
            def fetchall(self): return []
        class Connection:
            def __enter__(self): return self
            def __exit__(self, *args): pass
            def execute(self, sql, parameters):
                self.sql, self.parameters = sql, parameters
                return Cursor()
        connection = Connection()
        store = PgVectorStore("unused", EmbeddingConfig("http://local/v1", "mock", 2))
        store._connect = lambda: connection
        store.search(TASK, [1, 2], 3, ["other_eval"], ["other_id"])
        self.assertIn("ANY(%s::text[])", connection.sql)
        self.assertNotIn("heldout_family", connection.sql)
        self.assertEqual(connection.parameters[1], ["heldout_family", "other_eval"])
        self.assertEqual(connection.parameters[2], ["heldout", "other_id"])

    def test_paired_eval_counts_api_failures_and_excludes_all_eval(self):
        with tempfile.TemporaryDirectory() as directory:
            store = FakeStore()
            tasks = [TASK, dict(TASK, task_id="api_error", family="second_family")]
            result = asyncio.run(evaluate_tasks(tasks, FakeTeacher(), FakeSandbox(), directory,
                                                 store=store, embedder=FakeEmbedder(), workers=2))
            self.assertEqual(result["conditions"]["no_rag"]["attempts"], 2)
            self.assertEqual(result["conditions"]["rag"]["verified"], 1)
            self.assertEqual(result["conditions"]["rag"]["validation_pass_rate"], .5)
            self.assertEqual(result["paired_outcomes"]["rag_only_pass"], 1)
            self.assertEqual(result["paired_outcomes"]["both_fail"], 1)
            self.assertIsNone(result["conditions"]["rag"]["api_cost_usd"])
            self.assertTrue(all("second_family" in families and "api_error" in ids for families, ids in store.calls))
            rows = [json.loads(line) for line in (Path(directory) / "evaluation.jsonl").read_text().splitlines()]
            self.assertEqual(len(rows), 4)
            with self.assertRaises(FileExistsError):
                asyncio.run(evaluate_tasks(tasks, FakeTeacher(), FakeSandbox(), directory, no_rag_only=True))

    def test_retrieval_error_remains_failed_attempt(self):
        class BrokenEmbedder:
            async def embed(self, text): raise RuntimeError("embedding error")
        with tempfile.TemporaryDirectory() as directory:
            result = asyncio.run(evaluate_tasks([TASK], FakeTeacher(), FakeSandbox(), directory,
                                    store=FakeStore(), embedder=BrokenEmbedder()))
            self.assertEqual(result["conditions"]["rag"]["attempts"], 1)
            self.assertEqual(result["conditions"]["rag"]["verified"], 0)

    def test_training_tasks_rejected(self):
        with tempfile.TemporaryDirectory() as directory:
            with self.assertRaises(ValueError):
                asyncio.run(evaluate_tasks([dict(TASK, split="train")], FakeTeacher(), FakeSandbox(), directory, no_rag_only=True))

    def test_benchmark_keeps_errors(self):
        with tempfile.TemporaryDirectory() as directory:
            result = asyncio.run(benchmark([TASK, dict(TASK, task_id="api_error")], FakeTeacher(), directory,
                                           requests=4, workers=2, warmup=0))
            self.assertEqual(result["successful_requests"], 2)
            self.assertEqual(result["failed_requests"], 2)
            self.assertIsNone(result["api_cost_usd"])
            self.assertIsNone(result["ttft_seconds"])
            self.assertGreater(result["aggregate_output_tokens_per_second"], 0)


if __name__ == "__main__":
    unittest.main()
