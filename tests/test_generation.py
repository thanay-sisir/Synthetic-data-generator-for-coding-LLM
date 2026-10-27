import asyncio
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch
from kernel_lab.generate import run_generation
from kernel_lab.prompts import build_messages
from kernel_lab.sandbox import DockerSandbox, _read_limited
from kernel_lab.schema import screen_candidate, validate_task
from kernel_lab.tasks import build_tasks
from kernel_lab.teacher import extract_code, TeacherClient, TeacherConfig

CODE = '''import torch
import triton
import triton.language as tl
@triton.jit
def device_kernel(X, Y, N: tl.constexpr, B: tl.constexpr):
    i = tl.program_id(0) * B + tl.arange(0, B)
    value = tl.load(X + i, i < N, 0)
    tl.store(Y + i, value, i < N)
def kernel(x):
    y = torch.empty_like(x)
    device_kernel[(triton.cdiv(x.numel(), 256),)](x, y, x.numel(), 256)
    return y
'''


class GenerationTests(unittest.TestCase):
    def test_openrouter_free_only_rejects_paid_and_auto_models(self):
        for model in ("cohere/north-mini-code", "openrouter/auto", "openrouter/free"):
            with self.subTest(model=model), self.assertRaises(ValueError):
                TeacherClient(TeacherConfig(model=model))
        # A local vLLM model name remains valid and receives no OpenRouter fields.
        TeacherClient(TeacherConfig(base_url="http://127.0.0.1:8000/v1", model="kernel-lora"))

    def test_openrouter_request_caps_provider_prices_at_zero(self):
        import httpx
        sent = []
        class Client:
            def __init__(self, **kwargs):
                pass
            async def __aenter__(self):
                return self
            async def __aexit__(self, *args):
                pass
            async def post(self, url, **kwargs):
                sent.append(kwargs["json"])
                return httpx.Response(200, request=httpx.Request("POST", url), json={
                    "choices": [{"message": {"content": CODE}, "finish_reason": "stop"}],
                    "usage": {"prompt_tokens": 100, "completion_tokens": 20}})
        config = TeacherConfig(input_price_per_million=0, output_price_per_million=0)
        with patch("httpx.AsyncClient", Client):
            result = asyncio.run(TeacherClient(config).complete(build_tasks(1)[0]))
        self.assertEqual(sent[0]["model"], "cohere/north-mini-code:free")
        self.assertEqual(sent[0]["provider"]["max_price"], {"prompt": 0, "completion": 0})
        self.assertNotIn("models", sent[0])
        self.assertEqual(result["cost_usd"], 0)

    def test_openrouter_pacing_is_shared_between_workers(self):
        async def run():
            client = TeacherClient(TeacherConfig(request_interval=4))
            clock = [0.0]
            waits = []
            async def advance(delay):
                waits.append(delay)
                clock[0] += delay
            with patch("kernel_lab.teacher.time.monotonic", lambda: clock[0]), patch("kernel_lab.teacher.asyncio.sleep", advance):
                await asyncio.gather(*(client._pace() for _ in range(3)))
            self.assertEqual(waits, [4, 4])
        asyncio.run(run())

    def test_tasks_have_family_disjoint_splits_and_unique_ids(self):
        tasks = build_tasks(25)
        self.assertEqual(len(tasks), 375)
        self.assertEqual(len({t["task_id"] for t in tasks}), len(tasks))
        for family in {t["family"] for t in tasks}:
            self.assertEqual(len({t["split"] for t in tasks if t["family"] == family}), 1)
        for task in tasks:
            validate_task(task)

    def test_prompt_contains_actual_reference(self):
        task = build_tasks(1)[0]
        self.assertIn(task["reference_code"], build_messages(task)[-1]["content"])

    def test_screen_allows_real_triton_load_and_rejects_host_access(self):
        screen_candidate(CODE)
        with self.assertRaises(ValueError):
            screen_candidate(CODE + "\nimport os\n")
        with self.assertRaises(ValueError):
            screen_candidate(CODE.replace("torch.empty_like(x)", "torch.sin(x)"))

    def test_dummy_launch_cannot_hide_pytorch_compute(self):
        for returned in ["x * 0.125 - 1.25", "x.sin()", "x.sum()", "x[0]", "-x", "x > 0", "x.to(torch.float32)", "(x,)[0]"]:
            with self.subTest(returned=returned), self.assertRaises(ValueError):
                screen_candidate(CODE.replace("return y", "return " + returned))
        # Aliases and newly allocated tensors retain their tensor taint.
        for extra in ["alias = x\n    return alias + alias", "alias = y\n    return alias.sin()", "return y + 1"]:
            with self.subTest(extra=extra), self.assertRaises(ValueError):
                screen_candidate(CODE.replace("return y", extra))

    def test_screen_allows_reduction_metadata_and_multiple_launches(self):
        module = CODE[:CODE.index("def kernel(x):")] + '''def kernel(x):
    rows, cols = x.shape
    block = triton.next_power_of_2(cols)
    temporary = torch.empty((rows, cols), device=x.device, dtype=x.dtype)
    result = torch.empty((rows,), device=x.device, dtype=torch.float32)
    grid = lambda meta: (triton.cdiv(rows * cols, meta["B"]),)
    device_kernel[grid](x, temporary, rows * cols, B=block)
    device_kernel[(rows,)](temporary, result, cols, B=block)
    return result
'''
        screen_candidate(module)

    def test_extract_raw_fenced_and_json(self):
        self.assertEqual(extract_code("```python\n" + CODE + "```"), CODE)
        self.assertEqual(extract_code(json.dumps({"candidate_code": CODE})), CODE)

    def test_docker_limits_and_apptainer_gate(self):
        command = DockerSandbox().command(".", "test")
        for flag in ["--network", "--read-only", "--cap-drop", "--memory", "--pids-limit", "--user"]:
            self.assertIn(flag, command)
        with self.assertRaises(ValueError):
            DockerSandbox(backend="apptainer")

    def test_ledger_retains_failures_unknown_cost_and_verified_only(self):
        class Teacher:
            async def complete(self, task, examples=None):
                if task["family"] == "affine":
                    raise RuntimeError("mock API failure")
                return {"candidate_code": CODE, "cost_usd": None}
        class Sandbox:
            async def validate(self, task, code):
                return {"status": "passed", "cases_passed": 6}
        with tempfile.TemporaryDirectory() as directory:
            tasks = build_tasks(1)[:2]
            summary = asyncio.run(run_generation(tasks, Teacher(), Sandbox(), directory, workers=2))
            self.assertEqual(summary["attempts"], 2)
            self.assertEqual(summary["verified"], 1)
            self.assertIsNone(summary["response_token_cost_per_verified_pair_usd"])
            self.assertEqual(len(Path(directory, "attempts.jsonl").read_text().splitlines()), 2)
            rows = [json.loads(line) for line in Path(directory, "verified.jsonl").read_text().splitlines()]
            self.assertEqual(rows[0]["validation"]["status"], "passed")
            with self.assertRaises(ValueError):
                asyncio.run(run_generation(tasks, Teacher(), Sandbox(), directory))

    def test_failed_static_screen_does_not_launch_process(self):
        result = asyncio.run(DockerSandbox().validate(build_tasks(1)[0], "import os\n"))
        self.assertEqual(result["stage"], "static_screen")
        self.assertEqual(result["status"], "failed")

    def test_teacher_parse_failure_preserves_paid_token_usage(self):
        import httpx
        class Client:
            def __init__(self, **kwargs):
                pass
            async def __aenter__(self):
                return self
            async def __aexit__(self, *args):
                pass
            async def post(self, *args, **kwargs):
                return httpx.Response(200, request=httpx.Request("POST", "http://local"), json={
                    "choices": [{"message": {"content": '{"invalid":"format"}'}, "finish_reason": "stop"}],
                    "usage": {"prompt_tokens": 100, "completion_tokens": 20}})
        config = TeacherConfig(input_price_per_million=1, output_price_per_million=2)
        with patch("httpx.AsyncClient", Client):
            result = asyncio.run(TeacherClient(config).complete(build_tasks(1)[0]))
        self.assertEqual(result["candidate_code"], "")
        self.assertTrue(result["parse_error"])
        self.assertEqual(result["prompt_tokens"], 100)
        self.assertAlmostEqual(result["cost_usd"], 0.00014)

    def test_teacher_retry_cost_is_unknown_not_success_response_only(self):
        import httpx
        class Client:
            count = 0
            def __init__(self, **kwargs):
                pass
            async def __aenter__(self):
                return self
            async def __aexit__(self, *args):
                pass
            async def post(self, *args, **kwargs):
                Client.count += 1
                if Client.count == 1:
                    raise httpx.ReadTimeout("unknown billing")
                return httpx.Response(200, request=httpx.Request("POST", "http://local"), json={
                    "choices": [{"message": {"content": CODE}, "finish_reason": "stop"}],
                    "usage": {"prompt_tokens": 100, "completion_tokens": 20}})
        async def no_delay(*args):
            pass
        config = TeacherConfig(input_price_per_million=1, output_price_per_million=2)
        with patch("httpx.AsyncClient", Client), patch("kernel_lab.teacher.asyncio.sleep", no_delay):
            result = asyncio.run(TeacherClient(config).complete(build_tasks(1)[0]))
        self.assertEqual(result["api_attempts"], 2)
        self.assertIsNone(result["cost_usd"])
        self.assertAlmostEqual(result["known_successful_response_cost_usd"], 0.00014)

    def test_output_limit(self):
        async def run():
            stream = asyncio.StreamReader()
            stream.feed_data(b"12345")
            stream.feed_eof()
            with self.assertRaises(ValueError):
                await _read_limited(stream, limit=4)
        asyncio.run(run())

    def test_validator_never_accepts_skips_nonzero_or_missing_cases(self):
        async def check(payload, exit_code):
            class Process:
                returncode = exit_code
                def __init__(self):
                    self.stdout, self.stderr = asyncio.StreamReader(), asyncio.StreamReader()
                    self.stdout.feed_data(json.dumps(payload).encode())
                    self.stdout.feed_eof()
                    self.stderr.feed_eof()
                async def wait(self):
                    return exit_code
            async def spawn(*args, **kwargs):
                return Process()
            with patch("kernel_lab.sandbox.asyncio.create_subprocess_exec", spawn):
                return await DockerSandbox().validate(build_tasks(1)[0], CODE)
        for payload, code in [({"status": "skipped", "cases_passed": 6}, 0), ({"status": "passed", "cases_passed": 6}, 1), ({"status": "passed", "cases_passed": 3}, 0)]:
            self.assertNotEqual(asyncio.run(check(payload, code))["status"], "passed")
        self.assertEqual(asyncio.run(check({"status": "passed", "cases_passed": 6}, 0))["status"], "passed")


if __name__ == "__main__":
    unittest.main()
