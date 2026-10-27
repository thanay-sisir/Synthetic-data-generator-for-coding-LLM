"""Execute reviewed generated code in bounded GPU containers, never on the host.

Docker reduces accidental damage; sharing a GPU driver is not a hostile-code
security boundary. Apptainer is explicitly weaker and requires opt-in.
"""
import asyncio
import json
import os
import signal
from pathlib import Path
import tempfile
import time
import uuid
from .schema import screen_candidate, validate_task


async def _read_limited(stream, limit=1_048_576):
    chunks, size = [], 0
    while True:
        chunk = await stream.read(65536)
        if not chunk:
            return b"".join(chunks)
        size += len(chunk)
        if size > limit:
            raise ValueError("Container output exceeded 1 MiB limit")
        chunks.append(chunk)


class DockerSandbox:
    def __init__(self, image="kernel-lab-validator:latest", timeout=120, gpu_device="0", memory="4g", cpus=2, backend="docker", allow_weaker_isolation=False):
        if backend not in {"docker", "apptainer"}:
            raise ValueError("backend must be docker or apptainer")
        if backend == "apptainer" and not allow_weaker_isolation:
            raise ValueError("Apptainer needs allow_weaker_isolation=True and site approval; it is not equivalent to Docker isolation")
        if timeout <= 0 or cpus <= 0:
            raise ValueError("Positive limits required")
        self.image, self.timeout, self.gpu_device = image, timeout, gpu_device
        self.memory, self.cpus, self.backend = memory, cpus, backend

    def command(self, directory, name):
        directory = str(Path(directory).resolve())
        if self.backend == "docker":
            return ["docker", "run", "--name", name, "--rm", "--gpus", f"device={self.gpu_device}",
                    "--network", "none", "--read-only", "--cap-drop", "ALL", "--security-opt", "no-new-privileges",
                    "--pids-limit", "128", "--memory", self.memory, "--cpus", str(self.cpus),
                    "--user", "65534:65534", "--tmpfs", "/tmp:rw,nosuid,nodev,size=1g", "--shm-size", "256m",
                    "--mount", f"type=bind,source={directory},target=/work,readonly", self.image,
                    "python", "/opt/runner.py", "/work/task.json", "/work/candidate.py"]
        # KERNEL_LAB_SANDBOX_GPU pins the checker to one device when the parent
        # process can see several. A 27B eval keeps the model on GPU 0 and the
        # checker on GPU 1. Unset, the parent Slurm device list is unchanged.
        sandbox_gpu = os.environ.get("KERNEL_LAB_SANDBOX_GPU") or os.environ.get("CUDA_VISIBLE_DEVICES", self.gpu_device)
        return ["apptainer", "exec", "--nv", "--containall", "--cleanenv", "--no-home", "--pwd", "/tmp",
                "--env", "CUDA_VISIBLE_DEVICES=" + sandbox_gpu,
                "--env", "TRITON_CACHE_DIR=/tmp/triton-cache",
                "--bind", f"{directory}:/work:ro", self.image,
                "python", "/opt/runner.py", "/work/task.json", "/work/candidate.py"]

    async def validate(self, task, code):
        started = time.perf_counter()
        try:
            validate_task(task)
            screen_candidate(code)
        except (ValueError, SyntaxError) as error:
            return {"status": "failed", "stage": "static_screen", "error": str(error), "elapsed_seconds": time.perf_counter() - started}
        name = "kernel-lab-" + uuid.uuid4().hex
        with tempfile.TemporaryDirectory(prefix="kernel-lab-") as directory:
            root = Path(directory)
            root.chmod(0o755)
            (root / "task.json").write_text(json.dumps(task), encoding="utf-8")
            (root / "candidate.py").write_text(code, encoding="utf-8")
            for path in root.iterdir():
                path.chmod(0o644)
            try:
                process = await asyncio.create_subprocess_exec(*self.command(root, name), stdout=asyncio.subprocess.PIPE, stderr=asyncio.subprocess.PIPE,
                    start_new_session=(os.name == "posix"))
            except (OSError, ValueError) as error:
                return {"status": "error", "stage": "container_start", "error": str(error), "elapsed_seconds": time.perf_counter() - started}
            try:
                reader_tasks = [asyncio.create_task(_read_limited(process.stdout)), asyncio.create_task(_read_limited(process.stderr)), asyncio.create_task(process.wait())]
                readers = asyncio.gather(*reader_tasks)
                stdout, stderr, _ = await asyncio.wait_for(readers, timeout=self.timeout)
            except (asyncio.TimeoutError, asyncio.CancelledError, ValueError) as error:
                for reader in reader_tasks:
                    if not reader.done():
                        reader.cancel()
                await asyncio.gather(*reader_tasks, return_exceptions=True)
                if self.backend == "docker":
                    try:
                        cleanup = await asyncio.create_subprocess_exec("docker", "rm", "-f", name, stdout=asyncio.subprocess.DEVNULL, stderr=asyncio.subprocess.DEVNULL)
                        await asyncio.wait_for(cleanup.wait(), timeout=15)
                    except (OSError, asyncio.TimeoutError):
                        pass
                if os.name == "posix":
                    try:
                        os.killpg(process.pid, signal.SIGKILL)
                    except ProcessLookupError:
                        pass
                elif process.returncode is None:
                    process.kill()
                await process.communicate()
                if isinstance(error, asyncio.CancelledError):
                    raise
                return {"status": "error" if isinstance(error, ValueError) else "timeout", "error": str(error),
                        "elapsed_seconds": time.perf_counter() - started, "backend": self.backend}
            try:
                # Runner emits one JSON object. Noise, forged extra lines, or nonzero exit fail closed.
                result = json.loads(stdout.decode("utf-8"))
                if process.returncode != 0 or result.get("status") != "passed" or result.get("cases_passed", 0) < len(task["test_cases"]) + 2:
                    result["status"] = "failed"
            except (ValueError, TypeError, AttributeError):
                result = {"status": "error", "error": "Container did not produce a valid validator result"}
            result.update({"elapsed_seconds": time.perf_counter() - started, "backend": self.backend, "image": self.image,
                           "returncode": process.returncode, "stderr_tail": stderr.decode("utf-8", errors="replace")[-4000:]})
            return result
