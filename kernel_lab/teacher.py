"""Bounded OpenAI-compatible calls; prices must be supplied, never guessed."""
import asyncio
from dataclasses import dataclass
import json
import os
import random
import time
from urllib.parse import urlparse
from .prompts import build_messages


@dataclass
class TeacherConfig:
    base_url: str = "https://openrouter.ai/api/v1"
    model: str = "cohere/north-mini-code:free"
    api_key: str = "local"
    timeout: float = 180
    retries: int = 3
    max_tokens: int = 2048
    temperature: float = 0.2
    input_price_per_million: float | None = None
    output_price_per_million: float | None = None
    free_only: bool = True
    request_interval: float = 3.4
    reasoning_effort: str | None = None

    @classmethod
    def from_env(cls):
        def price(key):
            value = os.environ.get(key)
            return float(value) if value else None
        return cls(base_url=os.environ.get("TEACHER_BASE_URL", cls.base_url),
                   model=os.environ.get("TEACHER_MODEL", cls.model),
                   api_key=os.environ.get("TEACHER_API_KEY", "local"),
                   timeout=float(os.environ.get("TEACHER_TIMEOUT", "180")),
                   retries=int(os.environ.get("TEACHER_RETRIES", "3")),
                   max_tokens=int(os.environ.get("TEACHER_MAX_TOKENS", "2048")),
                   temperature=float(os.environ.get("TEACHER_TEMPERATURE", "0.2")),
                   free_only=os.environ.get("TEACHER_FREE_ONLY", "1") == "1",
                   request_interval=float(os.environ.get("TEACHER_REQUEST_INTERVAL", "3.4")),
                   input_price_per_million=price("TEACHER_INPUT_PRICE_PER_MILLION"),
                   output_price_per_million=price("TEACHER_OUTPUT_PRICE_PER_MILLION"),
                   reasoning_effort=os.environ.get("TEACHER_REASONING_EFFORT") or None)


def extract_code(content):
    content = content.strip()
    if content.startswith("{"):
        parsed = json.loads(content)
        content = parsed["candidate_code"]
    if content.startswith("```") and content.endswith("```"):
        content = content.split("\n", 1)[1].rsplit("```", 1)[0]
    if not content.strip():
        raise ValueError("Empty candidate")
    return content.strip() + "\n"


class TeacherClient:
    def __init__(self, config=None):
        self.config = config or TeacherConfig.from_env()
        if self.config.retries < 0 or self.config.max_tokens <= 0:
            raise ValueError("Invalid retries/max_tokens")
        if self.config.request_interval < 0:
            raise ValueError("Request interval must be nonnegative")
        self.is_openrouter = urlparse(self.config.base_url).hostname == "openrouter.ai"
        if self.is_openrouter and self.config.free_only and not self.config.model.endswith(":free"):
            raise ValueError("Free-only OpenRouter mode requires an explicit :free model; paid/automatic model routing is disabled")
        self._pace_lock = asyncio.Lock()
        self._last_request = None
        prices = [self.config.input_price_per_million, self.config.output_price_per_million]
        if any(price is not None and price < 0 for price in prices):
            raise ValueError("Prices must be nonnegative")

    async def _pace(self):
        # One gate across workers and retries, rather than one rate limit per worker.
        if not self.is_openrouter or not self.config.request_interval:
            return
        async with self._pace_lock:
            if self._last_request is not None:
                delay = self.config.request_interval - (time.monotonic() - self._last_request)
                if delay > 0:
                    await asyncio.sleep(delay)
            self._last_request = time.monotonic()

    async def complete(self, task, examples=None):
        import httpx
        cfg = self.config
        started = time.perf_counter()
        async with httpx.AsyncClient(timeout=cfg.timeout) as client:
            for attempt in range(cfg.retries + 1):
                try:
                    await self._pace()
                    payload = {"model": cfg.model, "messages": build_messages(task, examples), "temperature": cfg.temperature, "max_tokens": cfg.max_tokens}
                    if cfg.reasoning_effort:
                        payload["reasoning_effort"] = cfg.reasoning_effort
                    if self.is_openrouter and cfg.free_only:
                        payload["provider"] = {"max_price": {"prompt": 0, "completion": 0}}
                    response = await client.post(cfg.base_url.rstrip("/") + "/chat/completions",
                        headers={"Authorization": f"Bearer {cfg.api_key}"},
                        json=payload)
                    if response.status_code == 429 or response.status_code >= 500:
                        response.raise_for_status()
                    response.raise_for_status()
                    data = response.json()
                    usage = data.get("usage") or {}
                    prompt, completion = usage.get("prompt_tokens"), usage.get("completion_tokens")
                    cost = None
                    if prompt is not None and completion is not None and cfg.input_price_per_million is not None and cfg.output_price_per_million is not None:
                        cost = (prompt * cfg.input_price_per_million + completion * cfg.output_price_per_million) / 1_000_000
                    parse_error = None
                    try:
                        text = (data["choices"][0].get("message") or {}).get("content") or ""
                        code = extract_code(text)
                    except (ValueError, KeyError, IndexError, TypeError, AttributeError) as error:
                        code, parse_error = "", f"Malformed candidate response ({type(error).__name__})"
                    return {"candidate_code": code, "model": data.get("model", cfg.model), "prompt_tokens": prompt,
                            "completion_tokens": completion, "cost_usd": cost if attempt == 0 else None,
                            "known_successful_response_cost_usd": cost, "parse_error": parse_error, "latency_seconds": time.perf_counter() - started,
                            "api_attempts": attempt + 1, "finish_reason": (data.get("choices") or [{}])[0].get("finish_reason"),
                            "cost_scope": "successful response token usage; ambiguous failed HTTP attempts may also be billed"}
                except (httpx.TimeoutException, httpx.TransportError, httpx.HTTPStatusError) as error:
                    retryable = not isinstance(error, httpx.HTTPStatusError) or error.response.status_code == 429 or error.response.status_code >= 500
                    if not retryable or attempt >= cfg.retries:
                        # Do not leak request headers, API keys, or provider response text.
                        raise RuntimeError(f"Teacher request failed after {attempt + 1} attempts ({type(error).__name__}); usage/cost unknown") from None
                    retry_after = 0
                    if isinstance(error, httpx.HTTPStatusError):
                        try:
                            retry_after = float(error.response.headers.get("Retry-After", "0"))
                        except ValueError:
                            pass
                    # Do not hold a GPU job waiting through a daily quota reset.
                    if retry_after > 60:
                        raise RuntimeError("Provider quota unavailable for more than 60 seconds; stop this run and resume later with a fresh output directory") from None
                    await asyncio.sleep(max(retry_after, min(30, 2 ** attempt) + random.random()))
