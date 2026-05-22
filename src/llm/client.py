"""Groq client with round-robin key rotation, per-key rate limiting, and retry.

Why a custom wrapper instead of using groq.Groq directly:
- The free tier is rate-limited per API key. Rotating across N keys gives
  N * RPM of effective throughput while staying under each key's limit.
- Each key carries its own sliding-window of timestamps so the limiter is
  per-key, not global.
- Transient 429 / 5xx / connection errors retry with exponential backoff;
  Retry-After is honored when the server provides it.
"""

import json
import logging
import os
import threading
import time
from collections import deque
from dataclasses import dataclass
from typing import Callable, Optional

from groq import APIConnectionError, APIStatusError, Groq, RateLimitError

logger = logging.getLogger(__name__)


@dataclass
class CallLog:
    timestamp: float
    key_idx: int
    model: str
    prompt_tokens: int = 0
    completion_tokens: int = 0
    total_tokens: int = 0
    latency_ms: float = 0.0
    retries: int = 0
    error: Optional[str] = None


class GroqClient:
    DEFAULT_RPM = 25
    DEFAULT_MODEL = "llama-3.3-70b-versatile"

    def __init__(
        self,
        api_keys: list[str],
        model: str = DEFAULT_MODEL,
        rpm_per_key: int = DEFAULT_RPM,
        *,
        sleep_fn: Callable[[float], None] = time.sleep,
        time_fn: Callable[[], float] = time.monotonic,
        client_factory: Callable[[str], Groq] = lambda k: Groq(api_key=k),
    ) -> None:
        if not api_keys:
            raise ValueError("at least one API key is required")
        self._keys = list(api_keys)
        self._clients = [client_factory(k) for k in self._keys]
        self._model = model
        self._rpm = rpm_per_key
        self._next_key = 0
        self._history: list[deque[float]] = [deque() for _ in self._keys]
        self._logs: list[CallLog] = []
        self._lock = threading.Lock()
        self._sleep = sleep_fn
        self._now = time_fn

    @classmethod
    def from_env(cls, **kwargs) -> "GroqClient":
        raw = os.environ.get("GROQ_API_KEYS", "")
        keys = [k.strip() for k in raw.split(",") if k.strip()]
        if not keys:
            raise RuntimeError(
                "GROQ_API_KEYS env var is empty. Set it in .env (comma-separated)."
            )
        return cls(
            api_keys=keys,
            model=os.environ.get("GROQ_MODEL", cls.DEFAULT_MODEL),
            rpm_per_key=int(os.environ.get("GROQ_RPM_PER_KEY", cls.DEFAULT_RPM)),
            **kwargs,
        )

    # ---------- public API ----------

    def complete_json(
        self,
        *,
        system: str,
        user: str,
        temperature: float = 0.2,
        max_retries: int = 3,
        max_tokens: Optional[int] = None,
    ) -> dict:
        """Chat completion with response_format=json_object. Returns parsed dict."""
        attempt = 0
        last_err: Optional[Exception] = None
        while attempt <= max_retries:
            with self._lock:
                idx = self._acquire_slot()
            client = self._clients[idx]
            t0 = self._now()
            try:
                kwargs = dict(
                    model=self._model,
                    messages=[
                        {"role": "system", "content": system},
                        {"role": "user", "content": user},
                    ],
                    temperature=temperature,
                    response_format={"type": "json_object"},
                )
                if max_tokens is not None:
                    kwargs["max_tokens"] = max_tokens
                resp = client.chat.completions.create(**kwargs)
            except RateLimitError as e:
                last_err = e
                self._log(idx, t0, None, attempt, error=f"RateLimitError: {e}")
                if attempt >= max_retries:
                    break
                wait = self._retry_wait(e, attempt)
                logger.warning("RateLimitError key=%d attempt=%d sleep=%.2fs", idx, attempt, wait)
                self._sleep(wait)
                attempt += 1
                continue
            except APIStatusError as e:
                status = getattr(e, "status_code", None) or getattr(getattr(e, "response", None), "status_code", None)
                if status is not None and 500 <= status < 600:
                    last_err = e
                    self._log(idx, t0, None, attempt, error=f"APIStatusError {status}: {e}")
                    if attempt >= max_retries:
                        break
                    wait = self._retry_wait(e, attempt)
                    logger.warning("APIStatusError %s key=%d attempt=%d sleep=%.2fs", status, idx, attempt, wait)
                    self._sleep(wait)
                    attempt += 1
                    continue
                self._log(idx, t0, None, attempt, error=f"{type(e).__name__}: {e}")
                raise
            except APIConnectionError as e:
                last_err = e
                self._log(idx, t0, None, attempt, error=f"APIConnectionError: {e}")
                if attempt >= max_retries:
                    break
                wait = self._retry_wait(e, attempt)
                logger.warning("APIConnectionError key=%d attempt=%d sleep=%.2fs", idx, attempt, wait)
                self._sleep(wait)
                attempt += 1
                continue

            content = (resp.choices[0].message.content or "{}").strip()
            self._log(idx, t0, resp, attempt)
            try:
                return json.loads(content)
            except json.JSONDecodeError as e:
                raise ValueError(
                    f"LLM returned non-JSON despite response_format: {content[:300]}"
                ) from e

        raise RuntimeError(
            f"exceeded max_retries={max_retries}; last error: {last_err}"
        ) from last_err

    @property
    def logs(self) -> list[CallLog]:
        with self._lock:
            return list(self._logs)

    def summary(self) -> dict:
        with self._lock:
            logs = list(self._logs)
        per_key = {i: 0 for i in range(len(self._keys))}
        for entry in logs:
            per_key[entry.key_idx] += 1
        successful = [e for e in logs if e.error is None]
        return {
            "model": self._model,
            "total_calls": len(logs),
            "successful_calls": len(successful),
            "failed_calls": len(logs) - len(successful),
            "total_retries": sum(e.retries for e in logs),
            "total_prompt_tokens": sum(e.prompt_tokens for e in logs),
            "total_completion_tokens": sum(e.completion_tokens for e in logs),
            "total_tokens": sum(e.total_tokens for e in logs),
            "avg_latency_ms": (
                sum(e.latency_ms for e in successful) / len(successful)
                if successful
                else 0.0
            ),
            "calls_per_key": per_key,
        }

    def format_summary(self) -> str:
        s = self.summary()
        lines = [
            f"  Groq usage — model={s['model']}",
            f"    calls         : {s['successful_calls']} ok / {s['failed_calls']} failed (retries={s['total_retries']})",
            f"    tokens        : prompt={s['total_prompt_tokens']:,} completion={s['total_completion_tokens']:,} total={s['total_tokens']:,}",
            f"    avg latency   : {s['avg_latency_ms']:.0f} ms",
            f"    calls per key : {s['calls_per_key']}",
        ]
        return "\n".join(lines)

    # ---------- internals ----------

    def _acquire_slot(self) -> int:
        """Pick a key with rate budget; sleep if all are saturated. Caller holds lock."""
        while True:
            now = self._now()
            window_start = now - 60.0
            for q in self._history:
                while q and q[0] <= window_start:
                    q.popleft()
            n = len(self._keys)
            for offset in range(n):
                idx = (self._next_key + offset) % n
                if len(self._history[idx]) < self._rpm:
                    self._history[idx].append(now)
                    self._next_key = (idx + 1) % n
                    return idx
            earliest = min(q[0] for q in self._history if q)
            wait = max(0.0, earliest + 60.0 - now) + 0.05
            logger.info("all %d keys saturated at rpm=%d; sleeping %.2fs", n, self._rpm, wait)
            self._sleep(wait)

    def _retry_wait(self, err: Exception, attempt: int) -> float:
        # Honor server-provided Retry-After when available.
        resp = getattr(err, "response", None)
        if resp is not None:
            try:
                ra = resp.headers.get("retry-after")
                if ra:
                    return float(ra)
            except (AttributeError, ValueError, TypeError):
                pass
        # Exponential backoff capped at 60s.
        return min(60.0, 2 ** attempt)

    def _log(
        self,
        key_idx: int,
        t_start: float,
        resp,  # type: ignore[no-untyped-def]
        retries: int,
        error: Optional[str] = None,
    ) -> None:
        entry = CallLog(
            timestamp=t_start,
            key_idx=key_idx,
            model=self._model,
            latency_ms=(self._now() - t_start) * 1000.0,
            retries=retries,
            error=error,
        )
        usage = getattr(resp, "usage", None) if resp is not None else None
        if usage is not None:
            entry.prompt_tokens = int(getattr(usage, "prompt_tokens", 0) or 0)
            entry.completion_tokens = int(getattr(usage, "completion_tokens", 0) or 0)
            entry.total_tokens = int(getattr(usage, "total_tokens", 0) or 0)
        with self._lock:
            self._logs.append(entry)
        logger.info(
            "groq call key=%d latency=%.0fms prompt=%d completion=%d retries=%d%s",
            key_idx,
            entry.latency_ms,
            entry.prompt_tokens,
            entry.completion_tokens,
            retries,
            f" ERR={error}" if error else "",
        )
