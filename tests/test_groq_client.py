"""Tests for src.llm.client.GroqClient. No real network calls."""

from unittest.mock import MagicMock

import httpx
import pytest
from groq import APIConnectionError, APIStatusError, RateLimitError

from src.llm.client import GroqClient


# ---------- helpers ----------


def _ok_response(prompt_tokens: int = 10, completion_tokens: int = 5, content: str = '{"hello": "world"}'):
    r = MagicMock()
    r.choices = [MagicMock()]
    r.choices[0].message.content = content
    r.usage.prompt_tokens = prompt_tokens
    r.usage.completion_tokens = completion_tokens
    r.usage.total_tokens = prompt_tokens + completion_tokens
    return r


def _http_response(code: int, headers: dict | None = None) -> httpx.Response:
    return httpx.Response(
        status_code=code,
        request=httpx.Request("POST", "https://api.groq.com/openai/v1/chat/completions"),
        headers=headers or {},
    )


def _rate_limit_error(retry_after: str | None = None) -> RateLimitError:
    headers = {"retry-after": retry_after} if retry_after else {}
    return RateLimitError("rate limited", response=_http_response(429, headers), body=None)


def _api_status_error(code: int) -> APIStatusError:
    return APIStatusError(f"http {code}", response=_http_response(code), body=None)


def _connection_error() -> APIConnectionError:
    return APIConnectionError(request=httpx.Request("POST", "https://api.groq.com/"))


def _make_client(n_keys: int = 2, rpm: int = 25):
    fake_clients = [MagicMock() for _ in range(n_keys)]
    sleeps: list[float] = []
    t = [0.0]

    def sleep_fn(s: float) -> None:
        sleeps.append(s)
        t[0] += s

    def time_fn() -> float:
        return t[0]

    client = GroqClient(
        api_keys=[f"key{i}" for i in range(n_keys)],
        model="test-model",
        rpm_per_key=rpm,
        sleep_fn=sleep_fn,
        time_fn=time_fn,
        client_factory=lambda _k: fake_clients.pop(0) if fake_clients else MagicMock(),
    )
    # Reconstruct list since the factory popped them.
    return client, client._clients, sleeps, t  # type: ignore[attr-defined]


# ---------- construction ----------


def test_requires_at_least_one_key() -> None:
    with pytest.raises(ValueError, match="at least one"):
        GroqClient(api_keys=[], client_factory=lambda _k: MagicMock())


def test_from_env_reads_comma_separated_keys(monkeypatch) -> None:
    monkeypatch.setenv("GROQ_API_KEYS", "k1, k2 ,  k3")
    monkeypatch.setenv("GROQ_MODEL", "custom-model")
    monkeypatch.setenv("GROQ_RPM_PER_KEY", "10")
    c = GroqClient.from_env(client_factory=lambda _k: MagicMock())
    assert c._keys == ["k1", "k2", "k3"]
    assert c._model == "custom-model"
    assert c._rpm == 10


def test_from_env_raises_when_empty(monkeypatch) -> None:
    monkeypatch.setenv("GROQ_API_KEYS", "  ,  ")
    with pytest.raises(RuntimeError, match="GROQ_API_KEYS"):
        GroqClient.from_env(client_factory=lambda _k: MagicMock())


# ---------- round-robin rotation ----------


def test_round_robin_two_keys() -> None:
    client, fakes, sleeps, _ = _make_client(n_keys=2)
    for fc in fakes:
        fc.chat.completions.create.return_value = _ok_response()
    for _ in range(4):
        client.complete_json(system="s", user="u", temperature=0.1)
    assert [log.key_idx for log in client.logs] == [0, 1, 0, 1]
    assert sleeps == []


def test_round_robin_three_keys() -> None:
    client, fakes, sleeps, _ = _make_client(n_keys=3)
    for fc in fakes:
        fc.chat.completions.create.return_value = _ok_response()
    for _ in range(6):
        client.complete_json(system="s", user="u", temperature=0.1)
    assert [log.key_idx for log in client.logs] == [0, 1, 2, 0, 1, 2]


# ---------- rate limiting ----------


def test_rate_limit_sleeps_when_all_keys_saturated() -> None:
    client, fakes, sleeps, _ = _make_client(n_keys=2, rpm=2)
    for fc in fakes:
        fc.chat.completions.create.return_value = _ok_response()
    # 4 calls fill the 2-key x 2-rpm budget instantly.
    for _ in range(4):
        client.complete_json(system="s", user="u", temperature=0.1)
    assert sleeps == []
    # 5th call must wait until the oldest slot expires (~60s).
    client.complete_json(system="s", user="u", temperature=0.1)
    assert any(s >= 60.0 for s in sleeps)


# ---------- retry on errors ----------


def test_retry_on_rate_limit_then_success() -> None:
    client, fakes, sleeps, _ = _make_client(n_keys=1)
    fakes[0].chat.completions.create.side_effect = [_rate_limit_error(), _ok_response()]
    out = client.complete_json(system="s", user="u", temperature=0.1)
    assert out == {"hello": "world"}
    assert len(sleeps) == 1
    # Two CallLog entries: the failure and the success.
    assert [log.error is None for log in client.logs] == [False, True]


def test_retry_honors_retry_after_header() -> None:
    client, fakes, sleeps, _ = _make_client(n_keys=1)
    fakes[0].chat.completions.create.side_effect = [_rate_limit_error(retry_after="7"), _ok_response()]
    client.complete_json(system="s", user="u", temperature=0.1)
    assert sleeps[0] == 7.0


def test_retry_on_500_then_success() -> None:
    client, fakes, sleeps, _ = _make_client(n_keys=1)
    fakes[0].chat.completions.create.side_effect = [_api_status_error(503), _ok_response()]
    out = client.complete_json(system="s", user="u", temperature=0.1)
    assert out == {"hello": "world"}
    assert len(sleeps) == 1


def test_no_retry_on_400() -> None:
    client, fakes, _, _ = _make_client(n_keys=1)
    fakes[0].chat.completions.create.side_effect = _api_status_error(400)
    with pytest.raises(APIStatusError):
        client.complete_json(system="s", user="u", temperature=0.1)


def test_retry_on_connection_error() -> None:
    client, fakes, sleeps, _ = _make_client(n_keys=1)
    fakes[0].chat.completions.create.side_effect = [_connection_error(), _ok_response()]
    out = client.complete_json(system="s", user="u", temperature=0.1)
    assert out == {"hello": "world"}
    assert len(sleeps) == 1


def test_max_retries_exceeded_raises_runtime_error() -> None:
    client, fakes, _, _ = _make_client(n_keys=1)
    fakes[0].chat.completions.create.side_effect = _rate_limit_error()
    with pytest.raises(RuntimeError, match="max_retries"):
        client.complete_json(system="s", user="u", temperature=0.1, max_retries=2)


# ---------- JSON parsing ----------


def test_non_json_content_raises() -> None:
    client, fakes, _, _ = _make_client(n_keys=1)
    fakes[0].chat.completions.create.return_value = _ok_response(content="not json at all")
    with pytest.raises(ValueError, match="non-JSON"):
        client.complete_json(system="s", user="u", temperature=0.1)


def test_empty_content_returns_empty_dict() -> None:
    client, fakes, _, _ = _make_client(n_keys=1)
    fakes[0].chat.completions.create.return_value = _ok_response(content="")
    assert client.complete_json(system="s", user="u", temperature=0.1) == {}


# ---------- summary ----------


def test_summary_tracks_tokens_and_per_key_calls() -> None:
    client, fakes, _, _ = _make_client(n_keys=2)
    for fc in fakes:
        fc.chat.completions.create.return_value = _ok_response(prompt_tokens=10, completion_tokens=5)
    for _ in range(4):
        client.complete_json(system="s", user="u", temperature=0.1)
    s = client.summary()
    assert s["total_calls"] == 4
    assert s["successful_calls"] == 4
    assert s["failed_calls"] == 0
    assert s["total_prompt_tokens"] == 40
    assert s["total_completion_tokens"] == 20
    assert s["total_tokens"] == 60
    assert s["calls_per_key"] == {0: 2, 1: 2}


def test_format_summary_is_string() -> None:
    client, fakes, _, _ = _make_client(n_keys=2)
    for fc in fakes:
        fc.chat.completions.create.return_value = _ok_response()
    client.complete_json(system="s", user="u", temperature=0.1)
    out = client.format_summary()
    assert "Groq usage" in out
    assert "tokens" in out
