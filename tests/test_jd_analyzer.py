"""Tests for src.stages.jd_analyzer. Groq calls are mocked."""

from unittest.mock import MagicMock

import pytest

from src.models.schemas import JDAnalysis
from src.stages.jd_analyzer import analyze_jd


VALID_PAYLOAD = {
    "must_have": [
        {"tech": "Python", "category": "language", "evidence": "5+ years of Python"},
        {"tech": "Kafka", "category": "streaming", "evidence": "build Kafka pipelines"},
    ],
    "nice_to_have": [{"tech": "Rust", "category": "language"}],
    "soft_skills": ["ownership"],
    "domain_keywords": ["real-time analytics"],
    "seniority_level": "senior",
    "exact_phrases_to_mirror": ["distributed systems at scale"],
}


def _client(*responses) -> MagicMock:
    client = MagicMock()
    client.complete_json.side_effect = list(responses)
    return client


def test_empty_jd_rejected() -> None:
    with pytest.raises(ValueError, match="jd_text is empty"):
        analyze_jd("   \n  ", _client())


def test_valid_response_returns_jd_analysis() -> None:
    client = _client(VALID_PAYLOAD)
    result = analyze_jd("Senior backend engineer, 5+ years Python and Kafka.", client)
    assert isinstance(result, JDAnalysis)
    assert result.seniority_level == "senior"
    assert {m.tech for m in result.must_have} == {"Python", "Kafka"}
    assert client.complete_json.call_count == 1


def test_first_attempt_uses_temperature_point_one() -> None:
    client = _client(VALID_PAYLOAD)
    analyze_jd("some jd text", client)
    kwargs = client.complete_json.call_args.kwargs
    assert kwargs["temperature"] == 0.1


def test_jd_text_is_injected_into_user_prompt() -> None:
    client = _client(VALID_PAYLOAD)
    analyze_jd("UNIQUE_MARKER_42 in jd", client)
    kwargs = client.complete_json.call_args.kwargs
    assert "UNIQUE_MARKER_42 in jd" in kwargs["user"]


def test_invalid_then_valid_succeeds_after_retry() -> None:
    bad = {"must_have": "not a list", "seniority_level": "senior"}
    client = _client(bad, VALID_PAYLOAD)
    result = analyze_jd("jd", client)
    assert isinstance(result, JDAnalysis)
    assert client.complete_json.call_count == 2
    # Second attempt should drop temperature to 0.0.
    second_call_temp = client.complete_json.call_args_list[1].kwargs["temperature"]
    assert second_call_temp == 0.0


def test_two_invalid_attempts_raises_value_error() -> None:
    bad = {"must_have": "still not a list", "seniority_level": "wizard"}
    client = _client(bad, bad)
    with pytest.raises(ValueError, match="failed schema validation"):
        analyze_jd("jd", client)
    assert client.complete_json.call_count == 2


def test_system_prompt_contains_persona() -> None:
    client = _client(VALID_PAYLOAD)
    analyze_jd("jd", client)
    system = client.complete_json.call_args.kwargs["system"]
    assert "ATS" in system
    assert "STRICT JSON" in system
