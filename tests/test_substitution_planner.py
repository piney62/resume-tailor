"""Tests for src.stages.substitution. Groq calls are mocked."""

from pathlib import Path
from unittest.mock import MagicMock

import pytest

from src.models.schemas import (
    Experience,
    ExperienceIndices,
    Header,
    HeaderIndices,
    JDAnalysis,
    Resume,
    SkillsSection,
    SkillsSectionIndices,
    SubstitutionPlan,
    Summary,
)
from src.stages.resume_parser import parse_resume
from src.stages.substitution import plan_substitutions


SAMPLE_RESUME = Path(__file__).parent.parent / "profiles" / "sample" / "Katharine Berry.docx"


def _build_jd() -> JDAnalysis:
    return JDAnalysis(
        must_have=[
            {"tech": "Python", "category": "language", "evidence": "5+ years of Python"},
            {"tech": "Kafka", "category": "streaming", "evidence": "build Kafka pipelines"},
        ],
        nice_to_have=[{"tech": "Flink", "category": "streaming"}],
        soft_skills=["ownership"],
        domain_keywords=["real-time analytics"],
        seniority_level="senior",
        exact_phrases_to_mirror=["distributed systems at scale"],
    )


def _build_resume() -> Resume:
    return Resume(
        header=Header(
            name="Jane Doe", title="STAFF ENGINEER",
            contact_lines=["jane@example.com"],
            indices=HeaderIndices(name_idx=0, title_idx=1, contact_idxs=[2]),
        ),
        summary=Summary(text="Staff engineer with 12 years building backend systems.", paragraph_idxs=[4]),
        experience=[
            Experience(
                company="Acme", title="Staff Engineer", dates="2020 - Present",
                location="Remote", intro="Led platform team.",
                bullets=["Built a thing."],
                skills_line="Skills: Python, Redis, PostgreSQL",
                indices=ExperienceIndices(header_idxs=[5], intro_idx=6, bullet_idxs=[7], skills_line_idx=8),
            ),
        ],
        education=[],
        skills_section=SkillsSection(
            categories={"Languages": ["Python", "Go"], "Databases": ["PostgreSQL", "Redis"]},
            raw_lines=["Languages: Python, Go", "Databases: PostgreSQL, Redis"],
            indices=SkillsSectionIndices(header_idx=9, content_idxs=[10, 11]),
        ),
        raw_paragraphs=[],
    )


VALID_PLAN = {
    "title_modifier": "Backend Platforms",
    "substitutions": [
        {"old": "Redis", "new": "Kafka", "domain": "streaming", "apply_to": ["skills"]}
    ],
    "additions_to_skills": ["Real-Time Data Pipelines"],
    "summary_focus": "Emphasize distributed systems work; downplay frontend tooling.",
}


def _client(*responses) -> MagicMock:
    c = MagicMock()
    c.complete_json.side_effect = list(responses)
    return c


# ---------- happy path ----------


def test_valid_response_returns_plan() -> None:
    client = _client(VALID_PLAN)
    plan = plan_substitutions(_build_jd(), _build_resume(), client)
    assert isinstance(plan, SubstitutionPlan)
    assert plan.title_modifier == "Backend Platforms"
    assert plan.substitutions[0].old == "Redis"
    assert plan.substitutions[0].new == "Kafka"
    assert client.complete_json.call_count == 1


def test_first_attempt_uses_temperature_point_two() -> None:
    client = _client(VALID_PLAN)
    plan_substitutions(_build_jd(), _build_resume(), client)
    assert client.complete_json.call_args.kwargs["temperature"] == 0.2


def test_empty_plan_is_valid() -> None:
    empty = {
        "title_modifier": None,
        "substitutions": [],
        "additions_to_skills": [],
        "summary_focus": "Keep the original direction.",
    }
    plan = plan_substitutions(_build_jd(), _build_resume(), _client(empty))
    assert plan.substitutions == []
    assert plan.additions_to_skills == []


# ---------- retry ----------


def test_invalid_then_valid_retries_at_zero_temperature() -> None:
    bad = {"substitutions": "not a list"}
    client = _client(bad, VALID_PLAN)
    plan = plan_substitutions(_build_jd(), _build_resume(), client)
    assert isinstance(plan, SubstitutionPlan)
    assert client.complete_json.call_count == 2
    assert client.complete_json.call_args_list[1].kwargs["temperature"] == 0.0


def test_two_invalid_attempts_raises() -> None:
    bad = {"substitutions": "still not a list"}
    with pytest.raises(ValueError, match="failed schema validation"):
        plan_substitutions(_build_jd(), _build_resume(), _client(bad, bad))


# ---------- prompt content ----------


def test_prompt_includes_jd_must_have_tech() -> None:
    client = _client(VALID_PLAN)
    plan_substitutions(_build_jd(), _build_resume(), client)
    user = client.complete_json.call_args.kwargs["user"]
    assert "Kafka" in user
    assert "Python" in user
    assert "5+ years of Python" in user  # evidence quote included


def test_prompt_includes_resume_skills_and_title() -> None:
    client = _client(VALID_PLAN)
    plan_substitutions(_build_jd(), _build_resume(), client)
    user = client.complete_json.call_args.kwargs["user"]
    assert "STAFF ENGINEER" in user
    assert "Redis" in user
    assert "PostgreSQL" in user
    # Per-role skills line included.
    assert "Skills: Python, Redis, PostgreSQL" in user


def test_prompt_excludes_bullet_text_to_keep_compact() -> None:
    client = _client(VALID_PLAN)
    plan_substitutions(_build_jd(), _build_resume(), client)
    user = client.complete_json.call_args.kwargs["user"]
    # The bullet text "Built a thing." should NOT appear in the planner prompt.
    assert "Built a thing." not in user


def test_system_prompt_contains_domain_rules() -> None:
    client = _client(VALID_PLAN)
    plan_substitutions(_build_jd(), _build_resume(), client)
    system = client.complete_json.call_args.kwargs["system"]
    assert "domain" in system.lower()
    assert "STRICT JSON" in system
    assert "metric" in system.lower()


# ---------- integration with real parsed resume ----------


def test_integration_runs_against_real_resume() -> None:
    if not SAMPLE_RESUME.exists():
        pytest.skip("sample resume missing")
    resume = parse_resume(SAMPLE_RESUME)
    client = _client(VALID_PLAN)
    plan = plan_substitutions(_build_jd(), resume, client)
    assert isinstance(plan, SubstitutionPlan)
    user = client.complete_json.call_args.kwargs["user"]
    # Real resume's categorized skills should appear.
    assert "Languages & Frameworks" in user
    assert "Google" in user  # role company in per-role skills line
