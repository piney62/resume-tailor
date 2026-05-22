"""Tests for src.stages.holistic_rewriter. Groq calls are mocked."""

from unittest.mock import MagicMock

import pytest

from src.models.schemas import (
    Education,
    EducationIndices,
    Experience,
    ExperienceIndices,
    Header,
    HeaderIndices,
    JDAnalysis,
    Resume,
    SkillsSection,
    SkillsSectionIndices,
    Summary,
)
from src.stages.holistic_rewriter import (
    _guard_numbers,
    _numbers_preserved,
    _safe_title,
    holistic_rewrite,
)


# ---------- fixtures ----------


def _jd() -> JDAnalysis:
    return JDAnalysis(
        must_have=[
            {"tech": "Python", "category": "language", "evidence": "5+ years Python"},
            {"tech": "Kafka", "category": "streaming", "evidence": "Kafka pipelines"},
        ],
        nice_to_have=[],
        soft_skills=[],
        domain_keywords=["real-time analytics"],
        seniority_level="staff",
        exact_phrases_to_mirror=["distributed systems at scale"],
    )


def _role(company: str, *, intro: str, bullets: list[str], skills_line: str | None) -> Experience:
    return Experience(
        company=company,
        title="Engineer",
        dates="2020 - 2021",
        location="Remote",
        intro=intro,
        bullets=bullets,
        skills_line=skills_line,
        indices=ExperienceIndices(header_idxs=[0], intro_idx=1, bullet_idxs=[2, 3], skills_line_idx=4),
    )


def _resume(experience: list[Experience]) -> Resume:
    return Resume(
        header=Header(
            name="Alice", title="STAFF ENGINEER",
            contact_lines=["alice@example.com"],
            indices=HeaderIndices(name_idx=0),
        ),
        summary=Summary(text="Staff engineer with 12 years building backend.", paragraph_idxs=[4]),
        experience=experience,
        education=[
            Education(
                institution="MIT", degree="BS", field="CS",
                dates="2010 - 2014", location="Cambridge, MA",
                indices=EducationIndices(header_idx=10),
            )
        ],
        skills_section=SkillsSection(
            categories={"Languages": ["Python", "Go"]},
            raw_lines=["Languages: Python, Go"],
            indices=SkillsSectionIndices(header_idx=11, content_idxs=[12]),
        ),
        raw_paragraphs=[""] * 13,
    )


def _good_response(*, n_roles: int, extra_bullets: list[str] | None = None) -> dict:
    """Build a holistic-output JSON that matches a given role count."""
    extras = extra_bullets or []
    exp = []
    for i in range(n_roles):
        bullets = ["Cut p99 latency from 800ms to 200ms.", "Mentored 6 engineers."]
        if i == 0:
            bullets.extend(extras)
        exp.append({
            "intro": "Led platform team of 6.",
            "bullets": bullets,
            "skills_line": "Skills: Python, Kafka",
        })
    return {
        "header_title": "STAFF ENGINEER, BACKEND PLATFORMS",
        "summary_text": "Staff engineer with 12 years on distributed systems at scale.",
        "experience": exp,
        "skills_section_categories": {"Languages": ["Python", "Kafka"]},
    }


def _client_returning(*responses) -> MagicMock:
    c = MagicMock()
    c.complete_json.side_effect = list(responses)
    return c


# ---------- happy path ----------


def test_holistic_rewrite_returns_resume() -> None:
    resume = _resume([
        _role("Acme",
              intro="Led platform team of 6.",
              bullets=["Cut p99 latency from 800ms to 200ms.", "Mentored 6 engineers."],
              skills_line="Skills: Python, Redis"),
    ])
    client = _client_returning(_good_response(n_roles=1))
    out = holistic_rewrite(resume, _jd(), client)
    assert isinstance(out, Resume)
    assert "STAFF ENGINEER" in (out.header.title or "")
    assert "BACKEND PLATFORMS" in (out.header.title or "")
    assert "distributed systems" in out.summary.text


def test_holistic_uses_temperature_point_three_first() -> None:
    resume = _resume([_role("Acme", intro="Led.", bullets=["b1", "b2"], skills_line=None)])
    client = _client_returning(_good_response(n_roles=1))
    holistic_rewrite(resume, _jd(), client)
    assert client.complete_json.call_args.kwargs["temperature"] == 0.3


# ---------- per-field number-preservation guard ----------


def test_guard_reverts_summary_when_number_dropped() -> None:
    resume = _resume([_role("Acme", intro="Led.", bullets=["b1", "b2"], skills_line=None)])
    bad = _good_response(n_roles=1)
    bad["summary_text"] = "Staff engineer building backend systems."  # "12 years" dropped
    client = _client_returning(bad)
    out = holistic_rewrite(resume, _jd(), client)
    # Guard reverted summary to the original.
    assert out.summary.text == resume.summary.text


def test_guard_reverts_bullet_when_metric_altered() -> None:
    resume = _resume([_role("Acme", intro="Led.", bullets=["Cut p99 from 800ms to 200ms.", "b2"], skills_line=None)])
    bad = _good_response(n_roles=1)
    bad["experience"][0]["bullets"][0] = "Cut p99 from 800ms to 50ms."  # altered metric
    client = _client_returning(bad)
    out = holistic_rewrite(resume, _jd(), client)
    assert out.experience[0].bullets[0] == "Cut p99 from 800ms to 200ms."


# ---------- oldest-tier verbatim guard ----------


def test_oldest_role_reverts_even_if_model_edited() -> None:
    resume = _resume([
        _role("Recent",
              intro="Led recent team.",
              bullets=["Recent b1 with 5 reports.", "Recent b2."],
              skills_line="Skills: Python"),
        _role("Oldest",
              intro="ORIGINAL intro for oldest role.",
              bullets=["ORIG bullet A.", "ORIG bullet B."],
              skills_line="Skills: ORIGINAL"),
    ])
    # The model "tried" to edit the oldest role — our guard must revert.
    response = _good_response(n_roles=2)
    response["experience"][1] = {
        "intro": "TAMPERED intro",
        "bullets": ["TAMPERED b A", "TAMPERED b B"],
        "skills_line": "Skills: TAMPERED",
    }
    client = _client_returning(response)
    out = holistic_rewrite(resume, _jd(), client)
    oldest_out = out.experience[1]
    assert oldest_out.intro == "ORIGINAL intro for oldest role."
    assert oldest_out.bullets == ["ORIG bullet A.", "ORIG bullet B."]
    assert oldest_out.skills_line == "Skills: ORIGINAL"


# ---------- new bullets ----------


def test_new_bullets_appear_in_recent_role() -> None:
    resume = _resume([_role("Acme", intro="Led.", bullets=["b1", "b2"], skills_line=None)])
    extras = ["Built Kafka streaming pipeline.", "Led Python migration."]
    client = _client_returning(_good_response(n_roles=1, extra_bullets=extras))
    out = holistic_rewrite(resume, _jd(), client)
    # Existing 2 bullets + 2 new bullets = 4 total in recent role.
    assert len(out.experience[0].bullets) == 4
    assert "Built Kafka streaming pipeline." in out.experience[0].bullets
    assert "Led Python migration." in out.experience[0].bullets


def test_new_bullets_capped_at_two() -> None:
    resume = _resume([_role("Acme", intro="Led.", bullets=["b1", "b2"], skills_line=None)])
    extras = ["new 1", "new 2", "new 3", "new 4"]
    client = _client_returning(_good_response(n_roles=1, extra_bullets=extras))
    out = holistic_rewrite(resume, _jd(), client)
    assert len(out.experience[0].bullets) == 4  # 2 original + 2 capped


def test_extras_in_non_recent_role_are_dropped() -> None:
    resume = _resume([
        _role("Recent", intro="x", bullets=["b1", "b2"], skills_line=None),
        _role("Oldest", intro="ORIG.", bullets=["A", "B"], skills_line=None),
    ])
    response = _good_response(n_roles=2)
    # Sneak an extra bullet into the oldest role.
    response["experience"][1]["bullets"].append("smuggled new bullet")
    client = _client_returning(response)
    out = holistic_rewrite(resume, _jd(), client)
    # Oldest role's verbatim guard reverts to original bullets.
    assert out.experience[1].bullets == ["A", "B"]


# ---------- title guard ----------


def test_safe_title_keeps_modifier_when_original_is_prefix() -> None:
    assert _safe_title("STAFF ENGINEER", "STAFF ENGINEER, BACKEND") == "STAFF ENGINEER, BACKEND"


def test_safe_title_reapplies_original_as_prefix_when_dropped() -> None:
    out = _safe_title("STAFF ENGINEER", "PRINCIPAL ENGINEER")
    assert out.startswith("STAFF ENGINEER")
    assert "PRINCIPAL ENGINEER" in out


def test_safe_title_passthrough_when_one_is_none() -> None:
    assert _safe_title(None, "anything") == "anything"
    assert _safe_title("STAFF", None) == "STAFF"


# ---------- number-preservation helpers ----------


def test_numbers_preserved_matches_sorted_sets() -> None:
    assert _numbers_preserved("12 years, 10M users", "10M users — 12 years")
    assert not _numbers_preserved("12 years, 10M users", "10M users")


def test_guard_numbers_falls_back_on_empty() -> None:
    assert _guard_numbers("original 5 years", "", label="x") == "original 5 years"


# ---------- schema retry ----------


def test_invalid_then_valid_retries_at_zero_temperature() -> None:
    resume = _resume([_role("Acme", intro="Led.", bullets=["b1", "b2"], skills_line=None)])
    client = MagicMock()
    client.complete_json.side_effect = [
        {"experience": "not a list"},  # invalid shape
        _good_response(n_roles=1),
    ]
    out = holistic_rewrite(resume, _jd(), client)
    assert isinstance(out, Resume)
    assert client.complete_json.call_count == 2
    # Second call should be at temperature=0.0.
    assert client.complete_json.call_args_list[1].kwargs["temperature"] == 0.0


def test_two_invalid_attempts_raises() -> None:
    resume = _resume([_role("Acme", intro="Led.", bullets=["b1", "b2"], skills_line=None)])
    client = MagicMock()
    client.complete_json.side_effect = [
        {"experience": "broken"},
        {"experience": "still broken"},
    ]
    with pytest.raises(ValueError, match="failed validation"):
        holistic_rewrite(resume, _jd(), client)


# ---------- prompt content ----------


def test_prompt_contains_tier_labels_and_jd_must_have() -> None:
    resume = _resume([
        _role("Recent", intro="x", bullets=["b1", "b2"], skills_line=None),
        _role("Mid", intro="y", bullets=["b1", "b2"], skills_line=None),
        _role("Oldest", intro="z", bullets=["b1", "b2"], skills_line=None),
    ])
    client = _client_returning(_good_response(n_roles=3))
    holistic_rewrite(resume, _jd(), client)
    user = client.complete_json.call_args.kwargs["user"]
    # Tier annotations are uppercase in the rendered template.
    assert "[TIER: RECENT]" in user
    assert "[TIER: MID]" in user
    assert "[TIER: OLDEST]" in user
    # JD must-have terms appear.
    assert "Python" in user
    assert "Kafka" in user
    # Banned words listed.
    assert "leveraged" in user
