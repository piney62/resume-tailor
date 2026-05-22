"""Tests for src.stages.rewriter. LLM calls are mocked."""

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
from src.stages.rewriter import (
    _apply_subs_to_item,
    _apply_subs_to_skills_line,
    _apply_title_modifier,
    _extract_numbers,
    _numbers_preserved,
    _rewrite_skills_section,
    rewrite_resume,
)


# ---------- a) title modifier ----------


def test_apply_title_modifier_appends_suffix() -> None:
    assert _apply_title_modifier("Senior Software Engineer", "Backend Platforms") == (
        "Senior Software Engineer, Backend Platforms"
    )


def test_apply_title_modifier_matches_uppercase_style() -> None:
    assert _apply_title_modifier("SENIOR SOFTWARE ENGINEER", "Backend Platforms") == (
        "SENIOR SOFTWARE ENGINEER, BACKEND PLATFORMS"
    )


def test_apply_title_modifier_dedupes_if_already_present() -> None:
    assert _apply_title_modifier(
        "Senior Software Engineer, Backend Platforms", "backend platforms"
    ) == "Senior Software Engineer, Backend Platforms"


def test_apply_title_modifier_none_or_empty_returns_original() -> None:
    assert _apply_title_modifier("Engineer", None) == "Engineer"
    assert _apply_title_modifier("Engineer", "") == "Engineer"
    assert _apply_title_modifier(None, "Backend") is None


# ---------- e) skills line substitutions ----------


def test_apply_subs_to_skills_line_substitutes_matching_item() -> None:
    plan = SubstitutionPlan(
        summary_focus="x",
        substitutions=[
            {"old": "Redis", "new": "Kafka", "domain": "streaming", "apply_to": ["skills"]}
        ],
    )
    out = _apply_subs_to_skills_line("Skills: Python, Redis, PostgreSQL", plan)
    assert out == "Skills: Python, Kafka, PostgreSQL"


def test_apply_subs_to_skills_line_case_insensitive_match() -> None:
    plan = SubstitutionPlan(
        summary_focus="x",
        substitutions=[
            {"old": "redis", "new": "Kafka", "domain": "streaming", "apply_to": ["skills"]}
        ],
    )
    out = _apply_subs_to_skills_line("Skills: Python, Redis, PostgreSQL", plan)
    assert out == "Skills: Python, Kafka, PostgreSQL"


def test_apply_subs_to_skills_line_skips_subs_without_skills_target() -> None:
    plan = SubstitutionPlan(
        summary_focus="x",
        substitutions=[
            {"old": "Redis", "new": "Kafka", "domain": "streaming", "apply_to": ["bullets"]}
        ],
    )
    assert _apply_subs_to_skills_line("Skills: Python, Redis", plan) == "Skills: Python, Redis"


def test_apply_subs_to_skills_line_no_colon_returns_original() -> None:
    plan = SubstitutionPlan(summary_focus="x", substitutions=[])
    assert _apply_subs_to_skills_line("Python, Go", plan) == "Python, Go"


def test_apply_subs_to_item_unmatched_returns_original() -> None:
    plan = SubstitutionPlan(summary_focus="x", substitutions=[])
    assert _apply_subs_to_item("Python", plan) == "Python"


# ---------- f) skills section rewrite ----------


def _section(categories: dict[str, list[str]]) -> SkillsSection:
    return SkillsSection(
        categories=categories,
        raw_lines=[f"{k}: " + ", ".join(v) for k, v in categories.items()],
        indices=SkillsSectionIndices(header_idx=0, content_idxs=list(range(1, 1 + len(categories)))),
    )


def test_skills_section_substitutes_in_each_category() -> None:
    section = _section({"DB": ["Redis", "PostgreSQL"], "Languages": ["Python"]})
    plan = SubstitutionPlan(
        summary_focus="x",
        substitutions=[
            {"old": "Redis", "new": "Kafka", "domain": "streaming", "apply_to": ["skills"]}
        ],
        additions_to_skills=[],
    )
    new = _rewrite_skills_section(section, plan)
    assert new.categories["DB"] == ["Kafka", "PostgreSQL"]
    assert new.categories["Languages"] == ["Python"]


def test_skills_section_appends_additions_to_other_category() -> None:
    section = _section({"Languages": ["Python"]})
    plan = SubstitutionPlan(
        summary_focus="x",
        substitutions=[],
        additions_to_skills=["Real-Time Data Pipelines"],
    )
    new = _rewrite_skills_section(section, plan)
    assert "Other" in new.categories
    assert "Real-Time Data Pipelines" in new.categories["Other"]


def test_skills_section_drops_additions_already_present() -> None:
    section = _section({"DB": ["Redis", "PostgreSQL"]})
    plan = SubstitutionPlan(
        summary_focus="x",
        substitutions=[],
        additions_to_skills=["postgresql", "Kafka"],
    )
    new = _rewrite_skills_section(section, plan)
    # PostgreSQL already there (case-insensitive); only Kafka should land in Other.
    other = new.categories.get("Other", [])
    assert "Kafka" in other
    assert not any(x.lower() == "postgresql" for x in other)


def test_skills_section_rebuilds_raw_lines() -> None:
    section = _section({"DB": ["Redis"], "Languages": ["Python"]})
    plan = SubstitutionPlan(summary_focus="x", substitutions=[], additions_to_skills=[])
    new = _rewrite_skills_section(section, plan)
    assert new.raw_lines == ["DB: Redis", "Languages: Python"]


# ---------- number preservation ----------


def test_extract_numbers_handles_units_and_percentages() -> None:
    nums = _extract_numbers("Scaled from 1k to 100k rps, cutting p99 by 40% in 12 months")
    assert "1k" in nums
    assert "100k" in nums
    assert "40%" in nums
    assert "12" in nums


def test_numbers_preserved_true_when_same_set() -> None:
    assert _numbers_preserved(
        "Cut latency from 800ms to 200ms over 6 months",
        "Reduced latency from 800ms to 200ms across 6 months",
    )


def test_numbers_preserved_false_when_dropped() -> None:
    assert not _numbers_preserved("10M events per day, 200ms p99", "Massive event volume")


def test_numbers_preserved_false_when_altered() -> None:
    assert not _numbers_preserved("Reduced p99 from 800ms to 200ms", "Reduced p99 from 800ms to 50ms")


# ---------- full rewriter wiring ----------


def _build_jd() -> JDAnalysis:
    return JDAnalysis(
        must_have=[{"tech": "Kafka", "category": "streaming", "evidence": "Kafka pipelines"}],
        nice_to_have=[],
        soft_skills=[],
        domain_keywords=["real-time analytics"],
        seniority_level="senior",
        exact_phrases_to_mirror=["distributed systems at scale"],
    )


def _build_resume() -> Resume:
    return Resume(
        header=Header(
            name="Alice",
            title="STAFF ENGINEER",
            contact_lines=["alice@example.com"],
            indices=HeaderIndices(name_idx=0, title_idx=1, contact_idxs=[2]),
        ),
        summary=Summary(text="Staff engineer with 12 years building backend.", paragraph_idxs=[4]),
        experience=[
            Experience(
                company="Acme",
                title="Staff Engineer",
                dates="2020 - Present",
                location="Remote",
                intro="Led platform team of 6.",
                bullets=["Cut p99 latency from 800ms to 200ms.", "Mentored 6 engineers."],
                skills_line="Skills: Python, Redis, PostgreSQL",
                indices=ExperienceIndices(
                    header_idxs=[5], intro_idx=6, bullet_idxs=[7, 8], skills_line_idx=9,
                ),
            )
        ],
        education=[],
        skills_section=SkillsSection(
            categories={"DB": ["Redis", "PostgreSQL"]},
            raw_lines=["DB: Redis, PostgreSQL"],
            indices=SkillsSectionIndices(header_idx=10, content_idxs=[11]),
        ),
        raw_paragraphs=[],
    )


def _plan() -> SubstitutionPlan:
    return SubstitutionPlan(
        title_modifier="Backend Platforms",
        substitutions=[
            {"old": "Redis", "new": "Kafka", "domain": "streaming", "apply_to": ["skills"]}
        ],
        additions_to_skills=["Real-Time Pipelines"],
        summary_focus="Emphasize backend distributed systems",
    )


def test_full_rewrite_calls_llm_for_summary_intro_and_each_bullet() -> None:
    client = MagicMock()
    # Responses in order: summary, intro, bullet1, bullet2
    client.complete_json.side_effect = [
        {"text": "Senior backend engineer with 12 years of distributed-systems experience."},
        {"text": "Led Acme platform team of 6, focused on backend distributed systems."},
        {"text": "Cut p99 latency from 800ms to 200ms across the distributed request path."},
        {"text": "Mentored 6 engineers."},
    ]
    new = rewrite_resume(_build_resume(), _plan(), _build_jd(), client)

    # 1 summary + 1 intro + 2 bullets = 4 LLM calls.
    assert client.complete_json.call_count == 4
    # All LLM calls used the spec'd rewrite temperature.
    for call in client.complete_json.call_args_list:
        assert call.kwargs["temperature"] == 0.4


def test_full_rewrite_applies_title_modifier_in_uppercase() -> None:
    client = MagicMock()
    client.complete_json.return_value = {"text": "ok"}
    new = rewrite_resume(_build_resume(), _plan(), _build_jd(), client)
    assert new.header.title == "STAFF ENGINEER, BACKEND PLATFORMS"


def test_full_rewrite_falls_back_to_original_when_numbers_dropped() -> None:
    # LLM returns a summary that strips the "12 years" number.
    client = MagicMock()
    client.complete_json.return_value = {"text": "Senior backend engineer with deep experience."}
    new = rewrite_resume(_build_resume(), _plan(), _build_jd(), client)
    assert new.summary.text == "Staff engineer with 12 years building backend."


def test_full_rewrite_substitutes_skills_via_rules_not_llm() -> None:
    client = MagicMock()
    client.complete_json.return_value = {"text": "ok"}
    new = rewrite_resume(_build_resume(), _plan(), _build_jd(), client)
    # Per-role skills_line substituted.
    assert new.experience[0].skills_line == "Skills: Python, Kafka, PostgreSQL"
    # Skills section: Redis -> Kafka, and Real-Time Pipelines added to Other.
    assert new.skills_section.categories["DB"] == ["Kafka", "PostgreSQL"]
    assert "Real-Time Pipelines" in new.skills_section.categories["Other"]


def test_full_rewrite_does_not_mutate_input_resume() -> None:
    resume = _build_resume()
    snapshot_title = resume.header.title
    snapshot_summary = resume.summary.text
    snapshot_skills = list(resume.skills_section.categories["DB"])
    client = MagicMock()
    client.complete_json.return_value = {"text": "ok"}
    rewrite_resume(resume, _plan(), _build_jd(), client)
    assert resume.header.title == snapshot_title
    assert resume.summary.text == snapshot_summary
    assert resume.skills_section.categories["DB"] == snapshot_skills


def test_rewrite_uses_system_persona() -> None:
    client = MagicMock()
    client.complete_json.return_value = {"text": "ok"}
    rewrite_resume(_build_resume(), _plan(), _build_jd(), client)
    system = client.complete_json.call_args_list[0].kwargs["system"]
    assert "ATS" in system
    assert "PRESERVING" in system
    assert "STRICT JSON" in system


def test_rewrite_prompt_includes_banned_words_and_few_shot() -> None:
    client = MagicMock()
    client.complete_json.return_value = {"text": "ok"}
    rewrite_resume(_build_resume(), _plan(), _build_jd(), client)
    summary_user = client.complete_json.call_args_list[0].kwargs["user"]
    assert "leveraged" in summary_user
    assert "synergy" in summary_user
    # Few-shot example rendered.
    assert "Example 1" in summary_user


# ---------- progress callback ----------


def test_progress_callback_emits_one_event_per_llm_call_plus_final() -> None:
    client = MagicMock()
    client.complete_json.return_value = {"text": "ok"}
    events: list[tuple[str, int, int]] = []
    rewrite_resume(
        _build_resume(), _plan(), _build_jd(), client,
        progress_cb=lambda label, done, total: events.append((label, done, total)),
    )
    # _build_resume has 1 role with intro + 2 bullets, so total = 1+1+2 = 4
    # Events: summary, intro, bullet 1, bullet 2, "Rewrite complete" = 5
    assert len(events) == 5
    assert events[0][0] == "Rewriting summary"
    assert events[0][2] == 4
    assert events[-1] == ("Rewrite complete", 4, 4)
    # Done counts are monotonically non-decreasing.
    counts = [e[1] for e in events]
    assert counts == sorted(counts)


def test_progress_callback_optional() -> None:
    # Omitting the callback must not break anything.
    client = MagicMock()
    client.complete_json.return_value = {"text": "ok"}
    rewrite_resume(_build_resume(), _plan(), _build_jd(), client)  # no progress_cb
