"""Schema validation tests for src.models.schemas."""

import pytest
from pydantic import ValidationError

from src.models.schemas import (
    Education,
    EducationIndices,
    Experience,
    ExperienceIndices,
    Header,
    HeaderIndices,
    JDAnalysis,
    MustHaveItem,
    NiceToHaveItem,
    Resume,
    SkillsSection,
    SkillsSectionIndices,
    Substitution,
    SubstitutionPlan,
    Summary,
    ValidationIssue,
    ValidationReport,
)


# ---------- JDAnalysis ----------


def test_jd_analysis_minimal() -> None:
    jd = JDAnalysis(seniority_level="senior")
    assert jd.must_have == []
    assert jd.seniority_level == "senior"


def test_jd_analysis_full_roundtrip() -> None:
    payload = {
        "must_have": [{"tech": "Kafka", "category": "streaming", "evidence": "5+ years Kafka"}],
        "nice_to_have": [{"tech": "Flink", "category": "streaming"}],
        "soft_skills": ["ownership"],
        "domain_keywords": ["event streaming"],
        "seniority_level": "staff",
        "exact_phrases_to_mirror": ["distributed systems at scale"],
    }
    jd = JDAnalysis(**payload)
    again = JDAnalysis.model_validate_json(jd.model_dump_json())
    assert again == jd


def test_jd_analysis_rejects_bad_seniority() -> None:
    with pytest.raises(ValidationError):
        JDAnalysis(seniority_level="god-tier")


def test_jd_analysis_rejects_extra_fields() -> None:
    with pytest.raises(ValidationError):
        JDAnalysis(seniority_level="senior", unexpected="x")


def test_must_have_item_requires_all_fields() -> None:
    with pytest.raises(ValidationError):
        MustHaveItem(tech="Kafka", category="streaming")  # type: ignore[call-arg]


def test_nice_to_have_does_not_need_evidence() -> None:
    NiceToHaveItem(tech="Flink", category="streaming")


# ---------- Resume ----------


def _make_resume() -> Resume:
    return Resume(
        header=Header(
            name="Jane Doe",
            title="Senior Software Engineer",
            contact_lines=["jane@example.com", "linkedin.com/in/jane"],
            indices=HeaderIndices(name_idx=0, title_idx=1, contact_idxs=[2, 3]),
        ),
        summary=Summary(text="10 years building distributed systems.", paragraph_idxs=[4]),
        experience=[
            Experience(
                company="Acme",
                title="Staff Engineer",
                dates="2020 - Present",
                location="Remote",
                intro="Led platform team of 8.",
                bullets=["Scaled service from 1k to 100k rps.", "Cut p99 latency by 40%."],
                skills_line="Skills: Go, Kubernetes, Kafka",
                indices=ExperienceIndices(
                    header_idxs=[5],
                    intro_idx=6,
                    bullet_idxs=[7, 8],
                    skills_line_idx=9,
                ),
            )
        ],
        education=[
            Education(
                institution="MIT",
                degree="BS",
                field="CS",
                dates="2010 - 2014",
                indices=EducationIndices(header_idx=10, detail_idxs=[]),
            )
        ],
        skills_section=SkillsSection(
            categories={"Languages": ["Go", "Python"]},
            raw_lines=["Languages: Go, Python"],
            indices=SkillsSectionIndices(header_idx=11, content_idxs=[12]),
        ),
        raw_paragraphs=["Jane Doe", "Senior Software Engineer"],
    )


def test_resume_roundtrip() -> None:
    r = _make_resume()
    again = Resume.model_validate_json(r.model_dump_json())
    assert again == r


def test_summary_requires_at_least_one_index() -> None:
    with pytest.raises(ValidationError):
        Summary(text="hi", paragraph_idxs=[])


def test_experience_requires_at_least_one_header_idx() -> None:
    with pytest.raises(ValidationError):
        ExperienceIndices(header_idxs=[], bullet_idxs=[1])


def test_negative_paragraph_indices_rejected() -> None:
    with pytest.raises(ValidationError):
        HeaderIndices(name_idx=-1)


# ---------- SubstitutionPlan ----------


def test_substitution_plan_minimal() -> None:
    plan = SubstitutionPlan(summary_focus="Pivot toward streaming infra")
    assert plan.substitutions == []


def test_substitution_apply_to_must_not_be_empty() -> None:
    with pytest.raises(ValidationError):
        Substitution(old="Redis", new="Kafka", domain="streaming", apply_to=[])


def test_substitution_apply_to_literals_enforced() -> None:
    with pytest.raises(ValidationError):
        Substitution(old="Redis", new="Kafka", domain="streaming", apply_to=["header"])  # type: ignore[list-item]


# ---------- ValidationReport ----------


def test_validation_report_passed_with_no_issues() -> None:
    r = ValidationReport(passed=True, keyword_match_rate=0.85, issues=[])
    assert r.passed


def test_validation_report_keyword_rate_in_range() -> None:
    with pytest.raises(ValidationError):
        ValidationReport(passed=True, keyword_match_rate=1.5, issues=[])


def test_validation_report_passed_blocked_by_critical_issue() -> None:
    issue = ValidationIssue(
        severity="critical",
        section="experience[0].dates",
        issue="Date changed",
        original="2020 - Present",
        rewritten="2021 - Present",
    )
    with pytest.raises(ValidationError):
        ValidationReport(passed=True, keyword_match_rate=0.9, issues=[issue])


def test_validation_report_warning_does_not_block_pass() -> None:
    issue = ValidationIssue(
        severity="warning",
        section="summary",
        issue="Banned word used",
        original="leveraged Kafka",
        rewritten="leveraged Kafka",
    )
    ValidationReport(passed=True, keyword_match_rate=0.9, issues=[issue])
