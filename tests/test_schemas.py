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
    HolisticRewriteOutput,
    JDAnalysis,
    MustHaveItem,
    NiceToHaveItem,
    Resume,
    SkillsSection,
    SkillsSectionIndices,
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


# ---------- HolisticRewriteOutput ----------


def test_holistic_output_minimal() -> None:
    out = HolisticRewriteOutput(
        summary_text="Senior backend engineer.",
        experience=[],
        skills_section_categories={},
    )
    assert out.header_title is None
    assert out.summary_text.startswith("Senior")


def test_holistic_output_roundtrip() -> None:
    payload = {
        "header_title": "STAFF ENGINEER, BACKEND PLATFORMS",
        "summary_text": "12 years building distributed systems.",
        "experience": [
            {
                "intro": "Led platform team.",
                "bullets": ["Built Kafka pipeline.", "Cut latency by 40%."],
                "skills_line": "Skills: Python, Kafka",
            }
        ],
        "skills_section_categories": {"Languages": ["Python", "Java"]},
    }
    out = HolisticRewriteOutput(**payload)
    again = HolisticRewriteOutput.model_validate_json(out.model_dump_json())
    assert again == out


def test_holistic_output_rejects_extra_fields() -> None:
    with pytest.raises(ValidationError):
        HolisticRewriteOutput(
            summary_text="x",
            experience=[],
            skills_section_categories={},
            unknown_field=42,  # type: ignore[call-arg]
        )


def test_holistic_output_skills_line_optional() -> None:
    out = HolisticRewriteOutput(
        summary_text="x",
        experience=[{"intro": "Led team.", "bullets": ["did stuff"], "skills_line": None}],
        skills_section_categories={},
    )
    assert out.experience[0].skills_line is None


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
