"""Tests for src.stages.validator. No LLM, no network."""

from copy import deepcopy

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
from src.stages.validator import (
    _compute_keyword_match_rate,
    _extract_numbers,
    _numbers_match,
    validate,
)


# ---------- fixtures ----------


def _base_resume() -> Resume:
    return Resume(
        header=Header(
            name="Alice Example",
            title="STAFF ENGINEER",
            contact_lines=["alice@example.com | 555-0100"],
            indices=HeaderIndices(name_idx=0, title_idx=1, contact_idxs=[2]),
        ),
        summary=Summary(
            text="Staff engineer with 12 years building backend systems serving 10M users.",
            paragraph_idxs=[4],
        ),
        experience=[
            Experience(
                company="Acme",
                title="Staff Engineer",
                dates="2020 - Present",
                location="Remote",
                intro="Led platform team of 6 engineers.",
                bullets=[
                    "Cut API p99 latency from 800ms to 200ms.",
                    "Mentored 6 engineers, 2 promoted within 12 months.",
                ],
                skills_line="Skills: Python, Redis, PostgreSQL",
                indices=ExperienceIndices(
                    header_idxs=[5], intro_idx=6, bullet_idxs=[7, 8], skills_line_idx=9,
                ),
            )
        ],
        education=[
            Education(
                institution="MIT",
                degree="BS",
                field="Computer Science",
                dates="2010 - 2014",
                location="Cambridge, MA",
                indices=EducationIndices(header_idx=10, detail_idxs=[]),
            )
        ],
        skills_section=SkillsSection(
            categories={"DB": ["Redis", "PostgreSQL"], "Languages": ["Python"]},
            raw_lines=["DB: Redis, PostgreSQL", "Languages: Python"],
            indices=SkillsSectionIndices(header_idx=11, content_idxs=[12, 13]),
        ),
        raw_paragraphs=[""] * 14,
    )


def _jd() -> JDAnalysis:
    return JDAnalysis(
        must_have=[
            {"tech": "Python", "category": "language", "evidence": "5+ years Python"},
            {"tech": "Kafka", "category": "streaming", "evidence": "Kafka pipelines"},
        ],
        nice_to_have=[],
        soft_skills=[],
        domain_keywords=[],
        seniority_level="senior",
        exact_phrases_to_mirror=[],
    )


# ---------- identity / happy path ----------


def test_identical_resume_passes_with_no_issues() -> None:
    orig = _base_resume()
    new = deepcopy(orig)
    report = validate(orig, new, _jd())
    assert report.passed is True
    assert report.issues == []


def test_title_with_modifier_appended_is_not_an_issue() -> None:
    orig = _base_resume()
    new = deepcopy(orig)
    new.header.title = "STAFF ENGINEER, BACKEND PLATFORMS"
    report = validate(orig, new, _jd())
    assert report.passed is True


# ---------- critical: identity-level edits ----------


def test_company_change_is_critical() -> None:
    orig = _base_resume()
    new = deepcopy(orig)
    new.experience[0].company = "AcmeCorp"
    report = validate(orig, new, _jd())
    assert report.passed is False
    assert any(i.section == "experience[0].company" and i.severity == "critical" for i in report.issues)


def test_date_change_is_critical() -> None:
    orig = _base_resume()
    new = deepcopy(orig)
    new.experience[0].dates = "2019 - Present"
    report = validate(orig, new, _jd())
    assert report.passed is False
    assert any(i.section == "experience[0].dates" for i in report.issues)


def test_education_field_change_is_critical() -> None:
    orig = _base_resume()
    new = deepcopy(orig)
    new.education[0].degree = "PhD"
    report = validate(orig, new, _jd())
    assert report.passed is False
    assert any(i.section == "education[0].degree" for i in report.issues)


def test_education_count_change_is_critical() -> None:
    orig = _base_resume()
    new = deepcopy(orig)
    new.education = []
    report = validate(orig, new, _jd())
    assert report.passed is False
    assert any(i.section == "education" for i in report.issues)


def test_role_count_change_is_critical() -> None:
    orig = _base_resume()
    new = deepcopy(orig)
    new.experience = []
    report = validate(orig, new, _jd())
    assert report.passed is False


def test_bullet_count_change_is_critical() -> None:
    orig = _base_resume()
    new = deepcopy(orig)
    new.experience[0].bullets = new.experience[0].bullets[:1]
    report = validate(orig, new, _jd())
    assert report.passed is False
    assert any(i.section == "experience[0].bullets" for i in report.issues)


def test_header_name_change_is_critical() -> None:
    orig = _base_resume()
    new = deepcopy(orig)
    new.header.name = "Alice Different"
    report = validate(orig, new, _jd())
    assert any(i.section == "header.name" and i.severity == "critical" for i in report.issues)


def test_contact_lines_change_is_critical() -> None:
    orig = _base_resume()
    new = deepcopy(orig)
    new.header.contact_lines = ["alice@example.com | NEW-NUMBER"]
    report = validate(orig, new, _jd())
    assert any(i.section == "header.contact_lines" for i in report.issues)


def test_title_completely_replaced_is_critical() -> None:
    orig = _base_resume()
    new = deepcopy(orig)
    new.header.title = "PRINCIPAL ENGINEER"  # original "STAFF ENGINEER" is gone
    report = validate(orig, new, _jd())
    assert any(i.section == "header.title" and i.severity == "critical" for i in report.issues)


# ---------- critical: number preservation ----------


def test_bullet_number_altered_is_critical() -> None:
    orig = _base_resume()
    new = deepcopy(orig)
    new.experience[0].bullets[0] = "Cut API p99 latency from 800ms to 50ms."
    report = validate(orig, new, _jd())
    assert report.passed is False
    issues = [i for i in report.issues if i.section == "experience[0].bullets[0]"]
    assert any("numbers changed" in i.issue for i in issues)


def test_summary_numbers_dropped_is_critical() -> None:
    orig = _base_resume()
    new = deepcopy(orig)
    new.summary.text = "Staff engineer building backend systems for millions of users."
    report = validate(orig, new, _jd())
    assert report.passed is False
    assert any(i.section == "summary" and "numbers changed" in i.issue for i in report.issues)


def test_summary_number_preservation_with_reordered_text_is_fine() -> None:
    orig = _base_resume()
    new = deepcopy(orig)
    # Numbers preserved (12, 10M) but text rephrased.
    new.summary.text = "Building backend systems for 10M users — 12 years of experience."
    report = validate(orig, new, _jd())
    assert report.passed is True


def test_intro_number_altered_is_critical() -> None:
    orig = _base_resume()
    new = deepcopy(orig)
    new.experience[0].intro = "Led platform team of 10 engineers."
    report = validate(orig, new, _jd())
    assert report.passed is False
    assert any(i.section == "experience[0].intro" for i in report.issues)


# ---------- warnings: banned words ----------


def test_banned_word_introduced_in_rewrite_is_warning() -> None:
    orig = _base_resume()
    new = deepcopy(orig)
    new.summary.text = (
        "Staff engineer leveraged 12 years building backend systems serving 10M users."
    )
    report = validate(orig, new, _jd())
    # Numbers preserved → no critical from numbers; banned word is a warning.
    critical_count = sum(1 for i in report.issues if i.severity == "critical")
    warning_count = sum(1 for i in report.issues if i.severity == "warning")
    assert critical_count == 0
    assert warning_count >= 1
    assert report.passed is True
    assert any("leveraged" in i.issue for i in report.issues)


def test_banned_word_already_in_original_is_not_flagged() -> None:
    orig = _base_resume()
    orig.summary.text = "Staff engineer leveraged 12 years for 10M users."
    new = deepcopy(orig)
    new.summary.text = "Staff engineer leveraged 12 years across 10M users."  # leveraged was there
    report = validate(orig, new, _jd())
    assert not any("leveraged" in i.issue for i in report.issues)


def test_banned_word_with_word_boundary() -> None:
    # "leverage" should not match a word that contains it as a substring of
    # something else. Here the original lacks "leveraged" but "lever-arm" is
    # not flagged.
    orig = _base_resume()
    new = deepcopy(orig)
    new.summary.text = "Staff engineer at lever-arm consulting, 12 years, 10M users."
    report = validate(orig, new, _jd())
    assert not any("leverage" in i.issue for i in report.issues)


# ---------- skills section ----------


def test_skills_category_added_other_is_fine() -> None:
    orig = _base_resume()
    new = deepcopy(orig)
    new.skills_section.categories["Other"] = ["Real-Time Pipelines"]
    new.skills_section.raw_lines.append("Other: Real-Time Pipelines")
    report = validate(orig, new, _jd())
    assert report.passed is True


def test_skills_category_dropped_is_critical() -> None:
    orig = _base_resume()
    new = deepcopy(orig)
    del new.skills_section.categories["Languages"]
    report = validate(orig, new, _jd())
    assert report.passed is False
    assert any(i.section == "skills_section" and "dropped" in i.issue for i in report.issues)


# ---------- keyword match rate ----------


def test_keyword_match_rate_full_when_all_must_haves_present() -> None:
    orig = _base_resume()
    new = deepcopy(orig)
    # Add Kafka somewhere
    new.skills_section.categories["DB"].append("Kafka")
    new.skills_section.raw_lines = [
        "DB: Redis, PostgreSQL, Kafka", "Languages: Python"
    ]
    report = validate(orig, new, _jd())
    assert report.keyword_match_rate == 1.0


def test_keyword_match_rate_partial() -> None:
    orig = _base_resume()
    new = deepcopy(orig)
    # Python present but Kafka absent — 1/2 = 0.5
    report = validate(orig, new, _jd())
    assert report.keyword_match_rate == 0.5


def test_keyword_match_rate_empty_must_have_is_one() -> None:
    orig = _base_resume()
    new = deepcopy(orig)
    jd_empty = JDAnalysis(seniority_level="senior")
    assert _compute_keyword_match_rate(new, jd_empty) == 1.0
    report = validate(orig, new, jd_empty)
    assert report.keyword_match_rate == 1.0


# ---------- raw_paragraph count ----------


def test_raw_paragraph_count_change_is_critical() -> None:
    orig = _base_resume()
    new = deepcopy(orig)
    new.raw_paragraphs = new.raw_paragraphs[:-2]
    report = validate(orig, new, _jd())
    assert report.passed is False
    assert any(i.section == "raw_paragraphs" for i in report.issues)


# ---------- number-extraction helpers ----------


def test_extract_numbers_includes_units() -> None:
    nums = _extract_numbers("Scaled from 1k to 100k rps, cutting p99 by 40% over 6 months")
    assert "1k" in nums and "100k" in nums and "40%" in nums and "6" in nums


def test_numbers_match_handles_empty_strings() -> None:
    assert _numbers_match("", "")
    assert _numbers_match("just words", "different words")  # both have no numbers
    assert not _numbers_match("with 5 here", "no number here")


# ---------- multiple critical issues accumulate ----------


# ---------- tier-aware checks (v2 hybrid pipeline) ----------


def _three_role_resume() -> Resume:
    """Build a 3-role resume so tiers are [recent, mid, oldest]."""
    base = _base_resume()
    role_mid = Experience(
        company="Mid Co",
        title="Engineer",
        dates="2017 - 2019",
        location="Remote",
        intro="Mid intro.",
        bullets=["Mid bullet 1.", "Mid bullet 2."],
        skills_line="Skills: Go, Java",
        indices=ExperienceIndices(header_idxs=[20], intro_idx=21, bullet_idxs=[22, 23], skills_line_idx=24),
    )
    role_oldest = Experience(
        company="Old Co",
        title="Engineer",
        dates="2014 - 2016",
        location="Remote",
        intro="Old intro.",
        bullets=["Old bullet 1.", "Old bullet 2."],
        skills_line="Skills: C, Python",
        indices=ExperienceIndices(header_idxs=[30], intro_idx=31, bullet_idxs=[32, 33], skills_line_idx=34),
    )
    base.experience = [base.experience[0], role_mid, role_oldest]
    base.raw_paragraphs = [""] * 40
    return base


def test_recent_role_can_grow_bullets_by_up_to_two() -> None:
    orig = _three_role_resume()
    new = deepcopy(orig)
    new.experience[0].bullets = new.experience[0].bullets + ["new bullet 1", "new bullet 2"]
    report = validate(orig, new, _jd())
    assert report.passed is True


def test_recent_role_growth_over_two_is_critical() -> None:
    orig = _three_role_resume()
    new = deepcopy(orig)
    new.experience[0].bullets = new.experience[0].bullets + ["n1", "n2", "n3"]
    report = validate(orig, new, _jd())
    assert report.passed is False
    assert any(i.section == "experience[0].bullets" for i in report.issues)


def test_recent_role_bullet_drop_is_critical() -> None:
    orig = _three_role_resume()
    new = deepcopy(orig)
    new.experience[0].bullets = new.experience[0].bullets[:1]  # growth = -1
    report = validate(orig, new, _jd())
    assert report.passed is False


def test_mid_role_bullet_count_drift_is_critical() -> None:
    orig = _three_role_resume()
    new = deepcopy(orig)
    new.experience[1].bullets = new.experience[1].bullets + ["extra bullet"]
    report = validate(orig, new, _jd())
    assert report.passed is False
    assert any(i.section == "experience[1].bullets" for i in report.issues)


def test_oldest_role_intro_drift_is_critical() -> None:
    orig = _three_role_resume()
    new = deepcopy(orig)
    new.experience[2].intro = "Different intro for oldest."
    report = validate(orig, new, _jd())
    assert report.passed is False
    assert any(i.section == "experience[2].intro" and "verbatim" in i.issue for i in report.issues)


def test_oldest_role_bullet_drift_is_critical() -> None:
    orig = _three_role_resume()
    new = deepcopy(orig)
    new.experience[2].bullets[0] = "Tampered bullet"
    report = validate(orig, new, _jd())
    assert report.passed is False
    assert any(i.section == "experience[2].bullets" and "verbatim" in i.issue for i in report.issues)


def test_oldest_role_skills_line_drift_is_critical() -> None:
    orig = _three_role_resume()
    new = deepcopy(orig)
    new.experience[2].skills_line = "Skills: Tampered"
    report = validate(orig, new, _jd())
    assert report.passed is False
    assert any(i.section == "experience[2].skills_line" for i in report.issues)


def test_oldest_role_verbatim_passes() -> None:
    orig = _three_role_resume()
    new = deepcopy(orig)
    # Touch only the recent role; oldest is identical → no drift critical.
    new.summary.text = "New summary preserving 12 years and 10M users numbers."
    report = validate(orig, new, _jd())
    assert report.passed is True


def test_multiple_critical_issues_all_reported() -> None:
    orig = _base_resume()
    new = deepcopy(orig)
    new.experience[0].company = "Wrong Co"
    new.education[0].degree = "Wrong Degree"
    new.summary.text = "No numbers here."
    report = validate(orig, new, _jd())
    assert report.passed is False
    sections = {i.section for i in report.issues if i.severity == "critical"}
    assert "experience[0].company" in sections
    assert "education[0].degree" in sections
    assert "summary" in sections
