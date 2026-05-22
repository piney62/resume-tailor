"""Stage 5: Validator. Rule-based — no LLM.

Compares the rewritten Resume against the original and produces a
ValidationReport. The pipeline (Step 11) reads the report and
regenerates any section with a critical issue (up to 2 retries).

Severity policy:
  - critical: identity-level violations (company / dates / education
    changed, quantified metrics altered, paragraph counts changed,
    title's original text dropped, contact lines edited). Pipeline must
    regenerate.
  - warning: stylistic violations (banned word introduced). Pipeline
    notes them but does not block.

Skills section content is NOT deep-compared: the Substitution Planner
intentionally rewrites it, and the Validator is not given the plan, so
it cannot distinguish a legitimate swap from a violation. We only check
that no category was DROPPED (growth via "Other" is allowed).
"""

import re

from src.models.schemas import (
    Education,
    Experience,
    JDAnalysis,
    Resume,
    ValidationIssue,
    ValidationReport,
)
from src.style_rules import BANNED_WORDS


_NUMBER_RE = re.compile(r"\d+(?:[.,]\d+)?[%xkMK]?")
_BANNED_REGEXES = tuple(
    (word, re.compile(rf"\b{re.escape(word)}\b", re.IGNORECASE))
    for word in BANNED_WORDS
)


def validate(original: Resume, rewritten: Resume, jd: JDAnalysis) -> ValidationReport:
    issues: list[ValidationIssue] = []

    _check_header(original, rewritten, issues)
    _check_summary(original, rewritten, issues)
    _check_experience(original, rewritten, issues)
    _check_education(original, rewritten, issues)
    _check_skills_section(original, rewritten, issues)
    _check_paragraph_count(original, rewritten, issues)

    rate = _compute_keyword_match_rate(rewritten, jd)
    has_critical = any(i.severity == "critical" for i in issues)
    return ValidationReport(
        passed=not has_critical,
        keyword_match_rate=rate,
        issues=issues,
    )


# =========================================================================
# Section checks
# =========================================================================


def _check_header(orig: Resume, new: Resume, issues: list[ValidationIssue]) -> None:
    if orig.header.name != new.header.name:
        issues.append(_critical("header.name", "name changed", orig.header.name, new.header.name))

    # Title: original text must survive verbatim (modifier appended is fine).
    orig_title = (orig.header.title or "").strip()
    new_title = (new.header.title or "").strip()
    if orig_title and orig_title.lower() not in new_title.lower():
        issues.append(_critical("header.title", "original title text removed", orig_title, new_title))
    if not orig_title and new_title:
        issues.append(_critical("header.title", "title introduced where none existed", "", new_title))

    if orig.header.contact_lines != new.header.contact_lines:
        issues.append(_critical(
            "header.contact_lines",
            "contact lines changed",
            " | ".join(orig.header.contact_lines),
            " | ".join(new.header.contact_lines),
        ))


def _check_summary(orig: Resume, new: Resume, issues: list[ValidationIssue]) -> None:
    if not _numbers_match(orig.summary.text, new.summary.text):
        issues.append(_critical(
            "summary",
            f"numbers changed: {_extract_numbers(orig.summary.text)} -> {_extract_numbers(new.summary.text)}",
            orig.summary.text,
            new.summary.text,
        ))
    for issue in _banned_word_issues("summary", orig.summary.text, new.summary.text):
        issues.append(issue)


def _check_experience(orig: Resume, new: Resume, issues: list[ValidationIssue]) -> None:
    if len(orig.experience) != len(new.experience):
        issues.append(_critical(
            "experience",
            f"role count changed: {len(orig.experience)} -> {len(new.experience)}",
            str(len(orig.experience)),
            str(len(new.experience)),
        ))
        return

    for i, (o, n) in enumerate(zip(orig.experience, new.experience)):
        for field in ("company", "title", "dates", "location"):
            if getattr(o, field) != getattr(n, field):
                issues.append(_critical(
                    f"experience[{i}].{field}",
                    f"{field} changed",
                    getattr(o, field),
                    getattr(n, field),
                ))

        if not _numbers_match(o.intro, n.intro):
            issues.append(_critical(
                f"experience[{i}].intro",
                f"numbers changed: {_extract_numbers(o.intro)} -> {_extract_numbers(n.intro)}",
                o.intro,
                n.intro,
            ))
        for issue in _banned_word_issues(f"experience[{i}].intro", o.intro, n.intro):
            issues.append(issue)

        if len(o.bullets) != len(n.bullets):
            issues.append(_critical(
                f"experience[{i}].bullets",
                f"bullet count changed: {len(o.bullets)} -> {len(n.bullets)}",
                str(len(o.bullets)),
                str(len(n.bullets)),
            ))
            continue

        for j, (ob, nb) in enumerate(zip(o.bullets, n.bullets)):
            section = f"experience[{i}].bullets[{j}]"
            if not _numbers_match(ob, nb):
                issues.append(_critical(
                    section,
                    f"numbers changed: {_extract_numbers(ob)} -> {_extract_numbers(nb)}",
                    ob, nb,
                ))
            for issue in _banned_word_issues(section, ob, nb):
                issues.append(issue)


def _check_education(orig: Resume, new: Resume, issues: list[ValidationIssue]) -> None:
    if len(orig.education) != len(new.education):
        issues.append(_critical(
            "education",
            f"entry count changed: {len(orig.education)} -> {len(new.education)}",
            str(len(orig.education)),
            str(len(new.education)),
        ))
        return

    for i, (o, n) in enumerate(zip(orig.education, new.education)):
        for field in ("institution", "degree", "field", "dates", "location"):
            if getattr(o, field) != getattr(n, field):
                issues.append(_critical(
                    f"education[{i}].{field}",
                    f"{field} changed",
                    str(getattr(o, field) or ""),
                    str(getattr(n, field) or ""),
                ))


def _check_skills_section(orig: Resume, new: Resume, issues: list[ValidationIssue]) -> None:
    # Categories may grow (additions go to "Other"); they must not shrink.
    missing = set(orig.skills_section.categories.keys()) - set(new.skills_section.categories.keys())
    if missing:
        issues.append(_critical(
            "skills_section",
            f"categories dropped: {sorted(missing)}",
            ", ".join(orig.skills_section.categories.keys()),
            ", ".join(new.skills_section.categories.keys()),
        ))


def _check_paragraph_count(orig: Resume, new: Resume, issues: list[ValidationIssue]) -> None:
    # raw_paragraphs is sourced from the .docx and must not be edited by the
    # rewriter; if it drifts, something is structurally wrong upstream.
    if len(orig.raw_paragraphs) != len(new.raw_paragraphs):
        issues.append(_critical(
            "raw_paragraphs",
            f"paragraph count changed: {len(orig.raw_paragraphs)} -> {len(new.raw_paragraphs)}",
            str(len(orig.raw_paragraphs)),
            str(len(new.raw_paragraphs)),
        ))


# =========================================================================
# Keyword match rate
# =========================================================================


def _compute_keyword_match_rate(resume: Resume, jd: JDAnalysis) -> float:
    if not jd.must_have:
        return 1.0
    text = _full_resume_text(resume).lower()
    matched = sum(1 for item in jd.must_have if item.tech.lower() in text)
    return matched / len(jd.must_have)


def _full_resume_text(resume: Resume) -> str:
    parts: list[str] = [resume.summary.text]
    for r in resume.experience:
        parts.append(r.intro)
        parts.extend(r.bullets)
        if r.skills_line:
            parts.append(r.skills_line)
    parts.extend(resume.skills_section.raw_lines)
    return "\n".join(parts)


# =========================================================================
# Helpers
# =========================================================================


def _extract_numbers(text: str) -> list[str]:
    return _NUMBER_RE.findall(text or "")


def _numbers_match(original: str, new: str) -> bool:
    return sorted(_extract_numbers(original)) == sorted(_extract_numbers(new))


def _banned_word_issues(section: str, original: str, new: str) -> list[ValidationIssue]:
    """Flag banned words that appear in `new` but were not in `original`.

    This only fires on words the rewriter introduced. If the original
    resume said "leveraged", we do not penalize that — we only catch
    new style violations.
    """
    out: list[ValidationIssue] = []
    orig_lower = (original or "").lower()
    for word, rx in _BANNED_REGEXES:
        if rx.search(new or "") and word not in orig_lower:
            out.append(ValidationIssue(
                severity="warning",
                section=section,
                issue=f"banned word introduced: '{word}'",
                original=original,
                rewritten=new,
            ))
    return out


def _critical(section: str, issue: str, original: str, new: str) -> ValidationIssue:
    return ValidationIssue(
        severity="critical",
        section=section,
        issue=issue,
        original=original,
        rewritten=new,
    )
