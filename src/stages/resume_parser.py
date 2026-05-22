"""Stage 2: Resume parser — pure rule-based, no LLM.

Recognizes the common one-page resume layout used in profiles/sample:

  Name                                <Normal>
  TITLE (all caps)                    <Normal>
  city | phone | email | link         <Normal>
  (blank)
  Summary paragraph                   <Normal>
  PROFESSIONAL EXPERIENCE             <Normal, ALL CAPS — section header>
  Company {sep} Title                 <Normal — role header line 1>
  Dates {sep} Location                <Normal — role header line 2>
  Intro paragraph                     <Normal>
  Bullet ...                          <List Paragraph>
  ...
  Skills: a, b, c                     <Normal — role-level skills line>
  ...
  EDUCATION                           <section header>
  Institution {sep} Degree\nDates {sep} Location   <Normal — may contain soft line break>
  PROFESSIONAL SKILLS                 <section header>
  Category: a, b, c                   <List Paragraph — repeated per category>

Section headers and separators can vary across resumes; we accept several
common variants and surface a clear error if a section is missing.
"""

import re
from pathlib import Path

from src.docx.reader import ParagraphInfo, read_paragraphs
from src.models.schemas import (
    Education,
    EducationIndices,
    Experience,
    ExperienceIndices,
    Header,
    HeaderIndices,
    Resume,
    SkillsSection,
    SkillsSectionIndices,
    Summary,
)


# Strong field separators preferred when multiple separator characters
# appear on a single line. Resumes often use one of these for the "Dates ·
# Location" split while keeping an en-dash for the date range itself
# ("Jul 2018 – Present"). Middle dot (U+00B7), pipe, and em-dash all show
# up as "the field separator" in popular templates.
_FIELD_SEPARATORS = ("·", "|", "—")
# Any character that can serve as some kind of delimiter. Used to detect
# role-header lines (which usually carry exactly one separator).
_SEPARATORS = _FIELD_SEPARATORS + ("–", "•", "/")

# Patterns that identify the named section headers. We compare upper-cased,
# whitespace-collapsed text against this set.
_EXPERIENCE_HEADERS = {
    "PROFESSIONAL EXPERIENCE", "EXPERIENCE", "WORK EXPERIENCE", "EMPLOYMENT",
    "EMPLOYMENT HISTORY",
}
_EDUCATION_HEADERS = {"EDUCATION", "ACADEMIC BACKGROUND"}
_SKILLS_HEADERS = {
    "PROFESSIONAL SKILLS", "SKILLS", "TECHNICAL SKILLS", "CORE SKILLS",
    "COMPETENCIES", "TECHNICAL COMPETENCIES",
}

_BULLET_STYLES = {"List Paragraph", "List Bullet", "Bullet"}

_YEAR_RE = re.compile(r"\b(19|20)\d{2}\b")
_PRESENT_RE = re.compile(r"\b(present|current)\b", re.IGNORECASE)
_SKILLS_LINE_RE = re.compile(r"^\s*skills\s*:", re.IGNORECASE)


class ResumeParseError(ValueError):
    pass


# ---------- public entry point ----------


def parse_resume(path: Path | str) -> Resume:
    paragraphs = read_paragraphs(path)
    if not paragraphs:
        raise ResumeParseError(f"no paragraphs found in {path}")

    section_idxs = _find_section_headers(paragraphs)
    exp_idx = section_idxs["experience"]
    edu_idx = section_idxs["education"]
    skl_idx = section_idxs["skills"]

    header = _parse_header(paragraphs[:exp_idx])
    consumed = [header.indices.name_idx]
    if header.indices.title_idx is not None:
        consumed.append(header.indices.title_idx)
    consumed.extend(header.indices.contact_idxs)
    summary = _parse_summary(paragraphs[:exp_idx], skip_until=max(consumed) + 1)
    experience = _parse_experience(paragraphs[exp_idx + 1 : edu_idx])
    education = _parse_education(paragraphs[edu_idx + 1 : skl_idx])
    skills_section = _parse_skills_section(paragraphs[skl_idx:], header_local_idx=0)

    return Resume(
        header=header,
        summary=summary,
        experience=experience,
        education=education,
        skills_section=skills_section,
        raw_paragraphs=[p.text for p in paragraphs],
    )


# ---------- section detection ----------


def _norm(s: str) -> str:
    return re.sub(r"\s+", " ", s).strip().upper()


def _find_section_headers(paragraphs: list[ParagraphInfo]) -> dict[str, int]:
    exp = edu = skl = -1
    for p in paragraphs:
        if p.style in _BULLET_STYLES:
            continue
        norm = _norm(p.text)
        if not norm or len(norm) > 40:
            continue
        if exp < 0 and norm in _EXPERIENCE_HEADERS:
            exp = p.idx
        elif edu < 0 and norm in _EDUCATION_HEADERS:
            edu = p.idx
        elif skl < 0 and norm in _SKILLS_HEADERS:
            skl = p.idx

    missing = [name for name, i in (("experience", exp), ("education", edu), ("skills", skl)) if i < 0]
    if missing:
        raise ResumeParseError(
            f"could not find section headers: {missing}. "
            f"Expected one of {sorted(_EXPERIENCE_HEADERS)}, "
            f"{sorted(_EDUCATION_HEADERS)}, {sorted(_SKILLS_HEADERS)}."
        )
    if not (exp < edu < skl):
        raise ResumeParseError(
            f"sections out of expected order "
            f"(experience={exp}, education={edu}, skills={skl})"
        )
    return {"experience": exp, "education": edu, "skills": skl}


# ---------- header ----------


def _parse_header(pre_exp: list[ParagraphInfo]) -> Header:
    non_empty = [p for p in pre_exp if p.text.strip()]
    if len(non_empty) < 2:
        raise ResumeParseError("header needs at least a name and one more line")

    name_p = non_empty[0]
    # Detect: is the second non-empty line a title (no separators, short, often all-caps)?
    title_p: ParagraphInfo | None = None
    contact_start = 1
    candidate = non_empty[1]
    looks_like_contact = any(sep in candidate.text for sep in _SEPARATORS) and (
        "@" in candidate.text or "/" in candidate.text or re.search(r"\d{3,}", candidate.text)
    )
    if not looks_like_contact and len(candidate.text) <= 60:
        title_p = candidate
        contact_start = 2

    contact_only: list[ParagraphInfo] = []
    for p in non_empty[contact_start:]:
        if _is_contact_line(p.text):
            contact_only.append(p)
        else:
            break

    return Header(
        name=name_p.text.strip(),
        title=title_p.text.strip() if title_p else None,
        contact_lines=[p.text.strip() for p in contact_only],
        indices=HeaderIndices(
            name_idx=name_p.idx,
            title_idx=title_p.idx if title_p else None,
            contact_idxs=[p.idx for p in contact_only],
        ),
    )


_PROSE_RE = re.compile(r"\.\s+[A-Z]")


def _is_contact_line(text: str) -> bool:
    """Contact lines carry tokens (city/phone/email/url) separated by
    delimiters; summary paragraphs are sentence-flow prose. The two are
    easy to tell apart by markers, not by length."""
    if not _has_separator(text):
        return False
    has_marker = (
        "@" in text
        or "://" in text
        or "linkedin." in text.lower()
        or "github." in text.lower()
        or bool(re.search(r"\d{3,}", text))  # phone, zip
    )
    looks_like_prose = bool(_PROSE_RE.search(text))
    return has_marker and not looks_like_prose


# ---------- summary ----------


def _parse_summary(pre_exp: list[ParagraphInfo], skip_until: int) -> Summary:
    chunks = [p for p in pre_exp if p.idx >= skip_until and p.text.strip()]
    if not chunks:
        raise ResumeParseError("no summary paragraph found before the experience section")
    text = "\n".join(p.text.strip() for p in chunks)
    return Summary(text=text, paragraph_idxs=[p.idx for p in chunks])


# ---------- experience ----------


def _parse_experience(section: list[ParagraphInfo]) -> list[Experience]:
    """Split the experience block into roles, then parse each role.

    A role boundary is a Normal-style paragraph that contains a separator,
    no 4-digit year, and is not a "Skills:" line. The first such paragraph
    after the section header starts the first role; each subsequent one
    starts a new role.
    """
    role_starts: list[int] = []
    for i, p in enumerate(section):
        if p.style in _BULLET_STYLES:
            continue
        text = p.text.strip()
        if not text:
            continue
        if _SKILLS_LINE_RE.match(text):
            continue
        if not _has_separator(text):
            continue
        if _YEAR_RE.search(text) or _PRESENT_RE.search(text):
            continue
        # This looks like "Company – Title".
        role_starts.append(i)

    if not role_starts:
        raise ResumeParseError("no role headers detected in experience section")

    role_starts.append(len(section))  # sentinel for slicing
    roles: list[Experience] = []
    for a, b in zip(role_starts, role_starts[1:]):
        chunk = section[a:b]
        roles.append(_parse_role(chunk))
    return roles


def _parse_role(chunk: list[ParagraphInfo]) -> Experience:
    header_line = chunk[0]
    company, title = _split_on_separator(header_line.text, fallback=("", header_line.text))

    # Second non-bullet paragraph after the header is the dates / location line.
    dates_line: ParagraphInfo | None = None
    intro_line: ParagraphInfo | None = None
    skills_line: ParagraphInfo | None = None
    bullet_idxs: list[int] = []

    for p in chunk[1:]:
        text = p.text.strip()
        if not text:
            continue
        if p.style in _BULLET_STYLES:
            bullet_idxs.append(p.idx)
            continue
        if _SKILLS_LINE_RE.match(text):
            skills_line = p
            continue
        if dates_line is None and (_YEAR_RE.search(text) or _PRESENT_RE.search(text)):
            dates_line = p
            continue
        # Anything else before bullets is the intro.
        if intro_line is None and not bullet_idxs:
            intro_line = p

    dates, location = ("", "")
    if dates_line is not None:
        dates, location = _split_dates_and_location(dates_line.text)

    header_idxs = [header_line.idx]
    if dates_line is not None:
        header_idxs.append(dates_line.idx)

    return Experience(
        company=company.strip(),
        title=title.strip(),
        dates=dates.strip(),
        location=location.strip(),
        intro=intro_line.text.strip() if intro_line else "",
        bullets=[chunk_p.text.strip() for chunk_p in chunk if chunk_p.idx in bullet_idxs],
        skills_line=skills_line.text.strip() if skills_line else None,
        indices=ExperienceIndices(
            header_idxs=header_idxs,
            intro_idx=intro_line.idx if intro_line else None,
            bullet_idxs=bullet_idxs,
            skills_line_idx=skills_line.idx if skills_line else None,
        ),
    )


# ---------- education ----------


def _parse_education(section: list[ParagraphInfo]) -> list[Education]:
    out: list[Education] = []
    for p in section:
        if p.style in _BULLET_STYLES:
            continue
        text = p.text.strip()
        if not text:
            continue
        out.append(_parse_education_entry(p))
    if not out:
        raise ResumeParseError("no education entries detected")
    return out


def _parse_education_entry(p: ParagraphInfo) -> Education:
    # The entry may use soft line breaks (\n) inside a single paragraph,
    # so we split on '\n' before parsing.
    lines = [ln.strip() for ln in p.text.splitlines() if ln.strip()]
    institution, degree, field, dates, location = "", None, None, None, None

    if lines:
        inst_line = lines[0]
        institution, degree_field = _split_on_separator(inst_line, fallback=(inst_line, ""))
        institution = institution.strip()
        if degree_field:
            if "," in degree_field:
                d, _, f = degree_field.partition(",")
                degree = d.strip() or None
                field = f.strip() or None
            else:
                degree = degree_field.strip() or None

    if len(lines) >= 2:
        date_line = lines[1]
        dates_part, location_part = _split_dates_and_location(date_line)
        dates = dates_part or None
        location = location_part or None

    return Education(
        institution=institution,
        degree=degree,
        field=field,
        dates=dates,
        location=location,
        indices=EducationIndices(header_idx=p.idx, detail_idxs=[]),
    )


# ---------- skills section ----------


def _parse_skills_section(section: list[ParagraphInfo], header_local_idx: int) -> SkillsSection:
    header = section[header_local_idx]
    content = section[header_local_idx + 1 :]

    categories: dict[str, list[str]] = {}
    raw_lines: list[str] = []
    content_idxs: list[int] = []

    for p in content:
        text = p.text.strip()
        if not text:
            continue
        raw_lines.append(text)
        content_idxs.append(p.idx)
        if ":" in text:
            cat, _, items = text.partition(":")
            cat = cat.strip()
            item_list = [s.strip() for s in items.split(",") if s.strip()]
            if cat and item_list:
                categories[cat] = item_list

    return SkillsSection(
        categories=categories,
        raw_lines=raw_lines,
        indices=SkillsSectionIndices(header_idx=header.idx, content_idxs=content_idxs),
    )


# ---------- helpers ----------


def _has_separator(text: str) -> bool:
    return any(sep in text for sep in _SEPARATORS)


def _split_on_separator(text: str, *, fallback: tuple[str, str]) -> tuple[str, str]:
    """Split on the FIRST occurrence of any known separator."""
    best_idx = -1
    best_sep = ""
    for sep in _SEPARATORS:
        i = text.find(sep)
        if i >= 0 and (best_idx < 0 or i < best_idx):
            best_idx = i
            best_sep = sep
    if best_idx < 0:
        return fallback
    return text[:best_idx], text[best_idx + len(best_sep):]


def _all_separator_positions(text: str) -> list[tuple[int, str]]:
    out: list[tuple[int, str]] = []
    for sep in _SEPARATORS:
        i = 0
        while True:
            j = text.find(sep, i)
            if j < 0:
                break
            out.append((j, sep))
            i = j + len(sep)
    out.sort()
    return out


def _split_dates_and_location(text: str) -> tuple[str, str]:
    """Split "Dates · Location" — Dates itself may contain a range
    separator (e.g. "Jul 2018 – Present"), so prefer a strong field
    separator first. If none is present and only one separator exists,
    treat the whole thing as dates with no location.
    """
    for sep in _FIELD_SEPARATORS:
        i = text.rfind(sep)
        if i >= 0:
            return text[:i].strip(), text[i + len(sep):].strip()
    positions = _all_separator_positions(text)
    if len(positions) >= 2:
        last_idx, last_sep = positions[-1]
        return text[:last_idx].strip(), text[last_idx + len(last_sep):].strip()
    return text.strip(), ""
