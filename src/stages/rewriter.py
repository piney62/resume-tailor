"""Stage 4: Rewriter.

Order of operations (per spec):
  a) Title       — apply title_modifier (rule-based, no LLM)
  b) Summary     — full LLM rewrite (1 call)
  c) Role intros — LLM rewrite per role
  d) Bullets     — LLM rewrite per bullet (sequential; parallel-ready)
  e) Per-role skills_line — rule-based substitutions
  f) Skills section — rule-based substitutions + additions

LLM calls run at temperature=0.4. Every LLM-rewritten field is checked
for number preservation; if numbers are dropped or changed, we fall
back to the original text rather than ship a hallucinated metric.
"""

from __future__ import annotations

import logging
import re
from typing import Callable, Iterable, Optional

from src.llm.client import GroqClient
from src.llm.few_shot import load_examples
from src.llm.prompt_loader import render
from src.models.schemas import (
    Experience,
    JDAnalysis,
    Resume,
    SkillsSection,
    SubstitutionPlan,
    Substitution,
)
from src.style_rules import BANNED_WORDS

logger = logging.getLogger(__name__)


SYSTEM_PROMPT = (
    "You are an ATS-optimization expert with 10 years of senior technical "
    "recruiting experience. You rewrite resume sections so they align with "
    "a target JD while PRESERVING: every number, percentage, year, and "
    "quantified metric exactly as written; every company name; every date; "
    "and the candidate's seniority level. You never invent experience the "
    "candidate does not have. Always output STRICT JSON conforming exactly "
    "to the requested shape."
)

REWRITE_TEMPERATURE = 0.4

_NUMBER_RE = re.compile(r"\d+(?:[.,]\d+)?[%xkMK]?")
_ADDITIONS_CATEGORY = "Other"


# =========================================================================
# Public entry point
# =========================================================================


ProgressCallback = Callable[[str, int, int], None]


def rewrite_resume(
    resume: Resume,
    plan: SubstitutionPlan,
    jd: JDAnalysis,
    client: GroqClient,
    *,
    progress_cb: Optional[ProgressCallback] = None,
) -> Resume:
    """Rewrite the resume in spec order. If `progress_cb` is provided it is
    invoked as `(label, done, total)` before each LLM call so a UI can render
    a progress bar; `total` is the number of LLM calls scheduled (summary +
    intros + bullets), `done` is the count completed so far."""
    new = resume.model_copy(deep=True)
    new.header.title = _apply_title_modifier(resume.header.title, plan.title_modifier)

    n_intros = sum(1 for r in resume.experience if r.intro)
    n_bullets = sum(len(r.bullets) for r in resume.experience)
    total_calls = 1 + n_intros + n_bullets  # summary + each intro + each bullet
    done = 0

    def emit(label: str) -> None:
        if progress_cb is not None:
            progress_cb(label, done, total_calls)

    emit("Rewriting summary")
    new.summary.text = _rewrite_summary(resume.summary.text, plan, jd, client)
    done += 1

    for i, role in enumerate(resume.experience):
        if role.intro:
            emit(f"Rewriting intro: {role.company}")
            new.experience[i].intro = _rewrite_intro(role, plan, jd, client)
            done += 1
        new_bullets: list[str] = []
        for j, b in enumerate(role.bullets):
            emit(f"Rewriting bullet {j + 1}/{len(role.bullets)}: {role.company}")
            new_bullets.append(_rewrite_bullet(b, role, plan, jd, client))
            done += 1
        new.experience[i].bullets = new_bullets
        if role.skills_line is not None:
            new.experience[i].skills_line = _apply_subs_to_skills_line(role.skills_line, plan)

    new.skills_section = _rewrite_skills_section(resume.skills_section, plan)
    emit("Rewrite complete")
    return new


# =========================================================================
# a) Title — rule-based
# =========================================================================


def _apply_title_modifier(title: str | None, modifier: str | None) -> str | None:
    if not title or not modifier:
        return title
    if modifier.strip().lower() in title.lower():
        return title
    suffix = modifier.upper() if title.isupper() else modifier
    return f"{title}, {suffix}"


# =========================================================================
# b/c/d) LLM rewrites with number-preservation guard
# =========================================================================


def _rewrite_summary(
    original: str,
    plan: SubstitutionPlan,
    jd: JDAnalysis,
    client: GroqClient,
) -> str:
    user = render(
        "rewrite_summary.j2",
        original=original,
        jd=jd.model_dump(),
        substitutions=[s.model_dump() for s in _subs_for(plan, "summary")],
        summary_focus=plan.summary_focus,
        examples=load_examples("summary_rewrite"),
        banned_words=list(BANNED_WORDS),
    )
    return _call_text_rewrite(original, user, client, label="summary")


def _rewrite_intro(
    role: Experience,
    plan: SubstitutionPlan,
    jd: JDAnalysis,
    client: GroqClient,
) -> str:
    user = render(
        "rewrite_intro.j2",
        original=role.intro,
        role={"company": role.company, "title": role.title, "dates": role.dates},
        jd=jd.model_dump(),
        substitutions=[s.model_dump() for s in _subs_for(plan, "intro")],
        examples=load_examples("intro_rewrite"),
        banned_words=list(BANNED_WORDS),
    )
    return _call_text_rewrite(role.intro, user, client, label=f"intro:{role.company}")


def _rewrite_bullet(
    bullet: str,
    role: Experience,
    plan: SubstitutionPlan,
    jd: JDAnalysis,
    client: GroqClient,
) -> str:
    user = render(
        "rewrite_bullet.j2",
        original=bullet,
        role={"company": role.company, "title": role.title},
        jd=jd.model_dump(),
        substitutions=[s.model_dump() for s in _subs_for(plan, "bullets")],
        examples=load_examples("bullet_rewrite"),
        banned_words=list(BANNED_WORDS),
    )
    return _call_text_rewrite(bullet, user, client, label=f"bullet:{role.company}")


def _call_text_rewrite(
    original: str,
    user: str,
    client: GroqClient,
    *,
    label: str,
) -> str:
    """Single rewrite call with number-preservation fallback.

    If the model drops or alters any numeric token from the original,
    we keep the original verbatim rather than risk a hallucinated metric.
    """
    if not original.strip():
        return original
    raw = client.complete_json(
        system=SYSTEM_PROMPT,
        user=user,
        temperature=REWRITE_TEMPERATURE,
    )
    new = (raw.get("text") or "").strip()
    if not new:
        logger.warning("rewriter[%s] returned empty text; keeping original", label)
        return original
    if not _numbers_preserved(original, new):
        logger.warning(
            "rewriter[%s] dropped or altered numbers; falling back to original. "
            "original_nums=%s new_nums=%s",
            label, _extract_numbers(original), _extract_numbers(new),
        )
        return original
    return new


def _extract_numbers(text: str) -> list[str]:
    return _NUMBER_RE.findall(text)


def _numbers_preserved(original: str, new: str) -> bool:
    return sorted(_extract_numbers(original)) == sorted(_extract_numbers(new))


# =========================================================================
# e) Per-role skills_line — rule-based
# =========================================================================


def _apply_subs_to_skills_line(skills_line: str, plan: SubstitutionPlan) -> str:
    """Skills line format: "Skills: A, B, C". Apply substitutions to each item."""
    if not skills_line:
        return skills_line
    prefix, sep, body = skills_line.partition(":")
    if not sep:
        return skills_line
    items = [s.strip() for s in body.split(",") if s.strip()]
    new_items = [_apply_subs_to_item(item, plan) for item in items]
    return f"{prefix.strip()}: " + ", ".join(new_items)


def _apply_subs_to_item(item: str, plan: SubstitutionPlan) -> str:
    for sub in plan.substitutions:
        if "skills" in sub.apply_to and item.lower() == sub.old.lower():
            return sub.new
    return item


# =========================================================================
# f) Skills section — rule-based substitutions + additions
# =========================================================================


def _rewrite_skills_section(section: SkillsSection, plan: SubstitutionPlan) -> SkillsSection:
    new_categories: dict[str, list[str]] = {}
    for cat, items in section.categories.items():
        new_categories[cat] = [_apply_subs_to_item(item, plan) for item in items]

    # Drop additions that are already present somewhere in the section
    # (case-insensitive). Then deposit the remainder into the additions bucket.
    existing_lower = {
        s.lower() for items in new_categories.values() for s in items
    }
    additions = [a for a in plan.additions_to_skills if a.lower() not in existing_lower]
    if additions:
        bucket = _ADDITIONS_CATEGORY if _ADDITIONS_CATEGORY in new_categories else _ADDITIONS_CATEGORY
        new_categories.setdefault(bucket, [])
        new_categories[bucket] = new_categories[bucket] + additions

    raw_lines = [f"{cat}: " + ", ".join(items) for cat, items in new_categories.items()]
    return SkillsSection(
        categories=new_categories,
        raw_lines=raw_lines,
        indices=section.indices,
    )


# =========================================================================
# helpers
# =========================================================================


def _subs_for(plan: SubstitutionPlan, target: str) -> Iterable[Substitution]:
    return [s for s in plan.substitutions if target in s.apply_to]
