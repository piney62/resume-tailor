"""Stage 3 (v2): Holistic Rewriter.

Replaces the prior Substitution Planner + per-section Rewriter (33 LLM
calls) with a single recency-weighted LLM call that produces all editable
text fields in one shot.

Pipeline contract:
    holistic_rewrite(resume, jd, client, *, prior_issues=None) -> Resume

The returned Resume is a deep copy of the input with editable text fields
replaced from the LLM output. Paragraph indices, raw_paragraphs, header
identity (name + contact lines), and education are taken verbatim from
the source — these never round-trip through the model.

Safety net layered on top of the prompt:
  1. Pydantic validation of the LLM JSON against HolisticRewriteOutput.
  2. Per-field number-preservation guard: every existing bullet, intro,
     and the summary is compared against the original; on number drop or
     change, that field reverts to the original text verbatim.
  3. Oldest-role verbatim guard: regardless of what the model emits for
     the oldest role, its intro / bullets / skills_line are forced back
     to the original. This is a deterministic guard — the model is told
     not to touch the oldest role, this enforces it.
  4. Title prefix guard: if the model dropped the original title text
     from the new title, we re-apply the original as a prefix.
"""

from __future__ import annotations

import logging
import re
from typing import Optional

from pydantic import ValidationError

from src.llm.client import GroqClient
from src.llm.prompt_loader import render
from src.models.schemas import (
    HolisticRewriteOutput,
    JDAnalysis,
    Resume,
    SkillsSection,
    ValidationIssue,
)
from src.stages.tiers import classify_tiers
from src.style_rules import BANNED_WORDS

logger = logging.getLogger(__name__)


SYSTEM_PROMPT = (
    "You are an ATS-optimization expert with 10 years of senior technical "
    "recruiting experience. You rewrite resumes to align strongly with a "
    "target JD using a recency-weighted strategy: most-recent role gets "
    "aggressive JD injection, mid-tenure roles get balanced framing, and "
    "the oldest listed role stays byte-for-byte verbatim. You PRESERVE every "
    "company name, employment date, degree, location, and quantified metric "
    "exactly as written. Always output STRICT JSON in the requested shape."
)

# Two attempts: spec'd 0.3 first, then 0.0 deterministic on schema failure.
_TEMPERATURES = (0.3, 0.0)
_NUMBER_RE = re.compile(r"\d+(?:[.,]\d+)?[%xkMK]?")


def holistic_rewrite(
    resume: Resume,
    jd: JDAnalysis,
    client: GroqClient,
    *,
    prior_issues: Optional[list[ValidationIssue]] = None,
) -> Resume:
    tiers = classify_tiers(resume.experience)
    user_prompt = render(
        "holistic_rewrite.j2",
        resume=resume.model_dump(),
        jd=jd.model_dump(),
        tiers=tiers,
        banned_words=list(BANNED_WORDS),
        prior_issues=[i.model_dump() for i in (prior_issues or [])] if prior_issues else None,
    )

    last_err: Optional[ValidationError] = None
    for attempt, temperature in enumerate(_TEMPERATURES):
        raw = client.complete_json(
            system=SYSTEM_PROMPT,
            user=user_prompt,
            temperature=temperature,
            max_tokens=8192,
        )
        try:
            output = HolisticRewriteOutput.model_validate(raw)
        except ValidationError as e:
            last_err = e
            logger.warning(
                "holistic rewriter schema validation failed on attempt %d (temp=%s): %s",
                attempt + 1, temperature, e,
            )
            continue
        if len(output.experience) != len(resume.experience):
            logger.warning(
                "holistic rewriter returned %d roles but resume has %d; retrying",
                len(output.experience), len(resume.experience),
            )
            continue
        return _merge(resume, output, tiers)

    raise ValueError(
        f"holistic rewriter output failed validation after {len(_TEMPERATURES)} attempts"
    ) from last_err


# =========================================================================
# Merge: overlay LLM output onto a deep copy of the original
# =========================================================================


def _merge(original: Resume, out: HolisticRewriteOutput, tiers: list) -> Resume:
    new = original.model_copy(deep=True)

    new.header.title = _safe_title(original.header.title, out.header_title)
    new.summary.text = _guard_numbers(original.summary.text, out.summary_text, label="summary")

    for i, (orig_role, out_role) in enumerate(zip(original.experience, out.experience)):
        tier = tiers[i]

        if tier == "oldest":
            # Deterministic verbatim guard — regardless of what the LLM emitted.
            new.experience[i].intro = orig_role.intro
            new.experience[i].bullets = list(orig_role.bullets)
            new.experience[i].skills_line = orig_role.skills_line
            continue

        new.experience[i].intro = _guard_numbers(orig_role.intro, out_role.intro, label=f"intro:{orig_role.company}")

        # Existing bullets are number-guarded; any extras beyond the original
        # count are treated as NEW bullets (only allowed in the recent tier).
        merged_bullets: list[str] = []
        for j, orig_bullet in enumerate(orig_role.bullets):
            new_text = out_role.bullets[j] if j < len(out_role.bullets) else orig_bullet
            merged_bullets.append(
                _guard_numbers(orig_bullet, new_text, label=f"bullet[{i}][{j}]:{orig_role.company}")
            )
        extras = out_role.bullets[len(orig_role.bullets):]
        if extras and tier == "recent":
            merged_bullets.extend(extras[:2])  # cap at 2 new bullets per spec
        elif extras:
            logger.warning(
                "holistic rewriter proposed %d new bullets in non-recent role %r; dropped",
                len(extras), orig_role.company,
            )
        new.experience[i].bullets = merged_bullets

        # skills_line: if the role had one originally and the model emitted one,
        # take the model's; otherwise keep original.
        if orig_role.skills_line is not None:
            new.experience[i].skills_line = out_role.skills_line or orig_role.skills_line
        else:
            new.experience[i].skills_line = None

    new.skills_section = _merge_skills_section(original.skills_section, out.skills_section_categories)
    return new


def _safe_title(original: Optional[str], new: Optional[str]) -> Optional[str]:
    """If the model dropped the original title text, re-apply it as a prefix.
    Allows the model's JD-aligned suffix to survive."""
    if not original:
        return new
    if not new:
        return original
    if original.lower() in new.lower():
        return new
    # Model deviated — keep original and append the model's text as a suffix.
    return f"{original}, {new}"


def _merge_skills_section(
    original: SkillsSection, new_categories: dict[str, list[str]]
) -> SkillsSection:
    # Drop any category the model dropped (validator catches this as critical),
    # but here we accept the model's categories as-is so the validator can flag.
    raw_lines = [f"{cat}: " + ", ".join(items) for cat, items in new_categories.items()]
    return SkillsSection(
        categories=new_categories,
        raw_lines=raw_lines,
        indices=original.indices,
    )


# =========================================================================
# Number-preservation guard (per editable text field)
# =========================================================================


def _extract_numbers(text: str) -> list[str]:
    return _NUMBER_RE.findall(text or "")


def _numbers_preserved(original: str, new: str) -> bool:
    return sorted(_extract_numbers(original)) == sorted(_extract_numbers(new))


def _guard_numbers(original: str, new: str, *, label: str) -> str:
    new_stripped = (new or "").strip()
    if not new_stripped:
        logger.warning("holistic[%s] emitted empty text; keeping original", label)
        return original
    if not _numbers_preserved(original, new_stripped):
        logger.warning(
            "holistic[%s] dropped/altered numbers; reverting to original. "
            "orig_nums=%s new_nums=%s",
            label, _extract_numbers(original), _extract_numbers(new_stripped),
        )
        return original
    return new_stripped
