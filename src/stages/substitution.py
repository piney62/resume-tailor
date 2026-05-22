"""Stage 3: Substitution Planner.

Compares the JDAnalysis to the candidate's Resume and proposes safe
swaps. Substitutions are constrained to within a single technical
domain, with extra protection against changing quantified metrics or
identity-level fields (company, dates, seniority).
"""

import logging

from pydantic import ValidationError

from src.llm.client import GroqClient
from src.llm.prompt_loader import render
from src.models.schemas import JDAnalysis, Resume, SubstitutionPlan

logger = logging.getLogger(__name__)

SYSTEM_PROMPT = (
    "You are an ATS-optimization expert with 10 years of senior technical "
    "recruiting experience. You plan SAFE substitutions to align a candidate's "
    "resume with a target JD, without inventing experience or contradicting the "
    "candidate's record. "
    "Domain categories you reason within: frontend, backend, cloud, db, "
    "streaming, data, ml, devops, mobile, language, framework, tool, "
    "methodology, security, other. "
    "ABSOLUTE RULES: (1) never substitute across domains; (2) never alter a "
    "quantified metric, company name, date, degree, or the seniority word of a "
    "title; (3) prefer additions over substitutions when adjacency is plausible; "
    "(4) return empty lists when no safe action exists rather than inventing one. "
    "Always output STRICT JSON conforming exactly to the requested schema."
)

# Two attempts: spec'd 0.2, then 0.0 for deterministic recovery on schema failure.
_TEMPERATURES = (0.2, 0.0)


def plan_substitutions(
    jd: JDAnalysis,
    resume: Resume,
    client: GroqClient,
) -> SubstitutionPlan:
    user_prompt = render(
        "substitution_plan.j2",
        jd=jd.model_dump(),
        resume_ctx=_resume_context(resume),
    )
    last_err: ValidationError | None = None

    for attempt, temperature in enumerate(_TEMPERATURES):
        raw = client.complete_json(
            system=SYSTEM_PROMPT,
            user=user_prompt,
            temperature=temperature,
        )
        try:
            return SubstitutionPlan.model_validate(raw)
        except ValidationError as e:
            last_err = e
            logger.warning(
                "Substitution planner schema validation failed on attempt %d (temp=%s): %s",
                attempt + 1, temperature, e,
            )

    raise ValueError(
        f"Substitution planner output failed schema validation after "
        f"{len(_TEMPERATURES)} attempts"
    ) from last_err


def _resume_context(resume: Resume) -> dict:
    """Compact view of the resume tailored for the planner. Excludes bullets
    to keep the prompt small; the planner reasons about tech alignment from
    skills lines and the summary, not bullet-level achievements."""
    return {
        "title": resume.header.title or "",
        "summary": resume.summary.text,
        "skills_categories": resume.skills_section.categories,
        "roles": [
            {
                "company": r.company,
                "title": r.title,
                "skills_line": r.skills_line,
            }
            for r in resume.experience
        ],
    }
