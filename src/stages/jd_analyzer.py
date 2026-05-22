"""Stage 1: JD Analyzer.

Takes raw JD text, returns a validated JDAnalysis. On schema-validation
failure, retries once at a lower temperature before giving up.
"""

import logging

from pydantic import ValidationError

from src.llm.client import GroqClient
from src.llm.prompt_loader import render
from src.models.schemas import JDAnalysis

logger = logging.getLogger(__name__)

SYSTEM_PROMPT = (
    "You are an ATS-optimization expert with 10 years of senior technical "
    "recruiting experience. You read job descriptions and extract structured "
    "signals that candidates use to tailor their resumes. Always output STRICT "
    "JSON conforming exactly to the requested schema. Only extract what is "
    "explicitly stated in the JD — never invent skills, technologies, or "
    "requirements."
)

# Two passes: first at the spec'd 0.1, then drop to 0.0 (deterministic) on
# schema failure to maximize chance of recovery.
_TEMPERATURES = (0.1, 0.0)


def analyze_jd(jd_text: str, client: GroqClient) -> JDAnalysis:
    if not jd_text.strip():
        raise ValueError("jd_text is empty")

    user_prompt = render("jd_analyze.j2", jd_text=jd_text.strip())
    last_err: ValidationError | None = None

    for attempt, temperature in enumerate(_TEMPERATURES):
        raw = client.complete_json(
            system=SYSTEM_PROMPT,
            user=user_prompt,
            temperature=temperature,
        )
        try:
            return JDAnalysis.model_validate(raw)
        except ValidationError as e:
            last_err = e
            logger.warning(
                "JD analyzer schema validation failed on attempt %d (temp=%s): %s",
                attempt + 1, temperature, e,
            )

    raise ValueError(
        f"JD analyzer output failed schema validation after {len(_TEMPERATURES)} attempts"
    ) from last_err
