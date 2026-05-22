"""Pipeline orchestration.

Coordinates the six-stage flow:
  1. Analyze JD                (LLM)
  2. Parse resume              (rule-based)
  3. Plan substitutions        (LLM)
  4. Rewrite                   (LLM)
  5. Validate                  (rule-based)
     -> on critical issues: regenerate affected sections, max N passes
  6. Write DOCX + export PDF

Each stage's input and output is dumped as JSON to a per-run log dir
under {output_dir}/logs/{timestamp}/ so any production failure can be
diagnosed by replaying or inspecting the intermediate artifacts.
"""

import json
import logging
import re
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any, Callable, Optional

from src.docx.pdf_export import export_pdf
from src.docx.writer import write_resume
from src.llm.client import GroqClient
from src.models.schemas import (
    JDAnalysis,
    Resume,
    SubstitutionPlan,
    ValidationReport,
)
from src.stages.jd_analyzer import analyze_jd
from src.stages.resume_parser import parse_resume
from src.stages.rewriter import (
    _rewrite_bullet,
    _rewrite_intro,
    _rewrite_summary,
    rewrite_resume,
)
from src.stages.substitution import plan_substitutions
from src.stages.validator import validate

logger = logging.getLogger(__name__)


@dataclass
class TailorResult:
    docx_path: Path
    pdf_path: Optional[Path]
    report: ValidationReport
    log_dir: Path
    groq_summary: dict
    # Snapshots used by the UI's diff view; kept here so the caller does not
    # have to re-parse the source after the pipeline completes.
    original_resume: Optional[Any] = None
    rewritten_resume: Optional[Any] = None


# Callback signature surfaced to UIs: (label, fraction_complete_0_to_1).
PipelineProgressCallback = Callable[[str, float], None]


def run_tailor_pipeline(
    *,
    resume_path: Path,
    jd_text: str,
    output_dir: Path,
    client: GroqClient,
    pdf_backend: str = "auto",
    skip_pdf: bool = False,
    max_regen_passes: int = 2,
    progress_cb: Optional[PipelineProgressCallback] = None,
) -> TailorResult:
    output_dir.mkdir(parents=True, exist_ok=True)
    log_dir = output_dir / "logs" / datetime.now().strftime("%Y%m%d-%H%M%S")
    log_dir.mkdir(parents=True, exist_ok=True)

    def emit(label: str, frac: float) -> None:
        if progress_cb is not None:
            progress_cb(label, frac)

    emit("Analyzing JD…", 0.02)
    logger.info("[1/6] analyzing JD")
    jd = analyze_jd(jd_text, client)
    _dump_json(log_dir / "1_jd_analysis.json", jd)

    emit("Parsing resume…", 0.10)
    logger.info("[2/6] parsing resume")
    resume = parse_resume(resume_path)
    _dump_json(log_dir / "2_resume_parsed.json", resume)

    emit("Planning substitutions…", 0.15)
    logger.info("[3/6] planning substitutions")
    plan = plan_substitutions(jd, resume, client)
    _dump_json(log_dir / "3_substitution_plan.json", plan)

    # Rewriter occupies the largest share of wall time; map its
    # (done, total) calls into the 0.20..0.85 slice of the overall bar.
    def _rewriter_progress(label: str, done: int, total: int) -> None:
        frac_local = done / max(total, 1)
        emit(label, 0.20 + 0.65 * frac_local)

    logger.info("[4/6] rewriting (initial pass)")
    rewritten = rewrite_resume(resume, plan, jd, client, progress_cb=_rewriter_progress)
    _dump_json(log_dir / "4a_rewritten_initial.json", rewritten)

    emit("Validating…", 0.88)
    logger.info("[5/6] validating")
    report = validate(resume, rewritten, jd)
    for attempt in range(1, max_regen_passes + 1):
        if report.passed:
            break
        n_crit = sum(1 for i in report.issues if i.severity == "critical")
        logger.warning(
            "validation found %d critical issues; regen pass %d/%d",
            n_crit, attempt, max_regen_passes,
        )
        rewritten = _regenerate_critical_sections(
            original=resume, rewritten=rewritten, report=report,
            plan=plan, jd=jd, client=client,
        )
        _dump_json(log_dir / f"4b_rewritten_pass{attempt}.json", rewritten)
        report = validate(resume, rewritten, jd)
    _dump_json(log_dir / "5_validation_report.json", report)

    if not report.passed:
        logger.error(
            "validation still failing after %d regen passes; shipping anyway",
            max_regen_passes,
        )

    emit("Writing DOCX…", 0.92)
    logger.info("[6/6] writing DOCX")
    docx_out = output_dir / f"{resume_path.stem} (tailored).docx"
    write_resume(resume_path, rewritten, docx_out)

    pdf_out: Optional[Path] = None
    if not skip_pdf:
        emit("Exporting PDF…", 0.96)
        pdf_out = docx_out.with_suffix(".pdf")
        try:
            export_pdf(docx_out, pdf_out, backend=pdf_backend)  # type: ignore[arg-type]
            logger.info("PDF saved to %s", pdf_out)
        except Exception as e:  # noqa: BLE001
            logger.error("PDF export failed: %s", e)
            pdf_out = None

    groq_summary = client.summary()
    (log_dir / "groq_usage.json").write_text(
        json.dumps(groq_summary, indent=2), encoding="utf-8"
    )

    emit("Done", 1.0)
    return TailorResult(
        docx_path=docx_out,
        pdf_path=pdf_out,
        report=report,
        log_dir=log_dir,
        groq_summary=groq_summary,
        original_resume=resume,
        rewritten_resume=rewritten,
    )


# =========================================================================
# Regeneration of failed sections
# =========================================================================


_EXP_SECTION_RE = re.compile(r"^experience\[(\d+)\]\.(.+)$")
_BULLET_RE = re.compile(r"^bullets\[(\d+)\]$")


def _regenerate_critical_sections(
    *,
    original: Resume,
    rewritten: Resume,
    report: ValidationReport,
    plan: SubstitutionPlan,
    jd: JDAnalysis,
    client: GroqClient,
) -> Resume:
    """For each critical issue, either re-run the LLM rewriter on the
    affected section (summary / intro / bullet) or revert it to the
    original verbatim (identity fields, education, raw_paragraphs)."""
    new = rewritten.model_copy(deep=True)
    processed: set[str] = set()

    for issue in report.issues:
        if issue.severity != "critical":
            continue
        section = issue.section
        if section in processed:
            continue
        processed.add(section)

        try:
            _apply_remediation(section, original, new, plan, jd, client)
        except Exception as e:  # noqa: BLE001
            logger.error("regeneration failed for section %s: %s", section, e)

    return new


def _apply_remediation(
    section: str,
    original: Resume,
    new: Resume,
    plan: SubstitutionPlan,
    jd: JDAnalysis,
    client: GroqClient,
) -> None:
    if section == "summary":
        new.summary.text = _rewrite_summary(original.summary.text, plan, jd, client)
        return

    if section.startswith("header."):
        new.header = original.header.model_copy(deep=True)
        return

    if section == "experience":
        new.experience = [e.model_copy(deep=True) for e in original.experience]
        return

    m = _EXP_SECTION_RE.match(section)
    if m:
        i = int(m.group(1))
        sub = m.group(2)
        if sub == "intro":
            new.experience[i].intro = _rewrite_intro(original.experience[i], plan, jd, client)
            return
        bm = _BULLET_RE.match(sub)
        if bm:
            j = int(bm.group(1))
            new.experience[i].bullets[j] = _rewrite_bullet(
                original.experience[i].bullets[j],
                original.experience[i], plan, jd, client,
            )
            return
        if sub in ("company", "title", "dates", "location", "bullets"):
            # Identity field or count drift — revert wholesale.
            setattr(new.experience[i], sub, getattr(original.experience[i], sub))
            return
        return

    if section == "education" or section.startswith("education["):
        new.education = [e.model_copy(deep=True) for e in original.education]
        return

    if section == "skills_section":
        new.skills_section = original.skills_section.model_copy(deep=True)
        return

    if section == "raw_paragraphs":
        new.raw_paragraphs = list(original.raw_paragraphs)
        return


# =========================================================================
# JSON logging helpers
# =========================================================================


def _dump_json(path: Path, obj: Any) -> None:
    if hasattr(obj, "model_dump"):
        data = obj.model_dump()
    else:
        data = obj
    path.write_text(
        json.dumps(data, indent=2, ensure_ascii=False, default=str),
        encoding="utf-8",
    )
