"""Pipeline orchestration (v2: hybrid two-call flow).

Stages:
  1. Parse resume                  (no LLM)
  2. Analyze JD                    (LLM call #1)
  3. Holistic rewrite              (LLM call #2) — recency-weighted
  4. Validate                      (rule-based)
     → on critical issues: re-call holistic_rewrite with prior issues as
       context, up to `max_regen_passes` times.
  5. Write DOCX + export PDF       (no LLM)

Each stage's input/output is dumped as JSON under
{output_dir}/logs/{timestamp}/ for replay / debugging.
"""

import json
import logging
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any, Callable, Optional

from src.docx.pdf_export import export_pdf
from src.docx.writer import write_resume
from src.llm.client import GroqClient
from src.models.schemas import Resume, ValidationReport
from src.stages.holistic_rewriter import holistic_rewrite
from src.stages.jd_analyzer import analyze_jd
from src.stages.resume_parser import parse_resume
from src.stages.tiers import classify_tiers
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
    original_resume: Optional[Resume] = None
    rewritten_resume: Optional[Resume] = None


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

    emit("Parsing resume…", 0.05)
    logger.info("[1/5] parsing resume")
    resume = parse_resume(resume_path)
    _dump_json(log_dir / "1_resume_parsed.json", resume)

    tiers = classify_tiers(resume.experience)
    logger.info(
        "role tiers: %s",
        [(r.company, t) for r, t in zip(resume.experience, tiers)],
    )

    emit("Analyzing JD…", 0.15)
    logger.info("[2/5] analyzing JD")
    jd = analyze_jd(jd_text, client)
    _dump_json(log_dir / "2_jd_analysis.json", jd)

    emit("Holistic rewrite (initial pass)…", 0.30)
    logger.info("[3/5] holistic rewrite (initial pass)")
    rewritten = holistic_rewrite(resume, jd, client)
    _dump_json(log_dir / "3a_rewritten_initial.json", rewritten)

    emit("Validating…", 0.80)
    logger.info("[4/5] validating")
    report = validate(resume, rewritten, jd)
    for attempt in range(1, max_regen_passes + 1):
        if report.passed:
            break
        critical = [i for i in report.issues if i.severity == "critical"]
        logger.warning(
            "validation found %d critical issues; regen pass %d/%d",
            len(critical), attempt, max_regen_passes,
        )
        emit(f"Regenerating (pass {attempt}/{max_regen_passes})…", 0.80 + 0.05 * attempt)
        rewritten = holistic_rewrite(resume, jd, client, prior_issues=critical)
        _dump_json(log_dir / f"3b_rewritten_pass{attempt}.json", rewritten)
        report = validate(resume, rewritten, jd)
    _dump_json(log_dir / "4_validation_report.json", report)

    if not report.passed:
        logger.error(
            "validation still failing after %d regen passes; shipping anyway",
            max_regen_passes,
        )

    emit("Writing DOCX…", 0.92)
    logger.info("[5/5] writing DOCX")
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


def _dump_json(path: Path, obj: Any) -> None:
    if hasattr(obj, "model_dump"):
        data = obj.model_dump()
    else:
        data = obj
    path.write_text(
        json.dumps(data, indent=2, ensure_ascii=False, default=str),
        encoding="utf-8",
    )
