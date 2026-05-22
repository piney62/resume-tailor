"""End-to-end pipeline tests for the v2 hybrid (JD + holistic) flow.
Groq calls are mocked; the real DOCX parser and writer run against a
synthetic resume built in tmp_path."""

import json
from pathlib import Path
from unittest.mock import MagicMock

import pytest
from docx import Document

from src.pipeline import run_tailor_pipeline


# ---------- synthetic resume fixture (small to keep mock-call lists short) ----------


def _make_synthetic_resume(tmp_path: Path) -> Path:
    doc = Document()
    doc.add_paragraph("Alice Example")
    doc.add_paragraph("STAFF ENGINEER")
    doc.add_paragraph("NYC | (555) 010-2020 | alice@example.com | linkedin.com/in/alice")
    doc.add_paragraph("")
    doc.add_paragraph("Staff engineer with 12 years building backend systems.")
    doc.add_paragraph("PROFESSIONAL EXPERIENCE")
    doc.add_paragraph("AcmeCorp | Staff Engineer")
    doc.add_paragraph("Jan 2020 - Present | Remote")
    doc.add_paragraph("Led platform team of 6 engineers.")
    p1 = doc.add_paragraph("Cut p99 latency from 800ms to 200ms.")
    p1.style = doc.styles["List Paragraph"]
    p2 = doc.add_paragraph("Mentored 6 engineers, 2 promoted within 12 months.")
    p2.style = doc.styles["List Paragraph"]
    doc.add_paragraph("Skills: Python, Redis, PostgreSQL")
    doc.add_paragraph("EDUCATION")
    doc.add_paragraph("MIT | BS, Computer Science\n2008 - 2012 | Cambridge, MA")
    doc.add_paragraph("SKILLS")
    sk = doc.add_paragraph("Languages: Python, Go")
    sk.style = doc.styles["List Paragraph"]

    path = tmp_path / "alice.docx"
    doc.save(str(path))
    return path


# ---------- canned LLM responses ----------


JD_ANALYSIS = {
    "must_have": [
        {"tech": "Python", "category": "language", "evidence": "5+ years Python"},
        {"tech": "Kafka", "category": "streaming", "evidence": "build Kafka pipelines"},
    ],
    "nice_to_have": [],
    "soft_skills": [],
    "domain_keywords": ["real-time analytics"],
    "seniority_level": "staff",
    "exact_phrases_to_mirror": ["distributed systems at scale"],
}


def _holistic_response(*, intro="Led platform team of 6 engineers focused on backend systems.",
                       bullet_0="Cut p99 latency from 800ms to 200ms across the request path.",
                       bullet_1="Mentored 6 engineers, 2 promoted within 12 months.",
                       extra_bullets: list[str] | None = None) -> dict:
    bullets = [bullet_0, bullet_1]
    if extra_bullets:
        bullets.extend(extra_bullets)
    return {
        "header_title": "STAFF ENGINEER, BACKEND PLATFORMS",
        "summary_text": "Staff engineer with 12 years building backend systems and distributed pipelines.",
        "experience": [
            {"intro": intro, "bullets": bullets, "skills_line": "Skills: Python, Kafka, PostgreSQL"},
        ],
        "skills_section_categories": {"Languages": ["Python", "Go", "Kafka"]},
    }


def _make_mock_client(responses: list) -> MagicMock:
    client = MagicMock()
    client.complete_json.side_effect = list(responses)
    client.summary.return_value = {
        "model": "test-model", "total_calls": len(responses),
        "successful_calls": len(responses), "failed_calls": 0,
        "total_retries": 0, "total_prompt_tokens": 100,
        "total_completion_tokens": 50, "total_tokens": 150,
        "avg_latency_ms": 200.0, "calls_per_key": {0: len(responses)},
    }
    client.format_summary.return_value = "  Groq usage — model=test-model\n    calls: N"
    return client


# ---------- happy path: exactly 2 LLM calls (JD + holistic) ----------


def test_pipeline_runs_end_to_end_with_pdf_skipped(tmp_path: Path) -> None:
    resume_path = _make_synthetic_resume(tmp_path)
    out_dir = tmp_path / "out"
    client = _make_mock_client([JD_ANALYSIS, _holistic_response()])

    result = run_tailor_pipeline(
        resume_path=resume_path,
        jd_text="Senior backend engineer with Python and Kafka.",
        output_dir=out_dir,
        client=client,
        skip_pdf=True,
    )

    assert result.docx_path.exists()
    assert result.pdf_path is None
    assert result.report.passed is True
    # Exactly 2 LLM calls (JD analyze + holistic rewrite).
    assert client.complete_json.call_count == 2


def test_pipeline_writes_per_stage_json_logs(tmp_path: Path) -> None:
    resume_path = _make_synthetic_resume(tmp_path)
    out_dir = tmp_path / "out"
    client = _make_mock_client([JD_ANALYSIS, _holistic_response()])

    result = run_tailor_pipeline(
        resume_path=resume_path, jd_text="JD text", output_dir=out_dir,
        client=client, skip_pdf=True,
    )

    log_files = {p.name for p in result.log_dir.iterdir()}
    assert "1_resume_parsed.json" in log_files
    assert "2_jd_analysis.json" in log_files
    assert "3a_rewritten_initial.json" in log_files
    assert "4_validation_report.json" in log_files
    assert "groq_usage.json" in log_files

    jd_payload = json.loads((result.log_dir / "2_jd_analysis.json").read_text())
    assert jd_payload["seniority_level"] == "staff"


def test_pipeline_attempts_pdf_when_not_skipped(mocker, tmp_path: Path) -> None:
    resume_path = _make_synthetic_resume(tmp_path)
    out_dir = tmp_path / "out"
    mocker.patch(
        "src.pipeline.export_pdf",
        side_effect=lambda src, dst, **kw: Path(dst).write_bytes(b"%PDF-1.4 fake") or Path(dst),
    )
    client = _make_mock_client([JD_ANALYSIS, _holistic_response()])

    result = run_tailor_pipeline(
        resume_path=resume_path, jd_text="JD", output_dir=out_dir, client=client,
        skip_pdf=False,
    )
    assert result.pdf_path is not None
    assert result.pdf_path.exists()


def test_pipeline_records_pdf_as_none_when_export_fails(mocker, tmp_path: Path) -> None:
    resume_path = _make_synthetic_resume(tmp_path)
    out_dir = tmp_path / "out"
    mocker.patch("src.pipeline.export_pdf", side_effect=RuntimeError("no backend"))
    client = _make_mock_client([JD_ANALYSIS, _holistic_response()])

    result = run_tailor_pipeline(
        resume_path=resume_path, jd_text="JD", output_dir=out_dir, client=client,
        skip_pdf=False,
    )
    assert result.pdf_path is None
    assert result.docx_path.exists()


# ---------- regen loop ----------


def test_pipeline_regenerates_when_number_dropped(tmp_path: Path) -> None:
    """Holistic rewriter's own per-field number guard will revert dropped
    numbers to the original text. The regen loop only kicks in on critical
    issues the validator detects, which the guard prevents in practice. We
    still assert the validation passes."""
    resume_path = _make_synthetic_resume(tmp_path)
    out_dir = tmp_path / "out"
    # First holistic response drops "12" from summary → per-field guard
    # reverts to original. No critical issue, no regen needed.
    response_with_drop = _holistic_response()
    response_with_drop["summary_text"] = "Staff engineer building backend systems."  # no numbers

    client = _make_mock_client([JD_ANALYSIS, response_with_drop])
    result = run_tailor_pipeline(
        resume_path=resume_path, jd_text="JD", output_dir=out_dir, client=client,
        skip_pdf=True, max_regen_passes=2,
    )
    assert result.report.passed is True
    # Number-preservation guard kept the original "12 years".
    assert "12 years" in result.rewritten_resume.summary.text


# ---------- progress callback ----------


def test_pipeline_progress_callback_emits_monotonic_fractions(tmp_path: Path) -> None:
    resume_path = _make_synthetic_resume(tmp_path)
    out_dir = tmp_path / "out"
    client = _make_mock_client([JD_ANALYSIS, _holistic_response()])
    events: list[tuple[str, float]] = []

    run_tailor_pipeline(
        resume_path=resume_path, jd_text="JD", output_dir=out_dir,
        client=client, skip_pdf=True,
        progress_cb=lambda label, frac: events.append((label, frac)),
    )

    assert events
    fractions = [f for _, f in events]
    assert all(0.0 <= f <= 1.0 for f in fractions)
    assert events[-1][1] == 1.0
    assert fractions == sorted(fractions)


# ---------- groq summary surfaced ----------


def test_pipeline_returns_groq_summary_and_writes_json(tmp_path: Path) -> None:
    resume_path = _make_synthetic_resume(tmp_path)
    out_dir = tmp_path / "out"
    client = _make_mock_client([JD_ANALYSIS, _holistic_response()])
    result = run_tailor_pipeline(
        resume_path=resume_path, jd_text="JD", output_dir=out_dir, client=client,
        skip_pdf=True,
    )
    assert result.groq_summary["total_calls"] == 2
    usage_json = json.loads((result.log_dir / "groq_usage.json").read_text())
    assert usage_json["total_calls"] == 2
