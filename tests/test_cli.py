"""CLI tests using typer's CliRunner. Groq calls and pipeline internals
are mocked so the suite does not require an API key or network."""

import json
from pathlib import Path
from unittest.mock import MagicMock

import pytest
from typer.testing import CliRunner

import src.main as main_module
from src.main import app
from src.models.schemas import JDAnalysis, ValidationReport


runner = CliRunner()


# ---------- helpers ----------


def _stub_client() -> MagicMock:
    c = MagicMock()
    c.summary.return_value = {
        "model": "test", "total_calls": 0, "successful_calls": 0,
        "failed_calls": 0, "total_retries": 0, "total_prompt_tokens": 0,
        "total_completion_tokens": 0, "total_tokens": 0,
        "avg_latency_ms": 0.0, "calls_per_key": {},
    }
    c.format_summary.return_value = "groq summary stub"
    return c


# ---------- backends ----------


def test_backends_lists_available(monkeypatch) -> None:
    monkeypatch.setattr(main_module, "available_backends", lambda: ["docx2pdf", "libreoffice"])
    result = runner.invoke(app, ["backends"])
    assert result.exit_code == 0
    assert "docx2pdf" in result.stdout
    assert "libreoffice" in result.stdout


def test_backends_exits_nonzero_when_none_available(monkeypatch) -> None:
    monkeypatch.setattr(main_module, "available_backends", lambda: [])
    result = runner.invoke(app, ["backends"])
    assert result.exit_code == 1


# ---------- analyze-jd ----------


def test_analyze_jd_prints_json(monkeypatch, tmp_path: Path) -> None:
    jd_file = tmp_path / "jd.md"
    jd_file.write_text("Senior backend engineer with 5+ years of Python and Kafka.")

    monkeypatch.setattr(main_module.GroqClient, "from_env", classmethod(lambda cls, **kw: _stub_client()))
    monkeypatch.setattr(
        main_module, "analyze_jd",
        lambda text, client: JDAnalysis(
            must_have=[{"tech": "Python", "category": "language", "evidence": "5+ years"}],
            seniority_level="senior",
        ),
    )

    result = runner.invoke(app, ["analyze-jd", "--jd-file", str(jd_file)])
    assert result.exit_code == 0
    payload = json.loads(result.stdout)
    assert payload["seniority_level"] == "senior"
    assert payload["must_have"][0]["tech"] == "Python"


def test_analyze_jd_fails_on_missing_file() -> None:
    result = runner.invoke(app, ["analyze-jd", "--jd-file", "no_such_file.md"])
    assert result.exit_code != 0


# ---------- validate ----------


def _write_dummy_docx(path: Path) -> None:
    # CliRunner's `exists=True` check needs a real file; the parse_resume
    # mock intercepts the actual read.
    path.write_bytes(b"not a real docx")


def test_validate_command_runs(monkeypatch, tmp_path: Path) -> None:
    orig = tmp_path / "orig.docx"
    tail = tmp_path / "tail.docx"
    _write_dummy_docx(orig)
    _write_dummy_docx(tail)

    parsed = MagicMock()
    monkeypatch.setattr(main_module, "parse_resume", lambda p: parsed)
    monkeypatch.setattr(
        main_module, "validate",
        lambda o, t, jd: ValidationReport(passed=True, keyword_match_rate=0.9, issues=[]),
    )

    result = runner.invoke(app, ["validate", "--original", str(orig), "--tailored", str(tail)])
    assert result.exit_code == 0
    payload = json.loads(result.stdout)
    assert payload["passed"] is True


def test_validate_command_exits_nonzero_on_failure(monkeypatch, tmp_path: Path) -> None:
    orig = tmp_path / "orig.docx"
    tail = tmp_path / "tail.docx"
    _write_dummy_docx(orig)
    _write_dummy_docx(tail)

    monkeypatch.setattr(main_module, "parse_resume", lambda p: MagicMock())
    monkeypatch.setattr(
        main_module, "validate",
        lambda o, t, jd: ValidationReport(
            passed=False, keyword_match_rate=0.5, issues=[],
        ),
    )

    result = runner.invoke(app, ["validate", "--original", str(orig), "--tailored", str(tail)])
    assert result.exit_code == 1


# ---------- tailor ----------


def test_tailor_command_invokes_pipeline(monkeypatch, tmp_path: Path) -> None:
    # Build a fake profile dir with a .docx so _resolve_resume_path finds it.
    profile_dir = tmp_path / "profiles" / "test_profile"
    profile_dir.mkdir(parents=True)
    resume_doc = profile_dir / "resume.docx"
    resume_doc.write_bytes(b"fake docx")
    jd_file = tmp_path / "jd.md"
    jd_file.write_text("Senior backend role")

    monkeypatch.chdir(tmp_path)
    monkeypatch.setattr(main_module.GroqClient, "from_env", classmethod(lambda cls, **kw: _stub_client()))

    fake_result = MagicMock()
    fake_result.docx_path = tmp_path / "out.docx"
    fake_result.pdf_path = tmp_path / "out.pdf"
    fake_result.log_dir = tmp_path / "logs"
    fake_result.report = ValidationReport(passed=True, keyword_match_rate=0.9, issues=[])
    fake_result.groq_summary = {"total_calls": 5}
    mock_pipeline = MagicMock(return_value=fake_result)
    monkeypatch.setattr(main_module, "run_tailor_pipeline", mock_pipeline)

    result = runner.invoke(
        app, ["tailor", "--profile", "test_profile", "--jd-file", str(jd_file), "--skip-pdf"]
    )
    assert result.exit_code == 0
    assert mock_pipeline.called
    kwargs = mock_pipeline.call_args.kwargs
    # CLI passes the relative path it resolved from the profile dir.
    assert kwargs["resume_path"].name == "resume.docx"
    assert kwargs["resume_path"].parent.name == "test_profile"
    assert kwargs["skip_pdf"] is True


def test_tailor_command_missing_profile_dir(monkeypatch, tmp_path: Path) -> None:
    jd_file = tmp_path / "jd.md"
    jd_file.write_text("...")
    monkeypatch.chdir(tmp_path)

    result = runner.invoke(
        app, ["tailor", "--profile", "no_such_profile", "--jd-file", str(jd_file)]
    )
    assert result.exit_code != 0
    assert "profile directory" in result.stdout.lower() + result.output.lower() or "not found" in (result.stdout + result.output).lower()


def test_tailor_command_no_docx_in_profile_dir(monkeypatch, tmp_path: Path) -> None:
    profile_dir = tmp_path / "profiles" / "empty"
    profile_dir.mkdir(parents=True)
    jd_file = tmp_path / "jd.md"
    jd_file.write_text("...")
    monkeypatch.chdir(tmp_path)

    result = runner.invoke(
        app, ["tailor", "--profile", "empty", "--jd-file", str(jd_file)]
    )
    assert result.exit_code != 0


def test_tailor_command_exits_nonzero_when_validation_fails(monkeypatch, tmp_path: Path) -> None:
    profile_dir = tmp_path / "profiles" / "p"
    profile_dir.mkdir(parents=True)
    (profile_dir / "r.docx").write_bytes(b"fake")
    jd_file = tmp_path / "jd.md"
    jd_file.write_text("...")
    monkeypatch.chdir(tmp_path)
    monkeypatch.setattr(main_module.GroqClient, "from_env", classmethod(lambda cls, **kw: _stub_client()))

    fake_result = MagicMock()
    fake_result.docx_path = tmp_path / "out.docx"
    fake_result.pdf_path = None
    fake_result.log_dir = tmp_path / "logs"
    fake_result.report = ValidationReport(passed=False, keyword_match_rate=0.4, issues=[])
    fake_result.groq_summary = {}
    monkeypatch.setattr(main_module, "run_tailor_pipeline", lambda **kw: fake_result)

    result = runner.invoke(
        app, ["tailor", "--profile", "p", "--jd-file", str(jd_file), "--skip-pdf"]
    )
    assert result.exit_code == 1
