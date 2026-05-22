"""CLI entrypoint.

Three commands:
  resume-tailor tailor      — full pipeline (JD analysis → tailored .docx + .pdf)
  resume-tailor analyze-jd  — JD analysis only, prints JSON
  resume-tailor validate    — diff a tailored .docx against the original
  resume-tailor backends    — list PDF backends visible on this machine
"""

import logging
import os
import sys
from pathlib import Path
from typing import Optional

import typer
from dotenv import load_dotenv

from src.docx.pdf_export import available_backends
from src.llm.client import GroqClient
from src.models.schemas import JDAnalysis
from src.pipeline import run_tailor_pipeline
from src.stages.jd_analyzer import analyze_jd
from src.stages.resume_parser import parse_resume
from src.stages.validator import validate

app = typer.Typer(
    name="resume-tailor",
    help="Tailor a .docx resume to a JD via a multi-stage LLM pipeline.",
    no_args_is_help=True,
    add_completion=False,
)


def _setup_logging() -> None:
    load_dotenv()
    level = os.environ.get("LOG_LEVEL", "INFO").upper()
    logging.basicConfig(
        level=level,
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
        datefmt="%H:%M:%S",
    )


def _resolve_resume_path(profile: str) -> Path:
    profile_dir = Path("profiles") / profile
    if not profile_dir.exists():
        raise typer.BadParameter(f"profile directory not found: {profile_dir}")
    candidates = sorted(profile_dir.glob("*.docx"))
    candidates = [c for c in candidates if not c.name.startswith("~$")]
    if not candidates:
        raise typer.BadParameter(f"no .docx file found in {profile_dir}")
    if len(candidates) > 1:
        typer.echo(
            f"WARNING: multiple .docx files in {profile_dir}; using {candidates[0].name}",
            err=True,
        )
    return candidates[0]


# =========================================================================
# tailor
# =========================================================================


@app.command()
def tailor(
    profile: str = typer.Option(
        ..., "--profile", help="Profile name (subdirectory under profiles/)"
    ),
    jd_file: Path = typer.Option(
        ..., "--jd-file", exists=True, dir_okay=False,
        help="Path to JD text/markdown file",
    ),
    output: Optional[Path] = typer.Option(
        None, "--output",
        help="Output directory. Defaults to outputs/{profile}/",
    ),
    pdf_backend: str = typer.Option(
        "auto", "--pdf-backend",
        help="PDF conversion backend: auto | docx2pdf | libreoffice",
    ),
    skip_pdf: bool = typer.Option(False, "--skip-pdf", help="Skip PDF export"),
    max_regen: int = typer.Option(
        2, "--max-regen",
        help="Max regeneration passes after validation finds critical issues",
    ),
) -> None:
    """Run the full tailoring pipeline and emit a tailored .docx and .pdf."""
    _setup_logging()

    resume_path = _resolve_resume_path(profile)
    out_dir = output or (Path("outputs") / profile)
    jd_text = jd_file.read_text(encoding="utf-8")

    client = GroqClient.from_env()
    result = run_tailor_pipeline(
        resume_path=resume_path,
        jd_text=jd_text,
        output_dir=out_dir,
        client=client,
        pdf_backend=pdf_backend,
        skip_pdf=skip_pdf,
        max_regen_passes=max_regen,
    )

    status = "PASSED" if result.report.passed else "FAILED"
    typer.echo("")
    typer.echo(f"  DOCX:       {result.docx_path}")
    if result.pdf_path:
        typer.echo(f"  PDF:        {result.pdf_path}")
    else:
        typer.echo(f"  PDF:        (skipped or failed)")
    typer.echo(f"  Logs:       {result.log_dir}")
    typer.echo(
        f"  Validation: {status}  "
        f"({len(result.report.issues)} issues, "
        f"keyword_match={result.report.keyword_match_rate:.0%})"
    )
    typer.echo("")
    typer.echo(client.format_summary())

    if not result.report.passed:
        raise typer.Exit(code=1)


# =========================================================================
# analyze-jd
# =========================================================================


@app.command("analyze-jd")
def analyze_jd_cmd(
    jd_file: Path = typer.Option(..., "--jd-file", exists=True, dir_okay=False),
) -> None:
    """Run only JD analysis; print the JDAnalysis JSON."""
    _setup_logging()
    client = GroqClient.from_env()
    jd_text = jd_file.read_text(encoding="utf-8")
    result = analyze_jd(jd_text, client)
    typer.echo(result.model_dump_json(indent=2))


# =========================================================================
# validate
# =========================================================================


@app.command("validate")
def validate_cmd(
    original: Path = typer.Option(..., "--original", exists=True, dir_okay=False),
    tailored: Path = typer.Option(..., "--tailored", exists=True, dir_okay=False),
    jd_file: Optional[Path] = typer.Option(
        None, "--jd-file",
        help="Optional JD for keyword_match_rate computation",
    ),
) -> None:
    """Diff a tailored .docx against the original and print a ValidationReport."""
    _setup_logging()
    orig = parse_resume(original)
    tail = parse_resume(tailored)

    if jd_file:
        client = GroqClient.from_env()
        jd = analyze_jd(jd_file.read_text(encoding="utf-8"), client)
    else:
        jd = JDAnalysis(seniority_level="senior")

    report = validate(orig, tail, jd)
    typer.echo(report.model_dump_json(indent=2))
    if not report.passed:
        raise typer.Exit(code=1)


# =========================================================================
# backends
# =========================================================================


@app.command("backends")
def backends_cmd() -> None:
    """List PDF backends visible on this machine."""
    backends = available_backends()
    if not backends:
        typer.echo("No PDF backends found.")
        typer.echo("Install MS Word (for docx2pdf) or LibreOffice.")
        raise typer.Exit(code=1)
    typer.echo("Available PDF backends (in preference order):")
    for b in backends:
        typer.echo(f"  - {b}")


if __name__ == "__main__":
    app()
