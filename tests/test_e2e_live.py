"""Live end-to-end test against the real Groq API.

This test is gated behind the RUN_LIVE_TESTS env var because it costs
tokens and depends on external network. To run:

    set RUN_LIVE_TESTS=1
    pytest tests/test_e2e_live.py -v -s

It uses the real sample resume and sample JD and exercises the entire
pipeline (analyze → parse → plan → rewrite → validate → write DOCX).
PDF export is skipped so the test does not require LibreOffice/Word.
"""

import os
from pathlib import Path

import pytest
from dotenv import load_dotenv

from src.llm.client import GroqClient
from src.pipeline import run_tailor_pipeline


pytestmark = pytest.mark.skipif(
    os.environ.get("RUN_LIVE_TESTS", "0") != "1",
    reason="Live e2e tests are opt-in. Set RUN_LIVE_TESTS=1 to enable.",
)

PROJECT_ROOT = Path(__file__).parent.parent
SAMPLE_RESUME = PROJECT_ROOT / "profiles" / "sample" / "Katharine Berry.docx"
SAMPLE_JD = PROJECT_ROOT / "jd-archive" / "sample-developer-platform.md"


def test_live_e2e_tailor_pipeline(tmp_path: Path) -> None:
    if not SAMPLE_RESUME.exists():
        pytest.skip(f"sample resume missing at {SAMPLE_RESUME}")
    if not SAMPLE_JD.exists():
        pytest.skip(f"sample JD missing at {SAMPLE_JD}")

    load_dotenv(PROJECT_ROOT / ".env")
    if not os.environ.get("GROQ_API_KEYS"):
        pytest.skip("GROQ_API_KEYS not set")

    client = GroqClient.from_env()
    jd_text = SAMPLE_JD.read_text(encoding="utf-8")

    result = run_tailor_pipeline(
        resume_path=SAMPLE_RESUME,
        jd_text=jd_text,
        output_dir=tmp_path,
        client=client,
        skip_pdf=True,
        max_regen_passes=2,
    )

    # Assertions on outputs
    assert result.docx_path.exists(), "tailored .docx not produced"
    assert result.docx_path.stat().st_size > 10_000, "tailored .docx is suspiciously small"

    # Validation should pass — number-preservation guard + revert logic in
    # the regen loop should keep identity fields intact.
    critical = [i for i in result.report.issues if i.severity == "critical"]
    assert not critical, (
        f"live pipeline produced critical validation issues: "
        + "\n".join(f"{i.section}: {i.issue}" for i in critical)
    )

    # Keyword match rate should be reasonable for an aligned JD.
    assert result.report.keyword_match_rate >= 0.3, (
        f"keyword_match_rate too low: {result.report.keyword_match_rate}. "
        f"Either the JD/resume are poorly aligned or the rewriter is "
        f"dropping JD terms."
    )

    # Print a summary so `pytest -s` shows useful info.
    print("\n=== Live e2e summary ===")
    print(f"DOCX:         {result.docx_path}")
    print(f"Validation:   {'PASSED' if result.report.passed else 'FAILED'}")
    print(f"  issues:     {len(result.report.issues)} total, {len(critical)} critical")
    print(f"  kw match:   {result.report.keyword_match_rate:.0%}")
    print(client.format_summary())
