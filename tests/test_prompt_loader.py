"""Tests for src.llm.prompt_loader."""

import pytest
from jinja2 import UndefinedError

from src.llm.prompt_loader import render


def test_render_jd_analyze_substitutes_jd_text() -> None:
    out = render("jd_analyze.j2", jd_text="Senior Python role at FinTech co.")
    assert "Senior Python role at FinTech co." in out


def test_render_strict_undefined_raises_on_missing_var() -> None:
    with pytest.raises(UndefinedError):
        render("jd_analyze.j2")  # jd_text missing


def test_render_strips_trailing_whitespace() -> None:
    out = render("jd_analyze.j2", jd_text="x")
    assert not out.endswith("\n")
    assert not out.startswith("\n")
