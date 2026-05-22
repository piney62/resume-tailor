"""Tests for src.llm.few_shot loader."""

import pytest

from src.llm.few_shot import load_examples


def test_summary_rewrite_examples_load() -> None:
    examples = load_examples("summary_rewrite")
    assert len(examples) >= 1
    for ex in examples:
        assert "original" in ex
        assert "direction" in ex
        assert "rewritten" in ex


def test_intro_rewrite_examples_load() -> None:
    assert load_examples("intro_rewrite")


def test_bullet_rewrite_examples_load() -> None:
    examples = load_examples("bullet_rewrite")
    assert len(examples) >= 2
    # At least one example preserves numbers verbatim.
    assert any("10M" in ex["rewritten"] for ex in examples if "10M" in ex.get("original", ""))


def test_unknown_name_returns_empty_list() -> None:
    assert load_examples("does_not_exist") == []
