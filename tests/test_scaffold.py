"""Smoke test: every package imports cleanly."""

import importlib


def test_package_imports() -> None:
    for module in (
        "src",
        "src.main",
        "src.pipeline",
        "src.stages.jd_analyzer",
        "src.stages.resume_parser",
        "src.stages.holistic_rewriter",
        "src.stages.tiers",
        "src.stages.validator",
        "src.llm.client",
        "src.models.schemas",
        "src.docx.reader",
        "src.docx.writer",
        "src.docx.pdf_export",
    ):
        importlib.import_module(module)
