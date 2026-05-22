"""Stage 6: DOCX Writer — format-preserving edits via python-docx.

Opens the ORIGINAL .docx and edits only the paragraphs the Rewriter
changed. The document chrome (headers, footers, section breaks,
margins, page size, styles) is never regenerated — we only rewrite
paragraph text content using the indices captured by the Resume parser.

Run-level handling: each python-docx paragraph is a sequence of `runs`
(contiguous character ranges with identical formatting — bold, italic,
font, size, color). When we replace a paragraph's text, we put the new
text into run #0 and blank every subsequent run. Outcomes:
  - Uniform-formatted paragraphs (the vast majority of resume bullets,
    intros, and summary text): formatting fully preserved.
  - Paragraphs with mid-paragraph formatting changes (e.g., a bold
    "Skills:" prefix in front of plain items): the entire new text
    inherits run #0's formatting. Acceptable trade-off.

Why we compare against raw_paragraphs before writing: rewriting a run
always destroys the original run structure. For paragraphs whose text
is UNCHANGED from the source (the rewriter's number-preservation guard
falls back to the original verbatim more often than not), we skip the
write entirely so mid-paragraph bold/italic survives untouched.

Skills section growth: if the Rewriter added a new category and the
section therefore has more raw_lines than the source had paragraphs,
the surplus items are appended inline to the last existing category's
line. The .docx paragraph count stays stable.
"""

from pathlib import Path

from docx import Document

from src.models.schemas import Experience, Resume


def write_resume(source_path: Path | str, rewritten: Resume, output_path: Path | str) -> None:
    """Open `source_path`, apply edits from `rewritten`, save to `output_path`."""
    doc = Document(str(source_path))
    paragraphs = doc.paragraphs
    raw = rewritten.raw_paragraphs

    _write_header(paragraphs, rewritten, raw)
    _write_summary(paragraphs, rewritten, raw)
    for role in rewritten.experience:
        _write_role(paragraphs, role, raw)
    _write_skills_section(paragraphs, rewritten, raw)

    doc.save(str(output_path))


# ---------- per-section writers ----------


def _write_header(paragraphs, resume: Resume, raw: list[str]) -> None:
    h = resume.header
    if h.indices.title_idx is not None and h.title is not None:
        _maybe_set(paragraphs, h.indices.title_idx, h.title, raw)


def _write_summary(paragraphs, resume: Resume, raw: list[str]) -> None:
    idxs = resume.summary.paragraph_idxs
    if not idxs:
        return
    # Collapse multi-paragraph summaries into the first paragraph and clear
    # the rest. Validator's raw_paragraph_count check keeps the structural
    # paragraph count stable; we only blank text, never delete paragraphs.
    _maybe_set(paragraphs, idxs[0], resume.summary.text, raw)
    for idx in idxs[1:]:
        _maybe_set(paragraphs, idx, "", raw)


def _write_role(paragraphs, role: Experience, raw: list[str]) -> None:
    if role.indices.intro_idx is not None and role.intro:
        _maybe_set(paragraphs, role.indices.intro_idx, role.intro, raw)

    if len(role.bullets) != len(role.indices.bullet_idxs):
        raise ValueError(
            f"bullet count mismatch for role {role.company!r}: "
            f"{len(role.bullets)} bullets vs {len(role.indices.bullet_idxs)} paragraph indices"
        )
    for text, idx in zip(role.bullets, role.indices.bullet_idxs):
        _maybe_set(paragraphs, idx, text, raw)

    if role.indices.skills_line_idx is not None and role.skills_line is not None:
        _maybe_set(paragraphs, role.indices.skills_line_idx, role.skills_line, raw)


def _write_skills_section(paragraphs, resume: Resume, raw: list[str]) -> None:
    raw_lines = resume.skills_section.raw_lines
    content_idxs = resume.skills_section.indices.content_idxs
    n = len(content_idxs)

    if not content_idxs:
        return

    if len(raw_lines) <= n:
        for i, idx in enumerate(content_idxs):
            text = raw_lines[i] if i < len(raw_lines) else ""
            _maybe_set(paragraphs, idx, text, raw)
        return

    # raw_lines > content_idxs: a new category was added. Write the first
    # (n-1) lines as-is, then append every surplus line's items (the
    # substring after its "Category:" prefix) to the last paragraph.
    for i in range(n - 1):
        _maybe_set(paragraphs, content_idxs[i], raw_lines[i], raw)

    last_line = raw_lines[n - 1]
    for surplus in raw_lines[n:]:
        _, sep, items = surplus.partition(":")
        items = items.strip() if sep else surplus.strip()
        if items:
            last_line = f"{last_line}, {items}"
    _maybe_set(paragraphs, content_idxs[-1], last_line, raw)


# ---------- run-level text replacement ----------


def _maybe_set(paragraphs, idx: int, new_text: str, raw: list[str]) -> None:
    """Write `new_text` into paragraph[idx] only if it differs from the
    original paragraph text. Unchanged paragraphs are left untouched so
    mid-paragraph formatting (e.g. bold spans) survives."""
    if 0 <= idx < len(raw) and new_text == raw[idx]:
        return
    _set_paragraph_text(paragraphs[idx], new_text)


def _set_paragraph_text(paragraph, new_text: str) -> None:
    """Replace paragraph text while preserving the first run's formatting.

    Newlines in `new_text` are collapsed to single spaces — python-docx
    does not translate "\n" inside `run.text` into a soft line break.
    """
    normalized = (new_text or "").replace("\r\n", " ").replace("\n", " ")

    runs = paragraph.runs
    if not runs:
        if normalized:
            paragraph.add_run(normalized)
        return

    runs[0].text = normalized
    for run in runs[1:]:
        run.text = ""
