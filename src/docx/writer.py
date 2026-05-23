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

from copy import deepcopy
from pathlib import Path

from docx import Document
from docx.oxml import OxmlElement
from docx.oxml.ns import qn

from src.models.schemas import Experience, Resume

# Style applied to inserted new bullets. Matches what the resume parser
# expects from a List Paragraph in the source .docx.
_BULLET_STYLE_NAME = "List Paragraph"


def write_resume(source_path: Path | str, rewritten: Resume, output_path: Path | str) -> None:
    """Open `source_path`, apply edits from `rewritten`, save to `output_path`.

    Order matters:
      1. All index-based writes go FIRST, using the paragraph list captured
         from the source. This includes header, summary, every role's
         intro/existing-bullets/skills_line, and the skills section.
      2. New-bullet insertions for the recent role go LAST. Inserting earlier
         would shift downstream paragraph indices and invalidate the writes
         that follow.
    """
    doc = Document(str(source_path))
    paragraphs = doc.paragraphs
    raw = rewritten.raw_paragraphs

    _write_header(paragraphs, rewritten, raw)
    _write_summary(paragraphs, rewritten, raw)
    for role in rewritten.experience:
        _write_role(paragraphs, role, raw)
    _write_skills_section(paragraphs, rewritten, raw)

    # Insertions last so the index-based writes above always saw the original
    # paragraph layout.
    _insert_new_bullets_for_recent(doc, rewritten)

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

    # Existing bullets are written by index; any extras beyond the original
    # bullet_idxs length are handled by _insert_new_bullets_for_recent.
    n_existing = len(role.indices.bullet_idxs)
    if len(role.bullets) < n_existing:
        raise ValueError(
            f"bullet count mismatch for role {role.company!r}: "
            f"{len(role.bullets)} bullets vs {n_existing} paragraph indices "
            f"(rewriter dropped bullets — not supported)"
        )
    for text, idx in zip(role.bullets, role.indices.bullet_idxs):
        _maybe_set(paragraphs, idx, text, raw)

    if role.indices.skills_line_idx is not None and role.skills_line is not None:
        _maybe_set(paragraphs, role.indices.skills_line_idx, role.skills_line, raw)


def _insert_new_bullets_for_recent(doc, resume: Resume) -> None:
    """Insert new bullets in the most-recent role if the rewriter added any.

    The new bullets come from `role.bullets[len(role.indices.bullet_idxs):]`
    — i.e. anything beyond the count captured at parse time. We only allow
    this for the recent role (index 0); see Validator's tier check.

    Insertion point: immediately BEFORE the paragraph that follows the last
    existing bullet of the recent role. We prefer the skills_line if present;
    otherwise the next role's first header paragraph; otherwise append at
    the end of the document.
    """
    if not resume.experience:
        return
    recent = resume.experience[0]
    n_existing = len(recent.indices.bullet_idxs)
    extras = list(recent.bullets[n_existing:])
    if not extras:
        return

    paragraphs = doc.paragraphs

    anchor_idx: int | None = None
    if recent.indices.skills_line_idx is not None:
        anchor_idx = recent.indices.skills_line_idx
    elif len(resume.experience) > 1:
        next_role = resume.experience[1]
        if next_role.indices.header_idxs:
            anchor_idx = next_role.indices.header_idxs[0]

    if anchor_idx is None or anchor_idx >= len(paragraphs):
        # Fall back: append at the end of the document.
        for text in extras:
            p = doc.add_paragraph(text)
            _try_set_style(p, _BULLET_STYLE_NAME)
        return

    # Use the last existing bullet as a formatting template so the inserted
    # paragraphs pick up the same numPr (bullet symbol + list definition).
    last_bullet_idx = recent.indices.bullet_idxs[-1] if recent.indices.bullet_idxs else None
    template_para = paragraphs[last_bullet_idx] if last_bullet_idx is not None else None

    anchor = paragraphs[anchor_idx]
    for text in extras:
        if template_para is not None:
            _insert_cloned_bullet(anchor._p, template_para, text)
        else:
            new_para = anchor.insert_paragraph_before(text)
            _try_set_style(new_para, _BULLET_STYLE_NAME)


def _try_set_style(paragraph, style_name: str) -> None:
    try:
        paragraph.style = paragraph.part.document.styles[style_name]
    except KeyError:
        pass


def _insert_cloned_bullet(anchor_p, template_para, text: str) -> None:
    """Insert a new bullet paragraph immediately before anchor_p.

    Deep-clones the template paragraph's XML so the inserted paragraph
    inherits the full pPr (numPr, spacing, paragraph-level rPr giving the
    correct font/size/color) and uses a new run whose rPr is copied from
    the template's first run. This ensures the inserted bullet is visually
    identical to the surrounding bullets.
    """
    XML_SPACE = "{http://www.w3.org/XML/1998/namespace}space"

    # Clone the full paragraph element (pPr intact — numPr, style, spacing…).
    new_p = deepcopy(template_para._p)

    # Strip existing text content; preserve pPr.
    for r in new_p.findall(qn("w:r")):
        new_p.remove(r)
    for hl in new_p.findall(qn("w:hyperlink")):
        new_p.remove(hl)

    # Build a new run with the same rPr as the template's first run.
    new_r = OxmlElement("w:r")
    template_runs = template_para._p.findall(qn("w:r"))
    if template_runs:
        src_rPr = template_runs[0].find(qn("w:rPr"))
        if src_rPr is not None:
            new_r.append(deepcopy(src_rPr))

    t = OxmlElement("w:t")
    t.text = text
    if text and (text[0] == " " or text[-1] == " "):
        t.set(XML_SPACE, "preserve")
    new_r.append(t)
    new_p.append(new_r)

    anchor_p.addprevious(new_p)


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
