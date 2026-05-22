"""Low-level .docx paragraph reader.

Returns a flat list of ParagraphInfo. The Resume parser (Step 5) and
DOCX writer (Step 9) both work off these zero-based paragraph indices.
"""

from dataclasses import dataclass
from pathlib import Path

from docx import Document


@dataclass(frozen=True)
class ParagraphInfo:
    idx: int
    text: str
    style: str


def read_paragraphs(path: Path | str) -> list[ParagraphInfo]:
    doc = Document(str(path))
    out: list[ParagraphInfo] = []
    for i, p in enumerate(doc.paragraphs):
        style = p.style.name if p.style is not None else ""
        out.append(ParagraphInfo(idx=i, text=p.text, style=style))
    return out
