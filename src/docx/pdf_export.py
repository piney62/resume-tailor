"""Stage 7: PDF Export.

Two backends, with auto-detection:
  - docx2pdf  — uses Microsoft Word (Windows COM or macOS AppleScript).
                Highest fidelity; requires Word installed.
  - LibreOffice (headless) — cross-platform fallback. Invokes
                `soffice --headless --convert-to pdf`. Slightly lower
                fidelity but works on Linux and on Windows without Word.

Strategy: in `backend="auto"` mode we try docx2pdf first (best fidelity
when Word is present), then fall back to LibreOffice. The user can pin
either backend explicitly. A clear actionable error is raised if neither
is available.
"""

import logging
import shutil
import subprocess
from pathlib import Path
from typing import Literal

logger = logging.getLogger(__name__)

Backend = Literal["auto", "docx2pdf", "libreoffice"]

_DEFAULT_TIMEOUT_S = 120


def export_pdf(
    docx_path: Path | str,
    output_path: Path | str,
    *,
    backend: Backend = "auto",
    timeout: int = _DEFAULT_TIMEOUT_S,
) -> Path:
    """Convert `docx_path` to a PDF at `output_path`. Returns output_path."""
    docx_path = Path(docx_path)
    output_path = Path(output_path)
    if not docx_path.exists():
        raise FileNotFoundError(f"source .docx not found: {docx_path}")
    if output_path.suffix.lower() != ".pdf":
        raise ValueError(f"output_path must end in .pdf, got: {output_path}")
    output_path.parent.mkdir(parents=True, exist_ok=True)

    if backend == "auto":
        return _export_auto(docx_path, output_path, timeout)
    if backend == "docx2pdf":
        return _export_docx2pdf(docx_path, output_path)
    if backend == "libreoffice":
        return _export_libreoffice(docx_path, output_path, timeout)
    raise ValueError(f"unknown backend: {backend!r}")


def available_backends() -> list[Backend]:
    """Best-effort introspection of which backends could be used right now.
    Returns the list in preference order."""
    out: list[Backend] = []
    if _docx2pdf_importable():
        out.append("docx2pdf")
    if _find_soffice() is not None:
        out.append("libreoffice")
    return out


# ---------- auto-detect ----------


def _export_auto(docx_path: Path, output_path: Path, timeout: int) -> Path:
    errors: list[str] = []

    if _docx2pdf_importable():
        try:
            return _export_docx2pdf(docx_path, output_path)
        except Exception as e:  # noqa: BLE001 — broad: COM errors come in many shapes
            errors.append(f"docx2pdf: {type(e).__name__}: {e}")
            logger.info("docx2pdf failed, will try LibreOffice. %s", errors[-1])
    else:
        errors.append("docx2pdf: not importable")

    if _find_soffice() is not None:
        try:
            return _export_libreoffice(docx_path, output_path, timeout)
        except Exception as e:  # noqa: BLE001
            errors.append(f"libreoffice: {type(e).__name__}: {e}")
    else:
        errors.append("libreoffice: soffice not found on PATH or in standard locations")

    raise RuntimeError(
        "No PDF backend produced a result. Tried:\n  - "
        + "\n  - ".join(errors)
        + "\nInstall Microsoft Word (for docx2pdf) or LibreOffice "
        "(https://www.libreoffice.org/) and retry."
    )


# ---------- docx2pdf backend ----------


def _docx2pdf_importable() -> bool:
    try:
        import docx2pdf  # noqa: F401
        return True
    except ImportError:
        return False


def _export_docx2pdf(docx_path: Path, output_path: Path) -> Path:
    from docx2pdf import convert

    logger.info("PDF: docx2pdf %s -> %s", docx_path, output_path)
    convert(str(docx_path), str(output_path))
    if not output_path.exists():
        raise RuntimeError(f"docx2pdf returned but no PDF at {output_path}")
    return output_path


# ---------- LibreOffice backend ----------


def _export_libreoffice(docx_path: Path, output_path: Path, timeout: int) -> Path:
    soffice = _find_soffice()
    if soffice is None:
        raise FileNotFoundError(
            "soffice (LibreOffice) not found on PATH or standard install locations"
        )

    # `--convert-to pdf --outdir <dir>` produces "<basename>.pdf" in
    # the target directory; we rename if the user requested a different
    # filename.
    out_dir = output_path.parent
    cmd = [
        str(soffice),
        "--headless",
        "--convert-to", "pdf",
        "--outdir", str(out_dir),
        str(docx_path),
    ]
    logger.info("PDF: LibreOffice %s", " ".join(cmd))
    result = subprocess.run(
        cmd, capture_output=True, text=True, timeout=timeout, check=False,
    )
    if result.returncode != 0:
        raise RuntimeError(
            f"LibreOffice exited {result.returncode}.\n"
            f"stdout: {result.stdout.strip()}\nstderr: {result.stderr.strip()}"
        )

    produced = out_dir / f"{docx_path.stem}.pdf"
    if not produced.exists():
        raise RuntimeError(
            f"LibreOffice ran cleanly but produced no PDF at {produced}\n"
            f"stdout: {result.stdout.strip()}"
        )
    if produced != output_path:
        produced.replace(output_path)
    return output_path


def _find_soffice() -> Path | None:
    for name in ("soffice", "soffice.exe"):
        found = shutil.which(name)
        if found:
            return Path(found)
    for candidate in (
        Path(r"C:\Program Files\LibreOffice\program\soffice.exe"),
        Path(r"C:\Program Files (x86)\LibreOffice\program\soffice.exe"),
        Path("/Applications/LibreOffice.app/Contents/MacOS/soffice"),
        Path("/usr/bin/soffice"),
        Path("/usr/local/bin/soffice"),
    ):
        if candidate.exists():
            return candidate
    return None
