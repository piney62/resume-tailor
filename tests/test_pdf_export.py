"""Tests for src.docx.pdf_export. Backends are mocked so the suite does
not depend on Microsoft Word or LibreOffice being installed."""

from pathlib import Path
from unittest.mock import MagicMock

import pytest

import src.docx.pdf_export as pdf_export
from src.docx.pdf_export import available_backends, export_pdf


# ---------- helpers ----------


@pytest.fixture
def fake_docx(tmp_path: Path) -> Path:
    p = tmp_path / "src.docx"
    p.write_bytes(b"not a real docx, but it exists")
    return p


# ---------- input validation ----------


def test_missing_source_raises(tmp_path: Path) -> None:
    with pytest.raises(FileNotFoundError):
        export_pdf(tmp_path / "missing.docx", tmp_path / "out.pdf")


def test_output_must_end_in_pdf(fake_docx: Path, tmp_path: Path) -> None:
    with pytest.raises(ValueError, match=r"end in \.pdf"):
        export_pdf(fake_docx, tmp_path / "out.txt")


def test_unknown_backend_raises(fake_docx: Path, tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="unknown backend"):
        export_pdf(fake_docx, tmp_path / "out.pdf", backend="potato")  # type: ignore[arg-type]


# ---------- auto-detect ----------


def test_auto_prefers_docx2pdf_when_importable(
    monkeypatch, mocker, fake_docx: Path, tmp_path: Path
) -> None:
    out = tmp_path / "out.pdf"
    monkeypatch.setattr(pdf_export, "_docx2pdf_importable", lambda: True)
    mock_docx2pdf = mocker.patch.object(pdf_export, "_export_docx2pdf", return_value=out)
    mock_libre = mocker.patch.object(pdf_export, "_export_libreoffice", return_value=out)

    export_pdf(fake_docx, out)
    mock_docx2pdf.assert_called_once()
    mock_libre.assert_not_called()


def test_auto_falls_back_to_libreoffice_when_docx2pdf_fails(
    monkeypatch, mocker, fake_docx: Path, tmp_path: Path
) -> None:
    out = tmp_path / "out.pdf"
    monkeypatch.setattr(pdf_export, "_docx2pdf_importable", lambda: True)
    monkeypatch.setattr(pdf_export, "_find_soffice", lambda: Path("/usr/bin/soffice"))
    mocker.patch.object(pdf_export, "_export_docx2pdf", side_effect=RuntimeError("Word not installed"))
    mock_libre = mocker.patch.object(pdf_export, "_export_libreoffice", return_value=out)

    export_pdf(fake_docx, out)
    mock_libre.assert_called_once()


def test_auto_falls_back_to_libreoffice_when_docx2pdf_not_importable(
    monkeypatch, mocker, fake_docx: Path, tmp_path: Path
) -> None:
    out = tmp_path / "out.pdf"
    monkeypatch.setattr(pdf_export, "_docx2pdf_importable", lambda: False)
    monkeypatch.setattr(pdf_export, "_find_soffice", lambda: Path("/usr/bin/soffice"))
    mock_libre = mocker.patch.object(pdf_export, "_export_libreoffice", return_value=out)

    export_pdf(fake_docx, out)
    mock_libre.assert_called_once()


def test_auto_raises_when_both_backends_unavailable(
    monkeypatch, fake_docx: Path, tmp_path: Path
) -> None:
    monkeypatch.setattr(pdf_export, "_docx2pdf_importable", lambda: False)
    monkeypatch.setattr(pdf_export, "_find_soffice", lambda: None)

    with pytest.raises(RuntimeError, match="No PDF backend"):
        export_pdf(fake_docx, tmp_path / "out.pdf")


def test_auto_raises_when_both_backends_fail(
    monkeypatch, mocker, fake_docx: Path, tmp_path: Path
) -> None:
    monkeypatch.setattr(pdf_export, "_docx2pdf_importable", lambda: True)
    monkeypatch.setattr(pdf_export, "_find_soffice", lambda: Path("/usr/bin/soffice"))
    mocker.patch.object(pdf_export, "_export_docx2pdf", side_effect=RuntimeError("Word boom"))
    mocker.patch.object(pdf_export, "_export_libreoffice", side_effect=RuntimeError("libre boom"))

    with pytest.raises(RuntimeError, match="docx2pdf.*libreoffice"):
        export_pdf(fake_docx, tmp_path / "out.pdf")


# ---------- forced backends ----------


def test_force_libreoffice_backend(mocker, fake_docx: Path, tmp_path: Path) -> None:
    out = tmp_path / "out.pdf"
    mock_libre = mocker.patch.object(pdf_export, "_export_libreoffice", return_value=out)
    mock_docx2pdf = mocker.patch.object(pdf_export, "_export_docx2pdf", return_value=out)
    export_pdf(fake_docx, out, backend="libreoffice")
    mock_libre.assert_called_once()
    mock_docx2pdf.assert_not_called()


def test_force_docx2pdf_backend(mocker, fake_docx: Path, tmp_path: Path) -> None:
    out = tmp_path / "out.pdf"
    mock_docx2pdf = mocker.patch.object(pdf_export, "_export_docx2pdf", return_value=out)
    export_pdf(fake_docx, out, backend="docx2pdf")
    mock_docx2pdf.assert_called_once()


# ---------- LibreOffice subprocess wiring ----------


def test_libreoffice_invokes_correct_command(
    monkeypatch, mocker, fake_docx: Path, tmp_path: Path
) -> None:
    out = tmp_path / "out.pdf"
    monkeypatch.setattr(pdf_export, "_find_soffice", lambda: Path("/usr/bin/soffice"))

    fake_completed = MagicMock(returncode=0, stdout="ok", stderr="")
    mock_run = mocker.patch("src.docx.pdf_export.subprocess.run", return_value=fake_completed)

    # Pre-create the file LibreOffice would have produced.
    produced = tmp_path / f"{fake_docx.stem}.pdf"
    produced.write_bytes(b"%PDF-1.4 fake")

    export_pdf(fake_docx, out, backend="libreoffice")
    cmd = mock_run.call_args.args[0]
    # First arg is soffice path (string form is platform-dependent).
    assert "soffice" in cmd[0]
    assert "--headless" in cmd
    assert "--convert-to" in cmd and cmd[cmd.index("--convert-to") + 1] == "pdf"
    assert "--outdir" in cmd and cmd[cmd.index("--outdir") + 1] == str(tmp_path)
    assert cmd[-1] == str(fake_docx)
    assert out.exists()


def test_libreoffice_renames_to_requested_filename(
    monkeypatch, mocker, fake_docx: Path, tmp_path: Path
) -> None:
    requested = tmp_path / "tailored-acme.pdf"
    monkeypatch.setattr(pdf_export, "_find_soffice", lambda: Path("/usr/bin/soffice"))
    fake_completed = MagicMock(returncode=0, stdout="", stderr="")
    mocker.patch("src.docx.pdf_export.subprocess.run", return_value=fake_completed)
    (tmp_path / f"{fake_docx.stem}.pdf").write_bytes(b"%PDF fake")

    export_pdf(fake_docx, requested, backend="libreoffice")
    assert requested.exists()
    # The default-named output should be gone (renamed).
    assert not (tmp_path / f"{fake_docx.stem}.pdf").exists()


def test_libreoffice_nonzero_exit_raises(
    monkeypatch, mocker, fake_docx: Path, tmp_path: Path
) -> None:
    monkeypatch.setattr(pdf_export, "_find_soffice", lambda: Path("/usr/bin/soffice"))
    fake_completed = MagicMock(returncode=2, stdout="", stderr="boom")
    mocker.patch("src.docx.pdf_export.subprocess.run", return_value=fake_completed)

    with pytest.raises(RuntimeError, match="exited 2"):
        export_pdf(fake_docx, tmp_path / "out.pdf", backend="libreoffice")


def test_libreoffice_missing_output_raises(
    monkeypatch, mocker, fake_docx: Path, tmp_path: Path
) -> None:
    monkeypatch.setattr(pdf_export, "_find_soffice", lambda: Path("/usr/bin/soffice"))
    fake_completed = MagicMock(returncode=0, stdout="", stderr="")
    mocker.patch("src.docx.pdf_export.subprocess.run", return_value=fake_completed)
    # Note: we deliberately do NOT create the expected output file.

    with pytest.raises(RuntimeError, match="produced no PDF"):
        export_pdf(fake_docx, tmp_path / "out.pdf", backend="libreoffice")


def test_libreoffice_soffice_not_found_raises(
    monkeypatch, fake_docx: Path, tmp_path: Path
) -> None:
    monkeypatch.setattr(pdf_export, "_find_soffice", lambda: None)
    with pytest.raises(FileNotFoundError, match="soffice"):
        export_pdf(fake_docx, tmp_path / "out.pdf", backend="libreoffice")


# ---------- available_backends introspection ----------


def test_available_backends_returns_list(monkeypatch) -> None:
    monkeypatch.setattr(pdf_export, "_docx2pdf_importable", lambda: True)
    monkeypatch.setattr(pdf_export, "_find_soffice", lambda: Path("/usr/bin/soffice"))
    assert available_backends() == ["docx2pdf", "libreoffice"]


def test_available_backends_empty_when_neither_present(monkeypatch) -> None:
    monkeypatch.setattr(pdf_export, "_docx2pdf_importable", lambda: False)
    monkeypatch.setattr(pdf_export, "_find_soffice", lambda: None)
    assert available_backends() == []


# ---------- docx2pdf output verification ----------


def test_docx2pdf_raises_when_output_not_produced(
    mocker, fake_docx: Path, tmp_path: Path
) -> None:
    # Mock the docx2pdf.convert call to be a no-op so output is never created.
    mock_module = MagicMock()
    mock_module.convert = lambda *a, **kw: None
    mocker.patch.dict("sys.modules", {"docx2pdf": mock_module})

    with pytest.raises(RuntimeError, match="no PDF"):
        export_pdf(fake_docx, tmp_path / "out.pdf", backend="docx2pdf")
