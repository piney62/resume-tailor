"""Streamlit UI for Resume Tailor.

Run with:
    streamlit run src/ui/streamlit_app.py

Layout:
  Left  — title + compact inputs + results
  Right — PDF viewer from the very top of the page
"""

import difflib
import json
import os
import shutil
import sys
import tempfile
from datetime import datetime
from pathlib import Path

_ROOT = Path(__file__).parent.parent.parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

import streamlit as st
from dotenv import load_dotenv

from src.llm.client import GroqClient
from src.models.schemas import Resume
from src.pipeline import run_tailor_pipeline

load_dotenv(_ROOT / ".env")

_STATIC_DIR = Path(__file__).parent / "static"
_STATIC_DIR.mkdir(exist_ok=True)


# =========================================================================
# Helpers
# =========================================================================


def _resume_to_text(resume: Resume) -> str:
    lines: list[str] = []
    lines.append(f"NAME: {resume.header.name}")
    if resume.header.title:
        lines.append(f"TITLE: {resume.header.title}")
    for c in resume.header.contact_lines:
        lines.append(f"CONTACT: {c}")
    lines.append("")
    lines.append("--- SUMMARY ---")
    lines.append(resume.summary.text)
    lines.append("")
    for i, role in enumerate(resume.experience):
        lines.append(f"--- EXPERIENCE #{i+1}: {role.company} | {role.title} ---")
        lines.append(f"  dates:    {role.dates}")
        lines.append(f"  location: {role.location}")
        if role.intro:
            lines.append(f"  intro: {role.intro}")
        for j, b in enumerate(role.bullets):
            lines.append(f"  - [{j}] {b}")
        if role.skills_line:
            lines.append(f"  {role.skills_line}")
        lines.append("")
    lines.append("--- EDUCATION ---")
    for e in resume.education:
        lines.append(f"  {e.institution} — {e.degree or ''}, {e.field or ''}")
        if e.dates:
            lines.append(f"    {e.dates} · {e.location or ''}")
    lines.append("")
    lines.append("--- SKILLS ---")
    for line in resume.skills_section.raw_lines:
        lines.append(f"  {line}")
    return "\n".join(lines)


def _diff_resumes(original: Resume, rewritten: Resume) -> str:
    return "\n".join(
        difflib.unified_diff(
            _resume_to_text(original).splitlines(),
            _resume_to_text(rewritten).splitlines(),
            fromfile="original",
            tofile="tailored",
            lineterm="",
            n=2,
        )
    )


_HISTORY_DIR = _ROOT / "history"


def _save_history(pdf_path: Path, company: str, role: str) -> Path | None:
    """Copy PDF to history/ as '{company} {role}.pdf'. Returns dest path or None."""
    parts = [p.strip() for p in [company, role] if p.strip()]
    if not parts:
        return None
    _HISTORY_DIR.mkdir(exist_ok=True)
    stem = " ".join(parts)
    for ch in r'\/:*?"<>|':
        stem = stem.replace(ch, "_")
    dest = _HISTORY_DIR / f"{stem}.pdf"
    counter = 2
    while dest.exists():
        dest = _HISTORY_DIR / f"{stem} ({counter}).pdf"
        counter += 1
    shutil.copy2(pdf_path, dest)
    return dest


def _static_pdf_url(pdf_path: Path) -> str:
    """Copy PDF to the static dir (cache-busted) and return its /app/static/ URL."""
    dest_name = f"preview_{pdf_path.stat().st_mtime_ns}.pdf"
    dest = _STATIC_DIR / dest_name
    if not dest.exists():
        for old in _STATIC_DIR.glob("preview_*.pdf"):
            old.unlink(missing_ok=True)
        shutil.copy2(pdf_path, dest)
    return dest_name


def _show_pdf_inline(pdf_path: Path, height: int = 900) -> None:
    dest_name = _static_pdf_url(pdf_path)
    src_url = f"/app/static/{dest_name}#navpanes=0&view=FitH"
    st.markdown(
        f'<iframe src="{src_url}" width="100%" height="{height}px" '
        f'style="border:none;border-radius:6px;"></iframe>',
        unsafe_allow_html=True,
    )


def _auto_download(pdf_path: Path, filename: str, result_key: str) -> None:
    """Trigger a one-shot browser download via an injected anchor + script."""
    if st.session_state.get("last_auto_dl") == result_key:
        return
    st.session_state.last_auto_dl = result_key
    dest_name = _static_pdf_url(pdf_path)
    safe_filename = filename.replace('"', "")
    st.markdown(
        f'<a id="_adl" href="/app/static/{dest_name}" '
        f'download="{safe_filename}" style="display:none">dl</a>'
        f'<script>document.getElementById("_adl").click();</script>',
        unsafe_allow_html=True,
    )


def _label(text: str) -> None:
    st.markdown(
        f"<p style='font-weight:600;margin:10px 0 3px 0;font-size:0.9rem'>{text}</p>",
        unsafe_allow_html=True,
    )


def _build_client(api_keys: str, model: str) -> GroqClient:
    keys = [k.strip() for k in api_keys.split(",") if k.strip()]
    return GroqClient(api_keys=keys, model=model)


# =========================================================================
# Page config
# =========================================================================

st.set_page_config(page_title="Resume Tailor", layout="wide")

# Reduce default top padding so the right-column PDF viewer sits higher.
st.markdown(
    "<style>div[data-testid='stAppViewContainer'] > section > div:first-child"
    "{padding-top:1rem}</style>",
    unsafe_allow_html=True,
)

# =========================================================================
# Sidebar
# =========================================================================

with st.sidebar:
    st.header("Groq")
    env_keys = os.environ.get("GROQ_API_KEYS", "")
    api_keys = st.text_input(
        "API keys (comma-separated)",
        value=env_keys,
        type="password",
        help="Round-robin rotated. Load order: this field first, otherwise .env.",
    )
    model = st.text_input(
        "Model",
        value=os.environ.get("GROQ_MODEL", "llama-3.3-70b-versatile"),
    )

    st.divider()
    st.header("Pipeline")
    max_regen = st.slider("Max regen passes", 0, 3, 2)
    skip_pdf = st.checkbox("Skip PDF export", value=False)
    pdf_backend = st.selectbox("PDF backend", ["auto", "docx2pdf", "libreoffice"], index=0)

    st.divider()
    st.header("Output")
    auto_download = st.checkbox(
        "Auto-download PDF after tailoring",
        value=False,
        help="Automatically saves the PDF to your Downloads folder when the pipeline finishes.",
    )

    st.divider()
    st.header("History")
    history_pdfs = sorted(_HISTORY_DIR.glob("*.pdf"), key=lambda p: p.stat().st_mtime, reverse=True) if _HISTORY_DIR.exists() else []
    if history_pdfs:
        for hp in history_pdfs:
            st.download_button(
                label=hp.stem,
                data=hp.read_bytes(),
                file_name=hp.name,
                mime="application/pdf",
                use_container_width=True,
                key=f"hist_{hp.name}",
            )
    else:
        st.caption("No history yet.")


# =========================================================================
# Two-column layout (no top-level title — it lives inside the left column)
# =========================================================================

left_col, right_col = st.columns([1, 1], gap="large")

# =========================================================================
# LEFT — title + inputs + results
# =========================================================================

with left_col:

    st.markdown(
        "<h2 style='margin:0 0 2px 0'>Resume Tailor</h2>"
        "<p style='color:#888;margin:0 0 12px 0;font-size:0.82rem'>"
        "Numbers, company names, and dates are always preserved.</p>",
        unsafe_allow_html=True,
    )

    # ── Resume ───────────────────────────────────────────────────────────
    _label("Resume")

    profiles_dir = _ROOT / "profiles"
    existing_profiles = sorted(
        p.name for p in profiles_dir.iterdir()
        if p.is_dir() and any(p.glob("*.docx"))
    ) if profiles_dir.exists() else []

    r_src_col, r_pick_col = st.columns([1, 2])
    with r_src_col:
        resume_source = st.radio(
            "_rs", ["Profile", "Upload"],
            label_visibility="collapsed",
            disabled=not existing_profiles,
            index=0 if existing_profiles else 1,
        )
    with r_pick_col:
        resume_path: Path | None = None
        profile_name: str | None = None
        if resume_source == "Profile" and existing_profiles:
            profile_name = st.selectbox("_prof", existing_profiles,
                                        label_visibility="collapsed")
            candidates = [c for c in sorted((profiles_dir / profile_name).glob("*.docx"))
                          if not c.name.startswith("~$")]
            if candidates:
                resume_path = candidates[0]
                st.caption(f"`{resume_path.name}` · {resume_path.stat().st_size // 1024} KB")
        else:
            uploaded = st.file_uploader("_up", type=["docx"],
                                        label_visibility="collapsed")
            if uploaded is not None:
                if st.session_state.get("uploaded_name") != uploaded.name:
                    tmp = tempfile.NamedTemporaryFile(suffix=".docx", delete=False)
                    tmp.write(uploaded.getvalue())
                    tmp.close()
                    st.session_state.uploaded_path = Path(tmp.name)
                    st.session_state.uploaded_name = uploaded.name
                resume_path = st.session_state.uploaded_path
                profile_name = Path(uploaded.name).stem
                st.caption(f"`{uploaded.name}` · {len(uploaded.getvalue()) // 1024} KB")

    # ── Target job (for records) ──────────────────────────────────────────
    _label("Target job (for your records)")
    j_co, j_role = st.columns(2)
    with j_co:
        company_name = st.text_input("Company", placeholder="e.g. Google")
    with j_role:
        role_name = st.text_input("Role", placeholder="e.g. Senior Engineer")

    # ── Job description ───────────────────────────────────────────────────
    _label("Job description")

    jd_archive_dir = _ROOT / "jd-archive"
    existing_jds = sorted(
        [p for p in jd_archive_dir.glob("*.md") if not p.name.startswith(".")] +
        [p for p in jd_archive_dir.glob("*.txt") if not p.name.startswith(".")]
    ) if jd_archive_dir.exists() else []

    jd_source = st.radio(
        "_jds", ["jd-archive/", "Paste text"],
        horizontal=True,
        label_visibility="collapsed",
        disabled=not existing_jds,
        index=1,
    )

    jd_text: str = ""
    if jd_source == "jd-archive/" and existing_jds:
        selected_jd = st.selectbox("_jdf", [p.name for p in existing_jds],
                                   label_visibility="collapsed")
        jd_text = (jd_archive_dir / selected_jd).read_text(encoding="utf-8")
        with st.expander("Preview"):
            st.markdown(jd_text)
    else:
        jd_text = st.text_area(
            "_jdp", height=175,
            placeholder="Paste the job description here…",
            label_visibility="collapsed",
        )

    # ── Action row (Tailor | ↓ docx | ↓ pdf) ─────────────────────────────
    st.markdown("<div style='margin-top:6px'></div>", unsafe_allow_html=True)
    run_disabled = not (resume_path and api_keys.strip())

    result = st.session_state.get("result")

    # Derive resume owner name for download filename
    resume_owner = ""
    if result and result.original_resume:
        resume_owner = result.original_resume.header.name
    elif profile_name:
        resume_owner = profile_name
    dl_stem = resume_owner if resume_owner else "resume"

    act_col, docx_col, pdf_col = st.columns([3, 2, 2])
    with act_col:
        if not api_keys.strip():
            st.warning("Add Groq API key in sidebar.", icon="🔑")
        elif not resume_path:
            st.info("Select a resume above.", icon="📄")
        go = st.button("Tailor my resume", type="primary",
                       disabled=run_disabled, use_container_width=True)
    with docx_col:
        if result and result.docx_path and Path(result.docx_path).exists():
            st.download_button(
                "↓ .docx",
                data=Path(result.docx_path).read_bytes(),
                file_name=f"{dl_stem}.docx",
                mime="application/vnd.openxmlformats-officedocument.wordprocessingml.document",
                use_container_width=True,
            )
        else:
            st.button("↓ .docx", disabled=True, use_container_width=True)
    with pdf_col:
        if result and result.pdf_path and Path(result.pdf_path).exists():
            st.download_button(
                "↓ .pdf",
                data=Path(result.pdf_path).read_bytes(),
                file_name=f"{dl_stem}.pdf",
                mime="application/pdf",
                use_container_width=True,
            )
        else:
            st.button("↓ .pdf", disabled=True, use_container_width=True)

    # ── Pipeline run ──────────────────────────────────────────────────────
    if go and resume_path:
        if not jd_text.strip():
            st.error("Please paste a job description before running.")
            st.stop()
        output_dir = _ROOT / "outputs" / "_ui" / datetime.now().strftime("%Y%m%d-%H%M%S")
        progress_bar = st.progress(0.0, text="Starting…")

        def on_progress(label: str, frac: float) -> None:
            progress_bar.progress(min(max(frac, 0.0), 1.0), text=label)

        try:
            client = _build_client(api_keys, model)
            with st.status("Running pipeline…", expanded=False) as status:
                result = run_tailor_pipeline(
                    resume_path=resume_path,
                    jd_text=jd_text,
                    output_dir=output_dir,
                    client=client,
                    pdf_backend=pdf_backend,
                    skip_pdf=skip_pdf,
                    max_regen_passes=max_regen,
                    progress_cb=on_progress,
                )
                status.update(label="Pipeline complete", state="complete")
            st.session_state.result = result
            if result.pdf_path and Path(result.pdf_path).exists():
                _save_history(Path(result.pdf_path), company_name, role_name)
            st.rerun()
        except Exception as e:  # noqa: BLE001
            st.error(f"Pipeline failed: {type(e).__name__}: {e}")
            st.session_state.result = None
            st.exception(e)

    # ── Auto-download ─────────────────────────────────────────────────────
    if (
        auto_download
        and result is not None
        and result.pdf_path
        and Path(result.pdf_path).exists()
    ):
        _auto_download(
            Path(result.pdf_path),
            f"{dl_stem}.pdf",
            result_key=str(result.log_dir),
        )

    # ── Results summary ───────────────────────────────────────────────────
    if result is not None:
        st.divider()
        report = result.report
        badge = "PASSED" if report.passed else "FAILED"
        color = ":green" if report.passed else ":red"
        n_critical = sum(1 for i in report.issues if i.severity == "critical")
        n_warnings = sum(1 for i in report.issues if i.severity == "warning")

        if company_name or role_name:
            parts = [p for p in [company_name, role_name] if p]
            st.markdown(
                f"<p style='font-size:1rem;font-weight:600;margin:0 0 4px 0'>"
                f"{'  ·  '.join(parts)}</p>",
                unsafe_allow_html=True,
            )

        st.markdown(
            f"{color}[**{badge}**] · "
            f"{n_critical} critical, {n_warnings} warnings · "
            f"keyword match **{report.keyword_match_rate:.0%}**"
        )

        if report.issues:
            with st.expander(f"Validation issues ({len(report.issues)})",
                             expanded=n_critical > 0):
                for issue in report.issues:
                    tag = "[CRITICAL]" if issue.severity == "critical" else "[warn]"
                    st.markdown(f"**{tag} `{issue.section}`** — {issue.issue}")
                    if issue.original != issue.rewritten:
                        diff_text = "\n".join(
                            difflib.unified_diff(
                                (issue.original or "").splitlines(),
                                (issue.rewritten or "").splitlines(),
                                lineterm="", n=1,
                            )
                        )
                        if diff_text:
                            st.code(diff_text, language="diff")

        if result.original_resume and result.rewritten_resume:
            with st.expander("What changed"):
                diff = _diff_resumes(result.original_resume, result.rewritten_resume)
                st.code(diff if diff.strip() else "(no text changes)", language="diff")

        log_dir = Path(result.log_dir)
        jd_json = log_dir / "2_jd_analysis.json"
        if jd_json.exists():
            with st.expander("JD analysis"):
                st.json(json.loads(jd_json.read_text(encoding="utf-8")))

        rewritten_json = log_dir / "3a_rewritten_initial.json"
        if rewritten_json.exists():
            with st.expander("Holistic rewrite"):
                st.json(json.loads(rewritten_json.read_text(encoding="utf-8")))

        with st.expander("Groq usage"):
            st.json(result.groq_summary)


# =========================================================================
# RIGHT — PDF viewer (no title, starts from top)
# =========================================================================

with right_col:
    result = st.session_state.get("result")
    pdf_ready = (
        result is not None
        and result.pdf_path is not None
        and Path(result.pdf_path).exists()
    )

    if pdf_ready:
        _show_pdf_inline(Path(result.pdf_path), height=900)
    else:
        st.markdown(
            "<div style='"
            "height:600px;"
            "display:flex;align-items:center;justify-content:center;"
            "border:2px dashed #555;border-radius:10px;"
            "color:#777;font-size:0.95rem;"
            "margin-top:52px;"
            "'>PDF preview appears here after running</div>",
            unsafe_allow_html=True,
        )
