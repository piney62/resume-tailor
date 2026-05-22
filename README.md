# Resume Tailor

JD-aware resume tailoring CLI. Input a job description + your `.docx` resume; get back a `.docx` and `.pdf` rewritten for that JD — **without** breaking the original formatting and **without** changing companies, dates, numbers, or degrees.

## Status

Under construction. Built in 12 steps; currently at **Step 1 (scaffolding)**.

## Quick start

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install -e ".[dev]"
copy .env.example .env   # then fill in GROQ_API_KEYS
```

## Layout

```
src/
  stages/       # 5-stage pipeline: analyze JD → parse resume → plan → rewrite → validate
  llm/          # Groq client + Jinja2 prompt templates
  models/       # Pydantic schemas (stage I/O contracts)
  docx/         # .docx read/write (format-preserving) + PDF export
profiles/{name}/  # local-only: your own source-resume.docx (gitignored)
jd-archive/     # JD inputs
outputs/{name}/ # tailored outputs + per-run logs (gitignored)
tests/
```

`profiles/` contents are gitignored — drop your own `.docx` into
`profiles/{your-name}/` locally and it stays off the repo.

## Design rules

1. Format preservation: edit existing paragraphs in place; never regenerate the `.docx` from scratch.
2. No hallucinations: company names, dates, degrees, and numeric metrics are immutable.
3. Multi-stage pipeline: one prompt per concern. No single mega-call.
4. Determinism where possible: rule-based parser and validator; LLM only for analysis and rewriting.
