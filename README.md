# Resume Tailor

JD-aware resume tailoring CLI. You give it a job description and your `.docx` resume; it returns a rewritten `.docx` and `.pdf` that aligns with the JD — **without** breaking the original Word formatting and **without** changing any company name, date, degree, or quantified metric.

Built as a multi-stage pipeline (no single mega-prompt). Powered by [Groq](https://groq.com/) running `llama-3.3-70b-versatile` on the free tier; two API keys are rotated round-robin to stay under per-key rate limits.

## What it preserves

- Every company name, employment date, job title text, degree, and field of study.
- Every number, percentage, year, and quantified metric in summary, intro, and bullet text.
- The original `.docx` chrome — fonts, styles, margins, section breaks, headers/footers, paragraph numbering. Edits happen **in place** via python-docx; we never regenerate a document from scratch.
- The original paragraph count and structure (the Validator blocks structural drift).

## What it changes

- The job title is allowed to gain a JD-aligned **suffix** (e.g. `SENIOR SOFTWARE ENGINEER, BACKEND PLATFORMS`).
- The summary is rewritten to mirror the JD's focus.
- Each role's intro and each bullet are individually rewritten when JD alignment is possible, and left **verbatim** otherwise.
- Per-role `Skills:` lines and the `PROFESSIONAL SKILLS` section get rule-based substitutions and additions from the substitution plan — no LLM in this hop, so swaps are predictable.

## Quick start

```powershell
git clone https://github.com/piney62/resume-tailor.git
cd resume-tailor

python -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install -e ".[dev]"

copy .env.example .env
# Edit .env and fill in GROQ_API_KEYS (comma-separated for multiple keys)
```

Put your source resume in `profiles/<your-name>/your-resume.docx`. The filename does not matter — the CLI auto-detects the first `.docx` in the profile directory. The `profiles/` directory is gitignored, so personal resumes never get committed.

Place job descriptions under `jd-archive/` as plain text or markdown.

## Usage

### Tailor a resume to a JD

```powershell
python -m src.main tailor `
  --profile sample `
  --jd-file jd-archive/sample-developer-platform.md
```

Outputs land in `outputs/<profile>/`:

```
outputs/sample/
├── Katharine Berry (tailored).docx
├── Katharine Berry (tailored).pdf
└── logs/20260522-143015/
    ├── 1_jd_analysis.json
    ├── 2_resume_parsed.json
    ├── 3_substitution_plan.json
    ├── 4a_rewritten_initial.json
    ├── 5_validation_report.json
    └── groq_usage.json
```

Options:
- `--output <dir>` — override the default `outputs/<profile>/`
- `--pdf-backend auto|docx2pdf|libreoffice` — pin the PDF converter
- `--skip-pdf` — skip PDF export (faster, useful for iteration)
- `--max-regen N` — number of regeneration passes after validation failures (default 2)

### Run JD analysis only

```powershell
python -m src.main analyze-jd --jd-file jd-archive/sample-developer-platform.md
```

Prints the parsed `JDAnalysis` as JSON: must-have / nice-to-have technologies, soft skills, domain keywords, seniority level, and the JD phrases worth mirroring.

### Diff a tailored resume against the original

```powershell
python -m src.main validate `
  --original "profiles/sample/Katharine Berry.docx" `
  --tailored "outputs/sample/Katharine Berry (tailored).docx"
```

Prints a `ValidationReport`: `passed`, `keyword_match_rate`, and any critical or warning issues with their original-vs-rewritten diffs. Exits non-zero when `passed=False`.

### Check which PDF backends are visible on this machine

```powershell
python -m src.main backends
```

## Architecture

```
            +-------------------+
JD (text) ──►  1. JD Analyzer    │  Groq, temp=0.1, JSON schema-validated
            +-------------------+
                     │ JDAnalysis
                     ▼
            +-------------------+
.docx     ──►  2. Resume Parser  │  python-docx → Resume model (no LLM)
            +-------------------+
                     │ Resume
                     ▼
            +-------------------+
            │  3. Subst Planner  │  Groq, temp=0.2, domain-constrained swaps
            +-------------------+
                     │ SubstitutionPlan
                     ▼
            +-------------------+
            │  4. Rewriter       │  Groq, temp=0.4
            │   a) title         │    rule-based modifier
            │   b) summary       │    1 LLM call
            │   c) intros        │    1 LLM call per role
            │   d) bullets       │    1 LLM call per bullet
            │   e) skills_line   │    rule-based substitutions
            │   f) skills_section│    rule-based substitutions + additions
            +-------------------+
                     │ Resume (modified)
                     ▼
            +-------------------+
            │  5. Validator      │  Rule-based diff vs original
            +-------------------+
                     │  passed? no → regenerate failed sections (max 2)
                     │ ValidationReport
                     ▼
            +-------------------+
            │  6. DOCX Writer    │  python-docx in-place run-level edits
            │     PDF Export     │  docx2pdf or LibreOffice headless
            +-------------------+
```

## Hallucination prevention

Five layers, each at a different stage:

1. **System prompts** instruct the model to preserve every number, company, date, degree, and seniority word, with the persona of an ATS-optimization expert.
2. **JD evidence quotes**: every `must_have` technology in the JD analysis includes a short verbatim quote from the JD as evidence — discourages the analyzer from inventing requirements.
3. **Substitution planner constraints**: substitutions must stay within a technical domain (e.g. `db→db`, not `db→frontend`). Identity-level fields (company, dates, seniority) are protected by absolute rules.
4. **Number-preservation guard in the rewriter**: after every LLM call, the rewritten text's number-token set is compared against the original's. On any drop or alteration, we **fall back to the original verbatim** rather than ship a hallucinated metric.
5. **Rule-based validator**: diffs the rewritten resume against the original field by field. Identity-level changes are flagged critical and trigger section-level regeneration (max 2 passes); banned ATS-fatigue words are flagged as warnings.

## Multi-key rate limiting

The Groq client (`src/llm/client.py`) carries one sliding-window rate limiter per API key. The default budget is 25 RPM per key, configurable via `GROQ_RPM_PER_KEY`. With two keys you get 50 RPM aggregate, sufficient for a typical resume (≈ 33 LLM calls per tailor run).

When a key saturates, the client either rotates to the next or sleeps until the earliest call in that key's window expires. Transient `429`, `5xx`, and connection errors retry with exponential backoff (honoring `Retry-After` when provided), up to 3 attempts.

Every call's latency, token usage, key index, and retry count are captured in a `CallLog`. The end-of-run `groq_usage.json` aggregates the totals.

## Project layout

```
src/
  main.py             — typer CLI (tailor / analyze-jd / validate / backends)
  pipeline.py         — orchestration + section-level regeneration loop
  style_rules.py      — BANNED_WORDS shared by rewriter and validator
  stages/
    jd_analyzer.py
    resume_parser.py
    substitution.py
    rewriter.py
    validator.py
  llm/
    client.py         — multi-key Groq client with rate limit + retry
    prompt_loader.py  — Jinja2 environment (StrictUndefined)
    prompts/          — *.j2 user-prompt templates
    few_shot/         — *.yaml few-shot examples
    few_shot.py       — YAML loader
  models/
    schemas.py        — Pydantic v2 I/O contracts for every stage
  docx/
    reader.py         — .docx → ParagraphInfo[]
    writer.py         — in-place edit (run-level, format-preserving)
    pdf_export.py     — docx2pdf with LibreOffice fallback

profiles/{name}/      — your local resume (gitignored)
jd-archive/           — JD inputs
outputs/{name}/       — tailored output + per-run logs (gitignored)
tests/                — pytest suite (175+ tests)
```

## PDF backend

`pdf_export` auto-detects:
1. **docx2pdf** if importable and MS Word is installed (highest fidelity).
2. **LibreOffice** via `soffice --headless --convert-to pdf` (cross-platform fallback).

Verify what's available on your machine:
```powershell
python -m src.main backends
```

Pin a backend explicitly with `--pdf-backend docx2pdf|libreoffice`.

## Testing

```powershell
# Full unit + integration suite (no network, no API key required)
pytest -v

# Live end-to-end test against the real Groq API.
# Uses the sample resume and sample JD; costs ~17k tokens (free tier).
set RUN_LIVE_TESTS=1
pytest tests/test_e2e_live.py -v -s
```

The non-live suite mocks the Groq client entirely and runs in under 2 seconds. The live e2e test is opt-in via `RUN_LIVE_TESTS=1` and verifies:
- the full pipeline succeeds end-to-end against `profiles/sample/Katharine Berry.docx` and `jd-archive/sample-developer-platform.md`
- the produced `.docx` is non-trivial and not corrupted
- no critical validation issues remain after the regeneration loop
- `keyword_match_rate` is ≥ 0.3 (a sanity threshold)

## Known limitations

- **Mid-paragraph formatting on edited paragraphs**: paragraphs that mix formatting mid-text (e.g., a bold `Skills:` prefix in front of plain items) inherit the first run's formatting when edited. Resumes with uniform per-paragraph formatting — the overwhelming common case — are unaffected.
- **Table-based resumes** are not supported. The parser expects paragraph-based layouts.
- **Single-stage education parsing**: `degree, field` is split on the first comma. Formats like `MS in Computer Science` parse as `degree="MS in Computer Science"`, `field=None`.
- **No bullet skipping**: every bullet currently goes through an LLM rewrite (per the spec). When the bullet has no JD-relevant content, the rewriter prompt tells the model to return verbatim — and the number-preservation guard enforces it — but you still pay the token cost. A `--rewrite-only-relevant` option would be a reasonable future enhancement.
- **Provider lock-in to Groq**: the design isolates LLM calls behind `GroqClient.complete_json`, but no abstraction layer for OpenAI/Anthropic is in place yet.

## Configuration reference (`.env`)

| Variable | Default | Purpose |
|---|---|---|
| `GROQ_API_KEYS` | (required) | Comma-separated Groq API keys. Round-robin rotated. |
| `GROQ_MODEL` | `llama-3.3-70b-versatile` | Model id. |
| `GROQ_RPM_PER_KEY` | `25` | Per-key sliding-window rate budget. |
| `LOG_LEVEL` | `INFO` | `DEBUG` / `INFO` / `WARNING` / `ERROR`. |

## License

MIT.
