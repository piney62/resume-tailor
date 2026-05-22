"""Pydantic v2 schemas. Each model is the I/O contract for a pipeline stage.

Why every section carries `indices`: the Writer (Step 9) edits the original
.docx paragraph-by-paragraph using these zero-based indices. Without them
we cannot preserve formatting.
"""

from typing import Literal, Optional

from pydantic import BaseModel, ConfigDict, Field, model_validator


SeniorityLevel = Literal["junior", "mid", "senior", "staff", "principal"]
Severity = Literal["critical", "warning"]
ApplyTo = Literal["skills", "bullets", "summary", "intro"]


class _Base(BaseModel):
    model_config = ConfigDict(extra="forbid", str_strip_whitespace=True)


# ---------- Stage 1: JD Analyzer ----------


class MustHaveItem(_Base):
    tech: str
    category: str
    evidence: str


class NiceToHaveItem(_Base):
    tech: str
    category: str


class JDAnalysis(_Base):
    must_have: list[MustHaveItem] = Field(default_factory=list)
    nice_to_have: list[NiceToHaveItem] = Field(default_factory=list)
    soft_skills: list[str] = Field(default_factory=list)
    domain_keywords: list[str] = Field(default_factory=list)
    seniority_level: SeniorityLevel
    exact_phrases_to_mirror: list[str] = Field(default_factory=list)


# ---------- Stage 2: Resume Parser ----------


class HeaderIndices(_Base):
    name_idx: int = Field(ge=0)
    title_idx: Optional[int] = Field(default=None, ge=0)
    contact_idxs: list[int] = Field(default_factory=list)


class Header(_Base):
    name: str
    title: Optional[str] = None
    contact_lines: list[str] = Field(default_factory=list)
    indices: HeaderIndices


class Summary(_Base):
    text: str
    paragraph_idxs: list[int] = Field(min_length=1)


class ExperienceIndices(_Base):
    # The role header line(s). Most resumes use one line ("Company | Title |
    # Dates | Location"); some split company and title across two lines.
    header_idxs: list[int] = Field(min_length=1)
    intro_idx: Optional[int] = Field(default=None, ge=0)
    bullet_idxs: list[int] = Field(default_factory=list)
    skills_line_idx: Optional[int] = Field(default=None, ge=0)


class Experience(_Base):
    company: str
    title: str
    dates: str
    location: str
    intro: str = ""
    bullets: list[str] = Field(default_factory=list)
    skills_line: Optional[str] = None
    indices: ExperienceIndices


class EducationIndices(_Base):
    header_idx: int = Field(ge=0)
    detail_idxs: list[int] = Field(default_factory=list)


class Education(_Base):
    institution: str
    degree: Optional[str] = None
    field: Optional[str] = None
    dates: Optional[str] = None
    location: Optional[str] = None
    indices: EducationIndices


class SkillsSectionIndices(_Base):
    # The "Skills" header paragraph itself, if present.
    header_idx: Optional[int] = Field(default=None, ge=0)
    # Paragraphs that contain the actual skill lines / categories.
    content_idxs: list[int] = Field(default_factory=list)


class SkillsSection(_Base):
    # Categorized form: { "Languages": ["Python", ...], "Cloud": [...] }.
    # Empty dict if the section is a single ungrouped list.
    categories: dict[str, list[str]] = Field(default_factory=dict)
    # Verbatim lines, kept as a fallback for writers when categorization
    # is unreliable.
    raw_lines: list[str] = Field(default_factory=list)
    indices: SkillsSectionIndices


class Resume(_Base):
    header: Header
    summary: Summary
    experience: list[Experience] = Field(default_factory=list)
    education: list[Education] = Field(default_factory=list)
    skills_section: SkillsSection
    # The original ordered paragraph texts, kept so the Writer and Validator
    # can resolve indices and diff against the source.
    raw_paragraphs: list[str] = Field(default_factory=list)


# ---------- Stage 3: Substitution Planner ----------


class Substitution(_Base):
    old: str
    new: str
    domain: str
    apply_to: list[ApplyTo] = Field(min_length=1)


class SubstitutionPlan(_Base):
    title_modifier: Optional[str] = None
    substitutions: list[Substitution] = Field(default_factory=list)
    additions_to_skills: list[str] = Field(default_factory=list)
    summary_focus: str


# ---------- Stage 5: Validator ----------


class ValidationIssue(_Base):
    severity: Severity
    # Dotted path like "experience[2].bullets[4]" or "header.title".
    section: str
    issue: str
    original: str
    rewritten: str


class ValidationReport(_Base):
    passed: bool
    keyword_match_rate: float = Field(ge=0.0, le=1.0)
    issues: list[ValidationIssue] = Field(default_factory=list)

    @model_validator(mode="after")
    def _passed_implies_no_critical(self) -> "ValidationReport":
        if self.passed and any(i.severity == "critical" for i in self.issues):
            raise ValueError("passed=True but critical issues are present")
        return self
