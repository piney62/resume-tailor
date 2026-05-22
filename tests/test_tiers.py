"""Tests for src.stages.tiers.classify_tiers."""

from src.models.schemas import Experience, ExperienceIndices
from src.stages.tiers import classify_tiers


def _role(company: str) -> Experience:
    return Experience(
        company=company,
        title="Engineer",
        dates="2020 - 2021",
        location="Remote",
        intro="",
        bullets=[],
        skills_line=None,
        indices=ExperienceIndices(header_idxs=[0]),
    )


def test_empty_experience() -> None:
    assert classify_tiers([]) == []


def test_single_role_is_recent() -> None:
    assert classify_tiers([_role("Acme")]) == ["recent"]


def test_two_roles_recent_and_oldest() -> None:
    assert classify_tiers([_role("Now"), _role("Then")]) == ["recent", "oldest"]


def test_three_roles_recent_mid_oldest() -> None:
    assert classify_tiers(
        [_role("Now"), _role("Mid"), _role("Then")]
    ) == ["recent", "mid", "oldest"]


def test_four_roles_one_recent_two_mid_one_oldest() -> None:
    tiers = classify_tiers([_role(c) for c in ["Now", "Mid1", "Mid2", "Then"]])
    assert tiers == ["recent", "mid", "mid", "oldest"]


def test_five_roles_one_recent_three_mid_one_oldest() -> None:
    tiers = classify_tiers([_role(c) for c in ["A", "B", "C", "D", "E"]])
    assert tiers == ["recent", "mid", "mid", "mid", "oldest"]
