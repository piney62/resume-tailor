"""Role-recency tier classifier.

Maps a role list (assumed sorted most-recent-first by the resume parser)
to a parallel list of recency tiers used by the Holistic Rewriter to
apply different JD-injection strategies per role:

    recent: most recent role only          → inject every JD must-have
    mid:    everything between             → apply JD signals naturally
    oldest: the last role in the list      → keep byte-for-byte verbatim

The classifier is deliberately position-based (not date-based). It treats
the role at index 0 as the most recent and the role at index N-1 as the
oldest. The Resume Parser already establishes that ordering for the
common one-page resume layouts we support.
"""

from src.models.schemas import Experience, RoleTier


def classify_tiers(experience: list[Experience]) -> list[RoleTier]:
    n = len(experience)
    if n == 0:
        return []
    if n == 1:
        return ["recent"]
    if n == 2:
        return ["recent", "oldest"]
    return ["recent"] + ["mid"] * (n - 2) + ["oldest"]
