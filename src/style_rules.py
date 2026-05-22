"""Style rules shared by the Rewriter (Step 7) and Validator (Step 8).

BANNED_WORDS is intentionally conservative: each item is a recognized
ATS / hiring-manager fatigue phrase ("leveraged", "synergy", "rockstar")
or a generic filler ("results-driven", "team player") that adds no signal.
"""

BANNED_WORDS: tuple[str, ...] = (
    "leveraged", "leverage",
    "synergy", "synergies", "synergize",
    "utilized", "utilize",
    "spearheaded",
    "pivotal",
    "guru", "ninja", "rockstar",
    "thought leader", "thought-leader",
    "impactful",
    "world-class",
    "results-driven", "results-oriented",
    "self-starter",
    "team player",
    "go-getter",
    "value-add", "value-added",
    "deep dive",
    "hit the ground running",
    "circle back",
    "low-hanging fruit",
    "moving the needle",
    "out-of-the-box thinker",
)
