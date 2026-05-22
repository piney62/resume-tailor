"""Loader for few-shot examples stored as YAML.

Examples live in src/llm/few_shot/{name}.yaml and are rendered into
rewrite prompts. Storing them as data (not code) lets us tweak example
quality without touching prompt template logic.
"""

from pathlib import Path

import yaml

_DIR = Path(__file__).parent / "few_shot"


def load_examples(name: str) -> list[dict]:
    path = _DIR / f"{name}.yaml"
    if not path.exists():
        return []
    with path.open(encoding="utf-8") as f:
        data = yaml.safe_load(f) or []
    if not isinstance(data, list):
        raise ValueError(f"{path} must contain a YAML list at the top level")
    return data
