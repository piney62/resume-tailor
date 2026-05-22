"""Jinja2 template loader for LLM prompts.

StrictUndefined is intentional: a missing template variable should fail
loudly, not silently produce an empty string in the prompt.
"""

from pathlib import Path

from jinja2 import Environment, FileSystemLoader, StrictUndefined

_PROMPTS_DIR = Path(__file__).parent / "prompts"

_env = Environment(
    loader=FileSystemLoader(_PROMPTS_DIR),
    autoescape=False,
    undefined=StrictUndefined,
    trim_blocks=True,
    lstrip_blocks=True,
    keep_trailing_newline=False,
)


def render(template_name: str, **context: object) -> str:
    return _env.get_template(template_name).render(**context).strip()
