"""CLI entrypoint. Implemented in Step 11."""

import typer

app = typer.Typer(
    name="resume-tailor",
    help="Tailor a .docx resume to a JD using a multi-stage LLM pipeline.",
    no_args_is_help=True,
)


@app.command()
def tailor() -> None:
    """Run the full tailoring pipeline (not yet implemented)."""
    raise typer.Exit(code=1)


if __name__ == "__main__":
    app()
