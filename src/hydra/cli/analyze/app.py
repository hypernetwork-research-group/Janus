from pathlib import Path
from typing import Annotated

import typer

app = typer.Typer(help="MyCLI: a tiny example Typer app.")

@app.command()
def analyze(
    ctx: typer.Context,
    sample_path: Annotated[Path, typer.Option("--sample-path", "--sample", help="Path to the checkpoint to sample from.")],
):
    pass
