from pathlib import Path
from typing import Annotated

import typer

app = typer.Typer(help="MyCLI: a tiny example Typer app.")

@app.command()
def report(ctx: typer.Context):
    print(ctx.obj)
