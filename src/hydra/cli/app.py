from typing import Literal, Annotated, Optional
import typer

from .train import app as train_app
from .sample import app as sample_app

app = typer.Typer(help="MyCLI: a tiny example Typer app.")
app.add_typer(train_app, name="train")
app.add_typer(sample_app, name="sample")

def main() -> None:
    # Entry point for console scripts
    app()

if __name__ == "__main__":
    main()
