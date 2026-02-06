from typing import Literal, Annotated, Optional
import typer
import logging

import rich.logging

from .train.app import app as train_app
from .sample import app as sample_app

logger = logging.getLogger(__name__)

app = typer.Typer(help="MyCLI: a tiny example Typer app.")
app.add_typer(train_app, name="train")
app.add_typer(sample_app, name="sample")

@app.callback()
def main(
    ctx: typer.Context,
    log_level: Annotated[Literal["DEBUG", "INFO", "WARNING", "ERROR"], typer.Option("--log-level", help="Set the logging level.")] = "WARNING",
):
    """MyCLI: a tiny example Typer app."""
    root_logger = logging.getLogger()
    root_logger.setLevel(log_level)
    root_logger.addHandler(rich.logging.RichHandler())

def main() -> None:
    # Entry point for console scripts
    app()

if __name__ == "__main__":
    main()
