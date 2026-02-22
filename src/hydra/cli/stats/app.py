from typing import Literal, Annotated, Optional
import typer
import logging
from pathlib import Path

from .analysis.app import app as analyze_app
from hydra.core.configs import HuggingFaceDatasetsConfig, DataLoaderConfig, DataModuleConfig

app = typer.Typer(help="MyCLI: a tiny example Typer app for stats.")
app.add_typer(analyze_app)

def main() -> None:
    # Entry point for console scripts
    app()

if __name__ == "__main__":
    main()
