from pathlib import Path
from typing import Annotated

import typer

from hydra.core.analysis.utils import hif_discovery
from hydra.core.analysis.runner import analyze_hypergraph

app = typer.Typer(help="MyCLI: a tiny example Typer app.")

import xgi

@app.command()
def analyze(
    ctx: typer.Context,
    root_dir: Annotated[Path, typer.Option("--root-dir", help="Path to the checkpoint to sample from.")] = Path("samples"),
    include_metrics: Annotated[list[str] | None, typer.Option("--include-metrics", help="List of metrics to include in the analysis.")] = None,
    split: Annotated[str, typer.Option("--split", help="Split of the dataset to use.")] = "full",
    data_dir: Annotated[Path, typer.Option("--data-dir", help="Path to the data directory.")] = Path("data"),
    cache_dir: Annotated[Path, typer.Option("--cache-dir", help="Path to the cache directory.")] = Path("cache")
):
    for hypergraph in hif_discovery(root_dir):
        results = analyze_hypergraph(hypergraph,
                                     include_metrics=include_metrics,
                                     split=split,
                                     data_dir=data_dir,
                                     cache_dir=cache_dir)
