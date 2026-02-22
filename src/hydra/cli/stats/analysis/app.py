from pathlib import Path
from typing import Annotated
import json

import typer
from tqdm import tqdm

from hydra.core.analysis.utils import hif_discovery, results_discovery
from hydra.core.analysis.quantitative import quantitative_analysis

app = typer.Typer(help="MyCLI: a tiny example Typer app.")

@app.command()
def analyze(
    ctx: typer.Context,
    root_dir: Annotated[Path, typer.Option("--root-dir", help="Path to the checkpoint to sample from.")] = Path("samples"),
    include_metrics: Annotated[list[str], typer.Option("--include-metrics", "-m", help="List of metrics to include in the analysis.")] = [],
    exclude_metrics: Annotated[list[str], typer.Option("--exclude-metrics", "-e", help="List of metrics to exclude from the analysis.")] = [],
):
    for hypergraph in tqdm(hif_discovery(root_dir)):
        path = Path(hypergraph.xgi_hypergraph['path'])
        results_path = path.with_suffix(".results.json")
        results = dict()
        if results_path.exists():
            with open(results_path, "r") as f:
                results = json.load(f)
        exclude_metrics = list(results.keys()) + exclude_metrics
        results.update(quantitative_analysis(hypergraph,
                                     include_metrics=include_metrics,
                                     exclude_metrics=exclude_metrics))
        with open(results_path, "w") as f:
            json.dump(results, f, indent=2)

@app.command()
def clear(
    ctx: typer.Context,
    root_dir: Annotated[Path, typer.Option("--root-dir", help="Path to the checkpoint to sample from.")] = Path("samples"),
):
    for results in results_discovery(root_dir):
        path = Path(results['path'])
        path.unlink()
