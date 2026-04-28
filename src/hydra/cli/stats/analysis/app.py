from pathlib import Path
from typing import Annotated, Literal
import json

import typer
from tqdm import tqdm

from concurrent.futures import ProcessPoolExecutor

from hydra.core.analysis.utils import hif_discovery, results_discovery, comparison_discovery
from hydra.core.analysis.quantitative import quantitative_analysis
from hydra.core.analysis.comparative import comparative_analysis

app = typer.Typer(help="MyCLI: a tiny example Typer app.")

@app.command()
def analyze(
    ctx: typer.Context,
    root_dir: Annotated[Path, typer.Option("--root-dir", help="Path to the checkpoint to sample from.")] = Path("samples"),
    include_metrics: Annotated[list[str], typer.Option("--include-metric", "-m", help="List of metrics to include in the analysis.")] = [],
    exclude_metrics: Annotated[list[str], typer.Option("--exclude-metric", "-e", help="List of metrics to exclude from the analysis.")] = [],
    force_metrics: Annotated[list[str], typer.Option("--force-metric", "-f", help="List of metrics to force recompute in the analysis.")] = [],
):
    include_metrics.extend(force_metrics)
    for path, hypergraph in tqdm(hif_discovery(root_dir)):
        results_path = path.with_suffix(".results.json")
        results = dict()
        if results_path.exists():
            with open(results_path, "r") as f:
                results = json.load(f)
        original_keys = set(results.keys())
        results.update(quantitative_analysis(hypergraph,
                                     include_metrics=include_metrics,
                                     exclude_metrics=[metric for metric in list(results.keys()) + exclude_metrics if metric not in force_metrics]))
        if set(results.keys()) != original_keys or force_metrics:
            with open(results_path, "w") as f:
                json.dump(results, f, indent=None, sort_keys=True, separators=(",", ":"))

@app.command()
def compare(
    ctx: typer.Context,
    root_dir: Annotated[Path, typer.Option("--root-dir", help="Path to the checkpoint to sample from.")] = Path("samples"),
    references_dir: Annotated[Path, typer.Option("--references-dir", help="Path to the reference hypergraphs.")] = Path("references"),
    split: Annotated[str, typer.Option("--split", help="Split of the dataset to compare.")] = "full",
    include_metrics: Annotated[list[str], typer.Option("--include-metric", "-m", help="List of metrics to include in the analysis.")] = [],
    exclude_metrics: Annotated[list[str], typer.Option("--exclude-metric", "-e", help="List of metrics to exclude from the analysis.")] = [],
    force_metrics: Annotated[list[str], typer.Option("--force-metric", "-f", help="List of metrics to force recompute in the analysis.")] = [],
):
    references = dict()
    for path, results in results_discovery(references_dir):
        dataset_name = results['dataset_name']
        if results['name'] != split:
            continue
        references[dataset_name] = results
    include_metrics.extend(force_metrics)
    for path, results in tqdm(results_discovery(root_dir)):
        print(path)
        dataset_name = results['dataset_name']
        if 'daqh/' not in dataset_name:
            dataset_name = 'daqh/' + dataset_name
        reference = references[dataset_name]
        comparison_path = path.with_suffix(".comparison.json")
        comparison_results = dict()
        if comparison_path.exists():
            with open(comparison_path, "r") as f:
                comparison_results = json.load(f)
        comparison_results.update(comparative_analysis(results,
                                                       reference,
                                                       include_metrics=include_metrics,
                                                       exclude_metrics=[metric for metric in list(comparison_results.keys()) + exclude_metrics if metric not in force_metrics]))
        with open(comparison_path, "w") as f:
            json.dump(comparison_results, f, indent=None, sort_keys=True, separators=(",", ":"))

from datasets import load_dataset
import xgi

@app.command()
def parse(
    ctx: typer.Context,
    dataset_name: str,
    root_dir: Annotated[Path, typer.Option("--root-dir", help="Path to the checkpoint to sample from.")] = Path("references"),
    cache_dir: Annotated[Path, typer.Option("--cache-dir", help="Path to the checkpoint to sample from.")] = Path("cache"),
    split: Annotated[str, typer.Option("--split", help="Split of the dataset to parse.")] = "full",
):
    dataset = load_dataset(dataset_name,
                           split=split,
                           cache_dir=cache_dir)[0]
    hypergraph = xgi.from_hif_dict(dataset)
    hyperedges = hypergraph.edges.members()
    hypergraph = xgi.Hypergraph(hyperedges)
    hypergraph['dataset_name'] = dataset_name
    hypergraph['name'] = split
    hypergraph['kind'] = 'reference'
    path = root_dir / dataset_name / f"{split}.hif.json"
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as f:
        json.dump(xgi.to_hif_dict(hypergraph), f, indent=None, sort_keys=True, separators=(",", ":"))

@app.command()
def clear(
    ctx: typer.Context,
    root_dir: Annotated[Path, typer.Option("--root-dir", help="Path to the checkpoint to sample from.")] = Path("samples"),
    kind: Annotated[Literal["all", "results", "comparison"], typer.Option("--kind", "-k", help="Kind of results to clear.")] = "all",
):
    if kind in ("all", "results"):
        for path, _ in results_discovery(root_dir):
            path.unlink()
    if kind in ("all", "comparison"):
        for path, _ in comparison_discovery(root_dir):
            path.unlink()
