from typing import Annotated
from pathlib import Path
import json
from os import makedirs

import typer
import xgi

from janus.core.sample.bvae import sample_bvae
from janus.core.sample.ddm import sample_ddm
from janus.core.configs import RandomWalkConfig
from .utils import get_current_sample_path

app = typer.Typer(help="MyCLI: a tiny example Typer app.")

@app.callback()
def main(
    ctx: typer.Context,
    ckpt_path: Annotated[Path, typer.Option("--ckpt-path", "--ckpt", help="Path to the checkpoint to sample from.")],
    samples_path: Annotated[Path, typer.Option("--samples-path", help="Path to the samples dir to save into.")] = Path('samples')
):
    """MyCLI: a tiny example Typer app."""
    ctx.ensure_object(dict)
    ctx.obj['ckpt_path'] = ckpt_path
    ctx.obj['samples_path'] = samples_path

@app.command()
def bvae(
    ctx: typer.Context,
    # RandomWalkConfig options
    p: Annotated[float, typer.Option("-p", help="Biased random walk p parameter, controlling likelihood of immediately revisiting a node.")] = 2.0,
    q: Annotated[float, typer.Option("-q", help="Biased random walk q parameter, controlling likelihood of visiting nodes further away from the source node.")] = 0.5,
    alpha: Annotated[float, typer.Option("--alpha", help="Metropolis-Hastings acceptance probability parameter.")] = 0.0,
    walk_length: Annotated[int, typer.Option("--walk-length", help="Length of each random walk.")] = 256,
    samples_per_hyperedge: Annotated[int, typer.Option("--samples-per-hyperedge", help="Number of random walks to sample per hyperedge.")] = 1,
):
    """Train a model on the specified dataset."""
    ckpt_path = ctx.obj['ckpt_path']
    samples_path = ctx.obj['samples_path']
    random_walk_config = RandomWalkConfig(p=p,
                                         q=q,
                                         alpha=alpha,
                                         walk_length=walk_length,
                                         samples_per_hyperedge=samples_per_hyperedge)
    dataloader_config = ctx.obj['dataloader_config']
    huggingface_datasets_config = ctx.obj['huggingface_datasets_config']
    datamodule_config = ctx.obj['datamodule_config']
    hypergraph = sample_bvae(random_walk_config,
                datamodule_config,
                huggingface_datasets_config,
                dataloader_config,
                ckpt_path)
    model_samples_path = samples_path / hypergraph['dataset_name'].split("/")[-1] / hypergraph['name']
    makedirs(model_samples_path, exist_ok=True)
    current_sample_path = model_samples_path / get_current_sample_path(model_samples_path)
    makedirs(current_sample_path, exist_ok=True)
    hif_dict = xgi.to_hif_dict(hypergraph)
    with open(current_sample_path / "hypergraph.hif.json", "w") as f:
        json.dump(hif_dict, f, indent=None, separators=(',', ":"))

@app.command()
def ddm(ctx: typer.Context,
        walk_length: Annotated[int, typer.Option("--walk-length", help="Length of each random walk.")] = 256):
    """Train a node set constrained model on the specified dataset."""
    samples_path = ctx.obj['samples_path']
    huggingface_datasets_config = ctx.obj['huggingface_datasets_config']
    dataloader_config = ctx.obj['dataloader_config']
    datamodule_config = ctx.obj['datamodule_config']
    ckpt_path = ctx.obj['ckpt_path']
    hypergraphs = sample_ddm(
        datamodule_config,
        huggingface_datasets_config,
        dataloader_config,
        walk_length,
        ckpt_path
    )
    dataset_name = huggingface_datasets_config.dataset_name.split("/")[-1]
    for hypergraph in hypergraphs:
        model_samples_path = samples_path / dataset_name / hypergraph['name']
        makedirs(model_samples_path, exist_ok=True)
        current_sample_path = model_samples_path / get_current_sample_path(model_samples_path)
        makedirs(current_sample_path, exist_ok=True)
        hif_dict = xgi.to_hif_dict(hypergraph)
        with open(current_sample_path / "hypergraph.hif.json", "w") as f:
            json.dump(hif_dict, f, indent=None, separators=(',', ":"))
