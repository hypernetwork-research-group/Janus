from typing import Annotated
from pathlib import Path

import typer

from hydra.core.sample.bvae import sample_bvae
from hydra.core.sample.ddm import sample_ddm
from hydra.core.configs import DataModuleConfig, RandomWalkConfig

app = typer.Typer(help="MyCLI: a tiny example Typer app.")

@app.callback()
def main(
    ctx: typer.Context,
    ckpt_path: Annotated[Path, typer.Option("--ckpt-path", "--ckpt", help="Path to the checkpoint to sample from.")],
):
    """MyCLI: a tiny example Typer app."""
    ctx.ensure_object(dict)
    ctx.obj['ckpt_path'] = ckpt_path

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
    random_walk_config = RandomWalkConfig(p=p,
                                         q=q,
                                         alpha=alpha,
                                         walk_length=walk_length,
                                         samples_per_hyperedge=samples_per_hyperedge)
    dataloader_config = ctx.obj['dataloader_config']
    huggingface_datasets_config = ctx.obj['huggingface_datasets_config']
    datamodule_config = ctx.obj['datamodule_config']
    sample_bvae(random_walk_config,
                datamodule_config,
                huggingface_datasets_config,
                dataloader_config,
                ckpt_path)

@app.command()
def ddm(ctx: typer.Context,
        walk_length: Annotated[int, typer.Option("--walk-length", help="Length of each random walk.")] = 256):
    """Train a conditional model on the specified dataset."""
    huggingface_datasets_config = ctx.obj['huggingface_datasets_config']
    dataloader_config = ctx.obj['dataloader_config']
    datamodule_config = ctx.obj['datamodule_config']
    ckpt_path = ctx.obj['ckpt_path']
    sample_ddm(
        datamodule_config,
        huggingface_datasets_config,
        dataloader_config,
        walk_length,
        ckpt_path
    )
