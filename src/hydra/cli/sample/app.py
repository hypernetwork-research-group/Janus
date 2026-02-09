from typing import Annotated
from pathlib import Path

import typer

from hydra.core.sample.bvae import sample_bvae
from hydra.core.sample.ddm import sample_ddm
from hydra.core.configs import DataModuleConfig

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
    # DataModuleConfig options
    p: Annotated[float, typer.Option("-p", help="Biased random walk p parameter, controlling likelihood of immediately revisiting a node.")] = 2.0,
    q: Annotated[float, typer.Option("-q", help="Biased random walk q parameter, controlling likelihood of visiting nodes further away from the source node.")] = 0.5,
    alpha: Annotated[float, typer.Option("--alpha", help="Metropolis-Hastings acceptance probability parameter.")] = 0.0,
    walk_length: Annotated[int, typer.Option("--walk-length", help="Length of each random walk.")] = 256,
    samples_per_hyperedge: Annotated[int, typer.Option("--samples-per-hyperedge", help="Number of random walks to sample per hyperedge.")] = 1,
    data_dir: Annotated[Path, typer.Option("--data-dir", help="Data directory for datasets.")] = Path("./data"),
    retain_lcc: Annotated[bool, typer.Option("--retain-lcc/--no-retain-lcc", help="Whether to retain only the largest connected component of the hypergraph.")] = True,
    train_split: Annotated[str, typer.Option("--train-split", help="Dataset split(s) to use for training.")] = "full",
    val_split: Annotated[str, typer.Option("--val-split", help="Dataset split(s) to use for validation.")] = "full",
    predict_split: Annotated[str, typer.Option("--predict-split", help="Dataset split(s) to use for prediction.")] = "full",
    val_size: Annotated[float, typer.Option("--val-size", help="Ignored if train_split != val_split. Size of the validation set. If float, represents the proportion of the dataset to include in the validation split. If int, represents the absolute number of examples. If None, the value is set to 0.1.")] = None,
):
    """Train a model on the specified dataset."""
    ckpt_path = ctx.obj['ckpt_path']
    datamodule_config = DataModuleConfig(p=p,
                                         q=q,
                                         alpha=alpha,
                                         walk_length=walk_length,
                                         samples_per_hyperedge=samples_per_hyperedge,
                                         data_dir=data_dir,
                                         retain_lcc=retain_lcc,
                                         train_split=train_split,
                                         val_split=val_split,
                                         predict_split=predict_split,
                                         val_size=val_size)
    dataloader_config = ctx.obj['dataloader_config']
    huggingface_datasets_config = ctx.obj['huggingface_datasets_config']
    sample_bvae(datamodule_config,
                huggingface_datasets_config,
                dataloader_config,
                ckpt_path)

@app.command()
def ddm(ctx: typer.Context):
    """Train a conditional model on the specified dataset."""
    huggingface_datasets_config = ctx.obj['huggingface_datasets_config']
    print(ctx.obj)
    # sample_ddm(
    #
    # )
