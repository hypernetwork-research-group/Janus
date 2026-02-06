import typer
from typing import Annotated

import typer

from ...core.configs import DataModuleConfig, DataLoaderConfig, TrainerConfig, HuggingFaceDatasetsConfig, OptimizerConfig, EarlyStoppingConfig
from ...core.train.ddm import train_ddm

from .app import app

@app.command()
def ddm(ctx: typer.Context,
        T: Annotated[int, typer.Option("--T", help="Number of diffusion steps.")] = 1000):
    """Train a conditional model on the specified dataset."""
    datamodule_config: DataModuleConfig = ctx.obj["datamodule_config"]
    huggingface_datasets_config: HuggingFaceDatasetsConfig = ctx.obj["huggingface_datasets_config"]
    dataloader_config: DataLoaderConfig = ctx.obj["dataloader_config"]
    trainer_config: TrainerConfig = ctx.obj["trainer_config"]
