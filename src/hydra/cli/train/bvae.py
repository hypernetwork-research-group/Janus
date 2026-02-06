import typer
from typing import Annotated

import typer

from ...core.configs import DataModuleConfig, DataLoaderConfig, TrainerConfig, HuggingFaceDatasetsConfig, OptimizerConfig, EarlyStoppingConfig
from ...core.train.bvae import train_bvae
from ...core.configs import ModelSizeConfig

from .app import app

@app.command()
def bvae(ctx: typer.Context,
         vertex_encoding: Annotated[bool, typer.Option("--vertex-encoding/--no-vertex-encoding", help="Whether to encode nodes in the model.")] = False,
         kl_weight: Annotated[float, typer.Option("--kl-weight", help="Weight of the KL divergence term in the loss function.")] = 1e-6):
    """Train a model on the specified dataset."""
    datamodule_config: DataModuleConfig = ctx.obj["datamodule_config"]
    huggingface_datasets_config: HuggingFaceDatasetsConfig = ctx.obj["huggingface_datasets_config"]
    dataloader_config: DataLoaderConfig = ctx.obj["dataloader_config"]
    trainer_config: TrainerConfig = ctx.obj["trainer_config"]
    model_size_config: ModelSizeConfig = ctx.obj["model_size_config"]
    optimizer_config: OptimizerConfig = ctx.obj["optimizer_config"]
    early_stopping_config: EarlyStoppingConfig = ctx.obj["early_stopping_config"]
    train_bvae(datamodule_config=datamodule_config,
              dataloader_config=dataloader_config,
              trainer_config=trainer_config,
              huggingface_datasets_config=huggingface_datasets_config,
              model_size_config=model_size_config,
              optimizer_config=optimizer_config,
              early_stopping_config=early_stopping_config,
              vertex_encoding=vertex_encoding,
              kl_weight=kl_weight)
