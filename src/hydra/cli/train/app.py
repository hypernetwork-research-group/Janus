import logging
from typing import Annotated
from pathlib import Path

import typer

from hydra.core.configs import DataModuleConfig, DataLoaderConfig, TrainerConfig, HuggingFaceDatasetsConfig, OptimizerConfig, EarlyStoppingConfig
from hydra.core.models.enums import ModelSize

logger = logging.getLogger(__name__)
app = typer.Typer(help="MyCLI: a tiny example Typer app.")

@app.callback()
def train_callback(
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
    val_size: Annotated[float | None, typer.Option("--val-size", help="Ignored if train_split != val_split. Size of the validation set. If float, represents the proportion of the dataset to include in the validation split. If int, represents the absolute number of examples. If None, the value is set to 0.1.")] = None,
    # TrainerConfig options
    max_epochs: Annotated[int, typer.Option("--max-epochs", help="Maximum number of training epochs.")] = -1,
    accumulate_grad_batches: Annotated[int, typer.Option("--accumulate-grad-batches", help="Number of batches to accumulate gradients over.")] = 1,
    # ModelSizeConfig options
    model_size: Annotated[ModelSize, typer.Option("--model-size", help="Size of the model to use.")] = ModelSize.M,
    # OptimizerConfig options
    # Learning rate is --learning-rate or -lr
    learning_rate: Annotated[float | None, typer.Option("--learning-rate", "-lr", help="Learning rate for the optimizer. If not set, Learning Rate Finder will be used to determine it.")] = None,
    weight_decay: Annotated[float | None, typer.Option("--weight-decay", help="Weight decay (L2 regularization) for the optimizer. If not set, model defaults are used.")] = None,
    # EarlyStoppingConfig options
    patience: Annotated[int, typer.Option("--patience", help="Number of epochs with no improvement after which training will be stopped.")] = 70,
):
    """Common options for data loading."""
    # ctx.obj is the standard place to store shared state across commands :contentReference[oaicite:3]{index=3}
    ctx.ensure_object(dict)
    ctx.obj["datamodule_config"] = DataModuleConfig(p=p,
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
    ctx.obj["trainer_config"] = TrainerConfig(max_epochs=max_epochs,
                                              accumulate_grad_batches=accumulate_grad_batches)
    ctx.obj["model_size_config"] = model_size.value
    ctx.obj["optimizer_config"] = OptimizerConfig(learning_rate=learning_rate,
                                                  weight_decay=weight_decay)
    ctx.obj["early_stopping_config"] = EarlyStoppingConfig(patience=patience)

from hydra.core.train.bvae import train_bvae

@app.command()
def bvae(ctx: typer.Context,
         vertex_encoding: Annotated[bool, typer.Option("--vertex-encoding/--no-vertex-encoding", help="Whether to encode nodes in the model.")] = False,
         kl_weight: Annotated[float, typer.Option("--kl-weight", help="Weight of the KL divergence term in the loss function.")] = 1e-6,
         latent_dim: Annotated[int | None, typer.Option("--latent-dim", help="Dimensionality of the latent space. If None, use input feature dimension.")] = None):
    """Train a model on the specified dataset."""
    datamodule_config: DataModuleConfig = ctx.obj["datamodule_config"]
    huggingface_datasets_config: HuggingFaceDatasetsConfig = ctx.obj["huggingface_datasets_config"]
    dataloader_config: DataLoaderConfig = ctx.obj["dataloader_config"]
    trainer_config: TrainerConfig = ctx.obj["trainer_config"]
    model_size_config: str = ctx.obj["model_size_config"]
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
              kl_weight=kl_weight,
              latent_dim=latent_dim)

from hydra.core.train.ddm import train_ddm

@app.command()
def ddm(ctx: typer.Context,
        bvae_ckpt: Annotated[Path, typer.Option("--bvae-ckpt", help="Path to the pretrained BVAE checkpoint to use for the DDM.")],
        T: Annotated[int, typer.Option("--T", help="Number of diffusion steps.")] = 1000):
    """Train a conditional model on the specified dataset."""
    datamodule_config: DataModuleConfig = ctx.obj["datamodule_config"]
    huggingface_datasets_config: HuggingFaceDatasetsConfig = ctx.obj["huggingface_datasets_config"]
    dataloader_config: DataLoaderConfig = ctx.obj["dataloader_config"]
    trainer_config: TrainerConfig = ctx.obj["trainer_config"]
    optimizer_config: OptimizerConfig = ctx.obj["optimizer_config"]
    model_size_config: str = ctx.obj["model_size_config"]

    train_ddm(datamodule_config=datamodule_config,
              dataloader_config=dataloader_config,
              trainer_config=trainer_config,
              optimizer_config=optimizer_config,
              huggingface_datasets_config=huggingface_datasets_config,
              model_size_config=model_size_config,
              bvae_ckpt=bvae_ckpt,
              T=T)
