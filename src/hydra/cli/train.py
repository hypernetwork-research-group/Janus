from typing import Annotated, List
from pathlib import Path
from dataclasses import dataclass

import typer

app = typer.Typer(help="MyCLI: a tiny example Typer app.")

@dataclass
class DataModuleConfig:
    p: float
    q: float
    alpha: float
    walk_length: int
    samples_per_hyperedge: int
    data_dir: Path
    retain_lcc: bool
    train_split: List[str]
    val_split: List[str]
    predict_split: List[str]

@dataclass
class HuggingFaceDatasetsConfig:
    dataset_name: str
    cache_dir: Path

@dataclass
class DataLoaderConfig:
    pin_memory: bool
    num_workers: int
    persistent_workers: bool
    batch_size: int

@dataclass
class TrainerConfig:
    max_epochs: int
    accumulate_grad_batches: int

@app.callback()
def train_callback(
    ctx: typer.Context,
    # HuggingFaceDatasetsConfig options
    dataset_name: Annotated[str, typer.Argument(..., help="Name of the dataset to use.")],
    cache_dir: Annotated[Path, typer.Option("--cache-dir", help="Cache directory for datasets.")] = Path("./cache"),
    # DataModuleConfig options
    p: Annotated[float, typer.Option("-p", help="Biased random walk p parameter, controlling likelihood of immediately revisiting a node.")] = 1.0,
    q: Annotated[float, typer.Option("-q", help="Biased random walk q parameter, controlling likelihood of visiting nodes further away from the source node.")] = 1.0,
    alpha: Annotated[float, typer.Option("--alpha", help="Metropolis-Hastings acceptance probability parameter.")] = 0.0,
    walk_length: Annotated[int, typer.Option("--walk-length", help="Length of each random walk.")] = 256,
    samples_per_hyperedge: Annotated[int, typer.Option("--samples-per-hyperedge", help="Number of random walks to sample per hyperedge.")] = 1,
    data_dir: Annotated[Path, typer.Option("--data-dir", help="Data directory for datasets.")] = Path("./data"),
    retain_lcc: Annotated[bool, typer.Option("--retain-lcc/--no-retain-lcc", help="Whether to retain only the largest connected component of the hypergraph.")] = True,
    train_split: Annotated[List[str], typer.Option("--train-split", help="Dataset split(s) to use for training.")] = ["full"],
    val_split: Annotated[List[str], typer.Option("--val-split", help="Dataset split(s) to use for validation.")] = ["full"],
    predict_split: Annotated[List[str], typer.Option("--predict-split", help="Dataset split(s) to use for prediction.")] = ["full"],
    # DataLoaderConfig options
    pin_memory: Annotated[bool, typer.Option("--pin-memory/--no-pin-memory", help="Whether to pin memory in DataLoader.")] = True,
    num_workers: Annotated[int, typer.Option("--num-workers", help="Number of workers for DataLoader.")] = None,
    persistent_workers: Annotated[bool, typer.Option("--persistent-workers/--no-persistent-workers", help="Whether DataLoader should use persistent workers.")] = True,
    batch_size: Annotated[int, typer.Option("--batch-size", help="Batch size for DataLoader.")] = 128,
    # TrainerConfig options
    max_epochs: Annotated[int, typer.Option("--max-epochs", help="Maximum number of training epochs.")] = -1,
    accumulate_grad_batches: Annotated[int, typer.Option("--accumulate-grad-batches", help="Number of batches to accumulate gradients over.")] = 1,
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
                                                    predict_split=predict_split)
    ctx.obj["dataloader_config"] = DataLoaderConfig(pin_memory=pin_memory,
                                                    num_workers=num_workers,
                                                    persistent_workers=persistent_workers,
                                                    batch_size=batch_size)
    ctx.obj["trainer_config"] = TrainerConfig(max_epochs=max_epochs,
                                              accumulate_grad_batches=accumulate_grad_batches)
    ctx.obj["huggingface_datasets_config"] = HuggingFaceDatasetsConfig(dataset_name=dataset_name,
                                                                     cache_dir=cache_dir)

from ..core.datamodules import HypergraphDataModule

@app.command()
def bvae(ctx: typer.Context,
         encode_nodes: Annotated[bool, typer.Option("--encode-nodes/--no-encode-nodes", help="Whether to encode nodes in the model.")] = True):
    """Train a model on the specified dataset."""
    typer.echo(ctx.obj)
    datamodule_config: DataModuleConfig = ctx.obj["datamodule_config"]
    huggingface_datasets_config: HuggingFaceDatasetsConfig = ctx.obj["huggingface_datasets_config"]
    dataloader_config: DataLoaderConfig = ctx.obj["dataloader_config"]
    trainer_config: TrainerConfig = ctx.obj["trainer_config"]

    dataset = HypergraphDataModule(dataset_name=huggingface_datasets_config.dataset_name,
                        data_dir=datamodule_config.data_dir,
                        retain_lcc=datamodule_config.retain_lcc,
                        cache_dir=huggingface_datasets_config.cache_dir,
                        p=datamodule_config.p,
                        q=datamodule_config.q,
                        alpha=datamodule_config.alpha,
                        walk_length=datamodule_config.walk_length,
                        samples_per_hyperedge=datamodule_config.samples_per_hyperedge,
                        pin_memory=dataloader_config.pin_memory,
                        num_workers=dataloader_config.num_workers,
                        persistent_workers=dataloader_config.persistent_workers,
                        batch_size=dataloader_config.batch_size)
    dataset.prepare_data()
    dataset.setup("fit")

@app.command()
def ddm(ctx: typer.Context):
    """Train a conditional model on the specified dataset."""
    typer.echo(ctx.obj)
    datamodule_config: DataModuleConfig = ctx.obj["datamodule_config"]
    huggingface_datasets_config: HuggingFaceDatasetsConfig = ctx.obj["huggingface_datasets_config"]
    dataloader_config: DataLoaderConfig = ctx.obj["dataloader_config"]
    trainer_config: TrainerConfig = ctx.obj["trainer_config"]
