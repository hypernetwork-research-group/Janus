from typing import Literal, Annotated, Optional
import typer
import logging
from pathlib import Path
import resource

import psutil

import rich.logging

from .train.app import app as train_app
from .sample.app import app as sample_app
from hydra.core.configs import HuggingFaceDatasetsConfig, DataLoaderConfig, DataModuleConfig

logger = logging.getLogger(__name__)

app = typer.Typer(help="MyCLI: a tiny example Typer app.")
app.add_typer(train_app, name="train")
app.add_typer(sample_app, name="sample")

@app.callback()
def main_callback(
    ctx: typer.Context,
    # HuggingFaceDatasetsConfig options
    dataset_name: Annotated[str, typer.Argument(..., help="Name of the dataset to use.")],
    cache_dir: Annotated[Path, typer.Option("--cache-dir", help="Cache directory for datasets.")] = Path("./cache"),
    # DataLoaderConfig options
    pin_memory: Annotated[bool, typer.Option("--pin-memory/--no-pin-memory", help="Whether to pin memory in DataLoader.")] = True,
    num_workers: Annotated[int | None, typer.Option("--num-workers", help="Number of workers for DataLoader.")] = None,
    persistent_workers: Annotated[bool, typer.Option("--persistent-workers/--no-persistent-workers", help="Whether DataLoader should use persistent workers.")] = True,
    batch_size: Annotated[int | None, typer.Option("--batch-size", help="Batch size for DataLoader.")] = None,
    drop_last: Annotated[bool, typer.Option("--drop-last/--no-drop-last", help="Whether to drop the last incomplete batch in DataLoader.")] = False,
    node_feature: Annotated[str, typer.Option("--node-feature", help="Node feature to use for training.")] = "n2v",
    hyperedge_feature: Annotated[str, typer.Option("--hyperedge-feature", help="Hyperedge feature to use for training.")] = "n2v",
    # DataModuleConfig options
    data_dir: Annotated[Path, typer.Option("--data-dir", help="Data directory for datasets.")] = Path("./data"),
    train_split: Annotated[str, typer.Option("--train-split", help="Dataset split(s) to use for training.")] = "full",
    val_split: Annotated[str, typer.Option("--val-split", help="Dataset split(s) to use for validation.")] = "full",
    predict_split: Annotated[str, typer.Option("--predict-split", help="Dataset split(s) to use for prediction.")] = "full",
    val_size: Annotated[float | None, typer.Option("--val-size", help="Ignored if train_split != val_split. Size of the validation set. If float, represents the proportion of the dataset to include in the validation split. If int, represents the absolute number of examples. If None, the value is set to 0.1.")] = None,
    # Logging options
    log_level: Annotated[Literal["DEBUG", "INFO", "WARNING", "ERROR"], typer.Option("--log-level", help="Set the logging level.")] = "WARNING",
):
    """MyCLI: a tiny example Typer app."""
    root_logger = logging.getLogger()
    root_logger.setLevel(log_level)
    root_logger.addHandler(rich.logging.RichHandler())

    ctx.ensure_object(dict)
    huggingface_datasets_config = HuggingFaceDatasetsConfig(dataset_name=dataset_name,
                                                                    cache_dir=cache_dir,
                                                                    node_feature=node_feature,
                                                                    hyperedge_feature=hyperedge_feature)
    ctx.obj['huggingface_datasets_config'] = huggingface_datasets_config
    dataloader_config = DataLoaderConfig(pin_memory=pin_memory,
                                        num_workers=num_workers,
                                        persistent_workers=persistent_workers,
                                        batch_size=batch_size,
                                        drop_last=drop_last)
    ctx.obj['dataloader_config'] = dataloader_config

    datamodule_config = DataModuleConfig(data_dir=data_dir,
                                        train_split=train_split,
                                        val_split=val_split,
                                        predict_split=predict_split,
                                        val_size=val_size)
    ctx.obj['datamodule_config'] = datamodule_config

def main() -> None:
    # Entry point for console scripts
    app()

if __name__ == "__main__":
    main()
