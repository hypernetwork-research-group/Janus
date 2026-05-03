from pathlib import Path

import lightning as L
from lightning.pytorch.loggers import TensorBoardLogger
from lightning.pytorch.callbacks import BatchSizeFinder, LearningRateFinder, LearningRateMonitor, LambdaCallback, RichProgressBar

from janus.core.models.modules import HypergraphBetaVAE
from janus.core.configs import DataModuleConfig, DataLoaderConfig, RandomWalkConfig, TrainerConfig, HuggingFaceDatasetsConfig, ModelSizeConfig, OptimizerConfig, EarlyStoppingConfig
from janus.core.data.datamodules import HypergraphDataModule

def train_bvae(
    random_walk_config: RandomWalkConfig,
    datamodule_config: DataModuleConfig,
    dataloader_config: DataLoaderConfig,
    trainer_config: TrainerConfig,
    huggingface_datasets_config: HuggingFaceDatasetsConfig,
    model_size_config: str,
    optimizer_config: OptimizerConfig,
    early_stopping_config: EarlyStoppingConfig,
    vertex_encoding: bool,
    kl_weight: float,
    latent_dim: int | None,
    ckpt_path: Path | str | None
):
    datamodule = HypergraphDataModule(dataset_name=huggingface_datasets_config.dataset_name,
                                      node_feature=huggingface_datasets_config.node_feature,
                                      hyperedge_feature=huggingface_datasets_config.hyperedge_feature,
                        data_dir=datamodule_config.data_dir,
                        cache_dir=huggingface_datasets_config.cache_dir,
                        p=random_walk_config.p,
                        q=random_walk_config.q,
                        alpha=random_walk_config.alpha,
                        walk_length=random_walk_config.walk_length,
                        train_split=datamodule_config.train_split,
                        val_split=datamodule_config.val_split,
                        predict_split=datamodule_config.predict_split,
                        samples_per_hyperedge=random_walk_config.samples_per_hyperedge,
                        pin_memory=dataloader_config.pin_memory,
                        num_workers=dataloader_config.num_workers,
                        persistent_workers=dataloader_config.persistent_workers,
                        batch_size=dataloader_config.batch_size if dataloader_config.batch_size is not None else 1,
                        val_size=datamodule_config.val_size,
                        drop_last=vertex_encoding or dataloader_config.drop_last)
    # Since vertex encoding requires contrastive loss over nodes in the batch, we automatically set drop_last=True when vertex_encoding is enabled to ensure consistent batch sizes.

    # Here we should determine the model name:
    # BVAE-Janus-{model_size}/vertex_encoding

    model_name = f"Janus{'' if vertex_encoding else 'NC'}-BVAE"
    default_root_dir = f"logs/{huggingface_datasets_config.dataset_name}/{model_name}"

    trainer = L.Trainer(
        default_root_dir=default_root_dir,
        max_epochs=trainer_config.max_epochs,
        min_epochs=trainer_config.min_epochs,
        accumulate_grad_batches=trainer_config.accumulate_grad_batches,
        log_every_n_steps=trainer_config.log_every_n_steps,
        check_val_every_n_epoch=1,
        enable_checkpointing=False,
        logger=TensorBoardLogger(
            save_dir=default_root_dir, # base path
            name="logs",  # replaces the default "lightning_logs"
        ),
        callbacks=[
            # Save last every 50 epochs
            LearningRateMonitor(
                logging_interval='epoch',
                log_momentum=True,
                log_weight_decay=True
            ),
            BatchSizeFinder( # TODO: Add this to batch size finder configuration
                mode="binsearch",
                steps_per_trial=3,
                margin=0.45
            ) if dataloader_config.batch_size is None else LambdaCallback(),
            LearningRateFinder(
                mode="exponential",
                min_lr=5e-5,
                max_lr=1,
                num_training_steps=300,
            )  if optimizer_config.learning_rate is None else LambdaCallback(),
            RichProgressBar(
                refresh_rate=1,
            )
        ]
    )

    model = HypergraphBetaVAE(
        x_kl_weight=kl_weight,  # NOTE: Set both x_kl_weight and y_kl_weight to the same value for now, but we can experiment with different values for each to see if it improves performance
        y_kl_weight=kl_weight,
        learning_rate=optimizer_config.learning_rate,
        weight_decay=optimizer_config.weight_decay,
        vertex_encoding=vertex_encoding,
        model_size_config=model_size_config,
        patience=early_stopping_config.patience,
        latent_dim=latent_dim
    )

    trainer.fit(model,
                datamodule=datamodule,
                ckpt_path=ckpt_path)
