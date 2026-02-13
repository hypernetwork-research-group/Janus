from pathlib import Path

import lightning as L
from lightning.pytorch.callbacks import BatchSizeFinder, LearningRateFinder, LearningRateMonitor, LambdaCallback, RichProgressBar
from lightning.pytorch.loggers import TensorBoardLogger

from hydra.core.models.modules import DiffusionTransformer, HypergraphBetaVAE
from hydra.core.configs import DataModuleConfig, DataLoaderConfig, RandomWalkConfig, TrainerConfig, HuggingFaceDatasetsConfig, ModelSizeConfig, OptimizerConfig
from hydra.core.data.datamodules import HypergraphDataModule

def train_ddm(
    random_walk_config: RandomWalkConfig,
    datamodule_config: DataModuleConfig,
    dataloader_config: DataLoaderConfig,
    trainer_config: TrainerConfig,
    optimizer_config: OptimizerConfig,
    huggingface_datasets_config: HuggingFaceDatasetsConfig,
    model_size_config: str,
    T: int,
    bvae_ckpt: Path | str,
):
    datamodule = HypergraphDataModule(dataset_name=huggingface_datasets_config.dataset_name,
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
                        val_size=datamodule_config.val_size,)

    # Here we should determine the model name:
    # DDM-HyDRA-{model_size}/vertex_encoding

    bvae = HypergraphBetaVAE.load_from_checkpoint(str(bvae_ckpt))

    model_name = f"DDM-HyDRA{'-V' if bvae.vertex_encoding else ''}-{model_size_config}" # TODO: This should probably be defined in the model
    default_root_dir = f"logs/{huggingface_datasets_config.dataset_name}/{model_name}"

    del bvae # We don't need the bvae anymore, we just needed to load it to determine the model name

    trainer = L.Trainer(
        default_root_dir=default_root_dir,
        max_epochs=trainer_config.max_epochs,
        accumulate_grad_batches=trainer_config.accumulate_grad_batches,
        log_every_n_steps=10, # TODO: Add this to trainer configuration
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
            BatchSizeFinder(
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

    model = DiffusionTransformer(
        T=T,
        bvae_ckpt=str(bvae_ckpt),
        learning_rate=optimizer_config.learning_rate,
        model_size_config=model_size_config,
    )

    trainer.fit(model=model, datamodule=datamodule)

