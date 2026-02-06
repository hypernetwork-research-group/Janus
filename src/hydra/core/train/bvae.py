import lightning as L
from lightning.pytorch.callbacks import BatchSizeFinder, LearningRateFinder, LearningRateMonitor, EarlyStopping, ModelCheckpoint, LambdaCallback

from ..models.modules import HypergraphBetaVAE
from ..configs import DataModuleConfig, DataLoaderConfig, TrainerConfig, HuggingFaceDatasetsConfig, ModelSizeConfig, OptimizerConfig
from ..data.datamodule import HypergraphDataModule

def train_bvae(
    datamodule_config: DataModuleConfig,
    dataloader_config: DataLoaderConfig,
    trainer_config: TrainerConfig,
    huggingface_datasets_config: HuggingFaceDatasetsConfig,
    model_size_config: ModelSizeConfig,
    optimizer_config: OptimizerConfig,
    vertex_encoding: bool,
    kl_weight: float,
):
    datamodule = HypergraphDataModule(dataset_name=huggingface_datasets_config.dataset_name,
                        data_dir=datamodule_config.data_dir,
                        retain_lcc=datamodule_config.retain_lcc,
                        cache_dir=huggingface_datasets_config.cache_dir,
                        p=datamodule_config.p,
                        q=datamodule_config.q,
                        alpha=datamodule_config.alpha,
                        walk_length=datamodule_config.walk_length,
                        train_split=datamodule_config.train_split,
                        val_split=datamodule_config.val_split,
                        predict_split=datamodule_config.predict_split,
                        samples_per_hyperedge=datamodule_config.samples_per_hyperedge,
                        pin_memory=dataloader_config.pin_memory,
                        num_workers=dataloader_config.num_workers,
                        persistent_workers=dataloader_config.persistent_workers,
                        batch_size=dataloader_config.batch_size,
                        val_size=datamodule_config.val_size,)

    # Here we should determine the model name:
    # BVAE-HyDRA-{model_size}/vertex_encoding

    model_name = f"BVAE-HyDRA{'-V' if vertex_encoding else ''}-{model_size_config}"

    trainer = L.Trainer(
        gradient_clip_val=1.0,
        default_root_dir=f"logs/{huggingface_datasets_config.dataset_name}/{model_name}",
        max_epochs=trainer_config.max_epochs,
        accumulate_grad_batches=trainer_config.accumulate_grad_batches,
        log_every_n_steps=10, # TODO: Add this to trainer configuration
        callbacks=[
            EarlyStopping(
                monitor="validation/loss",
                patience=100, # TODO: Add this to trainer configuration
                mode="min",
                check_on_train_epoch_end=False, # Check only at the end of validation
            ),
            # Save last every 50 epochs
            ModelCheckpoint(
                filename="last",
                every_n_epochs=10, #
            ),
            ModelCheckpoint(
                mode="min",
                monitor="validation/loss",
                filename="best",
                save_top_k=1,
                every_n_epochs=10, #
            ),
            LearningRateMonitor(
                logging_interval='epoch',
                log_momentum=True,
                log_weight_decay=True
            ) if optimizer_config.learning_rate is None else LambdaCallback(),
            BatchSizeFinder(
                mode="binsearch",
                steps_per_trial=3,
                margin=0.45
            ) if dataloader_config.batch_size is None else LambdaCallback(),
            LearningRateFinder(
                mode="exponential",
                min_lr=5e-5,
                max_lr=1,
            )
        ]
    )

    # TODO: Read dataset and determine num_hyperedges

    model = HypergraphBetaVAE(
        num_hyperedges=1512,
        kl_weight=kl_weight,
        learning_rate=optimizer_config.learning_rate,
        weight_decay=optimizer_config.weight_decay,
        encode_nodes=vertex_encoding,
        model_size_config=model_size_config)

    trainer.fit(model, datamodule=datamodule)

