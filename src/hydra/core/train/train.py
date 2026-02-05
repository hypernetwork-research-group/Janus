import lightning as L
from lightning.pytorch.callbacks import BatchSizeFinder, LearningRateFinder, LearningRateMonitor

from ..models.modules import HypergraphBetaVAE
from ..configs import DataModuleConfig, DataLoaderConfig, TrainerConfig, HuggingFaceDatasetsConfig
from ..data.datamodule import HypergraphDataModule

def train_bvae(
    datamodule_config: DataModuleConfig,
    dataloader_config: DataLoaderConfig,
    trainer_config: TrainerConfig,
    huggingface_datasets_config: HuggingFaceDatasetsConfig,
    vertex_encoding: bool
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

    trainer = L.Trainer(
        max_epochs=trainer_config.max_epochs,
        accumulate_grad_batches=trainer_config.accumulate_grad_batches,
        callbacks=[
            BatchSizeFinder(
                mode="binsearch",
                steps_per_trial=3,
                margin=0.45
            )
        ]
    )

    model = HypergraphBetaVAE(
        num_hyperedges=1512,
        kl_weight=1e-6,
        learning_rate=1e-3,
        encode_nodes=vertex_encoding)

    trainer.fit(model, datamodule=datamodule)

def train_ddm(
    datamodule_config: DataModuleConfig,
    dataloader_config: DataLoaderConfig,
    trainer_config: TrainerConfig,
    huggingface_datasets_config: HuggingFaceDatasetsConfig,
    T: int,
):
    pass
