from pathlib import Path

import lightning as L

from hydra.core.configs import DataLoaderConfig, DataModuleConfig, HuggingFaceDatasetsConfig
from hydra.core.models.modules import DiffusionTransformer
from hydra.core.data.datamodules import HypergraphDataModule

# TODO: Implement this

def sample_ddm(
    datamodule_config: DataModuleConfig,
    huggingface_datasets_config: HuggingFaceDatasetsConfig,
    dataloader_config: DataLoaderConfig,
    ckpt_path: Path,
):

    model = DiffusionTransformer.load_from_checkpoint(ckpt_path)

    trainer = L.Trainer(
        default_root_dir="logs/daqh/email-Enron/BVAE-HyDRA-S/logs/version_0",
        logger=False,
        enable_checkpointing=False,
    )

    print(huggingface_datasets_config.dataset_name)
    print("aaa")

    datamodule = HypergraphDataModule(dataset_name=huggingface_datasets_config.dataset_name,
                        data_dir=datamodule_config.data_dir,
                        retain_lcc=datamodule_config.retain_lcc,
                        cache_dir=huggingface_datasets_config.cache_dir,
                        pin_memory=dataloader_config.pin_memory,
                        num_workers=dataloader_config.num_workers,
                        persistent_workers=dataloader_config.persistent_workers,
                        batch_size=dataloader_config.batch_size if dataloader_config.batch_size is not None else 1,
                        val_size=datamodule_config.val_size,)

    return

    predictions = trainer.predict(
        model,
        datamodule=datamodule,
        ckpt_path=ckpt_path,
    )
