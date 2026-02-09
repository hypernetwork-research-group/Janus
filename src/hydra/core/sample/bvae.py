from pathlib import Path

import lightning as L
from lightning.pytorch.callbacks.prediction_writer import WriteInterval
import torch
from tqdm.rich import tqdm

from hydra.core.configs import DataLoaderConfig, DataModuleConfig, HuggingFaceDatasetsConfig, RandomWalkConfig
from hydra.core.models.modules import HypergraphBetaVAE
from hydra.core.data.datamodules import HypergraphDataModule

def sample_bvae(
    random_walk_config: RandomWalkConfig,
    datamodule_config: DataModuleConfig,
    huggingface_datasets_config: HuggingFaceDatasetsConfig,
    dataloader_config: DataLoaderConfig,
    ckpt_path: Path,
):

    model = HypergraphBetaVAE.load_from_checkpoint(ckpt_path)

    trainer = L.Trainer(
        default_root_dir="logs/daqh/email-Enron/BVAE-HyDRA-S/logs/version_0",
        logger=False,
        enable_checkpointing=False,
    )

    datamodule = HypergraphDataModule(dataset_name=huggingface_datasets_config.dataset_name,
                        data_dir=datamodule_config.data_dir,
                        retain_lcc=datamodule_config.retain_lcc,
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

    predictions = trainer.predict(
        model,
        datamodule=datamodule,
        ckpt_path=ckpt_path,
    )

    hyperedges = set()
    for incidence_matrices, *_ in tqdm(predictions):
        for incidence_matrix in incidence_matrices:
            for col in incidence_matrix.T:
                nodes = torch.nonzero(col).squeeze().tolist()
                if isinstance(nodes, int):
                    nodes = [nodes]
                if len(nodes) < 1:
                    continue
                nodes = tuple(sorted(nodes))
                hyperedges.add(nodes)
    dist = [0] * 143
    for he in hyperedges:
        dist[len(he)] += 1
    print(dist)
