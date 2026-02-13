from pathlib import Path

import lightning as L
from tqdm.rich import tqdm
import torch

from hydra.core.configs import DataLoaderConfig, DataModuleConfig, HuggingFaceDatasetsConfig
from hydra.core.models.modules import DiffusionTransformer
from hydra.core.data.datamodules import HypergraphDataModule

# TODO: Implement this

def sample_ddm(
    datamodule_config: DataModuleConfig,
    huggingface_datasets_config: HuggingFaceDatasetsConfig,
    dataloader_config: DataLoaderConfig,
    walk_length: int,
    ckpt_path: Path,
):

    model = DiffusionTransformer.load_from_checkpoint(ckpt_path)

    trainer = L.Trainer(
        default_root_dir="logs/daqh/email-Enron/BVAE-HyDRA-S/logs/version_0",
        logger=False,
        enable_checkpointing=False,
    )

    datamodule = HypergraphDataModule(dataset_name=huggingface_datasets_config.dataset_name,
                        data_dir=datamodule_config.data_dir,
                        cache_dir=huggingface_datasets_config.cache_dir,
                        train_split=datamodule_config.train_split,
                        val_split=datamodule_config.val_split,
                        predict_split=datamodule_config.predict_split,
                        pin_memory=dataloader_config.pin_memory,
                        num_workers=dataloader_config.num_workers,
                        persistent_workers=dataloader_config.persistent_workers,
                        batch_size=dataloader_config.batch_size if dataloader_config.batch_size is not None else 1,
                        val_size=datamodule_config.val_size,
                        walk_length=walk_length,
                        only_node_features=True)

    hyperedges = set()

    while len(hyperedges) < 1512:
        predictions = trainer.predict(
            model,
            datamodule=datamodule,
            ckpt_path=ckpt_path,
        )

        for incidence_matrices, *_ in tqdm(predictions):
            for incidence_matrix in incidence_matrices:
                for col in incidence_matrix.T:
                    if len(hyperedges) >= 1512:
                        break
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
