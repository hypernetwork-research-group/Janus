from pathlib import Path
import logging

import lightning as L
from tqdm.rich import tqdm
import torch
import xgi

from hydra.core.configs import DataLoaderConfig, DataModuleConfig, HuggingFaceDatasetsConfig
from hydra.core.models.modules import DiffusionTransformer
from hydra.core.data.datamodules import FeaturesDataModule

logger = logging.getLogger(__name__)

# TODO: Implement this

def sample_ddm(
    datamodule_config: DataModuleConfig,
    huggingface_datasets_config: HuggingFaceDatasetsConfig,
    dataloader_config: DataLoaderConfig,
    walk_length: int,
    ckpt_path: Path,
) -> xgi.Hypergraph:

    model = DiffusionTransformer.load_from_checkpoint(ckpt_path)

    trainer = L.Trainer(
        default_root_dir="logs/daqh/email-Enron/DDM-HyDRA-S/logs/version_0", # TODO: remove this hardcoded path
        logger=False,
        enable_checkpointing=False,
    )

    datamodule = FeaturesDataModule(
        dataset_name=huggingface_datasets_config.dataset_name,
        node_feature=huggingface_datasets_config.node_feature,
        hyperedge_feature=huggingface_datasets_config.hyperedge_feature,
        data_dir=datamodule_config.data_dir,
        cache_dir=huggingface_datasets_config.cache_dir,
        train_split=datamodule_config.train_split,
        val_split=datamodule_config.val_split,
        predict_split=datamodule_config.predict_split,
        pin_memory=dataloader_config.pin_memory,
        num_workers=dataloader_config.num_workers,
        persistent_workers=dataloader_config.persistent_workers,
        batch_size=dataloader_config.batch_size if dataloader_config.batch_size is not None else 1,
    )

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
    
    hyperedges = list(hyperedges)
    hypergraph = xgi.Hypergraph(hyperedges)

    return hypergraph
