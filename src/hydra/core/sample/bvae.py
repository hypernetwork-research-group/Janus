from pathlib import Path

import lightning as L
import torch
from tqdm.rich import tqdm
import xgi

from hydra.core.configs import DataLoaderConfig, DataModuleConfig, HuggingFaceDatasetsConfig, RandomWalkConfig
from hydra.core.models.modules import HypergraphBetaVAE
from hydra.core.models.enums import BVAE_CONFIGS_REVERSE
from hydra.core.data.datamodules import HypergraphDataModule

def sample_bvae(
    random_walk_config: RandomWalkConfig,
    datamodule_config: DataModuleConfig,
    huggingface_datasets_config: HuggingFaceDatasetsConfig,
    dataloader_config: DataLoaderConfig,
    ckpt_path: Path,
) -> xgi.Hypergraph:

    model = HypergraphBetaVAE.load_from_checkpoint(ckpt_path)

    model_name = f"BVAE-HyDRA{'-V' if model.vertex_encoding else ''}-{BVAE_CONFIGS_REVERSE[model.model_size_config]}"

    trainer = L.Trainer(
        default_root_dir=f"logs/{huggingface_datasets_config.dataset_name}/{model_name}/logs/version_0", # TODO: remove this hardcoded path
        logger=False,
        enable_checkpointing=False,
    )

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
    hyperedges = list(hyperedges)

    hypergraph = xgi.Hypergraph(hyperedges)
    hypergraph['dataset_name'] = huggingface_datasets_config.dataset_name
    hypergraph['name'] = model_name
    hypergraph['kind'] = "reconstruction"
    
    return hypergraph
