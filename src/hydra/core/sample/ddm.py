from pathlib import Path
import logging

import lightning as L
from tqdm.rich import tqdm
import torch
import xgi
from datasets import load_dataset

from hydra.core.configs import DataLoaderConfig, DataModuleConfig, HuggingFaceDatasetsConfig
from hydra.core.models.modules import DiffusionTransformer
from hydra.core.models.enums import DDM_CONFIGS_REVERSE
from hydra.core.data.datamodules import FeaturesDataModule

logger = logging.getLogger(__name__)

# TODO: Implement this

def sample_ddm(
    datamodule_config: DataModuleConfig,
    huggingface_datasets_config: HuggingFaceDatasetsConfig,
    dataloader_config: DataLoaderConfig,
    walk_length: int,
    ckpt_path: Path | None = None,
    pl_module: DiffusionTransformer | None = None,
) -> xgi.Hypergraph:
    
    assert ckpt_path is not None or pl_module is not None, "Either ckpt_path or pl_module must be provided"
    assert not (ckpt_path is not None and pl_module is not None), "Only one of ckpt_path or pl_module can be provided"

    if ckpt_path is not None:
        model = DiffusionTransformer.load_from_checkpoint(ckpt_path)
    else:
        model = pl_module

    model_name = f"DDM-HyDRA{'-V' if model.bvae.vertex_encoding else ''}-{DDM_CONFIGS_REVERSE[model.model_size_config]}" # TODO: This should probably be defined in the model

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
    datamodule.prepare_data()
    datamodule.setup("predict")

    dataset = load_dataset(
        huggingface_datasets_config.dataset_name,
        cache_dir=huggingface_datasets_config.cache_dir,
        split=datamodule_config.predict_split
    )

    generated_hypergraphs = []

    for d in dataset:
        ds_hypergraph = xgi.from_hif_dict(d, nodetype=int, edgetype=int)
        logger.info(f"Sampling hypergraph with {ds_hypergraph.num_nodes} nodes and {ds_hypergraph.num_edges} hyperedges from {model_name}...")
        hyperedges = set()
        with tqdm(total=len(ds_hypergraph.edges), desc="Sampling hyperedges") as pbar:
            while len(hyperedges) < len(ds_hypergraph.edges):
                predictions = []
                for batch in datamodule.predict_dataloader():
                    batch = {key: value.to(model.device) for key, value in batch.items()}
                    incidence_matrices, *_ = model.predict_step(batch, 0)
                    predictions.append(incidence_matrices.cpu())

                for incidence_matrices in predictions:
                    for incidence_matrix in incidence_matrices:
                        for col in incidence_matrix.T:
                            nodes = torch.nonzero(col).squeeze().tolist()
                            if isinstance(nodes, int):
                                nodes = [nodes]
                            if len(nodes) < 1:
                                continue
                            nodes = tuple(sorted(nodes))
                            hyperedges.add(nodes)
                            if len(hyperedges) >= len(ds_hypergraph.edges):
                                break
                        if len(hyperedges) >= len(ds_hypergraph.edges):
                            break
                    if len(hyperedges) >= len(ds_hypergraph.edges):
                        break

                pbar.update(len(hyperedges) - pbar.n)
                logger.info(f"Sampled {len(hyperedges)} hyperedges so far...")

        hyperedges = list(hyperedges)
        hypergraph = xgi.Hypergraph(hyperedges)
        hypergraph['dataset_name'] = huggingface_datasets_config.dataset_name
        hypergraph['name'] = model_name
        hypergraph['kind'] = "unconditional" if model.bvae.vertex_encoding else "conditional"

        generated_hypergraphs.append(hypergraph)

    return generated_hypergraphs
