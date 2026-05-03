from pathlib import Path

import lightning as L
from datasets import load_dataset
import torch
from tqdm.rich import tqdm
import xgi
from sklearn.cluster import KMeans
import numpy as np

from janus.core.configs import DataLoaderConfig, DataModuleConfig, HuggingFaceDatasetsConfig, RandomWalkConfig
from janus.core.models.modules import HypergraphBetaVAE
from janus.core.models.enums import BVAE_CONFIGS_REVERSE
from janus.core.data.datamodules import HypergraphDataModule

def sample_bvae(
    random_walk_config: RandomWalkConfig,
    datamodule_config: DataModuleConfig,
    huggingface_datasets_config: HuggingFaceDatasetsConfig,
    dataloader_config: DataLoaderConfig,
    ckpt_path: Path,
) -> xgi.Hypergraph:

    model = HypergraphBetaVAE.load_from_checkpoint(ckpt_path)

    model_name = f"BVAE-Janus{'' if model.vertex_encoding else 'NC'}-{BVAE_CONFIGS_REVERSE[model.model_size_config]}"

    trainer = L.Trainer(
        default_root_dir=f"logs/{huggingface_datasets_config.dataset_name}/{model_name}/logs/version_0", # TODO: remove this hardcoded path
        logger=False,
        enable_checkpointing=False,
    )

    dataset = load_dataset(
        huggingface_datasets_config.dataset_name,
        cache_dir=huggingface_datasets_config.cache_dir,
        split=datamodule_config.predict_split
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
    datamodule.prepare_data()
    datamodule.setup("predict")

    predictions = trainer.predict(
        model,
        datamodule=datamodule,
        ckpt_path=ckpt_path,
    )

    num_nodes = xgi.from_hif_dict(dataset[0], nodetype=int, edgetype=int).num_nodes
    embeddings = []
    membership_masks = []

    hyperedges = set()
    for incidence_matrices, _, x_r, membership_mask, *_ in tqdm(predictions):
        if x_r is not None:
            embeddings.append(x_r.cpu())
        membership_masks.append(membership_mask.cpu())
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

    np_embeddings = torch.cat(embeddings, dim=0).cpu().numpy()  # [B, num_nodes, F]
    np_embeddings = np_embeddings.reshape(-1, np_embeddings.shape[-1]) # [B * num_nodes, F]
    np_membership_masks = torch.cat(membership_masks, dim=0).cpu().float().numpy()  # [B, num_nodes]
    np_membership_masks = np_membership_masks.reshape(-1)  # [B * num_nodes]
    kmeans = KMeans(n_clusters=num_nodes, random_state=0).fit(np_embeddings, sample_weight=np_membership_masks)
    kmeans_labels = np.array(kmeans.labels_).reshape(-1, num_nodes)  # [B, num_nodes]

    y_true = torch.arange(num_nodes).repeat(kmeans_labels.shape[0], 1)  # [B, num_nodes]

    from sklearn.metrics import adjusted_rand_score, normalized_mutual_info_score, adjusted_mutual_info_score

    ari = adjusted_rand_score(y_true.flatten()[np_membership_masks == 1], kmeans_labels.flatten()[np_membership_masks == 1])
    nmi = normalized_mutual_info_score(y_true.flatten()[np_membership_masks == 1], kmeans_labels.flatten()[np_membership_masks == 1])
    ami = adjusted_mutual_info_score(y_true.flatten()[np_membership_masks == 1], kmeans_labels.flatten()[np_membership_masks == 1])

    print(f"ARI: {ari:.4f}, NMI: {nmi:.4f}, AMI: {ami:.4f}")

    hypergraph = xgi.Hypergraph(hyperedges)
    hypergraph['dataset_name'] = huggingface_datasets_config.dataset_name
    hypergraph['name'] = model_name
    hypergraph['kind'] = "reconstruction"
    
    return hypergraph
