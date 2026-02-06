from pathlib import Path

import lightning as L
import torch
from tqdm.rich import tqdm

from ..configs import DataLoaderConfig, DataModuleConfig, HuggingFaceDatasetsConfig
from ..models.modules import HypergraphBetaVAE
from ..data.datamodules import HypergraphDataModule

def sample_bvae(
    datamodule_config: DataModuleConfig,
    huggingface_datasets_config: HuggingFaceDatasetsConfig,
    dataloader_config: DataLoaderConfig,
    ckpt_path: Path,
):
    trainer = L.Trainer(
        logger=False,
    )

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
                        batch_size=dataloader_config.batch_size if dataloader_config.batch_size is not None else 1,
                        val_size=datamodule_config.val_size,)

    model = HypergraphBetaVAE.load_from_checkpoint(ckpt_path)

    predictions = trainer.predict(
        model,
        datamodule=datamodule,
        ckpt_path=ckpt_path,
    )

    hyperedges = set()
    for h_logits, _, _, _, _, _, _, _ in tqdm(predictions):
        incidence_matrices = torch.distributions.Categorical(logits=h_logits).sample()
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
