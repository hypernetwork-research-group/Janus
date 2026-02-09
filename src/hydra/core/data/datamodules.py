from pathlib import Path
from typing import Union
from multiprocessing import cpu_count
import logging
from abc import ABC

import torch
import lightning as L
from datasets import load_dataset, load_from_disk, Dataset, DatasetDict

from .transforms import transform, process

logger = logging.getLogger(__name__)

class HypergraphDataModule(L.LightningDataModule):

    def __init__(self,
                 # HuggingFaceDatasetsConfig options
                 dataset_name: str,
                 cache_dir: Path = Path("./cache"),
                 # DataModuleConfig options
                 data_dir: Path = Path("./data"),
                 p: float = 2.0,
                 q: float = 0.5,
                 alpha: float = 0.0,
                 walk_length: int = 256,
                 samples_per_hyperedge: int = 1,
                 retain_lcc: bool = True,
                 train_split: str = "full",
                 val_split: str = "full",
                 predict_split: str = "full",
                 val_size: Union[float, int, None] = None,
                 # DataLoaderConfig options
                 pin_memory: bool = True,
                 num_workers: int | None = None,
                 persistent_workers: bool = True,
                 batch_size: int = 32,
                 drop_last: bool = False):
        super().__init__()
        # DataModuleConfig options
        self.data_dir = data_dir
        self.p = p
        self.q = q
        self.alpha = alpha
        self.walk_length = walk_length
        self.samples_per_hyperedge = samples_per_hyperedge
        self.retain_lcc = retain_lcc
        # DataLoaderConfig options
        self.pin_memory = pin_memory
        self.num_workers = num_workers or cpu_count()
        self.persistent_workers = persistent_workers
        self.batch_size = batch_size
        # HuggingFaceDatasetsConfig options
        self.dataset_name = dataset_name
        self.cache_dir = cache_dir
        # Dataset split
        self.train_split = train_split
        self.val_split = val_split
        self.predict_split = predict_split
        self.drop_last = drop_last
        # The validation size is only relevant if train_split == val_split, in which case we need to split the training set into a training and validation set
        # If it is >= 1, we interpret it as an absolute number of examples, if it is < 1, we interpret it as a proportion of the dataset
        # If it is None, we set it to 0.1 by default
        self.val_size = 0.1 if not val_size else int(val_size) if val_size >= 1 else val_size

        # Additional
        self.dataset_dir = data_dir / dataset_name
        self.processed_dataset_dir = self.dataset_dir / "processed"
        self.transformed_dataset_dir = self.dataset_dir / f"sph{samples_per_hyperedge}" / f"wl{walk_length}" / f"p{p}" / f"q{q}" / f"a{alpha}" / f"{self.val_size if train_split == val_split else 'fullvalsize'}"

    def prepare_data(self):
        # Loading the dataset from HuggingFace Datasets
        # This will be stored in the HuggingFace Datasets cache directory, so it won't be redownloaded every time
        datasets = load_dataset(self.dataset_name,
                               cache_dir=self.cache_dir / "datasets")

        # Preprocess the dataset
        # The preprocessing behaviour is described in the `process` function
        # At the end of this, the `hif`` columns will be removed
        # Additionally, `node_features` and `hyperedge_features` columns will be added
        # The `hif_dict` column will contain the hif representation of the hypergraph, which will be used in the next step to reconstruct the hypergraph and perform random walks on it
        processed = datasets.map(process,
                         load_from_cache_file=True,
                         remove_columns=["metadata", "network-type", "nodes", "edges", "incidences"],
                         fn_kwargs={
                            "retain_lcc": self.retain_lcc,
                         })

        # After preprocessing, we save the processed dataset to disk, so that we can load it later without having to redo the preprocessing step
        if not self.processed_dataset_dir.exists():
            processed.save_to_disk(self.processed_dataset_dir)

        # Here, we build a set of random walks for each hypergraph,
        # Each random walk will become an entry in the final dataset, associated with the matrices of corresponding node and hyperedge features
        transformed = DatasetDict({
            k: Dataset.from_generator(transform(v,
                                            samples_per_hyperedge=self.samples_per_hyperedge,
                                            walk_length=self.walk_length,
                                            p=self.p,
                                            q=self.q,
                                            alpha=self.alpha,
                                            num_workers=self.num_workers),
                                        cache_dir=self.cache_dir / "transformed" / self.dataset_name / f"sph{self.samples_per_hyperedge}" / f"wl{self.walk_length}" / f"p{self.p}" / f"q{self.q}" / f"a{self.alpha}")
            for k, v in processed.items()
        })

        # If the training and validation splits are the same, we need to split the transformed training set into a training and validation set
        if self.train_split == self.val_split:
            temp_ = transformed[self.train_split].train_test_split(test_size=self.val_size, shuffle=True, seed=42) # TODO: pass seed
            # Rename the splits to train and val
            transformed = DatasetDict({
                "train": temp_["train"],
                "val": temp_["test"],
                "predict": transformed[self.predict_split],
            })
        else: # If the training and validation splits are different, we can directly use them without splitting
            transformed = DatasetDict({
                "train": transformed[self.train_split],
                "val": transformed[self.val_split],
                "predict": transformed[self.predict_split],
            })

        # Set the format to PyTorch tensors, this will allow us to directly get PyTorch tensors when we access the elements of the dataset
        transformed.set_format(type='torch')

        # Similarly to the processed dataset, we save the transformed dataset to disk
        if not self.transformed_dataset_dir.exists():
            transformed.save_to_disk(self.transformed_dataset_dir)

    def setup(self, stage):
        self.dataset = load_from_disk(self.transformed_dataset_dir)

    def train_dataloader(self):
        return torch.utils.data.DataLoader(self.dataset["train"],
                                           batch_size=self.batch_size,
                                           pin_memory=self.pin_memory,
                                           num_workers=self.num_workers,
                                           persistent_workers=self.persistent_workers,
                                           shuffle=True,
                                           drop_last=self.drop_last)

    def val_dataloader(self):
        return torch.utils.data.DataLoader(self.dataset["val"],
                                           batch_size=self.batch_size,
                                           pin_memory=self.pin_memory,
                                           num_workers=self.num_workers,
                                           persistent_workers=self.persistent_workers,
                                           shuffle=False,
                                           drop_last=self.drop_last)

    def predict_dataloader(self):
        return torch.utils.data.DataLoader(self.dataset["predict"],
                                           batch_size=self.batch_size,
                                           pin_memory=self.pin_memory,
                                           num_workers=self.num_workers,
                                           persistent_workers=self.persistent_workers,
                                           shuffle=False)
