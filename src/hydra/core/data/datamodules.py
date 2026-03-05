from pathlib import Path
from typing import Union
from multiprocessing import cpu_count
import logging
import traceback

import torch
import lightning as L
from datasets import load_dataset, load_from_disk, Dataset, DatasetDict, Features, Array2D, List, Value

from .transforms import add_random_noise, transform, process

logger = logging.getLogger(__name__)

class FeaturesDataModule(L.LightningDataModule):

    def __init__(self,
                 dataset_name: str,
                 node_feature: str = "eigsh",
                 hyperedge_feature: str = "eigsh",
                 cache_dir: Path = Path("./cache"),
                 data_dir: Path = Path("./data"),
                 batch_size: int = 32,
                 num_workers: int | None = None,
                 persistent_workers: bool = True,
                 pin_memory: bool = True,
                 drop_last: bool = False,
                 train_split: str = "full",
                 val_split: str = "full",
                 predict_split: str = "full",
                 val_size: Union[float, int, None] = None):
        super().__init__()
        self.dataset_name = dataset_name
        self.node_feature = node_feature
        self.hyperedge_feature = hyperedge_feature
        self.cache_dir = cache_dir
        self.data_dir = data_dir
        self.batch_size = batch_size
        self.num_workers = num_workers if num_workers is not None else cpu_count() - 1
        self.persistent_workers = persistent_workers
        self.pin_memory = pin_memory
        self.drop_last = drop_last
        self.train_split = train_split
        self.val_split = val_split
        self.predict_split = predict_split
        self.val_size = 0.1 if not val_size else int(val_size) if val_size >= 1 else val_size

        self.dataset_dir = data_dir / dataset_name
        self.processed_dataset_dir = self.dataset_dir / "processed"
        self.node_features_dataset_dir = self.dataset_dir / "node_features"

    def prepare_data(self):
        # Loading the dataset from HuggingFace Datasets
        # This will be stored in the HuggingFace Datasets cache directory, so it won't be redownloaded every time
        dataset = load_dataset(self.dataset_name,
                               cache_dir=str(self.cache_dir / "datasets"))

        # Preprocess the dataset
        # The preprocessing behaviour is described in the `process` function
        # At the end of this, the `hif`` columns will be removed
        # Additionally, `node_features` and `hyperedge_features` columns will be added
        # The `hif_dict` column will contain the hif representation of the hypergraph, which will be used in the next step to reconstruct the hypergraph and perform random walks on it
        dataset = dataset.map(process,
                         load_from_cache_file=True,
                         remove_columns=["metadata", "network-type", "nodes", "edges", "incidences"],
                         fn_kwargs={
                             "node_feature": self.node_feature,
                             "hyperedge_feature": self.hyperedge_feature,
                         })

        dataset.set_format(type='torch')

        if not self.node_features_dataset_dir.exists():
            dataset.save_to_disk(self.node_features_dataset_dir)
    
    def setup(self, stage):
        self.dataset = load_from_disk(self.node_features_dataset_dir)

    def train_dataloader(self):
        dataset = self.dataset
        return torch.utils.data.DataLoader(dataset[self.train_split],
                                           batch_size=self.batch_size,
                                           pin_memory=self.pin_memory,
                                           num_workers=self.num_workers,
                                           persistent_workers=self.persistent_workers,
                                           shuffle=True,
                                           drop_last=self.drop_last)

    def val_dataloader(self):
        dataset = self.dataset
        return torch.utils.data.DataLoader(dataset[self.val_split],
                                           batch_size=self.batch_size,
                                           pin_memory=self.pin_memory,
                                           num_workers=self.num_workers,
                                           persistent_workers=self.persistent_workers,
                                           shuffle=False,
                                           drop_last=self.drop_last)

    def predict_dataloader(self):
        dataset = self.dataset
        return torch.utils.data.DataLoader(dataset[self.predict_split],
                                           batch_size=self.batch_size,
                                           pin_memory=self.pin_memory,
                                           num_workers=self.num_workers,
                                           persistent_workers=self.persistent_workers,
                                           shuffle=False)

class HypergraphDataModule(L.LightningDataModule):

    def __init__(self, # TODO: Pass the config objects, not all the options separately
                 # HuggingFaceDatasetsConfig options
                 dataset_name: str,
                 cache_dir: Path = Path("cache"),
                 node_feature: str = "eigsh",
                 hyperedge_feature: str = "eigsh",
                 # DataModuleConfig options
                 data_dir: Path = Path("data"),
                 p: float = 2.0,
                 q: float = 0.5,
                 alpha: float = 0.0,
                 walk_length: int = 256,
                 samples_per_hyperedge: int = 1,
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
        # DataLoaderConfig options
        self.pin_memory = pin_memory
        self.num_workers = num_workers if num_workers is not None else cpu_count() -1 # Set num_workers to the number of available CPU cores minus one, to avoid overloading the system
        self.persistent_workers = persistent_workers
        self.batch_size = batch_size
        # HuggingFaceDatasetsConfig options
        self.dataset_name = dataset_name
        self.cache_dir = cache_dir
        self.node_feature = node_feature
        self.hyperedge_feature = hyperedge_feature
        # Dataset split
        self.train_split = train_split
        self.val_split = val_split
        self.predict_split = predict_split
        self.drop_last = drop_last
        # The validation size is only relevant if train_split == val_split, in which case we need to split the training set into a training and validation set
        # If it is >= 1, we interpret it as an absolute number of examples, if it is < 1, we interpret it as a proportion of the dataset
        # If it is None, we set it to 0.1 by default
        self.val_size = 0.1 if val_size == None else int(val_size) if val_size >= 1 else val_size

        # Additional
        self.dataset_dir = data_dir / dataset_name
        self.processed_dataset_dir = self.dataset_dir / "processed" / f"n{node_feature}_h{hyperedge_feature}"
        self.transformed_dataset_dir = self.dataset_dir / "transformed" / f"sph{samples_per_hyperedge}" / f"wl{walk_length}" / f"p{p}" / f"q{q}" / f"a{alpha}" / f"{self.val_size if train_split == val_split else 'fullvalsize'}" / f"n{node_feature}_h{hyperedge_feature}"

    def prepare_data(self):
        # Loading the dataset from HuggingFace Datasets
        # This will be stored in the HuggingFace Datasets cache directory, so it won't be redownloaded every time
        dataset = load_dataset(self.dataset_name,
                               cache_dir=str(self.cache_dir / "datasets"))

        # Preprocess the dataset
        # The preprocessing behaviour is described in the `process` function
        # At the end of this, the `hif`` columns will be removed
        # Additionally, `node_features` and `hyperedge_features` columns will be added
        # The `hif_dict` column will contain the hif representation of the hypergraph, which will be used in the next step to reconstruct the hypergraph and perform random walks on it
        dataset = dataset.map(process,
                         load_from_cache_file=True,
                         remove_columns=["metadata", "network-type", "nodes", "edges", "incidences"],
                         fn_kwargs={
                            "node_feature": self.node_feature,
                            "hyperedge_feature": self.hyperedge_feature,
                         })

        # After preprocessing, we save the processed dataset to disk, so that we can load it later without having to redo the preprocessing step
        if not self.processed_dataset_dir.exists():
            dataset.save_to_disk(self.processed_dataset_dir, max_shard_size="1GB")

        # Here, we build a set of random walks for each hypergraph,
        # Each random walk will become an entry in the final dataset, associated with the matrices of corresponding node and hyperedge features
        dataset = DatasetDict({
            k: Dataset.from_generator(transform(v,
                                            samples_per_hyperedge=self.samples_per_hyperedge,
                                            walk_length=self.walk_length,
                                            p=self.p,
                                            q=self.q,
                                            alpha=self.alpha,
                                            num_workers=max(1, self.num_workers)),
                                        cache_dir=self.cache_dir / "transformed" / self.dataset_name / f"sph{self.samples_per_hyperedge}" / f"wl{self.walk_length}" / f"p{self.p}" / f"q{self.q}" / f"a{self.alpha}",
                                        writer_batch_size=500,
                                        keep_in_memory=False,
                                        features=Features({
                                            "touched_nodes": List(Value(dtype="int64")),
                                            "nodes_mask": List(Value(dtype="int64")),
                                            "touched_hyperedges": List(Value(dtype="int64"), length=self.walk_length),
                                            "node_features": Array2D(dtype="float32", shape=(None, 128)),
                                            "hyperedge_features": Array2D(dtype="float32", shape=(None, 128)),
                                            "incidence_matrix": Array2D(dtype="float32", shape=(None, self.walk_length)),
                                        }))
            for k, v in dataset.items()
        })

        # TODO: Adjust train, validation split logic
        # If the training and validation splits are the same, we need to split the transformed training set into a training and validation set
        if self.train_split == self.val_split:
            if len(dataset[self.train_split]) > 1 and self.val_size > 0:
                temp_ = dataset[self.train_split].train_test_split(test_size=self.val_size, shuffle=True) # TODO: pass seed
            else:
                temp_ = DatasetDict({
                    "train": dataset[self.train_split],
                    "test": dataset[self.train_split],
                })
            # Rename the splits to train and val
            dataset = DatasetDict({
                "train": dataset[self.train_split],
                "val": temp_["test"],
                "predict": dataset[self.predict_split],
            })
        else: # If the training and validation splits are different, we can directly use them without splitting
            dataset = DatasetDict({
                "train": dataset[self.train_split],
                "val": dataset[self.val_split],
                "predict": dataset[self.predict_split],
            })

        # Similarly to the processed dataset, we save the transformed dataset to disk
        if not self.transformed_dataset_dir.exists():
            dataset.save_to_disk(self.transformed_dataset_dir, max_shard_size="1GB")

    def setup(self, stage):
        # Drop edge_features_column
        self.dataset = load_from_disk(self.transformed_dataset_dir)
        self.dataset = self.dataset.with_format("torch")

    def train_dataloader(self):
        dataset = self.dataset
        return torch.utils.data.DataLoader(dataset["train"],
                                           batch_size=self.batch_size,
                                           pin_memory=self.pin_memory,
                                           num_workers=self.num_workers,
                                           persistent_workers=self.persistent_workers and self.num_workers > 0,
                                           shuffle=True,
                                           drop_last=self.drop_last,
                                           multiprocessing_context="spawn" if self.num_workers > 0 else None)

    def val_dataloader(self):
        dataset = self.dataset
        return torch.utils.data.DataLoader(dataset["val"],
                                           batch_size=self.batch_size,
                                           pin_memory=self.pin_memory,
                                           num_workers=self.num_workers,
                                           persistent_workers=self.persistent_workers and self.num_workers > 0,
                                           shuffle=False,
                                           drop_last=self.drop_last,
                                           multiprocessing_context="spawn" if self.num_workers > 0 else None)

    def predict_dataloader(self):
        dataset = self.dataset
        return torch.utils.data.DataLoader(dataset["predict"],
                                           batch_size=self.batch_size,
                                           pin_memory=self.pin_memory,
                                           num_workers=self.num_workers,
                                           persistent_workers=self.persistent_workers and self.num_workers > 0,
                                           shuffle=False,
                                           multiprocessing_context="spawn" if self.num_workers > 0 else None)
