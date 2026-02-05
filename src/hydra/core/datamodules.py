from pathlib import Path
from typing import List
from multiprocessing import cpu_count
from concurrent.futures import ProcessPoolExecutor

import torch
import lightning as L
import xgi
from datasets import load_dataset, load_from_disk, Dataset, DatasetDict
from tqdm import tqdm

from .utils import metropolis_hastings_biased_random_walk, patch_nodes

def process(batch,
            retain_lcc: bool):
    hif_dict = batch['hif']
    hypergraph = xgi.convert.from_hif_dict(hif_dict, nodetype=int)
    if retain_lcc:
        hypergraph = xgi.largest_connected_hypergraph(hypergraph)
        # TODO: Perform logging
    hyperedges = list(map(list, map(sorted, map(tuple, xgi.to_hyperedge_list(hypergraph)))))

    # 1) Reindex vertices to have consecutive indices
    vertices_idx = sorted(hypergraph.nodes)
    if vertices_idx != list(range(len(vertices_idx))):
        hyperedges = [[vertices_idx.index(v) for v in he] for he in hyperedges]
        hyperedges = list(set(map(tuple, hyperedges)))
    
    # 2) Reindex nodes to have ordered indices based on their occurrence in the hyperedges
    _, rowdict, coldict = xgi.convert.to_incidence_matrix(hypergraph, sparse=True, index=True)
    rowdict_inv = {value: key for key, value in rowdict.items()}
    ridx_hyperdedges = list()
    for edge in hypergraph.edges.members():
        ridx_edge = [rowdict_inv[node] for node in edge]
        ridx_hyperdedges.append(ridx_edge)
    ridx_hypergraph = xgi.Hypergraph(ridx_hyperdedges)
    ridx_hif_dict = xgi.convert.to_hif_dict(ridx_hypergraph)

    # Parse rowdict and coldict
    rowlist = [value for value in rowdict.values()]
    collist = [value for value in coldict.values()]

    return {
        "hyperedges": hyperedges,
        "ridx_hif": ridx_hif_dict,
        "rowlist": rowlist,
        "collist": collist
    }

def transform(dataset,
            samples_per_hyperedge: int,
            walk_length: int,
            p: float,
            q: float,
            alpha: float,
            num_workers: int):
    def _transform():
        for hypergraph_dataset in dataset:
            ridx_hif_dict = hypergraph_dataset['ridx_hif']
            ridx_hypergraph = xgi.convert.from_hif_dict(ridx_hif_dict, nodetype=int)
            # # 3) Sample random walks from the hypergraph
            linegraph = xgi.convert.to_line_graph(ridx_hypergraph)
            sources = list(linegraph.nodes)
            neighborhoods = {node: list(linegraph.neighbors(node)) for node in linegraph.nodes}
            members = ridx_hypergraph.edges.members()
            parallel_args = [(linegraph, samples_per_hyperedge, walk_length, p, q, alpha, 42, sources[i:i+10], neighborhoods, members) for i in range(0, len(sources), 10)]
            with ProcessPoolExecutor(max_workers=num_workers) as executor:
                paths = [path for result in tqdm(executor.map(metropolis_hastings_biased_random_walk, parallel_args), total=len(parallel_args)) for path in result]
            # Determine the target number of nodes
            target_lambda = 1.0
            max_touched_nodes = max(len(path['touched_nodes']) for path in paths)
            num_nodes = ridx_hypergraph.num_nodes
            target_num_nodes = int(max_touched_nodes + (num_nodes - max_touched_nodes) * target_lambda)
            for walk in paths:
                walk['touched_nodes'], walk['nodes_mask'] = patch_nodes(ridx_hypergraph, walk['touched_nodes'], target_num_nodes)
                yield walk
    return _transform

class HypergraphDataModule(L.LightningDataModule):

    def __init__(self,
                 # HuggingFaceDatasetsConfig options
                 dataset_name: str,
                 cache_dir: Path = Path("./cache"),
                 # DataModuleConfig options
                 data_dir: Path = Path("./data"),
                 p: float = 1.0,
                 q: float = 1.0,
                 alpha: float = 0.0,
                 walk_length: int = 256,
                 samples_per_hyperedge: int = 1,
                 retain_lcc: bool = True,
                 train_split: List[str] = ["full"],
                 val_split: List[str] = ["full"],
                 predict_split: List[str] = ["full"],
                 # DataLoaderConfig options
                 pin_memory: bool = True,
                 num_workers: int = 4,
                 persistent_workers: bool = True,
                 batch_size: int = 128):
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

        # Additional
        self.dataset_dir = data_dir / dataset_name

    def prepare_data(self):
        datasets = load_dataset(self.dataset_name,
                               cache_dir=self.cache_dir / "huggingface" / "datasets")
        
        processed = datasets.map(process,
                         remove_columns='hif',
                         load_from_cache_file=True,
                         fn_kwargs={
                            "retain_lcc": self.retain_lcc,
                            "samples_per_hyperedge": self.samples_per_hyperedge,
                            "walk_length": self.walk_length,
                            "p": self.p,
                            "q": self.q,
                            "alpha": self.alpha,
                            "num_workers": self.num_workers
                         })

        if not self.dataset_dir.exists():
            processed.save_to_disk(self.dataset_dir)
        
        transformed = DatasetDict({
            k: Dataset.from_generator(transform(v,
                                            samples_per_hyperedge=self.samples_per_hyperedge,
                                            walk_length=self.walk_length,
                                            p=self.p,
                                            q=self.q,
                                            alpha=self.alpha,
                                            num_workers=self.num_workers))
            for k, v in processed.items()
        })

        print(processed)
        print(transformed)

        # At this point we should have train/val/predict splits in the processed dataset

    def setup(self, stage):
        dataset = load_from_disk(self.dataset_dir)
        # print(dataset)
        # print(dataset['full'][0]['rowlist'])
