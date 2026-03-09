from functools import cache
from pathlib import Path
from os import listdir
from os.path import isdir
import json

import xgi
import hypernetx as hnx
import networkx as nx
import numpy as np
import torch

def file_discovery(
        path: Path | str,
        target_suffixes: list[str]):
    path = Path(path)
    if isdir(path):
        for f in listdir(path):
            yield from file_discovery(path / f, target_suffixes)
    elif str().join(path.suffixes) in target_suffixes:
        yield path

class HypergraphLazyParser:

    def __init__(self, xgi_hypergraph: xgi.Hypergraph | xgi.DiHypergraph):
        self.xgi_hypergraph = xgi_hypergraph
    
    @cache
    def to_graph(self):
        G = xgi.to_graph(self.xgi_hypergraph)
        return G
    
    @cache
    def to_bipartite_graph(self):
        return xgi.to_bipartite_graph(self.xgi_hypergraph)
    
    @cache
    def to_line_graph(self):
        return xgi.to_line_graph(self.xgi_hypergraph)

    @cache
    def to_hypernetx_hypergraph(self):
        hif_dict = xgi.to_hif_dict(self.xgi_hypergraph)
        hypernetx_hypergraph = hnx.from_hif(hif_dict) # Convert from xgi to hypernetx using HIF as an intermediate format
        df = hypernetx_hypergraph.edges.dataframe
        ps = hypernetx_hypergraph.edges.property_store
        df = ps.properties
        # Ensure weight exists and is float
        if "weight" not in df.columns:
            df["weight"] = 1.0
        else:
            # Convert the entire column to float in-place
            df["weight"] = df["weight"].astype("float64")

        # Also make the default weight float for any edges that get inserted later
        ps.set_defaults({"weight": 1.0})
        return hypernetx_hypergraph

    @cache
    def to_incidence_matrix(self):
        inc_np, nodeidx, edgeidx = xgi.incidence_matrix(self.xgi_hypergraph, sparse=False, index=True)
        incidence_matrix = torch.from_numpy(np.asarray(inc_np, dtype=np.float32))
        # Here, the incidence matrix indices may not be in the same order as the node and edge features
        # Permutations that sort rows/cols by the real IDs
        row_perm = torch.tensor(
            sorted(nodeidx.keys(), key=lambda k: nodeidx[k]),
            dtype=torch.long
        )
        col_perm = torch.tensor(
            sorted(edgeidx.keys(), key=lambda k: edgeidx[k]),
            dtype=torch.long
        )
        # Reorder incidence matrix to match the order of node and edge features
        incidence_matrix = incidence_matrix[row_perm][:, col_perm]
        return incidence_matrix

    def generate_random_paths(self, path_length: int, batch_size: int = 32):
        assert path_length >= 2, "Path length must be at least 2 to generate paths in the line graph."
        line_graph = self.to_line_graph()
        random_paths = list(nx.generate_random_paths(line_graph, batch_size, path_length - 1))
        incidence_matrix = self.to_incidence_matrix()
        return incidence_matrix[:, random_paths].permute(1, 0, 2).contiguous()

    def __str__(self):
        return f"{self.__class__.__name__}({str(self.xgi_hypergraph)})"
    
    def __repr__(self):
        return f"{self.__class__.__name__}({repr(self.xgi_hypergraph)})"
    
    def __getitem__(self, key):
        return self.xgi_hypergraph[key]

def hif_discovery(path: Path | str):
    for path in file_discovery(path, [".hif.json", ".hif.jsonl"]):
        hypergraph = xgi.read_hif(path, nodetype=int, edgetype=int)
        hypergraph['path'] = str(path)
        yield path, HypergraphLazyParser(hypergraph)

def results_discovery(path: Path | str):
    for path in file_discovery(path, [".hif.results.json"]):
        with open(path) as f:
            results = json.load(f)
            results['path'] = str(path)
        yield path, results

def comparison_discovery(path: Path | str):
    for path in file_discovery(path, [".hif.results.comparison.json"]):
        with open(path) as f:
            comparison_results = json.load(f)
        yield path, comparison_results
