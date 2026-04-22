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

import itertools
from math import comb
import numpy as np
import xgi
from tqdm import tqdm

def l_decomposed_hypergraph(
    H: xgi.Hypergraph,
    l: int,
    *,
    max_subsets_per_edge: int | None = None,
    sort_nodes: bool = True,
) -> nx.Graph:
    """
    Build the l-level decomposed graph of an XGI hypergraph.

    Assumptions for this implementation:
    - weighted = False
    - return_subset_labels = False

    Nodes in the returned graph are compact integer ids.
    Each node has attribute:
        subset -> tuple representing the l-subset

    This version avoids building a SciPy sparse matrix and avoids
    nx.from_scipy_sparse_array(...), which can create large temporary
    copies and cause OOM.
    """
    if not isinstance(l, int) or l < 1:
        raise ValueError("l must be a positive integer.")

    def iter_hyperedges(H):
        if hasattr(H, "edges") and hasattr(H.edges, "members"):
            members = H.edges.members()
            if hasattr(members, "items"):
                for _, nodes in members.items():
                    yield tuple(nodes)
                return
            for nodes in members:
                yield tuple(nodes)
            return

        if hasattr(H, "edges"):
            for e in H.edges:
                yield tuple(H.edges.members(e))
            return

        raise TypeError("Could not extract hyperedges from H.")

    subset_to_id: dict[tuple, int] = {}
    next_id = 0

    # Store each undirected edge only once as (min_id, max_id)
    edge_set: set[tuple[int, int]] = set()

    for edge_nodes in tqdm(iter_hyperedges(H), desc="Processing hyperedges"):
        m = len(edge_nodes)
        if m < l:
            continue

        n_subsets = comb(m, l)
        if max_subsets_per_edge is not None and n_subsets > max_subsets_per_edge:
            continue

        subset_ids = []
        for subset in itertools.combinations(edge_nodes, l):
            key = tuple(sorted(subset)) if sort_nodes else tuple(subset)
            node_id = subset_to_id.get(key)
            if node_id is None:
                node_id = next_id
                subset_to_id[key] = node_id
                next_id += 1
            subset_ids.append(node_id)

        # Add clique edges among all l-subsets of this hyperedge
        for u, v in itertools.combinations(subset_ids, 2):
            if u > v:
                u, v = v, u
            edge_set.add((u, v))

    n = next_id

    print(f"Building NetworkX graph directly from {len(edge_set)} undirected edges...")

    G = nx.Graph()
    print(f"Adding {n} nodes and {len(edge_set)} edges to the graph...")
    G.add_nodes_from(range(n))
    print("Adding edges...")
    # G.add_edges_from(edge_set)
    for u, v in tqdm(edge_set, desc="Adding edges to graph", mininterval=1.0):
        G.add_edge(u, v)

    # # Attach subset labels as node attributes
    # id_to_subset = {idx: subset for subset, idx in subset_to_id.items()}
    # nx.set_node_attributes(G, id_to_subset, name="subset")

    return G

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

    @cache
    def l_decomposed_hypergraph(self, l: int):
        decomposed_hg = l_decomposed_hypergraph(self.xgi_hypergraph, l)
        return decomposed_hg

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
