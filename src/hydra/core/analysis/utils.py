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
from collections import defaultdict
import networkx as nx

def l_decomposed_hypergraph(H: xgi.Hypergraph, l: int) -> nx.Graph:
    """
    Build the l-level decomposed graph of an XGI hypergraph.

    Parameters
    ----------
    H : xgi.Hypergraph
        Input hypergraph.
    l : int
        Decomposition level (called k in the paper).

    Returns
    -------
    G : networkx.Graph
        The l-level decomposed graph.

        - Each node is a frozenset of l original nodes.
        - Two nodes u, v are adjacent iff there exists a hyperedge e in H
          such that u ∪ v ⊆ e.
        - Edge attribute 'weight' counts how many hyperedges contain u ∪ v.
          If you want the simple unweighted version, you can ignore it.

    Notes
    -----
    This matches the paper's k-level decomposition:
        V^(k) = {v subset of V : |v| = k and v is contained in some hyperedge}
        E^(k) = {{u, v} : there exists a hyperedge e with u ∪ v ⊆ e}

    The weighted edge count is also useful because the appendix defines
    edge weights as the number of hyperedges containing u ∪ v.
    """

    if not isinstance(l, int) or l < 1:
        raise ValueError("l must be a positive integer.")

    def _get_hyperedges_as_sets(H):
        """
        Try a few common XGI access patterns and return a list of sets.
        """
        # Common XGI pattern
        if hasattr(H, "edges") and hasattr(H.edges, "members"):
            members = H.edges.members()
            # members may be a dict-like {edge_id: iterable_of_nodes}
            if hasattr(members, "items"):
                return [set(nodes) for _, nodes in members.items()]
            return [set(nodes) for nodes in members]

        # Fallback: iterate edge IDs and query members
        if hasattr(H, "edges"):
            try:
                return [set(H.edges.members(e)) for e in H.edges]
            except Exception:
                pass

        raise TypeError(
            "Could not extract hyperedges from H. "
            "Expected an XGI hypergraph with an accessible edge-members API."
        )

    hyperedges = _get_hyperedges_as_sets(H)

    G = nx.Graph()
    edge_weights = defaultdict(int)

    for e in hyperedges:
        if len(e) < l:
            continue

        # All l-subsets of this hyperedge become nodes in the decomposed graph
        l_subsets = [frozenset(s) for s in itertools.combinations(e, l)]

        # Add nodes
        for s in l_subsets:
            if not G.has_node(s):
                G.add_node(s, members=tuple(s), size=l)

        # Connect every pair of l-subsets whose union is contained in this hyperedge.
        # Since both subsets come from e, this condition is automatically satisfied.
        for u, v in itertools.combinations(l_subsets, 2):
            edge_weights[(u, v)] += 1

    # Add weighted edges
    for (u, v), w in edge_weights.items():
        G.add_edge(u, v, weight=w)

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
