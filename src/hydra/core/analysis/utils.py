from functools import cache
from pathlib import Path
from os import listdir
from os.path import isdir
import json

import xgi
import hypernetx as hnx

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
        return xgi.to_graph(self.xgi_hypergraph)
    
    @cache
    def to_bipartite_graph(self):
        return xgi.to_bipartite_graph(self.xgi_hypergraph)
    
    @cache
    def to_line_graph(self):
        return xgi.to_line_graph(self.xgi_hypergraph)

    @cache
    def to_hypernetx_hypergraph(self):
        hyperedges = self.xgi_hypergraph.edges.members()
        return hnx.Hypergraph(hyperedges)

    def __str__(self):
        return f"{self.__class__.__name__}({str(self.xgi_hypergraph)})"
    
    def __repr__(self):
        return f"{self.__class__.__name__}({repr(self.xgi_hypergraph)})"

def hif_discovery(path: Path | str):
    for path in file_discovery(path, [".hif.json", ".hif.jsonl"]):
        hypergraph = xgi.read_hif(path, nodetype=int, edgetype=int)
        hypergraph['path'] = str(path)
        yield HypergraphLazyParser(hypergraph)

def results_discovery(path: Path | str):
    for path in file_discovery(path, [".hif.results.json"]):
        with open(path) as f:
            results = json.load(f)
            results['path'] = str(path)
        yield results
