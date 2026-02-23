import xgi
import networkx as nx
import hypernetx.algorithms.hypergraph_modularity as hmod

from hydra.core.analysis.metrics.registry import register_metric
from hydra.core.analysis.metrics.protocols import Metric, MetricResult
from hydra.core.analysis.utils import HypergraphLazyParser

@register_metric()
class DatasetName(Metric):

    def compute(self, hg: HypergraphLazyParser) -> MetricResult:
        return hg.xgi_hypergraph['dataset_name']

    def compare(self, a: MetricResult, b: MetricResult) -> MetricResult:
        return a if a == b else [a, b]

@register_metric()
class Name(Metric):

    def compute(self, hg: HypergraphLazyParser) -> MetricResult:
        return hg.xgi_hypergraph['name']

    def compare(self, a: MetricResult, b: MetricResult) -> MetricResult:
        return a if a == b else [a, b]

@register_metric()
class NumberOfNodes(Metric):

    def compute(self, hg: HypergraphLazyParser) -> MetricResult:
        return hg.xgi_hypergraph.num_nodes
    
    def compare(self, a: MetricResult, b: MetricResult) -> MetricResult:
        return abs(a - b)

@register_metric()
class NumberOfHyperedges(Metric):

    def compute(self, hg: HypergraphLazyParser) -> MetricResult:
        return hg.xgi_hypergraph.num_edges

    def compare(self, a: MetricResult, b: MetricResult) -> MetricResult:
        return abs(a - b)

@register_metric()
class LargestConnectedComponentDiameter(Metric):

    def compute(self, hg: HypergraphLazyParser) -> MetricResult:
        xgi_hypergraph = hg.xgi_hypergraph
        largest_cc_nodes = xgi.largest_connected_component(xgi_hypergraph)
        largest_cc = xgi.subhypergraph(xgi_hypergraph, largest_cc_nodes)
        largest_cc_graph = xgi.to_graph(largest_cc)
        diameter = nx.diameter(largest_cc_graph)
        return diameter

    def compare(self, a: MetricResult, b: MetricResult) -> MetricResult:
        return abs(a - b)

@register_metric()
class IncidenceMatrixDensity(Metric):

    def compute(self, hg: HypergraphLazyParser) -> MetricResult:
        xgi_hypergraph = hg.xgi_hypergraph
        incidence_matrix = xgi.incidence_matrix(xgi_hypergraph)
        num_nodes, num_edges = incidence_matrix.shape
        density = incidence_matrix.nnz / (num_nodes * num_edges)
        return density

    def compare(self, a: MetricResult, b: MetricResult) -> MetricResult:
        return abs(a - b)

@register_metric()
class CliqueExpansionNumberOfEdges(Metric):

    def compute(self, hg: HypergraphLazyParser) -> MetricResult:
        xgi_hypergraph = hg.xgi_hypergraph
        graph = hg.to_graph()
        return graph.number_of_edges()

    def compare(self, a: MetricResult, b: MetricResult) -> MetricResult:
        return abs(a - b)

@register_metric()
class LineGraphNumberOfEdges(Metric):

    def compute(self, hg: HypergraphLazyParser) -> MetricResult:
        xgi_hypergraph = hg.xgi_hypergraph
        line_graph = hg.to_line_graph()
        return line_graph.number_of_edges()

    def compare(self, a: MetricResult, b: MetricResult) -> MetricResult:
        return abs(a - b)

@register_metric()
class CliqueExpansionModularity(Metric):

    def compute(self, hg: HypergraphLazyParser) -> MetricResult:
        graph = hg.to_graph()
        communities = nx.algorithms.community.louvain_communities(graph)
        modularity = nx.algorithms.community.modularity(graph, communities)
        return modularity

    def compare(self, a: MetricResult, b: MetricResult) -> MetricResult:
        return abs(a - b)

@register_metric()
class LineGraphModularity(Metric):

    def compute(self, hg: HypergraphLazyParser) -> MetricResult:
        line_graph = hg.to_bipartite_graph()
        communities = nx.algorithms.community.louvain_communities(line_graph)
        modularity = nx.algorithms.community.modularity(line_graph, communities)
        return modularity

    def compare(self, a: MetricResult, b: MetricResult) -> MetricResult:
        return abs(a - b)

@register_metric()
class BipartiteGraphModularity(Metric):

    def compute(self, hg: HypergraphLazyParser) -> MetricResult:
        xgi_hypergraph = hg.xgi_hypergraph
        bipartite_graph = hg.to_bipartite_graph()
        communities = nx.algorithms.community.louvain_communities(bipartite_graph)
        modularity = nx.algorithms.community.modularity(bipartite_graph, communities)
        return modularity

    def compare(self, a: MetricResult, b: MetricResult) -> MetricResult:
        return abs(a - b)

@register_metric()
class HypergraphModularity(Metric):

    def compute(self, hg: HypergraphLazyParser) -> MetricResult:
        hypernetx_hypergraph = hg.to_hypernetx_hypergraph()
        graph = hg.to_graph()
        communities = nx.algorithms.community.louvain_communities(graph)
        modularity = hmod.modularity(hypernetx_hypergraph, communities)
        return modularity

    def compare(self, a: MetricResult, b: MetricResult) -> MetricResult:
        return abs(a - b)

from .utils import pad_h_portraits, feature_vec, hyperedge_portrait
from scipy.spatial.distance import pdist

@register_metric()
class HyperNetSimile(Metric):

    def compute(self, hg: HypergraphLazyParser) -> MetricResult:
        feature_vector = list(feature_vec(hg.xgi_hypergraph))
        return {
            "feature_vector": feature_vector
        }

    def compare(self, a: MetricResult, b: MetricResult) -> MetricResult:
        hns = pdist([b["feature_vector"], a["feature_vector"]], metric='canberra')[0]
        return hns

from scipy.spatial import distance
import numpy as np

@register_metric()
class HyperPortraitDivergence(Metric):

    def compute(self, hg: HypergraphLazyParser) -> MetricResult:
        potrait = np.array(hyperedge_portrait(hg.xgi_hypergraph)).tolist()
        return {
            "hyperedge_portrait": potrait
        }

    def compare(self, a: MetricResult, b: MetricResult) -> MetricResult:
        a, b = np.array(a["hyperedge_portrait"]), np.array(b["hyperedge_portrait"])
        a, b = pad_h_portraits(a, b)
        P1 = np.ravel(a)
        P2 = np.ravel(b)
        JSD = distance.jensenshannon(P1, P2, base=2)
        hpd = JSD * JSD
        return hpd
