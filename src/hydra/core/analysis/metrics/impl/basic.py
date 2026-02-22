import xgi
import networkx as nx

from hydra.core.analysis.metrics.registry import register_metric
from hydra.core.analysis.metrics.protocols import Metric, MetricResult
from hydra.core.analysis.utils import HypergraphLazyParser

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
        graph = xgi.to_graph(xgi_hypergraph)
        return graph.number_of_edges()

    def compare(self, a: MetricResult, b: MetricResult) -> MetricResult:
        return abs(a - b)

@register_metric()
class LineGraphNumberOfEdges(Metric):

    def compute(self, hg: HypergraphLazyParser) -> MetricResult:
        xgi_hypergraph = hg.xgi_hypergraph
        line_graph = xgi.to_line_graph(xgi_hypergraph)
        return line_graph.number_of_edges()

    def compare(self, a: MetricResult, b: MetricResult) -> MetricResult:
        return abs(a - b)
