from hydra.core.analysis.metrics.registry import register_metric
from hydra.core.analysis.metrics.protocols import Metric, MetricResult
from hydra.core.analysis.utils import HypergraphLazyParser

@register_metric()
class NumNodes(Metric):

    def compute(self, hg: HypergraphLazyParser) -> MetricResult:
        return hg.xgi_hypergraph.num_nodes
    
    def compare(self, a: MetricResult, b: MetricResult) -> MetricResult:
        if isinstance(a, float) and isinstance(b, float):
            return abs(a - b)
        raise TypeError(f"Unexpected types for comparison: {type(a)}, {type(b)}")

@register_metric()
class NumHyperedges(Metric):

    def compute(self, hg: HypergraphLazyParser) -> MetricResult:
        return hg.xgi_hypergraph.num_edges

    def compare(self, a: MetricResult, b: MetricResult) -> MetricResult:
        if isinstance(a, float) and isinstance(b, float):
            return abs(a - b)
        raise TypeError(f"Unexpected types for comparison: {type(a)}, {type(b)}")
