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
class Kind(Metric):
    
    def compute(self, hg: HypergraphLazyParser) -> MetricResult:
        return hg.xgi_hypergraph['kind']

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
class LargestConnectedComponentNumberOfNodes(Metric):

    def compute(self, hg: HypergraphLazyParser) -> MetricResult:
        xgi_hypergraph = hg.xgi_hypergraph
        largest_cc_nodes = xgi.largest_connected_component(xgi_hypergraph)
        return len(largest_cc_nodes)

    def compare(self, a: MetricResult, b: MetricResult) -> MetricResult:
        return abs(a - b)

@register_metric()
class LargestConnectedComponentNumberOfHyperedges(Metric):

    def compute(self, hg: HypergraphLazyParser) -> MetricResult:
        xgi_hypergraph = hg.xgi_hypergraph
        largest_cc_nodes = xgi.largest_connected_component(xgi_hypergraph)
        largest_cc = xgi.subhypergraph(xgi_hypergraph, largest_cc_nodes)
        return largest_cc.num_edges

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
        line_graph = hg.to_line_graph()
        return line_graph.number_of_edges()

    def compare(self, a: MetricResult, b: MetricResult) -> MetricResult:
        return abs(a - b)

@register_metric()
class HypergraphDegreeAssortativity(Metric):

    def compute(self, hg: HypergraphLazyParser) -> MetricResult:
        xgi_hypergraph = hg.xgi_hypergraph
        degree_assortativity = xgi.degree_assortativity(xgi_hypergraph)
        return degree_assortativity

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
class CliqueExpansionAverageClusteringCoefficient(Metric):

    def compute(self, hg: HypergraphLazyParser) -> MetricResult:
        graph = hg.to_graph()
        avg_clustering = nx.average_clustering(graph)
        return avg_clustering

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
        communities = hmod.kumar(hypernetx_hypergraph)
        communities = hmod.last_step(hypernetx_hypergraph, communities)
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
        potrait = hyperedge_portrait(hg.xgi_hypergraph)
        potrait = np.array(potrait).tolist()
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

from sklearn.metrics import normalized_mutual_info_score

@register_metric()
class NormalizedMutualInformation(Metric):

    def compute(self, hg: HypergraphLazyParser) -> MetricResult:
        hypernetx_hypergraph = hg.to_hypernetx_hypergraph()

        communities = hmod.kumar(hypernetx_hypergraph)
        communities = hmod.last_step(hypernetx_hypergraph, communities)

        # Stable node order for the output vector
        node_order = list(hypernetx_hypergraph.nodes)

        # Safer: convert partition (list[set]) -> node -> community_id
        node_to_community = hmod.part2dict(communities)

        # Keep label vector aligned with node_order
        labels = [node_to_community.get(node, -1) for node in node_order]

        return {
            "labels": labels
        }

    def compare(self, a: MetricResult, b: MetricResult) -> MetricResult:
        val_a, val_b = a["labels"], b["labels"]
        max_length = max(len(val_a), len(val_b))
        val_a = val_a + [-1] * (max_length - len(val_a))
        val_b = val_b + [-1] * (max_length - len(val_b))
        nmi = normalized_mutual_info_score(val_a, val_b)
        return nmi

@register_metric()
class HyperedgeRecovery(Metric):

    def compute(self, hg: HypergraphLazyParser) -> MetricResult:
        if hg['kind'] not in ['conditional', 'reconstruction', 'reference']:
            return []
        xgi_hypergraph = hg.xgi_hypergraph
        hyperedges = list(map(list, map(sorted, xgi_hypergraph.edges.members())))
        return hyperedges

    def compare(self, a: MetricResult, b: MetricResult) -> MetricResult:
        set_a = set(tuple(sorted(edge)) for edge in a)
        set_b = set(tuple(sorted(edge)) for edge in b)
        intersection = set_a.intersection(set_b)
        union = set_a.union(set_b)
        jaccard_similarity = len(intersection) / len(union) if union else 1.0
        return {
            "jaccard_similarity": jaccard_similarity,
            "intersection": len(intersection),
            "union": len(union),
            "card_a": len(set_a),
            "card_b": len(set_b),
        }

from .utils import number_of_closed_triangles

@register_metric()
class NumberOfClosedTriangles(Metric):

    def compute(self, hg: HypergraphLazyParser) -> MetricResult:
        return number_of_closed_triangles(hg)

    def compare(self, a: MetricResult, b: MetricResult) -> MetricResult:
        return abs(a - b)

@register_metric()
class NumberOfOpenTriangles(Metric):

    def compute(self, hg: HypergraphLazyParser) -> MetricResult:
        # Triangles in the pairwise projection / 1-skeleton
        G = hg.to_graph()
        total_projected_triangles = sum(nx.triangles(G).values()) // 3

        # Open = projected triangles that are not closed in the hypergraph
        closed_triangles = number_of_closed_triangles(hg)
        return total_projected_triangles - closed_triangles

    def compare(self, a: MetricResult, b: MetricResult) -> MetricResult:
        return abs(a - b)

from hydra.core.models.components import StructureOnlyHypergraphRegressor

class GeneratedHypergraphDetection(Metric):

    def compute(self, hg: HypergraphLazyParser) -> MetricResult:
        hypergraph_detector = StructureOnlyHypergraphRegressor(
            out_channels=16,
            hidden_channels=16,
            in_channels=16,
            num_classes=1,
            num_blocks=3
        )

from .utils import empirical_integer_distribution, discrete_wasserstein_distance

@register_metric()
class NodeDegreeDistribution(Metric):

    def compute(self, hg: HypergraphLazyParser) -> MetricResult:
        incidence_matrix = hg.to_incidence_matrix()
        # Node degree = number of incident hyperedges = row sum
        degrees = incidence_matrix.sum(dim=1).cpu().numpy().astype(np.int64)
        return empirical_integer_distribution(degrees)

    def compare(self, a: MetricResult, b: MetricResult) -> MetricResult:
        return discrete_wasserstein_distance(a, b)

@register_metric()
class HyperedgeSizeDistribution(Metric):

    def compute(self, hg: HypergraphLazyParser) -> MetricResult:
        incidence_matrix = hg.to_incidence_matrix()
        # Hyperedge size = number of incident nodes = column sum
        sizes = incidence_matrix.sum(dim=0).cpu().numpy().astype(np.int64)
        return empirical_integer_distribution(sizes)

    def compare(self, a: MetricResult, b: MetricResult) -> MetricResult:
        return discrete_wasserstein_distance(a, b)

# Structural Patterns

@register_metric()
class OneLevelDecomposedHypergraphClusteringCoefficient(Metric):

    def compute(self, hg: HypergraphLazyParser) -> MetricResult:
        decomposed_hg: nx.Graph = hg.l_decomposed_hypergraph(l=1)
        clustering_coeffs = nx.clustering(decomposed_hg)
        avg_clustering_coeff = sum(clustering_coeffs.values()) / len(clustering_coeffs)
        return avg_clustering_coeff

    def compare(self, a: MetricResult, b: MetricResult) -> MetricResult:
        return abs(a - b)

@register_metric()
class OneLevelDecomposedHypergraphLargestConnectedComponent(Metric):

    def compute(self, hg: HypergraphLazyParser) -> MetricResult:
        decomposed_hg: nx.Graph = hg.l_decomposed_hypergraph(l=1)
        largest_cc_nodes = max(nx.connected_components(decomposed_hg), key=len)
        # return lcc size as a fraction of total nodes in the decomposed graph
        return len(largest_cc_nodes) / decomposed_hg.number_of_nodes()

    def compare(self, a: MetricResult, b: MetricResult) -> MetricResult:
        return abs(a - b)

@register_metric()
class OneLevelDecomposedHypergraphDiameter(Metric):

    def compute(self, hg: HypergraphLazyParser) -> MetricResult:
        G = hg.l_decomposed_hypergraph(l=1)

        if G.number_of_nodes() == 0:
            return 0

        largest_cc = max(nx.connected_components(G), key=len)
        G = G.subgraph(largest_cc)

        if G.number_of_nodes() <= 1:
            return 0

        # Collect shortest path lengths
        lengths = []
        for _, dist_dict in nx.all_pairs_shortest_path_length(G):
            lengths.extend(d for d in dist_dict.values() if d > 0)

        if not lengths:
            return 0

        # Histogram-based CDF
        values, counts = np.unique(lengths, return_counts=True)
        cum_counts = np.cumsum(counts)
        total = cum_counts[-1]

        target = 0.9 * total

        # Find where CDF crosses 90%
        idx = np.searchsorted(cum_counts, target)

        if idx == 0:
            return float(values[0])

        # Linear interpolation
        x0, x1 = values[idx - 1], values[idx]
        y0, y1 = cum_counts[idx - 1], cum_counts[idx]

        effective_diameter = x0 + (target - y0) * (x1 - x0) / (y1 - y0)

        return float(effective_diameter)

    def compare(self, a: MetricResult, b: MetricResult) -> MetricResult:
        return abs(a - b)

@register_metric()
class OneLevelDecomposedHypergraphClosedTriangles(Metric):

    def compute(self, hg: HypergraphLazyParser) -> MetricResult:
        G: nx.Graph = hg.l_decomposed_hypergraph(l=1)
        triangles_per_node = nx.triangles(G)
        total_triangles = sum(triangles_per_node.values()) // 3
        return total_triangles

    def compare(self, a: MetricResult, b: MetricResult) -> MetricResult:
        return abs(a - b)

@register_metric()
class OneLevelDecomposedHypergraphOpenTriangles(Metric):

    def compute(self, hg: HypergraphLazyParser) -> MetricResult:
        G: nx.Graph = hg.l_decomposed_hypergraph(l=1)
        
        # count all connected triplets
        triplets = 0
        for node in G:
            k = G.degree(node)
            triplets += k * (k - 1) // 2

        triangles_per_node = nx.triangles(G)
        # total number of closed triangles
        closed_triangles = sum(triangles_per_node.values()) // 3

        # open triangles = triplets - closed triangles
        open_triangles = triplets - closed_triangles
        return open_triangles

    def compare(self, a: MetricResult, b: MetricResult) -> MetricResult:
        return abs(a - b)

