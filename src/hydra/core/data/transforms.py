import torch
import xgi
from concurrent.futures import ProcessPoolExecutor
from tqdm.auto import tqdm

from .utils import metropolis_hastings_biased_random_walk, patch_nodes

def process(batch,
            retain_lcc: bool):
    hif_dict = batch
    hypergraph = xgi.convert.from_hif_dict(hif_dict, nodetype=int)
    # if retain_lcc:
    #     hypergraph: xgi.Hypergraph = xgi.largest_connected_hypergraph(hypergraph)
    #     hif_dict = xgi.convert.to_hif_dict(hypergraph)
        # TODO: Perform logging
        # TODO: After lcc retaining
        # TODO: Perform reindexing of nodes and lcc retaining in the preprocessing script

    X = []
    for node in sorted(hypergraph.nodes):
        X.append(hypergraph.nodes[node]['eigsh'])
    Y = []
    for edge in sorted(hypergraph.edges):
        Y.append(hypergraph.edges[edge]['eigsh'])

    return {
        "hif_dict": hif_dict,
        "node_features": X,
        "hyperedge_features": Y,
    }

def transform(dataset,
            samples_per_hyperedge: int,
            walk_length: int,
            p: float,
            q: float,
            alpha: float,
            num_workers: int):
    def _transform():
        for hypergraph_entry in dataset:
            ridx_hif_dict = hypergraph_entry['hif_dict']
            node_features = torch.tensor(hypergraph_entry['node_features'], dtype=torch.float)
            hyperedge_features = torch.tensor(hypergraph_entry['hyperedge_features'], dtype=torch.float)
            # 2) Reconstruct the reindexed hypergraph
            ridx_hypergraph = xgi.convert.from_hif_dict(ridx_hif_dict, nodetype=int)
            incidence_matrix = torch.tensor(xgi.convert.to_incidence_matrix(ridx_hypergraph, sparse=False).tolist(), dtype=torch.float32)
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
                touched_nodes, walk['nodes_mask'] = patch_nodes(ridx_hypergraph, walk['touched_nodes'], target_num_nodes)
                touched_hyperedges = walk['touched_hyperedges']
                walk_incidence_matrix = incidence_matrix[touched_nodes][:, touched_hyperedges]
                walk['touched_nodes'] = touched_nodes
                walk['node_features'] = node_features[touched_nodes]
                walk['hyperedge_features'] = hyperedge_features[touched_hyperedges]
                walk['incidence_matrix'] = walk_incidence_matrix
                yield walk
    return _transform

def add_random_noise(dataset, walk_length):
    node_features = dataset["node_features"]
    random_noise = torch.randn(walk_length, node_features.shape[-1])
    return {
        "random_noise": random_noise
    }
