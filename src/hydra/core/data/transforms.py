import logging

logger = logging.getLogger(__name__)

import torch
import xgi
from concurrent.futures import ProcessPoolExecutor, as_completed
from tqdm.auto import tqdm
import numpy as np

from .utils import metropolis_hastings_biased_random_walk, patch_nodes

def process(batch, node_feature: str, hyperedge_feature: str):
    hif_dict = batch
    hypergraph = xgi.convert.from_hif_dict(hif_dict, nodetype=int)
    # if retain_lcc:
    #     hypergraph: xgi.Hypergraph = xgi.largest_connected_hypergraph(hypergraph)
    #     hif_dict = xgi.convert.to_hif_dict(hypergraph)
        # TODO: This will be performed in the dataset creation part
        # TODO: Perform logging

    X = []
    for node in sorted(hypergraph.nodes):
        X.append(hypergraph.nodes[node][node_feature])
    Y = []
    for edge in sorted(hypergraph.edges):
        Y.append(hypergraph.edges[edge][hyperedge_feature])

    return {
        "hif_dict": hif_dict,
        "node_features": X,
        "hyperedge_features": Y,
    }

def transform(
    dataset,
    samples_per_hyperedge: int,
    walk_length: int,
    p: float,
    q: float,
    alpha: float,
    num_workers: int,
    target_lambda: float = 1.0,
    seed: int = 42,
):
    def _transform():
        logger.info("Starting transformation")
        for hypergraph_entry in dataset:
            ridx_hif_dict = hypergraph_entry["hif_dict"]
            node_features = torch.tensor(hypergraph_entry["node_features"], dtype=torch.float)
            hyperedge_features = torch.tensor(hypergraph_entry["hyperedge_features"], dtype=torch.float)

            # Reconstruct the reindexed hypergraph
            ridx_hypergraph = xgi.convert.from_hif_dict(ridx_hif_dict, nodetype=int, edgetype=int)
            num_nodes = ridx_hypergraph.num_nodes

            inc_np = xgi.convert.to_incidence_matrix(ridx_hypergraph, sparse=False)
            incidence_matrix = torch.from_numpy(np.asarray(inc_np, dtype=np.float32))

            # Sample random walks from the hypergraph
            logger.info("Computing line graph")
            linegraph = xgi.convert.to_line_graph(ridx_hypergraph)
            sources = list(linegraph.nodes)
            logger.info("Precomputing neighborhoods")
            neighborhoods = {node: list(linegraph.neighbors(node)) for node in linegraph.nodes}
            logger.info("Precomputing members")
            members = ridx_hypergraph.edges.members()
            logger.info("Precomputing degrees")
            degrees = dict(linegraph.degree())

            logger.info("Computing parallel args for random walks")
            parallel_args = [
                (
                    linegraph,
                    samples_per_hyperedge,
                    walk_length,
                    p,
                    q,
                    alpha,
                    seed,
                    sources[i : i + 10],
                    neighborhoods,
                    members,
                    degrees
                )
                for i in range(0, len(sources), 10)
            ]

            logger.info("Starting parallel random walks")

            with tqdm(total=len(parallel_args), desc="Random Walks") as pbar:

                for i in range(0, len(parallel_args), num_workers * 2):
                    sub_parallel_args = parallel_args[i : i + num_workers * 2]
                    # Stream walks as each worker returns its chunk (still using ProcessPoolExecutor + map)
                    with ProcessPoolExecutor(
                        max_workers=num_workers,
                    ) as executor:
                        for result in executor.map(metropolis_hastings_biased_random_walk, sub_parallel_args):
                            # `result` is the list of walks produced for this chunk
                            for walk in result:
                                # Compute target_num_nodes per-walk (instead of globally)
                                touched_nodes_raw = walk["touched_nodes"]
                                touched_hyperedges = walk["touched_hyperedges"]

                                target_num_nodes = int(
                                    len(touched_nodes_raw)
                                    + (num_nodes - len(touched_nodes_raw)) * target_lambda
                                )

                                touched_nodes, walk["nodes_mask"] = patch_nodes(
                                    ridx_hypergraph, touched_nodes_raw, target_num_nodes
                                )

                                walk_incidence_matrix = incidence_matrix[touched_nodes][:, touched_hyperedges].tolist()

                                walk["node_features"] = node_features[touched_nodes].tolist()
                                walk["hyperedge_features"] = hyperedge_features[touched_hyperedges].tolist()
                                walk["incidence_matrix"] = walk_incidence_matrix

                                yield walk
                            pbar.update(1)
    return _transform

def add_random_noise(dataset, walk_length):
    node_features = dataset["node_features"]
    random_noise = torch.randn(walk_length, node_features.shape[-1])
    return {
        "random_noise": random_noise
    }
