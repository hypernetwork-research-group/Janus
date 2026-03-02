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
        for hypergraph_entry in dataset:
            ridx_hif_dict = hypergraph_entry["hif_dict"]
            node_features = torch.tensor(hypergraph_entry["node_features"], dtype=torch.float)
            hyperedge_features = torch.tensor(hypergraph_entry["hyperedge_features"], dtype=torch.float)

            # Reconstruct hypergraph
            ridx_hypergraph = xgi.convert.from_hif_dict(ridx_hif_dict, nodetype=int)

            # Incidence matrix (avoid .tolist(); keep it numpy -> torch)
            inc_np = xgi.convert.to_incidence_matrix(ridx_hypergraph, sparse=False)
            incidence_matrix = torch.from_numpy(np.asarray(inc_np, dtype=np.float32))

            # Line graph + helpers
            linegraph = xgi.convert.to_line_graph(ridx_hypergraph)
            sources = list(linegraph.nodes)
            neighborhoods = {n: list(linegraph.neighbors(n)) for n in linegraph.nodes}
            members = ridx_hypergraph.edges.members(dtype=dict)  # edge_id -> members :contentReference[oaicite:3]{index=3}

            # Chunk sources so each task returns a list of walks for that chunk
            parallel_args = [
                (
                    linegraph,
                    samples_per_hyperedge,
                    walk_length,
                    p,
                    q,
                    alpha,
                    seed + chunk_idx,                 # (optional) avoid identical RNG streams per chunk
                    sources[i : i + 10],
                    neighborhoods,
                    members,
                )
                for chunk_idx, i in enumerate(range(0, len(sources), 10))
            ]

            num_nodes = ridx_hypergraph.num_nodes

            def _emit_walk(walk):
                # Streaming-friendly target: per-walk (no need for global max)
                touched_count = len(walk["touched_nodes"])
                target_num_nodes = int(touched_count + (num_nodes - touched_count) * target_lambda)
                target_num_nodes = max(touched_count, min(num_nodes, target_num_nodes))

                touched_nodes, walk["nodes_mask"] = patch_nodes(
                    ridx_hypergraph, walk["touched_nodes"], target_num_nodes
                )

                touched_hyperedges = walk["touched_hyperedges"]
                walk_incidence_matrix = incidence_matrix[touched_nodes][:, touched_hyperedges]

                walk["touched_nodes"] = touched_nodes
                walk["node_features"] = node_features[touched_nodes]
                walk["hyperedge_features"] = hyperedge_features[touched_hyperedges]
                walk["incidence_matrix"] = walk_incidence_matrix

                return walk

            if num_workers == 1:
                for args in tqdm(parallel_args, total=len(parallel_args)):
                    for walk in metropolis_hastings_biased_random_walk(args):
                        yield _emit_walk(walk)
            else:
                with ProcessPoolExecutor(max_workers=num_workers) as executor:
                    futures = [executor.submit(metropolis_hastings_biased_random_walk, args) for args in parallel_args]

                    # as_completed yields futures as soon as each finishes :contentReference[oaicite:4]{index=4}
                    for fut in tqdm(as_completed(futures), total=len(futures)):
                        chunk_paths = fut.result()
                        for walk in chunk_paths:
                            yield _emit_walk(walk)
    return _transform

def add_random_noise(dataset, walk_length):
    node_features = dataset["node_features"]
    random_noise = torch.randn(walk_length, node_features.shape[-1])
    return {
        "random_noise": random_noise
    }
