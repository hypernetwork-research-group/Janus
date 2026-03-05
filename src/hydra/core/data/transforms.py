import logging
from concurrent.futures import ProcessPoolExecutor, wait, FIRST_COMPLETED
import multiprocessing as mp

import torch
import xgi
from tqdm.auto import tqdm
import numpy as np

from .utils import metropolis_hastings_biased_random_walk, patch_nodes

logger = logging.getLogger(__name__)

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

_WALK_CTX = {}

def _init_walk_worker(
    linegraph,
    neighborhoods,
    members,
    degrees,
    samples_per_hyperedge: int,
    walk_length: int,
    p: float,
    q: float,
    alpha: float,
    seed: int,
):
    # Big read-only objects (sent once per worker)
    _WALK_CTX["linegraph"] = linegraph
    _WALK_CTX["neighborhoods"] = neighborhoods
    _WALK_CTX["members"] = members
    _WALK_CTX["degrees"] = degrees

    # Small constants (also avoid re-sending per task)
    _WALK_CTX["samples_per_hyperedge"] = samples_per_hyperedge
    _WALK_CTX["walk_length"] = walk_length
    _WALK_CTX["p"] = p
    _WALK_CTX["q"] = q
    _WALK_CTX["alpha"] = alpha
    _WALK_CTX["seed"] = seed

def _walk_task(sources_chunk):
    """
    Task input is *only* the small sources chunk.
    Everything else comes from the per-worker initializer globals.
    """
    args = (
        _WALK_CTX["linegraph"],
        _WALK_CTX["samples_per_hyperedge"],
        _WALK_CTX["walk_length"],
        _WALK_CTX["p"],
        _WALK_CTX["q"],
        _WALK_CTX["alpha"],
        _WALK_CTX["seed"],
        sources_chunk,
        _WALK_CTX["neighborhoods"],
        _WALK_CTX["members"],
        _WALK_CTX["degrees"],
    )
    return metropolis_hastings_biased_random_walk(args)

from concurrent.futures import ProcessPoolExecutor, wait, FIRST_COMPLETED

def transform(
    dataset,
    samples_per_hyperedge: int,
    walk_length: int,
    p: float,
    q: float,
    alpha: float,
    num_workers: int,
    seed: int = 42,
):
    def _transform():
        logger.info("Starting transformation")
        ctx = mp.get_context("fork")

        for hypergraph_entry in dataset:
            logger.info("Processing hypergraph entry")

            ridx_hif_dict = hypergraph_entry["hif_dict"]
            node_features = torch.tensor(hypergraph_entry["node_features"], dtype=torch.float)
            hyperedge_features = torch.tensor(hypergraph_entry["hyperedge_features"], dtype=torch.float)

            logger.info("Converting to xgi hypergraph")
            ridx_hypergraph = xgi.convert.from_hif_dict(ridx_hif_dict, nodetype=int, edgetype=int)
            num_nodes = ridx_hypergraph.num_nodes

            logger.info("Computing incidence matrix")
            inc_np = xgi.convert.to_incidence_matrix(ridx_hypergraph, sparse=False)
            incidence_matrix = torch.from_numpy(np.asarray(inc_np, dtype=np.float32))

            logger.info("Computing line graph")
            linegraph = xgi.to_line_graph(ridx_hypergraph)
            sources = list(linegraph.nodes)

            logger.info("Precomputing neighborhoods")
            neighborhoods = {node: list(linegraph.neighbors(node)) for node in linegraph.nodes}
            logger.info("Precomputing members")
            members = ridx_hypergraph.edges.members()
            logger.info("Precomputing degrees")
            degrees = dict(linegraph.degree())

            source_chunks = [sources[i : i + 10] for i in range(0, len(sources), 10)]
            it_source_chunks = iter(source_chunks)

            logger.info("Starting parallel random walks")
            with tqdm(total=len(source_chunks), desc="Random Walks") as pbar:
                with ProcessPoolExecutor(
                    max_workers=num_workers,
                    mp_context=ctx,
                    initializer=_init_walk_worker,
                    initargs=(
                        linegraph,
                        neighborhoods,
                        members,
                        degrees,
                        samples_per_hyperedge,
                        walk_length,
                        p,
                        q,
                        alpha,
                        seed,
                    ),
                ) as executor:
                    logger.info(f"Executor initialized with {num_workers} workers")

                    # Cap how many tasks/results can be outstanding at once.
                    # 2x workers keeps the pool busy while still bounding memory.
                    max_in_flight = max(1, num_workers * 2)
                    in_flight = set()

                    def submit_one() -> bool:
                        """Submit one chunk if available; return True if submitted."""
                        try:
                            chunk = next(it_source_chunks)
                        except StopIteration:
                            return False
                        in_flight.add(executor.submit(_walk_task, chunk))
                        return True

                    # Prime the pipeline (handle case: fewer chunks than workers)
                    for _ in range(min(max_in_flight, len(source_chunks))):
                        if not submit_one():
                            break

                    # Drain + refill
                    while in_flight:
                        done, _ = wait(in_flight, return_when=FIRST_COMPLETED)

                        for fut in done:
                            # IMPORTANT: remove future so it (and its result) can be GC’d
                            in_flight.remove(fut)

                            result = fut.result()  # list of walks for that finished chunk

                            for walk in result:
                                touched_nodes_raw = walk["touched_nodes"]
                                touched_hyperedges = walk["touched_hyperedges"]

                                target_num_nodes = num_nodes

                                touched_nodes, nodes_mask = patch_nodes(
                                    ridx_hypergraph, touched_nodes_raw, target_num_nodes
                                )

                                # KEEPING tolist() AS REQUESTED
                                walk["touched_nodes"] = touched_nodes
                                walk["nodes_mask"] = nodes_mask
                                walk["incidence_matrix"] = incidence_matrix[touched_nodes][:, touched_hyperedges].tolist()
                                walk["node_features"] = node_features[touched_nodes].tolist()
                                walk["hyperedge_features"] = hyperedge_features[touched_hyperedges].tolist()

                                yield walk

                            pbar.update(1)

                            # Refill one task per completed task (keeps bounded pressure)
                            submit_one()
    return _transform

def add_random_noise(dataset, walk_length):
    node_features = dataset["node_features"]
    random_noise = torch.randn(walk_length, node_features.shape[-1])
    return {
        "random_noise": random_noise
    }
