import random
from copy import deepcopy
from math import isfinite

import xgi

def metropolis_hastings_biased_random_walk(args):
    # TODO: Generalize random walk to k-th order
    G, num_paths, walk_length, p, q, a, seed, sources, neighborhoods, members, weights = args
    assert p > 0, "p must be positive"
    assert q > 0, "q must be positive"
    walks = []
    rng = random.Random(seed)
    inv_p = 1 / p
    inv_q = 1 / q
    for source in sources:
        for _ in range(num_paths):
            walk_touched_nodes = set()
            walk = [source]
            previous = None
            current = source
            touched_nodes = members[current]
            walk_touched_nodes.update(touched_nodes)
            for _ in range(walk_length - 1):
                neighbors = neighborhoods[current]
                if not neighbors:
                    # Keep looping
                    walk.append(current) # Insert at the beginning
                    continue
                cumulative_alphas = []
                total_alpha = 0.0
                for neighbor in neighbors:
                    if previous is None:
                        alpha = 1.0
                    elif neighbor == previous:
                        alpha = inv_p
                    elif G.has_edge(previous, neighbor):
                    # elif previous in neighborhoods[neighbor]:
                        alpha = 1
                    else:
                        alpha = inv_q
                    total_alpha += alpha
                    cumulative_alphas.append(total_alpha)
                if not isfinite(total_alpha) or total_alpha <= 0:
                    raise ValueError(f"Invalid total_alpha={total_alpha} for node {current} with neighbors {neighbors}. Check p, q values and graph structure.")
                next_node = rng.choices(neighbors, k=1, cum_weights=cumulative_alphas)[0]
                if a == 0.0:
                    accept = True
                else:
                    # If a node has degree 0 we are accepting it directly because it has no neighbors
                    acceptance_prob = min(1, (weights[current] / weights[next_node]) ** a)
                    accept = rng.random() < acceptance_prob
                if accept:
                    previous = current
                    current = next_node
                if current not in walk:
                    touched_nodes = members[current]
                    walk_touched_nodes.update(touched_nodes)
                walk.append(current) # Insert at the beginning
            walks.append({
                "touched_hyperedges": sorted(walk), # This sorted is not strictly necessary but it makes it easier to compare walks and debug
                "touched_nodes": sorted(list(walk_touched_nodes))
            })
    return walks

def patch_nodes(hypergraph: xgi.Hypergraph, touched_nodes: list, target_num_nodes: int):
    _temp_nodes = deepcopy(touched_nodes)
    missing_nodes = list(hypergraph.nodes.ids - set(touched_nodes))
    num_to_add = target_num_nodes - len(touched_nodes)
    mask = [True] * target_num_nodes # This mask indicates which nodes are original touched nodes
    if num_to_add > 0:
        nodes_to_add = random.sample(missing_nodes, num_to_add)
        _temp_nodes.extend(nodes_to_add)
        _temp_nodes = sorted(_temp_nodes)
        for node in nodes_to_add:
            index = _temp_nodes.index(node)
            mask[index] = False
    else:
        _temp_nodes = sorted(_temp_nodes)
    return _temp_nodes, mask

import numpy as np
from scipy import sparse

def hypergraph_laplacian_zhou(A: sparse.spmatrix, w=None, eps=1e-12):
    """
    Build the normalized hypergraph Laplacian Δ = I - Dv^{-1/2} A W De^{-1} A^T Dv^{-1/2}
    from incidence matrix A (n_vertices x n_hyperedges).

    A: scipy.sparse array/matrix (CSR/CSC/COO ok)
    w: optional length-m array of hyperedge weights (defaults to 1)
    """
    A = A.tocsr().astype(float)
    n, m = A.shape

    # hyperedge sizes δ(e) = sum_v A[v,e]
    delta = np.asarray(A.sum(axis=0)).ravel()
    if np.any(delta == 0):
        raise ValueError("Found an empty hyperedge (size 0). Remove it or fix A.")

    # hyperedge weights W (default all ones)
    if w is None:
        w = np.ones(m, dtype=float)
    else:
        w = np.asarray(w, dtype=float).ravel()
        if w.shape != (m,):
            raise ValueError(f"w must have shape ({m},), got {w.shape}")

    # vertex degrees d(v) = sum_{e} w(e) * A[v,e]
    d = np.asarray(A @ w).ravel()
    if np.any(d <= 0):
        # If isolated vertices exist, normalized Laplacian needs special handling
        raise ValueError("Found vertex with non-positive degree; check incidence/weights.")

    inv_sqrt_d = 1.0 / np.sqrt(d + eps)
    inv_delta  = 1.0 / (delta + eps)

    Dv_inv_sqrt = sparse.diags(inv_sqrt_d, format="csr")
    W           = sparse.diags(w,          format="csr")
    De_inv      = sparse.diags(inv_delta,  format="csr")

    Theta = Dv_inv_sqrt @ A @ W @ De_inv @ A.T @ Dv_inv_sqrt
    Delta = sparse.eye(n, format="csr") - Theta
    return Delta
