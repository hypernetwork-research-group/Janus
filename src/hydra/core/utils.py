import random
from copy import deepcopy

import xgi

def metropolis_hastings_biased_random_walk(args):
    # TODO: Generalize random walk to k-th order
    G, num_paths, walk_length, p, q, a, seed, sources, neighborhoods, members = args
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
            for _ in range(walk_length - 1):
                neighbors = neighborhoods[current]
                if not neighbors:
                    break
                cumulative_alphas = []
                total_alpha = 0.0
                for neighbor in neighbors:
                    if neighbor == previous:
                        alpha = inv_p
                    # elif G.has_edge(previous, neighbor):
                    elif previous in neighborhoods[neighbor]:
                        alpha = 1
                    else:
                        alpha = inv_q
                    total_alpha += alpha
                    cumulative_alphas.append(total_alpha)
                next_node = rng.choices(neighbors, k=1, cum_weights=cumulative_alphas)[0]
                if a == 0:
                    accept = True
                else:
                    acceptance_prob = min(1, (G.degree[current] / G.degree[next_node]) ** a)
                    accept = rng.random() < acceptance_prob
                if accept:
                    previous = current
                    current = next_node
                if current not in walk_touched_nodes:
                    touched_nodes = members[current]
                    walk_touched_nodes.update(touched_nodes)
                walk.append(current) # Insert at the beginning
            walks.append({
                "walk": sorted(walk),
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
