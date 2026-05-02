from __future__ import annotations

import numpy as np
import networkx as nx
import xgi
from scipy.cluster import hierarchy
from scipy.spatial import distance
from itertools import combinations

# From the https://github.com/cosimoagostinelli/Hor_dissimilarity_measures repository.
# Code to reproduce the results presented in the paper "Higher-order dissimilarity measures for hypergraph comparison", C. Agostinelli, M. Mancastroppa, A. Barrat (2025), https://arxiv.org/abs/2503.16959.

#  ------------------   Hyper-NetSimile  -------------------  #

from multiprocessing import Pool
from tqdm import tqdm

from hydra.core.analysis.utils import HypergraphLazyParser

def feature_vec (H):
    """""""""
    Feature vector related to the given hypergraph. This vector is a list of
    21 values corrsponding to median, mean, and standard deviation of 7 
    distributions. Each of them is a distribution of local structural
    features over all the nodes 'i' in H. The selected features are: number of i's  
    neighbors; i's hyperdegree; average size of hyperedges involving i;
    std of sizes of hyperedges involving i; average number of neighbors' of
    i's neighbors;  average hyperdegree of i's neighbors; number of neighbors 
    of i's egonet (i.e., number of nodes at distance 2 from i).

    Parameters:
    -----------
    H (xgi.Hypergraph) : the input hypergraph.
    -----------
    
    Returns:
        hgraph_fvec (list) : feature vector.
            
    """""""""
    deg_dict = H.nodes.degree.asdict()

    # 1 - list of number of neighbors of each node
    pd1_nneig = []
    
    # 2 - list of hyperdegree of each node
    pd2_hdeg = list(deg_dict.values())
    
    # 3 - list of hyper clustering coefficient of each node
    #clst_dict = xgi.local_clustering_coefficient(H)
    #pd3_hclst = list(clst_dict.values())
    
    # 4 - list of average size of nodes' hyperedges
    pd4_avg_hsize = []
    
    # 5 - list of std of size of nodes' hyperedges
    pd5_std_hsize = []
    
    # 6 - list of average neighbors' number of nieghbors
    pd6_neig_nneig = []

    # 7 - list of average neighbors' hyperdegree
    pd7_neig_hdeg = H.nodes.average_neighbor_degree.aslist()

    # 8 - list of average neighbors' hyper clustering coefficient
    #pd8_neig_hclst = []

    # 9 - list of number of neighbors of a node's egonet
    pd9_ego_neig = []

    for i in tqdm(H.nodes, desc="Nodes", leave=False):

        neig_i = H.nodes.neighbors(i)
        pd1_nneig.append( len(neig_i) )

        # for isolated nodes
        if len(neig_i)==0:
            pd4_avg_hsize.append(0.)
            pd5_std_hsize.append(0.)
            pd6_neig_nneig.append(0.)
            #pd8_neig_hclst.append(0.)
            pd9_ego_neig.append(0.)
        else:
            edge_neig_i = xgi.edge_neighborhood(H, i, include_self=True)
            hsizes = [len(j) for j in edge_neig_i]
            pd4_avg_hsize.append( np.mean(hsizes) )
            pd5_std_hsize.append( np.std(hsizes) )
            
            neig_nneig_i = [len(H.nodes.neighbors(j)) for j in neig_i]
            pd6_neig_nneig.append( np.mean(neig_nneig_i) )
            
            #neig_clst = [clst_dict[j] for j in neig_i]
            #pd8_neig_hclst.append( np.mean( neig_clst ) )

            neig_neig_i = set()
            for j in neig_i:
                neig_j = H.nodes.neighbors(j)
                neig_neig_i = neig_neig_i.union(neig_j) 
            # do not count i and its neighbors
            pd9_ego_neig.append(len(neig_neig_i-neig_i)-1)

    features_distr_list = [pd1_nneig, pd2_hdeg, #pd3_hclst, 
                           pd4_avg_hsize, pd5_std_hsize,
                           pd6_neig_nneig, pd7_neig_hdeg, #pd8_neig_hclst, 
                           pd9_ego_neig]
    hgraph_fvec = []

    for f_distr in features_distr_list:

        hgraph_fvec += [np.median(f_distr),
                     np.mean(f_distr),
                     np.std(f_distr),
                     #skew(f_distr),
                     #kurtosis(f_distr, fisher=False)
                       ]
        
    return (hgraph_fvec)

#  ------------------   Hyper-Portrait Divergence  -------------------  #

#####################################################################

from collections import defaultdict

def H_to_G_mapping(H):
    """
    Map hypergraph H to a graph G where each node is a hyperedge of H,
    and two nodes are connected iff the corresponding hyperedges intersect.
    Node attribute 'size' is the size of the hyperedge.
    """
    # hyperedge sizes (xgi provides this)
    sizes = H.edges.size.asdict()

    # Build node -> incident hyperedges index in one pass over incidences
    node_to_edges = defaultdict(list)
    for eid in tqdm(H.edges, desc="Building node-edge index", leave=False):
        for n in H.edges.members(eid):
            node_to_edges[n].append(eid)

    # Collect intersecting hyperedge pairs via shared nodes
    # Use frozenset to deduplicate pairs regardless of order
    edge_pairs = set()
    for inc in tqdm(node_to_edges.values(), desc="Collecting edge pairs", leave=False):
        if len(inc) > 1:
            for e1, e2 in combinations(inc, 2):
                edge_pairs.add(frozenset((e1, e2)))

    # Build graph
    G = nx.Graph()

    # Add all hyperedges as nodes (keeps isolated ones too) + attach size attribute
    G.add_nodes_from((eid, {"size": sizes.get(eid)}) for eid in H.edges)

    # Add overlap edges
    G.add_edges_from(tuple(p) for p in edge_pairs)

    return G

import numpy as np
import networkx as nx
import multiprocessing as mp
import os
from tqdm import tqdm

_WORKER_G = None
_WORKER_SIZE_IDX = None
_WORKER_SMAX = None

def _init_worker(G, size_idx, s_max):
    global _WORKER_G, _WORKER_SIZE_IDX, _WORKER_SMAX
    _WORKER_G = G
    _WORKER_SIZE_IDX = size_idx
    _WORKER_SMAX = s_max

def _portrait_counter_from_source(src):
    # single-source shortest path lengths (unweighted BFS)
    dist = dict(nx.single_source_shortest_path_length(_WORKER_G, src))
    dmax = max(dist.values()) if dist else 0

    # Build exactly what the parent ultimately uses (counts by size-group and distance),
    # instead of returning the full dist dict (huge pickles + RAM).
    counter = np.zeros((_WORKER_SMAX - 1, dmax + 1), dtype=np.uint32)
    for v, d in dist.items():
        counter[_WORKER_SIZE_IDX[v], d] += 1

    m = _WORKER_SIZE_IDX[src]
    return m, counter, dmax

def hyperedge_portrait(H):
    G = H_to_G_mapping(H)
    N = G.number_of_nodes()

    sizes_dict = nx.get_node_attributes(G, "size")
    s_max = int(np.max(xgi.unique_edge_sizes(H)))

    # Precompute "size -> index" once (used everywhere)
    size_idx = {u: sizes_dict[u] - 2 for u in G.nodes()}

    # connected components: store node lists (no subgraph .copy())
    CC = [list(c) for c in tqdm(nx.connected_components(G), desc="Components", leave=False)]

    dia = 0
    rows = []  # store (m, counter) per source; much smaller than storing all dist dicts

    n_jobs = os.cpu_count() - 1 if os.cpu_count() > 1 else 1

    # Use fork on POSIX to avoid pickling/copying the whole graph to each worker.
    start_method = "fork" if os.name != "nt" else "spawn"
    ctx = mp.get_context(start_method)

    with ctx.Pool(
        processes=n_jobs,
        initializer=_init_worker,
        initargs=(G, size_idx, s_max),
    ) as pool:
        for nodes in tqdm(CC, desc="Computing distances", leave=False):
            if not nodes:
                continue

            chunksize = max(1, len(nodes) // (n_jobs * 8))
            it = pool.imap_unordered(_portrait_counter_from_source, nodes, chunksize=chunksize)

            for m, counter, dmax in tqdm(
                it,
                total=len(nodes),
                desc="Distances",
                leave=False,
                mininterval=1.0,
            ):
                rows.append((m, counter))
                if dmax > dia:
                    dia = dmax

    B = np.zeros((s_max - 1, s_max - 1, dia + 1, N), dtype=int)

    # Same aggregation logic as your original code, but using counters produced by workers.
    # IMPORTANT: we must also account for distances > dmax where the count is 0,
    # because your original loops add B[..., l, 0] for those l as well.
    for m, counter in rows:
        L = counter.shape[1]
        for n in range(s_max - 1):
            for l in range(L):
                k = int(counter[n, l])
                B[m, n, l, k] += 1
            for l in range(L, dia + 1):
                B[m, n, l, 0] += 1

    return B

#####################################################################

def pad_h_portraits (B1,B2):
    """""""""
    Make sure that two tensors are padded with zeros and/or trimmed of
    zeros in order to have the same dimensions.
    
    Parameters: 
    --------------
    B1, B2 : the two hyperedge portraits of the hypergraphs to compare.
    --------------
    
    Returns (B1, B2) : the two hyperedge portraits with same dimensions.
    
    """""""""
    
    # the B tensors have last dimension = E (number of hyperedges)
    # by default. Find last occupied "column" and trim both down:
    lastcol1 = max(np.nonzero(B1)[3])
    lastcol2 = max(np.nonzero(B2)[3])
    lastcol = max(lastcol1,lastcol2)
    B1 = B1[:,:,:,:lastcol+1]
    B2 = B2[:,:,:,:lastcol+1]
    
    for i in range(4):
        max_dim = max(B1.shape[i],B2.shape[i])
        
        dims = list(B1.shape)
        dims[i] = max_dim-dims[i]   
        to_stack = np.zeros(dims, dtype=int)
        B1 = np.append(B1, to_stack, axis=i)
        
        dims = list(B2.shape)
        dims[i] = max_dim-dims[i]    
        to_stack = np.zeros(dims, dtype=int)
        B2 = np.append(B2, to_stack, axis=i)
      
    return (B1, B2)



def hyper_portrait_divergence(B1, B2):
    
    """""""""
    Dissimilarity measure between the two hypergraphs H1, H2, based
    on the generalization of the portrait divergence. It is defined
    as the Jensen-Shannon divergence between the distributions P1, P2
    associated to the two hypergraphs. P is built upon the hyperedge
    portrait of the hypergraph (see edge_portrait(H) function):
    P(m,n,l,k) = B_{m,n,l,k} / normalization.
    
    Parameters :
    --------------
    B1, B2 : can be either the two hypergraphs to compare (xgi.Hypergraph)
             or their hyperedge portraits, obtained via the
             hyperedge_portrait(H) function.
    --------------
    
    Returns (float) : the hyperedge portrait divergence between H1 and H2.
        
    """""""""

    if isinstance(B1, xgi.Hypergraph):
        B1 = hyperedge_portrait(B1)
        B2 = hyperedge_portrait(B2)    
        
    B1, B2 = pad_h_portraits(B1,B2)
    P1 = np.ravel(B1)
    P2 = np.ravel(B2)
    JSD = distance.jensenshannon(P1,P2, base=2)
    
    return (JSD*JSD)






def DunnIndex(distances, n_clusters, method='single'):
    """
    Compute the Dunn index (DI) for the given number of clusters. DI is computed
    as num/denom, where num is the minimum inter-cluster distance (i.e. the minimum
    distance between two points belonging to different clusters), and denom is the
    maximum intra-cluster distance (i.e. the maximium distance between two points
    belonging to the same cluster).
    The clusters are computed according to the scipy.cluster.hierarchy.linkage()
    function, with the given method (default is 'single').
    
    Parameters :
    -------------
    distances (array-like) : a 1-d array-like object containing the values of the
                             upper triangular distance matrix, that is, the list
                             of distances between all possible pairs of elements
                             (see scipy.spatial.distance.pdist).
                
    n_clusters (int) : number of clusters to group the elements.
    
    method (str) : method to perform the hierarchical clustering (default is 'single').
                   See the documentation of scipy.cluster.hierarchy.linkage for
                   available options.
    ------------
    
    Returns (float) : the Dunn index.
    
    """
    # build distance matrix
    D = distance.squareform(distances)
    # assign elements to clusters
    Z = hierarchy.linkage(np.array(distances), method=method)
    clst_assign = hierarchy.cut_tree(Z, n_clusters=n_clusters)
    idxs = dict()
    denoms = []
    nums = []
    
    for clst in range(n_clusters):      
        idxs[clst] = [idx for idx in np.where(clst_assign==clst)[0]]
        # compute denominator
        if len(idxs[clst])>1:
            dists_in_c = [D[i,j] for i,j in combinations(idxs[clst], 2)]
            denoms.append( np.max(dists_in_c) )
       
    for c1,c2 in combinations(range(n_clusters),2):
        # compute numerator
        dists_out_c = [D[i,j] for i in idxs[c1] for j in idxs[c2]]
        nums.append( np.min(dists_out_c) )
        
    DI = np.min(nums) / np.max(denoms)

    return DI

def _iter_edge_node_sets(xgi_hg):
    """
    Yield the set of nodes incident to each hyperedge.

    - Hypergraph: use edge members directly
    - DiHyperg:contentReference[oaicite:0]{index=0}s
    """
    edges = xgi_hg.edges

    if hasattr(edges, "members"):
        for members in edges.members(dtype=dict).values():
            yield set(members)
        return

    if hasattr(edges, "dimembers"):
        for tail, head in edges.dimembers(dtype=dict).values():
            yield set(tail) | set(head)
        return

    raise TypeError(f"Unsupported hypergraph type: {type(xgi_hg)!r}")

def number_of_closed_triangles(hg: HypergraphLazyParser) -> int:
    """
    Count distinct closed triangles, i.e. distinct node triples {u,v,w}
    such that some hyperedge contains all three nodes.
    """
    closed_triangles = set()

    for nodes in _iter_edge_node_sets(hg.xgi_hypergraph):
        if len(nodes) < 3:
            continue

        for triple in combinations(nodes, 3):
            closed_triangles.add(frozenset(triple))

    return len(closed_triangles)

def empirical_integer_distribution(values: np.ndarray) -> list[float]:
    """
    Returns a PMF as a dense list where index k contains P(X = k).
    JSON-safe.
    """
    values = np.asarray(values, dtype=np.int64)

    if values.size == 0:
        return []

    counts = np.bincount(values).astype(np.float64)
    probs = counts / counts.sum()
    return probs.tolist()

def discrete_wasserstein_distance(a: list[float], b: list[float]) -> float:
    """
    Wasserstein-1 distance between two discrete 1D distributions
    represented as dense PMF lists.
    """
    a = np.asarray(a, dtype=np.float64)
    b = np.asarray(b, dtype=np.float64)

    n = max(len(a), len(b))
    if len(a) < n:
        a = np.pad(a, (0, n - len(a)))
    if len(b) < n:
        b = np.pad(b, (0, n - len(b)))

    cdf_a = np.cumsum(a)
    cdf_b = np.cumsum(b)

    return float(np.abs(cdf_a - cdf_b).sum())

import networkx as nx

import math
import multiprocessing as mp
from collections import defaultdict
from itertools import islice
from typing import Iterable, List, Tuple

import networkx as nx
from tqdm import tqdm


# Worker globals, initialized once per process
_FORWARD: dict | None = None
_NODES: list | None = None


def _init_triangle_workers(forward: dict, nodes: list) -> None:
    """
    Initializer for worker processes.

    Using globals avoids sending the graph data with every task. On Unix-like
    systems, using the 'fork' start method keeps memory usage low because the
    read-only data is shared copy-on-write across workers.
    """
    global _FORWARD, _NODES
    _FORWARD = forward
    _NODES = nodes


def _count_triangles_for_node_range(start_stop: tuple[int, int]) -> int:
    """
    Count triangles for nodes in _NODES[start:stop].
    """
    assert _FORWARD is not None
    assert _NODES is not None

    start, stop = start_stop
    triangles = 0

    for i in range(start, stop):
        u = _NODES[i]
        fu = _FORWARD[u]
        if not fu:
            continue

        for v in fu:
            fv = _FORWARD[v]

            # Intersect the smaller set into the larger one
            if len(fu) < len(fv):
                triangles += sum(1 for w in fu if w in fv)
            else:
                triangles += sum(1 for w in fv if w in fu)

    return triangles


def _make_ranges(n: int, chunk_size: int) -> Iterable[tuple[int, int]]:
    """
    Yield (start, stop) index ranges.
    """
    for start in range(0, n, chunk_size):
        yield start, min(start + chunk_size, n)


def count_closed_triangles(
    G: nx.Graph,
    processes: int | None = None,
    chunk_size: int = 256,
    show_progress: bool = True,
) -> int:
    """
    Count the number of closed triangles in an undirected NetworkX graph.

    Only the SECOND loop (the outer counting loop over nodes) is parallelized.

    Parameters
    ----------
    G : nx.Graph
        Undirected simple graph.
    processes : int | None
        Number of worker processes. Default is max(1, cpu_count() - 1).
    chunk_size : int
        Number of nodes handled per task sent to workers.
    show_progress : bool
        Whether to display tqdm progress bars.

    Returns
    -------
    int
        Number of unique triangles.
    """
    if G.is_directed():
        raise ValueError("count_closed_triangles expects an undirected graph")

    if processes is None:
        processes = max(1, mp.cpu_count() - 1)
    else:
        processes = max(1, processes)

    # First phase: build the oriented forward adjacency (kept serial)
    degree = dict(G.degree())
    order = {node: (degree[node], node) for node in G.nodes()}
    forward = {u: set() for u in G.nodes()}

    edge_iter = G.edges()
    if show_progress:
        try:
            total_edges = G.number_of_edges()
        except Exception:
            total_edges = None
        edge_iter = tqdm(edge_iter, total=total_edges, desc="Orienting edges")

    for u, v in edge_iter:
        if u == v:
            continue
        if order[u] < order[v]:
            forward[u].add(v)
        else:
            forward[v].add(u)

    nodes = list(G.nodes())

    # If only one process is requested, run serially but keep tqdm
    if processes == 1 or len(nodes) == 0:
        triangles = 0
        node_iter = nodes
        if show_progress:
            node_iter = tqdm(node_iter, total=len(nodes), desc="Counting triangles", mininterval=1.0)

        for u in node_iter:
            fu = forward[u]
            if not fu:
                continue

            for v in fu:
                fv = forward[v]
                if len(fu) < len(fv):
                    triangles += sum(1 for w in fu if w in fv)
                else:
                    triangles += sum(1 for w in fv if w in fu)

        return triangles

    # Parallelize ONLY the second loop by splitting the node list into ranges
    ranges = list(_make_ranges(len(nodes), chunk_size))

    # Prefer fork on Unix to reduce memory usage via copy-on-write sharing
    try:
        ctx = mp.get_context("fork")
    except ValueError:
        ctx = mp.get_context()

    triangles = 0
    with ctx.Pool(
        processes=processes,
        initializer=_init_triangle_workers,
        initargs=(forward, nodes),
    ) as pool:
        results = pool.imap_unordered(_count_triangles_for_node_range, ranges)

        if show_progress:
            results = tqdm(results, total=len(ranges), desc="Counting triangles", mininterval=1.0)

        for partial in results:
            triangles += partial

    return triangles

from collections import defaultdict
from itertools import combinations
from typing import Any, Dict, Hashable, Optional, Tuple, Union


def hypertrans(
    H,
    *,
    return_wedge_scores: bool = False,
) -> Union[float, Tuple[float, Dict[Tuple[Hashable, Hashable], float]]]:
    """
    Compute HyperTrans for an undirected xgi.Hypergraph.

    This implements:
        T(G) = (1 / |W|) * sum_{w in W} T(w)

    where W is the set of hyperwedges, i.e. intersecting pairs of
    hyperedges (ei, ej) such that neither is a subset of the other.

    For a hyperwedge w = {ei, ej}:
        L = ei \\ ej
        R = ej \\ ei
        P(w) = L x R

    The hyperwedge-level HyperTrans score is:
        T(w) = (1 / |P(w)|) * sum_{(u, v) in P(w)}
               max_{e in E} f(w, e) * 1[u in e and v in e]

    The default group interaction score f is Eq. (5) from the paper:
        f(w, e) =
            |L ∩ e| * |R ∩ e|
            -----------------------------------------------
            |L ∪ (e \\ R)| * |R ∪ (e \\ L)|

    Parameters
    ----------
    H : xgi.Hypergraph
        Input undirected hypergraph.
    return_wedge_scores : bool, default False
        If True, also return a dictionary mapping hyperedge-id pairs
        to their hyperwedge-level HyperTrans scores.

    Returns
    -------
    float
        Global HyperTrans score. Returns 0.0 if the hypergraph has
        no hyperwedges.
    dict, optional
        Only returned when return_wedge_scores=True. Keys are
        (edge_id_1, edge_id_2), values are hyperwedge scores.
    """

    # XGI edge IDs and edge member sets.
    edge_ids = list(H.edges)
    edges = {eid: set(H.edges.members(eid)) for eid in edge_ids}

    # Node -> incident edge IDs, used both to enumerate intersecting
    # hyperedge pairs and to find candidate hyperedges efficiently.
    node_to_edges = defaultdict(set)
    for eid, members in edges.items():
        for node in members:
            node_to_edges[node].add(eid)

    # Enumerate hyperedge pairs that intersect.
    intersecting_pairs = set()
    for incident_edges in node_to_edges.values():
        for e1, e2 in combinations(sorted(incident_edges, key=repr), 2):
            intersecting_pairs.add((e1, e2))

    wedge_scores: Dict[Tuple[Hashable, Hashable], float] = {}

    for e1, e2 in intersecting_pairs:
        E1 = edges[e1]
        E2 = edges[e2]

        # Hyperwedge condition: intersecting, and neither is subset of the other.
        # Intersection is guaranteed by construction, but keep this for clarity.
        if not (E1 & E2):
            continue
        if E1 <= E2 or E2 <= E1:
            continue

        L = E1 - E2
        R = E2 - E1

        # For a valid hyperwedge both wings are nonempty.
        if not L or not R:
            continue

        # Phi[(u, v)] stores the best score found for the cross-wing pair.
        # Use ordered tuples for deterministic dictionary keys.
        phi = {(u, v): 0.0 for u in L for v in R}

        # Candidate hyperedges that intersect both wings.
        left_candidates = set().union(*(node_to_edges[u] for u in L))
        right_candidates = set().union(*(node_to_edges[v] for v in R))
        candidates = left_candidates & right_candidates

        for c in candidates:
            Ec = edges[c]
            Lcap = L & Ec
            Rcap = R & Ec

            if not Lcap or not Rcap:
                continue

            # Eq. (5) denominator.
            denom_left = len(L | (Ec - R))
            denom_right = len(R | (Ec - L))
            denom = denom_left * denom_right

            if denom == 0:
                continue

            f = (len(Lcap) * len(Rcap)) / denom

            # Fast-HyperTrans update: only cross-wing pairs covered by c.
            for u in Lcap:
                for v in Rcap:
                    if f > phi[(u, v)]:
                        phi[(u, v)] = f

        wedge_scores[(e1, e2)] = sum(phi.values()) / (len(L) * len(R))

    if not wedge_scores:
        global_score = 0.0
    else:
        global_score = sum(wedge_scores.values()) / len(wedge_scores)

    if return_wedge_scores:
        return global_score, wedge_scores

    return global_score
