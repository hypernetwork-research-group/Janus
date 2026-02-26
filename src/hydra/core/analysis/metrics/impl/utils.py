import numpy as np
import networkx as nx
import xgi
from scipy.stats import kurtosis, skew
from scipy.cluster import hierarchy
from scipy.spatial import distance
from itertools import combinations

# From the https://github.com/cosimoagostinelli/Hor_dissimilarity_measures repository.
# Code to reproduce the results presented in the paper "Higher-order dissimilarity measures for hypergraph comparison", C. Agostinelli, M. Mancastroppa, A. Barrat (2025), https://arxiv.org/abs/2503.16959.

#  ------------------   Hyper-NetSimile  -------------------  #

from multiprocessing import Pool
from tqdm import tqdm

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

def hyperedge_portrait(H):
    
    """""""""""
    The hyperedge-portrait of the given hypergraph H.
    The hyperedge portrait is a tensor with four indices whose entry B_{m,n,l,k}
    gives the number of hyperedges of size m having k hyperedges of size n at
    distance l. Two hyperedges are at distance 1 if they share at least one node.
    
    Parameters:
    ---------------
    H (xgi.Hypergraph) : the input hypergraph.
    ---------------
    
    Returns: 
        B (numpy.array) : the hyperedge-portrait of H, as a 4-dimensional array. 
        
    """""""""""
    
    G = H_to_G_mapping(H)
    N = G.number_of_nodes()
    sizes_dict = nx.get_node_attributes(G, 'size')
    s_max = np.max( xgi.unique_edge_sizes(H) )
    # connected components
    CC = [G.subgraph(c).copy() for c in nx.connected_components(G)]

    # compute all shortest paths and get diameter to inizialize B
    dist_dict = dict()
    lengths = set()
    for Gc in tqdm(CC, desc="Computing shortest paths", leave=False):
        for i in tqdm(Gc.nodes, desc="Nodes in CC", leave=False):
            dist_dict[i] = nx.shortest_path_length(Gc, i)
            lengths |= set( dist_dict[i].values() )

    dia = max(lengths)
    B = np.zeros((s_max-1, s_max-1, dia+1, N), dtype=int) 
    
    for Gc in tqdm(CC, desc="Processing connected components", leave=False):
        for i in tqdm(Gc.nodes, desc="Nodes in CC", leave=False):
            m = sizes_dict[i]-2
            dd_i = dist_dict[i]
            counter = np.zeros((s_max-1, dia+1), dtype=int)

            for j in Gc.nodes:
                counter[sizes_dict[j]-2][dd_i[j]] += 1

            for n in range(s_max-1):
                for l in range(dia+1):
                    k = counter[n][l]
                    B[m][n][l][k] += 1 
                    
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
