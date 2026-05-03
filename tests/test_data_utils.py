"""Tests for hydra.core.data.utils."""
import random
import networkx as nx
import numpy as np
import pytest
import xgi
from scipy import sparse

from janus.core.data.utils import (
    hypergraph_laplacian_zhou,
    metropolis_hastings_biased_random_walk,
    patch_nodes,
)


# ---------------------------------------------------------------------------
# hypergraph_laplacian_zhou
# ---------------------------------------------------------------------------

class TestHypergraphLaplacianZhou:

    def _simple_incidence(self):
        """3 nodes x 2 hyperedges, all nodes have positive degree."""
        # node0 in e0,e1 | node1 in e0 | node2 in e1
        data = np.array([1, 1, 1, 1], dtype=float)
        row = np.array([0, 0, 1, 2])
        col = np.array([0, 1, 0, 1])
        return sparse.csr_matrix((data, (row, col)), shape=(3, 2))

    def test_output_shape(self):
        A = self._simple_incidence()
        L = hypergraph_laplacian_zhou(A)
        assert L.shape == (3, 3)

    def test_output_is_sparse(self):
        A = self._simple_incidence()
        L = hypergraph_laplacian_zhou(A)
        assert sparse.issparse(L)

    def test_diagonal_is_one_for_uniform_walk(self):
        # For a simple incidence, diagonal of Theta should not exceed 1,
        # so diagonal of Delta = I - Theta should be >= 0.
        A = self._simple_incidence()
        L = hypergraph_laplacian_zhou(A)
        diag = L.diagonal()
        assert np.all(diag >= -1e-9)

    def test_custom_weights_correct_shape(self):
        A = self._simple_incidence()
        w = np.array([2.0, 0.5])
        L = hypergraph_laplacian_zhou(A, w=w)
        assert L.shape == (3, 3)

    def test_empty_hyperedge_raises(self):
        # Column with all zeros → empty hyperedge
        A = sparse.csr_matrix(np.array([[1, 0], [1, 0], [0, 0]], dtype=float))
        with pytest.raises(ValueError, match="empty hyperedge"):
            hypergraph_laplacian_zhou(A)

    def test_isolated_vertex_raises(self):
        # Row with all zeros → node has zero degree
        A = sparse.csr_matrix(np.array([[1, 1], [1, 0], [0, 0]], dtype=float))
        with pytest.raises(ValueError, match="non-positive degree"):
            hypergraph_laplacian_zhou(A)

    def test_wrong_weight_shape_raises(self):
        A = self._simple_incidence()
        w_bad = np.array([1.0, 1.0, 1.0])  # shape (3,) instead of (2,)
        with pytest.raises(ValueError, match="shape"):
            hypergraph_laplacian_zhou(A, w=w_bad)

    def test_accepts_csc_input(self):
        A = self._simple_incidence().tocsc()
        L = hypergraph_laplacian_zhou(A)
        assert L.shape == (3, 3)

    def test_single_hyperedge_spanning_all_nodes(self):
        # n=3, m=1: all nodes in one hyperedge
        A = sparse.csr_matrix(np.ones((3, 1), dtype=float))
        L = hypergraph_laplacian_zhou(A)
        assert L.shape == (3, 3)


# ---------------------------------------------------------------------------
# patch_nodes
# ---------------------------------------------------------------------------

class TestPatchNodes:

    def _make_hypergraph(self):
        return xgi.Hypergraph([[0, 1, 2], [1, 2, 3], [0, 3]])

    def test_exact_size_no_padding(self):
        H = self._make_hypergraph()
        touched = [0, 1]
        nodes, mask = patch_nodes(H, touched, target_num_nodes=2)
        assert len(nodes) == 2
        assert all(mask)

    def test_padding_increases_length(self):
        H = self._make_hypergraph()
        touched = [0, 1]
        target = 4
        nodes, mask = patch_nodes(H, touched, target_num_nodes=target)
        assert len(nodes) == target
        assert len(mask) == target

    def test_mask_has_false_for_padded_positions(self):
        H = self._make_hypergraph()
        touched = [0]
        target = 3
        nodes, mask = patch_nodes(H, touched, target_num_nodes=target)
        # Exactly 2 positions should be padded (False)
        assert mask.count(False) == 2

    def test_returned_nodes_are_sorted(self):
        H = self._make_hypergraph()
        touched = [3, 0]
        nodes, mask = patch_nodes(H, touched, target_num_nodes=2)
        assert nodes == sorted(nodes)

    def test_touched_nodes_appear_in_result(self):
        H = self._make_hypergraph()
        touched = [0, 2]
        nodes, mask = patch_nodes(H, touched, target_num_nodes=4)
        for n in touched:
            assert n in nodes

    def test_original_touched_nodes_have_true_mask(self):
        H = self._make_hypergraph()
        touched = [1]
        nodes, mask = patch_nodes(H, touched, target_num_nodes=3)
        idx = nodes.index(1)
        assert mask[idx] is True


# ---------------------------------------------------------------------------
# metropolis_hastings_biased_random_walk
# ---------------------------------------------------------------------------

class TestMetropolisHastingsBiasedRandomWalk:

    def _make_args(self, walk_length=5, num_paths=1, p=1.0, q=1.0, a=0.0, seed=42):
        """Build a small line graph (hyperedges as nodes) for walking."""
        H = xgi.Hypergraph([[0, 1], [1, 2], [2, 3]])
        # Build line graph: nodes are hyperedge IDs, edges connect intersecting hyperedges
        G = nx.Graph()
        edge_ids = list(H.edges)
        G.add_nodes_from(edge_ids)
        members = {eid: list(H.edges.members(eid)) for eid in edge_ids}
        # Connect edges that share a node
        for i, e1 in enumerate(edge_ids):
            for e2 in edge_ids[i + 1:]:
                if set(members[e1]) & set(members[e2]):
                    G.add_edge(e1, e2)
        neighborhoods = {n: list(G.neighbors(n)) for n in G.nodes}
        sources = [edge_ids[0]]
        return (G, num_paths, walk_length, p, q, a, seed, sources, neighborhoods, members)

    def test_returns_list(self):
        args = self._make_args()
        result = metropolis_hastings_biased_random_walk(args)
        assert isinstance(result, list)

    def test_result_count_equals_num_paths(self):
        args = self._make_args(num_paths=3)
        result = metropolis_hastings_biased_random_walk(args)
        assert len(result) == 3

    def test_each_walk_has_required_keys(self):
        args = self._make_args()
        result = metropolis_hastings_biased_random_walk(args)
        for walk in result:
            assert "touched_hyperedges" in walk
            assert "touched_nodes" in walk

    def test_walk_length_is_respected(self):
        wl = 6
        args = self._make_args(walk_length=wl)
        result = metropolis_hastings_biased_random_walk(args)
        for walk in result:
            assert len(walk["touched_hyperedges"]) == wl

    def test_same_seed_produces_same_result(self):
        args1 = self._make_args(seed=7)
        args2 = self._make_args(seed=7)
        r1 = metropolis_hastings_biased_random_walk(args1)
        r2 = metropolis_hastings_biased_random_walk(args2)
        assert r1 == r2

    def test_different_seeds_may_differ(self):
        args1 = self._make_args(seed=1, walk_length=10, num_paths=5)
        args2 = self._make_args(seed=99, walk_length=10, num_paths=5)
        r1 = metropolis_hastings_biased_random_walk(args1)
        r2 = metropolis_hastings_biased_random_walk(args2)
        # Very unlikely to be equal with different seeds and longer walks
        assert r1 != r2

    def test_touched_nodes_are_sorted(self):
        args = self._make_args()
        result = metropolis_hastings_biased_random_walk(args)
        for walk in result:
            assert walk["touched_nodes"] == sorted(walk["touched_nodes"])
