"""Shared pytest fixtures for the hydra test suite."""
import pytest
import xgi
import numpy as np
from scipy import sparse

from hydra.core.analysis.utils import HypergraphLazyParser


@pytest.fixture
def simple_hypergraph():
    """A small, connected xgi.Hypergraph with known structure.

    Nodes: 0, 1, 2, 3
    Hyperedges:
        e0: {0, 1, 2}
        e1: {1, 2, 3}
        e2: {0, 3}
    """
    H = xgi.Hypergraph([[0, 1, 2], [1, 2, 3], [0, 3]])
    H["dataset_name"] = "test_dataset"
    H["name"] = "test_hypergraph"
    H["kind"] = "undirected"
    return H


@pytest.fixture
def lazy_parser(simple_hypergraph):
    """HypergraphLazyParser wrapping the simple_hypergraph fixture."""
    return HypergraphLazyParser(simple_hypergraph)


@pytest.fixture
def simple_incidence_matrix():
    """A small (3 nodes x 2 hyperedges) scipy sparse incidence matrix.

    Incidence:
        node 0: edges 0, 1
        node 1: edge 0
        node 2: edge 1
    """
    data = np.array([1, 1, 1, 1], dtype=float)
    row = np.array([0, 0, 1, 2])
    col = np.array([0, 1, 0, 1])
    return sparse.csr_matrix((data, (row, col)), shape=(3, 2))
