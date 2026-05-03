"""Tests for the lightweight built-in metrics in janus.core.analysis.metrics.impl.basic."""
import pytest
import xgi

# Trigger metric registration
import janus.core.analysis.metrics.builtins  # noqa: F401

from janus.core.analysis.metrics.registry import get_metric
from janus.core.analysis.utils import HypergraphLazyParser


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture(scope="module")
def hg_parser():
    """A small connected hypergraph wrapped in HypergraphLazyParser."""
    H = xgi.Hypergraph([[0, 1, 2], [1, 2, 3], [0, 3]])
    H["dataset_name"] = "test_dataset"
    H["name"] = "test_name"
    H["kind"] = "undirected"
    return HypergraphLazyParser(H)


# ---------------------------------------------------------------------------
# DatasetName
# ---------------------------------------------------------------------------

class TestDatasetName:

    def test_compute_returns_string(self, hg_parser):
        metric = get_metric("dataset_name")()
        result = metric.compute(hg_parser)
        assert isinstance(result, str)

    def test_compute_returns_correct_value(self, hg_parser):
        metric = get_metric("dataset_name")()
        assert metric.compute(hg_parser) == "test_dataset"

    def test_compare_equal_values_returns_value(self):
        metric = get_metric("dataset_name")()
        assert metric.compare("foo", "foo") == "foo"

    def test_compare_different_values_returns_list(self):
        metric = get_metric("dataset_name")()
        result = metric.compare("foo", "bar")
        assert isinstance(result, list)
        assert "foo" in result
        assert "bar" in result


# ---------------------------------------------------------------------------
# Name
# ---------------------------------------------------------------------------

class TestName:

    def test_compute_returns_correct_value(self, hg_parser):
        metric = get_metric("name")()
        assert metric.compute(hg_parser) == "test_name"

    def test_compare_equal(self):
        metric = get_metric("name")()
        assert metric.compare("x", "x") == "x"

    def test_compare_different(self):
        metric = get_metric("name")()
        result = metric.compare("x", "y")
        assert isinstance(result, list)


# ---------------------------------------------------------------------------
# Kind
# ---------------------------------------------------------------------------

class TestKind:

    def test_compute_returns_correct_value(self, hg_parser):
        metric = get_metric("kind")()
        assert metric.compute(hg_parser) == "undirected"


# ---------------------------------------------------------------------------
# NumberOfNodes
# ---------------------------------------------------------------------------

class TestNumberOfNodes:

    def test_compute_returns_int(self, hg_parser):
        metric = get_metric("number_of_nodes")()
        result = metric.compute(hg_parser)
        assert isinstance(result, int)

    def test_compute_correct_value(self, hg_parser):
        metric = get_metric("number_of_nodes")()
        assert metric.compute(hg_parser) == 4

    def test_compare_returns_absolute_difference(self):
        metric = get_metric("number_of_nodes")()
        assert metric.compare(4, 6) == 2
        assert metric.compare(6, 4) == 2
        assert metric.compare(5, 5) == 0


# ---------------------------------------------------------------------------
# NumberOfHyperedges
# ---------------------------------------------------------------------------

class TestNumberOfHyperedges:

    def test_compute_returns_int(self, hg_parser):
        metric = get_metric("number_of_hyperedges")()
        result = metric.compute(hg_parser)
        assert isinstance(result, int)

    def test_compute_correct_value(self, hg_parser):
        metric = get_metric("number_of_hyperedges")()
        assert metric.compute(hg_parser) == 3

    def test_compare_returns_absolute_difference(self):
        metric = get_metric("number_of_hyperedges")()
        assert metric.compare(10, 7) == 3


# ---------------------------------------------------------------------------
# IncidenceMatrixDensity
# ---------------------------------------------------------------------------

class TestIncidenceMatrixDensity:

    def test_compute_returns_float(self, hg_parser):
        metric = get_metric("incidence_matrix_density")()
        result = metric.compute(hg_parser)
        assert isinstance(result, float)

    def test_compute_value_in_range(self, hg_parser):
        metric = get_metric("incidence_matrix_density")()
        result = metric.compute(hg_parser)
        assert 0.0 < result <= 1.0

    def test_compare_returns_absolute_difference(self):
        metric = get_metric("incidence_matrix_density")()
        assert metric.compare(0.8, 0.5) == pytest.approx(0.3)


# ---------------------------------------------------------------------------
# LargestConnectedComponentDiameter
# ---------------------------------------------------------------------------

class TestLargestConnectedComponentDiameter:

    def test_compute_returns_int(self, hg_parser):
        metric = get_metric("largest_connected_component_diameter")()
        result = metric.compute(hg_parser)
        assert isinstance(result, int)

    def test_compute_returns_non_negative(self, hg_parser):
        metric = get_metric("largest_connected_component_diameter")()
        result = metric.compute(hg_parser)
        assert result >= 0

    def test_compare_returns_absolute_difference(self):
        metric = get_metric("largest_connected_component_diameter")()
        assert metric.compare(3, 5) == 2
