"""Tests for hydra.core.analysis.metrics.registry."""
import pytest

# Importing builtins triggers registration of all built-in metrics
import hydra.core.analysis.metrics.builtins  # noqa: F401

from hydra.core.analysis.metrics.registry import (
    get_metric,
    list_metrics,
    register_metric,
)
from hydra.core.analysis.metrics.protocols import Metric


class TestListMetrics:

    def test_returns_list(self):
        assert isinstance(list_metrics(), list)

    def test_returns_non_empty(self):
        assert len(list_metrics()) > 0

    def test_returns_sorted(self):
        metrics = list_metrics()
        assert metrics == sorted(metrics)

    def test_contains_expected_builtins(self):
        metrics = list_metrics()
        for expected in ("number_of_nodes", "number_of_hyperedges", "incidence_matrix_density"):
            assert expected in metrics


class TestGetMetric:

    def test_returns_class_for_known_metric(self):
        cls = get_metric("number_of_nodes")
        # Protocol subclass check isn't supported; verify the class has the expected API
        assert callable(getattr(cls, "compute", None))
        assert callable(getattr(cls, "compare", None))

    def test_returned_class_is_instantiable(self):
        cls = get_metric("number_of_nodes")
        instance = cls()
        assert hasattr(instance, "compute")
        assert hasattr(instance, "compare")

    def test_unknown_metric_raises_key_error(self):
        with pytest.raises(KeyError, match="unknown_xyz_metric"):
            get_metric("unknown_xyz_metric")

    def test_error_message_lists_known_metrics(self):
        with pytest.raises(KeyError) as exc_info:
            get_metric("does_not_exist")
        assert "Known" in str(exc_info.value)


class TestRegisterMetric:

    def test_double_registration_raises_key_error(self):
        @register_metric(name="_test_unique_metric_abc")
        class _TestMetric(Metric):
            def compute(self, hg): return 0
            def compare(self, a, b): return abs(a - b)

        with pytest.raises(KeyError, match="_test_unique_metric_abc"):
            @register_metric(name="_test_unique_metric_abc")
            class _DuplicateMetric(Metric):
                def compute(self, hg): return 0
                def compare(self, a, b): return abs(a - b)

    def test_registered_metric_appears_in_list(self):
        @register_metric(name="_test_appears_in_list")
        class _AnotherMetric(Metric):
            def compute(self, hg): return 1
            def compare(self, a, b): return 0

        assert "_test_appears_in_list" in list_metrics()

    def test_registered_metric_retrievable(self):
        @register_metric(name="_test_retrievable_metric")
        class _RetrievableMetric(Metric):
            def compute(self, hg): return 42
            def compare(self, a, b): return a + b

        cls = get_metric("_test_retrievable_metric")
        assert cls is _RetrievableMetric

    def test_register_uses_class_name_as_default_key(self):
        # Use a name starting with uppercase so the snake_case regex produces a clean result.
        # re.sub inserts '_' before every uppercase that's NOT at string position 0.
        # "AutoNamedTestMetricAbc" → "auto_named_test_metric_abc"
        @register_metric()
        class AutoNamedTestMetricAbc(Metric):
            def compute(self, hg): return None
            def compare(self, a, b): return None

        assert "auto_named_test_metric_abc" in list_metrics()
