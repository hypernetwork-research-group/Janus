"""Tests for hydra.core.analysis.utils."""
import json
import pytest
import networkx as nx
import xgi
from pathlib import Path

from janus.core.analysis.utils import file_discovery, HypergraphLazyParser


# ---------------------------------------------------------------------------
# file_discovery
# ---------------------------------------------------------------------------

class TestFileDiscovery:

    def test_finds_matching_file_in_flat_dir(self, tmp_path):
        (tmp_path / "a.hif.json").write_text("{}")
        results = list(file_discovery(tmp_path, [".hif.json"]))
        assert len(results) == 1
        assert results[0].name == "a.hif.json"

    def test_ignores_non_matching_extension(self, tmp_path):
        (tmp_path / "a.txt").write_text("hello")
        results = list(file_discovery(tmp_path, [".hif.json"]))
        assert results == []

    def test_recursive_discovery(self, tmp_path):
        sub = tmp_path / "sub"
        sub.mkdir()
        (sub / "b.hif.json").write_text("{}")
        (tmp_path / "a.hif.json").write_text("{}")
        results = list(file_discovery(tmp_path, [".hif.json"]))
        assert len(results) == 2

    def test_multiple_target_suffixes(self, tmp_path):
        (tmp_path / "a.hif.json").write_text("{}")
        (tmp_path / "b.hif.jsonl").write_text("{}")
        (tmp_path / "c.txt").write_text("ignored")
        results = list(file_discovery(tmp_path, [".hif.json", ".hif.jsonl"]))
        assert len(results) == 2

    def test_no_files_yields_nothing(self, tmp_path):
        results = list(file_discovery(tmp_path, [".hif.json"]))
        assert results == []

    def test_single_file_path(self, tmp_path):
        f = tmp_path / "x.hif.json"
        f.write_text("{}")
        results = list(file_discovery(f, [".hif.json"]))
        assert len(results) == 1

    def test_single_file_path_wrong_suffix(self, tmp_path):
        f = tmp_path / "x.txt"
        f.write_text("{}")
        results = list(file_discovery(f, [".hif.json"]))
        assert results == []


# ---------------------------------------------------------------------------
# HypergraphLazyParser
# ---------------------------------------------------------------------------

class TestHypergraphLazyParser:

    def _make_parser(self):
        H = xgi.Hypergraph([[0, 1, 2], [1, 2, 3], [0, 3]])
        return HypergraphLazyParser(H)

    def test_to_graph_returns_networkx_graph(self):
        parser = self._make_parser()
        G = parser.to_graph()
        assert isinstance(G, nx.Graph)

    def test_to_bipartite_graph_returns_networkx_graph(self):
        parser = self._make_parser()
        B = parser.to_bipartite_graph()
        assert isinstance(B, nx.Graph)

    def test_to_line_graph_returns_networkx_graph(self):
        parser = self._make_parser()
        L = parser.to_line_graph()
        assert isinstance(L, nx.Graph)

    def test_to_graph_is_cached(self):
        parser = self._make_parser()
        G1 = parser.to_graph()
        G2 = parser.to_graph()
        assert G1 is G2

    def test_to_bipartite_graph_is_cached(self):
        parser = self._make_parser()
        B1 = parser.to_bipartite_graph()
        B2 = parser.to_bipartite_graph()
        assert B1 is B2

    def test_to_line_graph_is_cached(self):
        parser = self._make_parser()
        L1 = parser.to_line_graph()
        L2 = parser.to_line_graph()
        assert L1 is L2

    def test_getitem_delegates_to_xgi_hypergraph(self):
        H = xgi.Hypergraph([[0, 1], [1, 2]])
        H["custom_key"] = "custom_value"
        parser = HypergraphLazyParser(H)
        assert parser["custom_key"] == "custom_value"

    def test_str_contains_class_name(self):
        parser = self._make_parser()
        assert "HypergraphLazyParser" in str(parser)

    def test_repr_contains_class_name(self):
        parser = self._make_parser()
        assert "HypergraphLazyParser" in repr(parser)

    def test_to_graph_has_correct_number_of_nodes(self):
        """Clique expansion of simple_hypergraph has 4 nodes."""
        parser = self._make_parser()
        G = parser.to_graph()
        assert G.number_of_nodes() == 4
