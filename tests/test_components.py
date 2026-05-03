"""Tests for hydra.core.models.components."""
import pytest
import torch
import torch.nn as nn

from janus.core.models.components import (
    BatchedHypergraphConvAttn,
    DiT,
    DiTBlock,
    HGAT,
    HypergraphDecoder,
    MLP,
    TimeConditioning,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_incidence(B, N, M, seed=0):
    """Random binary incidence matrix ensuring all nodes and hyperedges are active."""
    torch.manual_seed(seed)
    H = (torch.rand(B, N, M) > 0.4).float()
    # Ensure every node is in at least one hyperedge and every hyperedge has >= 1 node
    H[:, :, 0] = 1.0  # All nodes in e0
    H[:, 0, :] = 1.0  # Node 0 in all hyperedges
    return H


# ---------------------------------------------------------------------------
# BatchedHypergraphConvAttn — conv mode
# ---------------------------------------------------------------------------

class TestBatchedHypergraphConvAttnConv:

    def _layer(self, in_ch=8, out_ch=16):
        return BatchedHypergraphConvAttn(in_ch, out_ch, mode="conv")

    def test_output_shape(self):
        B, N, M, in_ch, out_ch = 2, 5, 3, 8, 16
        layer = self._layer(in_ch, out_ch)
        X = torch.randn(B, N, in_ch)
        H = _make_incidence(B, N, M)
        out = layer(X, H)
        assert out.shape == (B, N, out_ch)

    def test_symmetric_norm_false(self):
        B, N, M, in_ch, out_ch = 2, 5, 3, 8, 16
        layer = BatchedHypergraphConvAttn(in_ch, out_ch, mode="conv", symmetric_norm=False)
        X = torch.randn(B, N, in_ch)
        H = _make_incidence(B, N, M)
        out = layer(X, H)
        assert out.shape == (B, N, out_ch)

    def test_with_hyperedge_weight_1d(self):
        B, N, M, in_ch, out_ch = 2, 5, 3, 8, 16
        layer = self._layer(in_ch, out_ch)
        X = torch.randn(B, N, in_ch)
        H = _make_incidence(B, N, M)
        w = torch.ones(M)
        out = layer(X, H, hyperedge_weight=w)
        assert out.shape == (B, N, out_ch)

    def test_with_hyperedge_weight_2d(self):
        B, N, M, in_ch, out_ch = 2, 5, 3, 8, 16
        layer = self._layer(in_ch, out_ch)
        X = torch.randn(B, N, in_ch)
        H = _make_incidence(B, N, M)
        w = torch.ones(B, M)
        out = layer(X, H, hyperedge_weight=w)
        assert out.shape == (B, N, out_ch)

    def test_invalid_hyperedge_weight_shape_raises(self):
        B, N, M, in_ch, out_ch = 2, 5, 3, 8, 16
        layer = self._layer(in_ch, out_ch)
        X = torch.randn(B, N, in_ch)
        H = _make_incidence(B, N, M)
        w_bad = torch.ones(M + 1)
        with pytest.raises(ValueError):
            layer(X, H, hyperedge_weight=w_bad)

    def test_non_3d_input_raises(self):
        layer = self._layer()
        X = torch.randn(5, 8)  # 2D, not 3D
        H = torch.ones(2, 5, 3)
        with pytest.raises(ValueError):
            layer(X, H)

    def test_shape_mismatch_raises(self):
        layer = self._layer()
        X = torch.randn(2, 5, 8)
        H = torch.ones(3, 5, 3)  # B mismatch
        with pytest.raises(ValueError):
            layer(X, H)

    def test_invalid_mode_raises(self):
        with pytest.raises(ValueError, match="mode"):
            BatchedHypergraphConvAttn(8, 16, mode="invalid")

    def test_out_channels_not_divisible_by_heads_raises(self):
        with pytest.raises(ValueError, match="divisible"):
            BatchedHypergraphConvAttn(8, 15, mode="attn", heads=4)

    def test_output_has_no_nan(self):
        B, N, M, in_ch, out_ch = 2, 5, 3, 8, 16
        layer = self._layer(in_ch, out_ch)
        X = torch.randn(B, N, in_ch)
        H = _make_incidence(B, N, M)
        out = layer(X, H)
        assert not torch.isnan(out).any()


# ---------------------------------------------------------------------------
# BatchedHypergraphConvAttn — attn mode
# ---------------------------------------------------------------------------

class TestBatchedHypergraphConvAttnAttn:

    def _layer(self, in_ch=8, out_ch=8, heads=4):
        return BatchedHypergraphConvAttn(in_ch, out_ch, mode="attn", heads=heads)

    def test_output_shape(self):
        B, N, M, in_ch, out_ch = 2, 5, 3, 8, 8
        layer = self._layer(in_ch, out_ch)
        X = torch.randn(B, N, in_ch)
        H = _make_incidence(B, N, M)
        out = layer(X, H)
        assert out.shape == (B, N, out_ch)

    def test_return_hyperedge_true(self):
        B, N, M, in_ch, out_ch = 2, 5, 3, 8, 8
        layer = self._layer(in_ch, out_ch)
        X = torch.randn(B, N, in_ch)
        H = _make_incidence(B, N, M)
        out, Y1 = layer(X, H, return_hyperedge=True)
        assert out.shape == (B, N, out_ch)
        assert Y1.shape[0] == B
        assert Y1.shape[1] == M

    def test_with_explicit_Y(self):
        B, N, M, in_ch, out_ch = 2, 5, 3, 8, 8
        layer = self._layer(in_ch, out_ch)
        X = torch.randn(B, N, in_ch)
        H = _make_incidence(B, N, M)
        Y = torch.randn(B, M, in_ch)
        out = layer(X, H, Y=Y)
        assert out.shape == (B, N, out_ch)

    def test_output_has_no_nan(self):
        B, N, M, in_ch, out_ch = 2, 5, 3, 8, 8
        layer = self._layer(in_ch, out_ch)
        X = torch.randn(B, N, in_ch)
        H = _make_incidence(B, N, M)
        out = layer(X, H)
        assert not torch.isnan(out).any()


# ---------------------------------------------------------------------------
# HGAT
# ---------------------------------------------------------------------------

class TestHGAT:

    def test_output_shape_single_layer(self):
        B, N, M = 2, 5, 3
        in_ch, hidden_ch, out_ch = 8, 16, 4
        model = HGAT(in_ch, hidden_ch, out_ch, num_layers=1, heads=4)
        X = torch.randn(B, N, in_ch)
        H = _make_incidence(B, N, M)
        out = model(X, H)
        assert out.shape == (B, N, out_ch)

    def test_output_shape_two_layers(self):
        B, N, M = 2, 5, 3
        in_ch, hidden_ch, out_ch = 8, 16, 4
        model = HGAT(in_ch, hidden_ch, out_ch, num_layers=2, heads=4)
        X = torch.randn(B, N, in_ch)
        H = _make_incidence(B, N, M)
        out = model(X, H)
        assert out.shape == (B, N, out_ch)

    def test_output_has_no_nan(self):
        B, N, M = 2, 5, 3
        model = HGAT(8, 16, 4, num_layers=1, heads=4)
        X = torch.randn(B, N, 8)
        H = _make_incidence(B, N, M)
        out = model(X, H)
        assert not torch.isnan(out).any()


# ---------------------------------------------------------------------------
# HypergraphDecoder
# ---------------------------------------------------------------------------

class TestHypergraphDecoder:

    def test_output_shape(self):
        B, N, M, F = 2, 5, 3, 8
        decoder = HypergraphDecoder(in_channels=F, num_classes=2)
        x = torch.randn(B, N, F)
        y = torch.randn(B, M, F)
        out = decoder(x, y)
        assert out.shape == (B, N, M, 2)

    def test_custom_num_classes(self):
        B, N, M, F = 2, 5, 3, 8
        decoder = HypergraphDecoder(in_channels=F, num_classes=3)
        x = torch.randn(B, N, F)
        y = torch.randn(B, M, F)
        out = decoder(x, y)
        assert out.shape == (B, N, M, 3)

    def test_output_has_no_nan(self):
        B, N, M, F = 2, 5, 3, 8
        decoder = HypergraphDecoder(in_channels=F)
        x = torch.randn(B, N, F)
        y = torch.randn(B, M, F)
        out = decoder(x, y)
        assert not torch.isnan(out).any()


# ---------------------------------------------------------------------------
# MLP
# ---------------------------------------------------------------------------

class TestMLP:

    def test_output_shape_matches_input(self):
        # MLP: in_channels -> hidden*2 -> hidden -> in_channels*2 -> in_channels
        in_ch, hidden_ch = 8, 16
        mlp = MLP(in_ch, hidden_ch, dropout=0.0)
        x = torch.randn(2, 5, in_ch)
        out = mlp(x)
        assert out.shape == (2, 5, in_ch)

    def test_output_has_no_nan(self):
        mlp = MLP(8, 16, dropout=0.0)
        x = torch.randn(2, 5, 8)
        out = mlp(x)
        assert not torch.isnan(out).any()

    def test_1d_input(self):
        mlp = MLP(8, 16, dropout=0.0)
        x = torch.randn(8)
        out = mlp(x)
        assert out.shape == (8,)


# ---------------------------------------------------------------------------
# TimeConditioning
# ---------------------------------------------------------------------------

class TestTimeConditioning:

    def test_output_shape(self):
        B, emb_dim = 4, 16
        tc = TimeConditioning(embedding_dim=emb_dim, hidden_channels=emb_dim)
        t = torch.randint(0, 1000, (B,))
        out = tc(t)
        assert out.shape == (B, emb_dim)

    def test_output_has_no_nan(self):
        tc = TimeConditioning(embedding_dim=16, hidden_channels=16)
        t = torch.randint(0, 1000, (4,))
        out = tc(t)
        assert not torch.isnan(out).any()

    def test_sinusoidal_embedding_shape(self):
        tc = TimeConditioning(embedding_dim=16, hidden_channels=16)
        t = torch.randint(0, 1000, (4,))
        emb = tc.sinusoidal_embedding(t, dim=16)
        assert emb.shape == (4, 16)

    def test_sinusoidal_embedding_no_nan(self):
        tc = TimeConditioning(embedding_dim=16, hidden_channels=16)
        t = torch.tensor([0, 500, 999])
        emb = tc.sinusoidal_embedding(t, dim=16)
        assert not torch.isnan(emb).any()

    def test_odd_dim_sinusoidal_embedding(self):
        tc = TimeConditioning(embedding_dim=15, hidden_channels=15)
        t = torch.tensor([1, 2, 3])
        emb = tc.sinusoidal_embedding(t, dim=15)
        assert emb.shape == (3, 15)


# ---------------------------------------------------------------------------
# DiTBlock — no cross-attention
# ---------------------------------------------------------------------------
# NOTE: In the actual training loop (DiffusionTransformer.training_step),
# t is passed as t.unsqueeze(-1) → shape [B, 1]. This makes TimeConditioning
# output [B, 1, channels], which broadcasts correctly with x of [B, N, channels].
# Therefore, the conditioning c fed to DiTBlock must have shape [B, 1, channels]
# (or more generally [B, cond_seq, channels] where cond_seq broadcasts with N).

class TestDiTBlockNoCrossAttn:

    def _block(self, channels=16, heads=4):
        return DiTBlock(num_channels=channels, num_heads=heads, cross_attention_layer=False)

    def test_output_shape(self):
        # c must be [B, 1, ch] to broadcast correctly with x of [B, N, ch]
        B, seq, ch = 2, 5, 16
        block = self._block(ch)
        x = torch.randn(B, seq, ch)
        c = torch.randn(B, 1, ch)  # [B, 1, ch] broadcasts to [B, seq, ch]
        out = block(x, c)
        assert out.shape == (B, seq, ch)

    def test_output_has_no_nan(self):
        B, seq, ch = 2, 5, 16
        block = self._block(ch)
        x = torch.randn(B, seq, ch)
        c = torch.randn(B, 1, ch)
        out = block(x, c)
        assert not torch.isnan(out).any()

    def test_providing_y_raises_when_no_cross_attn(self):
        block = self._block()
        x = torch.randn(2, 5, 16)
        c = torch.randn(2, 1, 16)
        y = torch.randn(2, 3, 16)
        with pytest.raises(ValueError, match="cross_attention_layer is False"):
            block(x, c, y)


# ---------------------------------------------------------------------------
# DiTBlock — with cross-attention
# ---------------------------------------------------------------------------

class TestDiTBlockWithCrossAttn:

    def _block(self, channels=16, heads=4):
        return DiTBlock(num_channels=channels, num_heads=heads, cross_attention_layer=True)

    def test_output_shape(self):
        B, seq, cond_seq, ch = 2, 5, 3, 16
        block = self._block(ch)
        x = torch.randn(B, seq, ch)
        c = torch.randn(B, 1, ch)  # [B, 1, ch] broadcasts to [B, seq, ch]
        y = torch.randn(B, cond_seq, ch)
        out = block(x, c, y)
        assert out.shape == (B, seq, ch)

    def test_missing_y_raises(self):
        block = self._block()
        x = torch.randn(2, 5, 16)
        c = torch.randn(2, 1, 16)
        with pytest.raises(ValueError, match="cross_attention_layer is True"):
            block(x, c)

    def test_output_has_no_nan(self):
        B, seq, cond_seq, ch = 2, 5, 3, 16
        block = self._block(ch)
        x = torch.randn(B, seq, ch)
        c = torch.randn(B, 1, ch)
        y = torch.randn(B, cond_seq, ch)
        out = block(x, c, y)
        assert not torch.isnan(out).any()


# ---------------------------------------------------------------------------
# DiT — no cross-attention
# ---------------------------------------------------------------------------
# In DiffusionTransformer, t is passed as t.unsqueeze(-1) → [B, 1], which
# makes TimeConditioning produce [B, 1, channels] for correct broadcasting.

class TestDiTNoCrossAttn:

    def _model(self, in_ch=8, hidden_ch=16, blocks=2, heads=4):
        return DiT(in_ch, hidden_ch, num_blocks=blocks, num_heads=heads, cross_attention=False)

    def test_output_shape(self):
        B, seq, in_ch = 2, 5, 8
        model = self._model(in_ch)
        x = torch.randn(B, seq, in_ch)
        t = torch.randint(0, 1000, (B, 1))  # [B, 1] as used in actual training
        out = model(x, t)
        assert out.shape == (B, seq, in_ch)

    def test_returns_tensor_not_tuple(self):
        model = self._model()
        x = torch.randn(2, 5, 8)
        t = torch.randint(0, 1000, (2, 1))
        out = model(x, t)
        assert isinstance(out, torch.Tensor)

    def test_output_has_no_nan(self):
        model = self._model()
        x = torch.randn(2, 5, 8)
        t = torch.randint(0, 1000, (2, 1))
        out = model(x, t)
        assert not torch.isnan(out).any()


# ---------------------------------------------------------------------------
# DiT — with cross-attention
# ---------------------------------------------------------------------------

class TestDiTWithCrossAttn:

    def _model(self, in_ch=8, hidden_ch=16, blocks=2, heads=4):
        return DiT(in_ch, hidden_ch, num_blocks=blocks, num_heads=heads, cross_attention=True)

    def test_output_is_tuple(self):
        B, seq, cond_seq, in_ch = 2, 5, 3, 8
        model = self._model(in_ch)
        x = torch.randn(B, seq, in_ch)
        t = torch.randint(0, 1000, (B, 1))  # [B, 1] as used in actual training
        y = torch.randn(B, cond_seq, in_ch)
        result = model(x, t, y)
        assert isinstance(result, tuple)
        assert len(result) == 2

    def test_output_shapes(self):
        B, seq, cond_seq, in_ch = 2, 5, 3, 8
        model = self._model(in_ch)
        x = torch.randn(B, seq, in_ch)
        t = torch.randint(0, 1000, (B, 1))
        y = torch.randn(B, cond_seq, in_ch)
        x_out, y_out = model(x, t, y)
        assert x_out.shape == (B, seq, in_ch)

    def test_output_has_no_nan(self):
        B, seq, cond_seq, in_ch = 2, 5, 3, 8
        model = self._model(in_ch)
        x = torch.randn(B, seq, in_ch)
        t = torch.randint(0, 1000, (B, 1))
        y = torch.randn(B, cond_seq, in_ch)
        x_out, y_out = model(x, t, y)
        assert not torch.isnan(x_out).any()
