"""Tests for hydra.core.models.utils."""
import pytest
import torch

from hydra.core.models.utils import batch_index_contrastive_loss


class TestBatchIndexContrastiveLoss:

    def _make_input(self, B=2, N=3, F=8, seed=0):
        torch.manual_seed(seed)
        x = torch.randn(B, N, F)
        mask = torch.ones(B, N, dtype=torch.bool)
        return x, mask

    def test_raises_when_batch_size_less_than_two(self):
        x = torch.randn(1, 3, 8)
        mask = torch.ones(1, 3, dtype=torch.bool)
        with pytest.raises(ValueError, match="B>=2"):
            batch_index_contrastive_loss(x, mask)

    def test_returns_scalar_tensor(self):
        x, mask = self._make_input()
        loss = batch_index_contrastive_loss(x, mask)
        assert loss.ndim == 0

    def test_loss_is_finite(self):
        x, mask = self._make_input()
        loss = batch_index_contrastive_loss(x, mask)
        assert torch.isfinite(loss)

    def test_all_zero_mask_returns_zero(self):
        x = torch.randn(2, 3, 8)
        mask = torch.zeros(2, 3, dtype=torch.bool)
        loss = batch_index_contrastive_loss(x, mask)
        assert loss.item() == pytest.approx(0.0)

    def test_all_same_embeddings_is_finite(self):
        # All embeddings identical → logits all equal → loss should still be finite
        x = torch.ones(2, 4, 8)
        mask = torch.ones(2, 4, dtype=torch.bool)
        loss = batch_index_contrastive_loss(x, mask)
        assert torch.isfinite(loss)

    def test_larger_batch_shape(self):
        x, mask = self._make_input(B=4, N=5, F=16)
        loss = batch_index_contrastive_loss(x, mask)
        assert loss.ndim == 0
        assert torch.isfinite(loss)

    def test_integer_mask_supported(self):
        x, _ = self._make_input()
        mask_int = torch.ones(2, 3, dtype=torch.int32)
        loss = batch_index_contrastive_loss(x, mask_int)
        assert torch.isfinite(loss)

    def test_partial_mask(self):
        x, _ = self._make_input(B=3, N=4, F=8)
        mask = torch.ones(3, 4, dtype=torch.bool)
        mask[0, 0] = False
        mask[1, 3] = False
        loss = batch_index_contrastive_loss(x, mask)
        assert torch.isfinite(loss)

    def test_temperature_affects_loss(self):
        torch.manual_seed(0)
        x = torch.randn(2, 3, 8)
        mask = torch.ones(2, 3, dtype=torch.bool)
        loss_low_temp = batch_index_contrastive_loss(x, mask, temperature=0.01)
        loss_high_temp = batch_index_contrastive_loss(x, mask, temperature=10.0)
        # Losses should differ
        assert loss_low_temp.item() != pytest.approx(loss_high_temp.item())
