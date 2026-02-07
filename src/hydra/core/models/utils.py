import torch
import torch.nn.functional as F

def batch_index_contrastive_loss(
    x: torch.Tensor,                # [B, N, F]
    mask: torch.Tensor,             # [B, N] (bool or 0/1)
    temperature: float = 0.1,
    eps: float = 1e-12,
) -> torch.Tensor:
    """
    Positives for anchor (b,n): all active (b',n) with b' != b
    Negatives: all other active (b',n') with n' != n
    Samples where mask[b,n] == 0 are excluded entirely (as anchors and as candidates).
    """
    B, N, Fdim = x.shape
    if B < 2:
        raise ValueError("Need B>=2 to have positives across the batch.")

    device = x.device
    M = B * N

    # Flatten
    z = x.reshape(M, Fdim)
    z = F.normalize(z, dim=1)

    active = mask.reshape(M).to(device=device)
    if active.dtype != torch.bool:
        active = active != 0  # support 0/1 or floats

    # Labels = n index (shared across batches)
    labels = torch.arange(N, device=device).repeat(B)  # [M]

    # Similarity logits
    logits = (z @ z.t()) / temperature

    eye = torch.eye(M, dtype=torch.bool, device=device)

    # Candidate mask for denominators: active keys (j) excluding self
    denom_mask = active[None, :] & (~eye)  # [M, M]
    # Positive mask: active pairs with same label, excluding self
    pos_mask = active[:, None] & active[None, :] & (labels[:, None] == labels[None, :]) & (~eye)

    # Stabilize with row-wise max (safe even if some rows are inactive)
    logits = logits - logits.max(dim=1, keepdim=True).values

    # Denominator: sum over active candidates (excluding self)
    logits_denom = logits.masked_fill(~denom_mask, float("-inf"))
    log_denom = torch.logsumexp(logits_denom, dim=1, keepdim=True)
    # Rows with no candidates -> log_denom = -inf; set to 0 so log_prob is finite (row will be ignored anyway)
    log_denom = torch.where(torch.isfinite(log_denom), log_denom, torch.zeros_like(log_denom))

    log_prob = logits - log_denom  # [M, M]

    # Average over positives for each anchor
    pos_count = pos_mask.sum(dim=1)  # [M]
    valid_anchor = active & (pos_count > 0)

    # Compute per-anchor loss (won't be used if invalid_anchor)
    pos_sum = (log_prob * pos_mask.float()).sum(dim=1)
    loss_per_anchor = -pos_sum / (pos_count.clamp_min(1).float() + eps)

    if valid_anchor.any():
        return loss_per_anchor[valid_anchor].mean()
    else:
        # No valid anchors -> return 0 (keeps training running; gradient is zero)
        return x.new_zeros(())
