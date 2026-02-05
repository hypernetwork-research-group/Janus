import torch
import torch.nn.functional as F

def batch_index_contrastive_loss(x: torch.Tensor, temperature: float = 0.1) -> torch.Tensor:
    """
    x: [B, N, F]
    Positives for anchor (b,n): all (b',n) with b' != b
    Negatives: all other (b', n') with n' != n  (and excluding self)
    """
    B, N, Fdim = x.shape
    M = B * N  # total samples

    if B < 2:
        raise ValueError("Need B>=2 to have positives across the batch.")

    # Flatten to [M, F] where each row is one (b,n)
    z = x.reshape(M, Fdim)
    z = F.normalize(z, dim=1)  # cosine similarity

    # Similarity logits [M, M]
    logits = (z @ z.t()) / temperature

    # Masks
    device = x.device
    eye = torch.eye(M, dtype=torch.bool, device=device)

    # Label = n (index in second dimension), repeated for each batch
    labels = torch.arange(N, device=device).repeat(B)  # [M]

    # positives: same label (same n), excluding self
    pos_mask = (labels[:, None] == labels[None, :]) & (~eye)  # [M, M]

    # all candidates except self for denominator
    denom_mask = ~eye

    # For numerical stability: subtract row-wise max (doesn't change softmax)
    logits = logits - logits.max(dim=1, keepdim=True).values

    # log_prob[i,j] = log( exp(logits[i,j]) / sum_{a!=i} exp(logits[i,a]) )
    logits_denom = logits.masked_fill(~denom_mask, float("-inf"))
    log_prob = logits - torch.logsumexp(logits_denom, dim=1, keepdim=True)

    # For each anchor i, average log_prob over its positives
    pos_count = pos_mask.sum(dim=1)  # [M], should be B-1 for every i
    # (safety in case some anchors have 0 positives)
    pos_count = pos_count.clamp_min(1)

    loss_per_anchor = -(log_prob * pos_mask.float()).sum(dim=1) / pos_count
    loss = loss_per_anchor.mean()
    return loss
