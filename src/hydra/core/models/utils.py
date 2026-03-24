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

from diffusers import DDPMScheduler

def compute_snr(noise_scheduler: DDPMScheduler, timesteps: torch.LongTensor) -> torch.Tensor:
    """
    Returns SNR(t) for each element in `timesteps` (shape: [batch]).
    """
    alphas_cumprod = noise_scheduler.alphas_cumprod.to(device=timesteps.device)  # [num_train_timesteps]
    sqrt_alphas_cumprod = alphas_cumprod.sqrt()
    sqrt_one_minus_alphas_cumprod = (1.0 - alphas_cumprod).sqrt()

    # Gather per-sample values
    alpha = sqrt_alphas_cumprod[timesteps].float()                  # [batch]
    sigma = sqrt_one_minus_alphas_cumprod[timesteps].float()        # [batch]

    # SNR = (alpha / sigma)^2
    snr = (alpha / sigma) ** 2
    return snr  # [batch]

def min_snr_weighted_v_mse_loss(
    noise_scheduler: DDPMScheduler,
    model_pred_v: torch.Tensor,   # UNet output (v), shape [B,C,H,W]
    latents: torch.Tensor,        # clean latents x0, shape [B,C,H,W]
    noise: torch.Tensor,          # eps, shape [B,C,H,W]
    timesteps: torch.LongTensor,  # [B]
    snr_gamma: float = 5.0,
) -> torch.Tensor:
    assert noise_scheduler.config.prediction_type == "v_prediction", \
        "This loss is for v_prediction. Set scheduler.config.prediction_type='v_prediction'."

    # Target for v-prediction (Diffusers provides this convenience method)
    target_v = noise_scheduler.get_velocity(latents, noise, timesteps)  # [B,C,H,W]

    # Per-pixel MSE (no reduction yet)
    loss = F.mse_loss(model_pred_v.float(), target_v.float(), reduction="none")  # [B,C,H,W]

    # Reduce to per-sample loss
    loss = loss.mean(dim=tuple(range(1, loss.ndim)))  # [B]

    # Compute Min-SNR-γ weights for v-prediction:
    # w(t) = min(SNR(t), gamma) / (SNR(t) + 1)
    snr = compute_snr(noise_scheduler, timesteps)  # [B]
    weights = torch.minimum(snr, torch.full_like(snr, snr_gamma)) / (snr + 1.0)  # [B]

    # Apply weights and average
    return (loss * weights).mean()