import math
import torch
import torch.nn as nn
import math
import torch
import torch.nn as nn
from typing import Optional, Literal, Tuple

class BatchedHypergraphConvAttn(nn.Module):
    """
    Batched Hypergraph layer supporting:
      - Hypergraph Convolution (HGConv): Dv^{-1/2} H We De^{-1} H^T Dv^{-1/2} X W   (symmetric)
                                   or: Dv^{-1}   H We De^{-1} H^T             X W   (row)
      - Hypergraph Attention (HGAttn): build a dynamic incidence weighting using QKV attention
        on the bipartite (node <-> hyperedge) incidence, then aggregate hyperedge->node.

    Shapes:
      X: [B, N, Fin]     node features
      H: [B, N, M]       incidence (0/1 or weighted)
      Y: [B, M, Ein]     optional hyperedge features

    hyperedge_weight:
      - None, or [M], or [B, M]  (diagonal We)

    Forward returns:
      out: [B, N, Fout]
      (optionally) y_out: [B, M, edge_out_dim] if return_hyperedge=True
    """

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        *,
        mode: Literal["conv", "attn"] = "conv",
        bias: bool = True,

        # Convolution normalization
        symmetric_norm: bool = True,

        # Attention config
        heads: int = 4,
        edge_in_channels: Optional[int] = None,   # if Y is provided (or you want a different edge feature dim)
        attn_symmetric_degree_scaling: bool = False,
        attn_dropout: float = 0.0,
        proj_dropout: float = 0.0,

        eps: float = 1e-12,
    ):
        super().__init__()
        if mode not in ("conv", "attn"):
            raise ValueError("mode must be 'conv' or 'attn'")
        self.mode = mode
        self.symmetric_norm = symmetric_norm
        self.eps = eps

        if bias:
            self.bias = nn.Parameter(torch.zeros(out_channels))
        else:
            self.register_parameter("bias", None)

        # --- Convolution path ---
        self.lin_conv = nn.Linear(in_channels, out_channels, bias=False)

        # --- Attention path ---
        self.heads = heads
        if out_channels % heads != 0:
            raise ValueError(f"out_channels ({out_channels}) must be divisible by heads ({heads}).")
        self.head_dim = out_channels // heads
        self.scale = 1.0 / math.sqrt(self.head_dim)

        self.edge_in_channels = edge_in_channels if edge_in_channels is not None else in_channels
        self.attn_symmetric_degree_scaling = attn_symmetric_degree_scaling

        self.attn_drop = nn.Dropout(attn_dropout)
        self.proj_drop = nn.Dropout(proj_dropout)

        # If Y is None, we initialize edge feats by pooling X then projecting to edge_in_channels if needed
        if self.edge_in_channels != in_channels:
            self.edge_init = nn.Linear(in_channels, self.edge_in_channels, bias=False)
        else:
            self.edge_init = None

        # Node->Edge attention (update hyperedges from nodes)
        self.q_edge = nn.Linear(self.edge_in_channels, heads * self.head_dim, bias=False)
        self.k_node_for_edge = nn.Linear(in_channels, heads * self.head_dim, bias=False)
        self.v_node_for_edge = nn.Linear(in_channels, heads * self.head_dim, bias=False)
        self.edge_out = nn.Linear(heads * self.head_dim, self.edge_in_channels, bias=False)

        # Edge->Node attention (update nodes from hyperedges)
        self.q_node = nn.Linear(in_channels, heads * self.head_dim, bias=False)
        self.k_edge_for_node = nn.Linear(self.edge_in_channels, heads * self.head_dim, bias=False)
        self.v_edge_for_node = nn.Linear(self.edge_in_channels, heads * self.head_dim, bias=False)
        self.node_out = nn.Linear(heads * self.head_dim, out_channels, bias=False)

    @staticmethod
    def _expand_hyperedge_weight(w: torch.Tensor, B: int, M: int, device, dtype) -> torch.Tensor:
        if w is None:
            return torch.ones(B, M, device=device, dtype=dtype)
        if w.dim() == 1:
            if w.numel() != M:
                raise ValueError(f"hyperedge_weight has shape {w.shape}, expected [{M}]")
            return w.to(device=device, dtype=dtype).unsqueeze(0).expand(B, -1)
        if w.dim() == 2:
            if w.shape != (B, M):
                raise ValueError(f"hyperedge_weight has shape {w.shape}, expected [{B}, {M}]")
            return w.to(device=device, dtype=dtype)
        raise ValueError("hyperedge_weight must be None, [M], or [B, M]")

    def _masked_softmax(self, logits: torch.Tensor, mask: torch.Tensor, dim: int) -> torch.Tensor:
        """
        logits: [...]
        mask: same shape as logits (bool), True where valid
        """
        if logits.dtype in (torch.float16, torch.bfloat16):
            neg = -1e4
        else:
            neg = -1e9

        logits = logits.masked_fill(~mask, neg)
        attn = torch.softmax(logits, dim=dim)
        attn = attn * mask.to(attn.dtype)
        attn = attn / attn.sum(dim=dim, keepdim=True).clamp_min(self.eps)
        return attn

    def forward(
        self,
        X: torch.Tensor,                 # [B, N, Fin]
        H: torch.Tensor,                 # [B, N, M]
        Y: Optional[torch.Tensor] = None,# [B, M, Ein] optional
        hyperedge_weight: Optional[torch.Tensor] = None,  # None, [M], or [B, M]
        return_hyperedge: bool = False,
    ) -> torch.Tensor | Tuple[torch.Tensor, torch.Tensor]:
        if X.dim() != 3 or H.dim() != 3:
            raise ValueError(f"Expected X and H to be 3D, got X:{X.shape}, H:{H.shape}")
        B, N, _ = X.shape
        if H.shape[0] != B or H.shape[1] != N:
            raise ValueError(f"Shape mismatch: X is {X.shape}, but H is {H.shape}")

        if not torch.is_floating_point(H):
            H = H.to(dtype=X.dtype)
        B, N, M = H.shape
        w = self._expand_hyperedge_weight(hyperedge_weight, B, M, H.device, H.dtype)  # [B, M]

        # Degrees (for conv, and optional scaling in attn)
        De = H.sum(dim=1)                                  # [B, M]
        De_inv = 1.0 / De.clamp_min(self.eps)              # [B, M]
        Dv = (H * w.unsqueeze(1)).sum(dim=2)               # [B, N]
        Dv_inv = 1.0 / Dv.clamp_min(self.eps)              # [B, N]
        Dv_inv_sqrt = torch.rsqrt(Dv.clamp_min(self.eps))  # [B, N]

        if self.mode == "conv":
            # Xw first
            Xw = self.lin_conv(X)  # [B, N, out]
            Ht = H.transpose(1, 2) # [B, M, N]

            if self.symmetric_norm:
                X0 = Xw * Dv_inv_sqrt.unsqueeze(-1)
                Xe = torch.bmm(Ht, X0)  # [B, M, out]
                Xe = Xe * (w * De_inv).unsqueeze(-1)
                Xn = torch.bmm(H, Xe)   # [B, N, out]
                out = Xn * Dv_inv_sqrt.unsqueeze(-1)
            else:
                Xe = torch.bmm(Ht, Xw)
                Xe = Xe * (w * De_inv).unsqueeze(-1)
                Xn = torch.bmm(H, Xe)
                out = Xn * Dv_inv.unsqueeze(-1)

            if self.bias is not None:
                out = out + self.bias.view(1, 1, -1)
            return out

        # ---------------- Attention mode ----------------
        # Initialize / accept hyperedge features
        if Y is None:
            # incidence pooling: mean over incident nodes
            Y0 = torch.bmm(H.transpose(1, 2), X) * De_inv.unsqueeze(-1)  # [B, M, Fin]
            if self.edge_init is not None:
                Y0 = self.edge_init(Y0)  # -> [B, M, Ein]
        else:
            if Y.dim() != 3 or Y.shape[0] != B or Y.shape[1] != M:
                raise ValueError(f"Y must be [B, M, Ein], got {Y.shape}")
            if Y.shape[2] != self.edge_in_channels:
                raise ValueError(
                    f"Y last dim {Y.shape[2]} must match edge_in_channels={self.edge_in_channels}"
                )
            Y0 = Y

        # Optional degree scaling to mimic symmetric degree factors around attention
        X_for_attn = X
        if self.attn_symmetric_degree_scaling:
            X_for_attn = X_for_attn * Dv_inv_sqrt.unsqueeze(-1)

        # ----- Node -> Edge (update hyperedges from incident nodes) -----
        # Q from hyperedges (Y0), K/V from nodes (X_for_attn)
        qE = self.q_edge(Y0).view(B, M, self.heads, self.head_dim)                 # [B,M,H,D]
        kV = self.k_node_for_edge(X_for_attn).view(B, N, self.heads, self.head_dim)# [B,N,H,D]
        vV = self.v_node_for_edge(X_for_attn).view(B, N, self.heads, self.head_dim)# [B,N,H,D]

        # logits_e: [B, H, M, N] (only where H^T indicates incidence)
        logits_e = torch.einsum("bmhd,bnhd->bhmn", qE, kV) * self.scale
        mask_e = (H.transpose(1, 2) > 0).unsqueeze(1)  # [B,1,M,N]
        attn_e = self._masked_softmax(logits_e, mask_e, dim=-1)
        attn_e = self.attn_drop(attn_e)

        y_msg = torch.einsum("bhmn,bnhd->bmhd", attn_e, vV).contiguous()           # [B,M,H,D]
        y_msg = y_msg.view(B, M, self.heads * self.head_dim)                      # [B,M,HD]
        Y1 = self.edge_out(y_msg)                                                 # [B,M,Ein]
        Y1 = self.proj_drop(Y1)

        if self.attn_symmetric_degree_scaling:
            # apply De^{-1} and hyperedge weights like the conv operator's middle term (We De^{-1})
            Y1 = Y1 * (w * De_inv).unsqueeze(-1)

        # ----- Edge -> Node (update nodes from incident hyperedges) -----
        qN = self.q_node(X_for_attn).view(B, N, self.heads, self.head_dim)         # [B,N,H,D]
        kE = self.k_edge_for_node(Y1).view(B, M, self.heads, self.head_dim)        # [B,M,H,D]
        vE = self.v_edge_for_node(Y1).view(B, M, self.heads, self.head_dim)        # [B,M,H,D]

        # logits_n: [B, H, N, M] masked by incidence
        logits_n = torch.einsum("bnhd,bmhd->bhnm", qN, kE) * self.scale
        mask_n = (H > 0).unsqueeze(1)  # [B,1,N,M]
        attn_n = self._masked_softmax(logits_n, mask_n, dim=-1)
        attn_n = self.attn_drop(attn_n)

        out_msg = torch.einsum("bhnm,bmhd->bnhd", attn_n, vE).contiguous()         # [B,N,H,D]
        out_msg = out_msg.view(B, N, self.heads * self.head_dim)                  # [B,N,HD]
        out = self.node_out(out_msg)                                              # [B,N,Fout]
        out = self.proj_drop(out)

        if self.attn_symmetric_degree_scaling:
            out = out * Dv_inv_sqrt.unsqueeze(-1)

        if self.bias is not None:
            out = out + self.bias.view(1, 1, -1)

        if return_hyperedge:
            return out, Y1
        return out

class HGAT(nn.Module):

    def __init__(self,
                 in_channels: int,
                 hidden_channels: int,
                 out_channels: int,
                 num_layers: int,
                 heads: int = 4):
        super(HGAT, self).__init__()
        self.input_norm = nn.LayerNorm(in_channels, elementwise_affine=False)
        self.input_proj = nn.Linear(in_channels, hidden_channels)
        self.layers = nn.ModuleList([
            nn.ModuleDict({
                'conv': BatchedHypergraphConvAttn(hidden_channels,
                                                  hidden_channels,
                                                  bias=True,
                                                  mode='attn',
                                                  heads=heads,
                                                  symmetric_norm=True),
                'norm': nn.LayerNorm(hidden_channels, elementwise_affine=False),
                'activation': nn.LeakyReLU(),
                'skip_proj': nn.Linear(hidden_channels, hidden_channels)
            }) for _ in range(num_layers)
        ])
        self.output_norm = nn.LayerNorm(hidden_channels, elementwise_affine=False)
        self.output_proj = nn.Linear(hidden_channels, out_channels)

    def forward(self, x: torch.Tensor, h: torch.Tensor) -> torch.Tensor:
        x = self.input_norm(x)
        x = self.input_proj(x)
        for layer in self.layers:
            residual = x
            x = layer['conv'](x, h)
            x = layer['norm'](x)
            x = x + layer['skip_proj'](residual)
            x = layer['activation'](x)
        x = self.output_norm(x)
        x = self.output_proj(x)
        return x

class TransformerBlock(nn.Module):

    def __init__(self,
                 num_channels: int,
                 num_heads: int):
        super(TransformerBlock, self).__init__()
        # Transfomer block with cross-attention
        self.norm_1 = nn.LayerNorm(num_channels, elementwise_affine=False)
        self.mha = nn.MultiheadAttention(num_channels, num_heads=num_heads, bias=True, add_bias_kv=True, batch_first=True)
        self.norm_2 = nn.LayerNorm(num_channels, elementwise_affine=False)
        self.mlp = MLP(num_channels, num_channels * 4, dropout=0.0)

    def forward(self, x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        x = x + self.mha(
            self.norm_1(x), y, y
        )[0]
        x = x + self.mlp(self.norm_2(x))
        return x

class Adapter(nn.Module):

    def __init__(self,
                 in_channels: int,
                 hidden_channels: int,
                 num_heads: int,
                 num_blocks: int = 1):
        super(Adapter, self).__init__()
        self.x_in_proj = nn.Linear(in_channels, hidden_channels)
        self.y_in_proj = nn.Linear(in_channels, hidden_channels)
        self.blocks = nn.ModuleList([
            TransformerBlock(hidden_channels, num_heads) for _ in range(num_blocks)
        ])
        self.out_norm = nn.LayerNorm(hidden_channels, elementwise_affine=False)
        self.out_proj = nn.Linear(hidden_channels, in_channels)

    def forward(self, x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        x = self.x_in_proj(x)
        y = self.y_in_proj(y)
        for block in self.blocks:
            x = block(x, y)
        x = self.out_norm(x)
        x = self.out_proj(x)
        return x

class HypergraphDecoder(nn.Module):

    def __init__(self,
                 in_channels: int,
                 num_classes: int = 2):
        super(HypergraphDecoder, self).__init__()
        self.final = nn.Sequential(
            nn.LayerNorm(in_channels, elementwise_affine=False),
            nn.Linear(in_channels, num_classes)
        )

    def forward(self, x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        h_logits = torch.einsum("taf,tbf->tabf", x, y)
        h_logits = self.final(h_logits)
        return h_logits

class MLP(nn.Module):

    def __init__(self, in_channels: int, hidden_channels, dropout: float = 0.1):
        super(MLP, self).__init__()
        self.linear_1 = nn.Linear(in_channels, hidden_channels * 2)
        self.activation = nn.SiLU()
        self.dropout = nn.Dropout(p=dropout)
        self.linear_2 = nn.Linear(hidden_channels, in_channels * 2)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        y, g = self.linear_1(x).chunk(2, dim=-1)
        y = self.activation(y) * torch.sigmoid(g)
        y = self.dropout(y)
        y, g = self.linear_2(y).chunk(2, dim=-1)
        y = self.activation(y) * torch.sigmoid(g)
        y = self.dropout(y)
        return y

class TimeConditioning(nn.Module):

    def __init__(self, embedding_dim: int, hidden_channels: int):
        super(TimeConditioning, self).__init__()
        self.embedding_dim = embedding_dim
        self.mlp = MLP(embedding_dim, hidden_channels, dropout=0.0)

    def sinusoidal_embedding(self, t: torch.Tensor, dim: int, max_period: int = 10000) -> torch.Tensor:
        half = dim // 2
        freqs = torch.exp(
            -math.log(max_period) * torch.arange(start=0, end=half, dtype=torch.float32, device=t.device) / half
        )
        args = t[:, None].float() * freqs[None]
        embedding = torch.cat([torch.cos(args), torch.sin(args)], dim=-1)
        if dim % 2:
            embedding = torch.cat([embedding, torch.zeros_like(embedding[:, :1])], dim=-1)
        return embedding

    def forward(self, t: torch.Tensor) -> torch.Tensor:
        t_freq = self.sinusoidal_embedding(t, self.embedding_dim)
        t_emb = self.mlp(t_freq) + t_freq
        return t_emb

class DiTBlock(nn.Module):

    def __init__(self,
                 num_channels: int,
                 num_heads: int,
                 cross_attention_layer: bool = False):
        super(DiTBlock, self).__init__()
        self.norm_1 = nn.LayerNorm(num_channels, elementwise_affine=False)
        self.mha = nn.MultiheadAttention(num_channels, num_heads=num_heads, bias=True, batch_first=True)

        self.cross_attention_layer = cross_attention_layer
        if cross_attention_layer:
            self.norm_2 = nn.LayerNorm(num_channels, elementwise_affine=False)
            self.cross_mha = nn.MultiheadAttention(
                num_channels,
                num_heads=num_heads,
                bias=True,
                add_bias_kv=True,
                batch_first=True
            )
            adaln_out = 3
        else:
            adaln_out = 2

        self.norm_3 = nn.LayerNorm(num_channels, elementwise_affine=False)
        self.mlp = MLP(num_channels, num_channels * 4, dropout=0.0)
        self.adaLN = nn.Sequential(
            nn.SiLU(),
            nn.Linear(num_channels, num_channels * 3 * adaln_out)
        )

    def forward(self, x: torch.Tensor, c: torch.Tensor, y: torch.Tensor = None) -> torch.Tensor:
        if self.cross_attention_layer and y is None:
            raise ValueError("cross_attention_layer is True but no y is provided.")
        if not self.cross_attention_layer and y is not None:
            raise ValueError("cross_attention_layer is False but y is provided.")
        if self.cross_attention_layer:
            shift_mha, scale_mha, gate_mha, shift_cross, scale_cross, gate_cross, shift_mlp, scale_mlp, gate_mlp = self.adaLN(c).chunk(9, dim=-1)
        else:
            shift_mha, scale_mha, gate_mha, shift_mlp, scale_mlp, gate_mlp = self.adaLN(c).chunk(6, dim=-1)

        z = (1 + scale_mha) * self.norm_1(x) + shift_mha # Modulate
        x = x + gate_mha * self.mha(
            z, z, z
        )[0]

        if self.cross_attention_layer:
            z = (1 + scale_cross) * self.norm_2(x) + shift_cross # Modulate
            x = x + gate_cross * self.cross_mha(
                z, y, y
            )[0]

        z = (1 + scale_mlp) * self.norm_3(x) + shift_mlp # Modulate
        x = x + gate_mlp * self.mlp(z)
        return x

class DiT(nn.Module):

    def __init__(self,
                 in_channels: int,
                 hidden_channels: int,
                 num_blocks: int,
                 num_heads: int,
                 cross_attention: bool = False):
        super(DiT, self).__init__()
        self.input_proj = nn.Linear(in_channels, hidden_channels)
        self.time_cond = TimeConditioning(hidden_channels, hidden_channels)
        self.blocks = nn.ModuleList([
            DiTBlock(hidden_channels, num_heads, cross_attention) for _ in range(num_blocks)
        ])
        if cross_attention:
            self.y_proj = nn.Sequential(
                nn.LayerNorm(in_channels, elementwise_affine=False),
                nn.Linear(in_channels, hidden_channels),
                nn.SiLU(),
            )
            self.y_cond = nn.ModuleList([
                nn.Sequential(
                    nn.Linear(hidden_channels, hidden_channels)
                ) for _ in range(num_blocks)
            ])
        self.cross_attention = cross_attention
        self.adaLN = nn.Sequential(
            nn.SiLU(),
            nn.Linear(hidden_channels, hidden_channels * 2, bias=True)
        )
        self.out_norm = nn.LayerNorm(hidden_channels, elementwise_affine=False)
        self.linear = nn.Linear(hidden_channels, in_channels)

    def forward(self, x: torch.Tensor, t: torch.Tensor, y: torch.Tensor = None) -> torch.Tensor:
        x = self.input_proj(x)
        t_emb = self.time_cond(t)
        if self.cross_attention:
            y = self.y_proj(y)
        for i, block in enumerate(self.blocks):
            if self.cross_attention:
                x = block(x, t_emb, self.y_cond[i](y))
            else:
                x = block(x, t_emb)
        # Final adaptive layer norm
        shift, scale = self.adaLN(t_emb).chunk(2, dim=-1)
        x = (1 + scale) * self.out_norm(x) + shift
        x = self.linear(x)
        if self.cross_attention:
            return x, y
        return x
