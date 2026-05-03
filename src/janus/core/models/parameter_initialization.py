import torch
import torch.nn as nn

@torch.no_grad()
def init_hypergraph_encoder(model: nn.Module, *, negative_slope: float = 0.01) -> None:
    """
    Applies init_weights, then does a couple of stability tweaks that are common in residual stacks:
      - Make residual branch output projections start small.
    """
    def init_weights_(module: nn.Module, *, negative_slope: float = 0.01) -> None:
        """
        Best-practice-ish initialization for the provided HGAT and its submodules.

        - nn.Linear: Kaiming init (fan_in) suited for LeakyReLU, bias zeros
        - LayerNorm: no-op if affine=False; if affine=True, weight=1 bias=0
        """
        if isinstance(module, nn.Linear):
            # He/Kaiming init for LeakyReLU networks
            nn.init.kaiming_uniform_(module.weight, a=negative_slope, mode="fan_in", nonlinearity="leaky_relu")
            if module.bias is not None:
                nn.init.zeros_(module.bias)

        elif isinstance(module, nn.LayerNorm):
            # Only applies if elementwise_affine=True
            if getattr(module, "elementwise_affine", False):
                if module.weight is not None:
                    nn.init.ones_(module.weight)
                if module.bias is not None:
                    nn.init.zeros_(module.bias)

    model.apply(lambda m: init_weights_(m, negative_slope=negative_slope))

    # Optional residual-stability tweaks:
    # Start the last projection in each attention layer small so the residual dominates initially.
    for m in model.modules():
        # Your BatchedHypergraphConvAttn has these projections in attn path:
        if hasattr(m, "node_out") and isinstance(m.node_out, nn.Linear):
            nn.init.zeros_(m.node_out.weight)  # residual-friendly start
            if m.node_out.bias is not None:
                nn.init.zeros_(m.node_out.bias)

        # Edge update projection can also be damped a bit (not strictly necessary)
        if hasattr(m, "edge_out") and isinstance(m.edge_out, nn.Linear):
            # small std normal is a gentle start (alternative: zeros like node_out)
            nn.init.normal_(m.edge_out.weight, mean=0.0, std=0.02)
            if m.edge_out.bias is not None:
                nn.init.zeros_(m.edge_out.bias)

    # Optional: keep the very last output projection modest too
    if hasattr(model, "output_proj") and isinstance(model.output_proj, nn.Linear):
        nn.init.kaiming_uniform_(model.output_proj.weight, a=negative_slope, mode="fan_in", nonlinearity="leaky_relu")
        if model.output_proj.bias is not None:
            nn.init.zeros_(model.output_proj.bias)

@torch.no_grad()
def init_hypergraph_decoder(model: nn.Module) -> None:
    def init_transformer_weights_(m: nn.Module) -> None:
        """Common best-practice init for Transformer-style blocks."""
        # Linear projections / MLP
        if isinstance(m, nn.Linear):
            nn.init.xavier_uniform_(m.weight)
            if m.bias is not None:
                nn.init.zeros_(m.bias)

        # MultiheadAttention (PyTorch init covers QKV, but out_proj.weight may remain default,
        # so we set it explicitly for consistency.)
        elif isinstance(m, nn.MultiheadAttention):
            # QKV projection(s)
            if getattr(m, "in_proj_weight", None) is not None:
                nn.init.xavier_uniform_(m.in_proj_weight)
            else:
                # Separate proj weights when kdim/vdim != embed_dim
                if getattr(m, "q_proj_weight", None) is not None:
                    nn.init.xavier_uniform_(m.q_proj_weight)
                if getattr(m, "k_proj_weight", None) is not None:
                    nn.init.xavier_uniform_(m.k_proj_weight)
                if getattr(m, "v_proj_weight", None) is not None:
                    nn.init.xavier_uniform_(m.v_proj_weight)

            if getattr(m, "in_proj_bias", None) is not None:
                nn.init.zeros_(m.in_proj_bias)

            # Output projection
            if getattr(m, "out_proj", None) is not None:
                nn.init.xavier_uniform_(m.out_proj.weight)
                if m.out_proj.bias is not None:
                    nn.init.zeros_(m.out_proj.bias)

            # add_bias_kv parameters
            if getattr(m, "bias_k", None) is not None:
                nn.init.xavier_normal_(m.bias_k)
            if getattr(m, "bias_v", None) is not None:
                nn.init.xavier_normal_(m.bias_v)

        # LayerNorm (no params if elementwise_affine=False, but handle the general case)
        elif isinstance(m, nn.LayerNorm):
            if getattr(m, "elementwise_affine", False):
                nn.init.ones_(m.weight)
                nn.init.zeros_(m.bias)
        model.apply(init_transformer_weights_)

        if isinstance(model.final, nn.Sequential) and isinstance(model.final[-1], nn.Linear):
            nn.init.zeros_(model.final[-1].weight)
            if model.final[-1].bias is not None:
                nn.init.zeros_(model.final[-1].bias)

@torch.no_grad()
def init_dit_weights(model: nn.Module) -> None:
    """
    Initialization following (1) DiT paper best practices and (2) common ViT init conventions:

    - All Linear layers: Xavier uniform, bias = 0  (ViT-style "standard" init)  :contentReference[oaicite:1]{index=1}
    - Time embedding MLP: Normal(std=0.02) on the MLP linear weights (as in DiT codebases) :contentReference[oaicite:2]{index=2}
    - adaLN-Zero: zero-init the *last* Linear in each adaLN modulation so blocks start as identity :contentReference[oaicite:3]{index=3}
    - Final output Linear: weight = 0, bias = 0 (explicitly stated in the paper) :contentReference[oaicite:4]{index=4}
    """

    # ---------------------------------------------------------------------
    # 1) "Standard ViT" transformer init: Linear -> Xavier uniform, bias -> 0
    # ---------------------------------------------------------------------
    def _basic_init(m: nn.Module) -> None:
        if isinstance(m, nn.Linear):
            nn.init.xavier_uniform_(m.weight)
            if m.bias is not None:
                nn.init.constant_(m.bias, 0.0)

    model.apply(_basic_init)  # recursively hits all nn.Linear submodules

    # ---------------------------------------------------------------------
    # 2) nn.MultiheadAttention: init the parameter tensors not covered by nn.Linear
    #    (in_proj_weight / in_proj_bias are Parameters, not nn.Linear modules)
    # ---------------------------------------------------------------------
    for m in model.modules():
        if isinstance(m, nn.MultiheadAttention):
            # QKV packed projection:
            if m.in_proj_weight is not None:
                nn.init.xavier_uniform_(m.in_proj_weight)
            if m.in_proj_bias is not None:
                nn.init.constant_(m.in_proj_bias, 0.0)

            # out_proj is an nn.Linear (already covered by _basic_init), but harmless to be explicit:
            nn.init.xavier_uniform_(m.out_proj.weight)
            if m.out_proj.bias is not None:
                nn.init.constant_(m.out_proj.bias, 0.0)

            # Optional learned key/value biases (you enabled add_bias_kv=True):
            if getattr(m, "bias_k", None) is not None:
                nn.init.constant_(m.bias_k, 0.0)
            if getattr(m, "bias_v", None) is not None:
                nn.init.constant_(m.bias_v, 0.0)

    # ---------------------------------------------------------------------
    # 3) Time embedding MLP: Normal(std=0.02) on weights, bias = 0
    #    (Mirrors DiT implementations for timestep embed MLPs) :contentReference[oaicite:5]{index=5}
    # ---------------------------------------------------------------------
    if hasattr(model, "time_cond") and hasattr(model.time_cond, "mlp"):
        tmlp = model.time_cond.mlp
        if hasattr(tmlp, "linear_1"):
            nn.init.normal_(tmlp.linear_1.weight, std=0.02)
            if tmlp.linear_1.bias is not None:
                nn.init.constant_(tmlp.linear_1.bias, 0.0)
        if hasattr(tmlp, "linear_2"):
            nn.init.normal_(tmlp.linear_2.weight, std=0.02)
            if tmlp.linear_2.bias is not None:
                nn.init.constant_(tmlp.linear_2.bias, 0.0)

    # ---------------------------------------------------------------------
    # 4) adaLN-Zero: zero-out the modulation Linear so shift/scale/gates start at 0
    #    -> each DiTBlock starts as identity (residuals are gated off) :contentReference[oaicite:6]{index=6}
    # ---------------------------------------------------------------------
    if hasattr(model, "blocks"):
        for blk in model.blocks:
            if hasattr(blk, "adaLN") and isinstance(blk.adaLN, nn.Sequential):
                last = blk.adaLN[-1]
                if isinstance(last, nn.Linear):
                    nn.init.constant_(last.weight, 0.0)
                    if last.bias is not None:
                        nn.init.constant_(last.bias, 0.0)

    # Final (post-block) adaptive LN modulation should also start as identity:
    if hasattr(model, "adaLN") and isinstance(model.adaLN, nn.Sequential):
        last = model.adaLN[-1]
        if isinstance(last, nn.Linear):
            nn.init.constant_(last.weight, 0.0)
            if last.bias is not None:
                nn.init.constant_(last.bias, 0.0)

    # ---------------------------------------------------------------------
    # 5) Final output projection: explicitly zero-init (paper) :contentReference[oaicite:7]{index=7}
    # ---------------------------------------------------------------------
    if hasattr(model, "linear") and isinstance(model.linear, nn.Linear):
        nn.init.constant_(model.linear.weight, 0.0)
        if model.linear.bias is not None:
            nn.init.constant_(model.linear.bias, 0.0)

