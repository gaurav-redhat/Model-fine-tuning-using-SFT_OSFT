"""
Strategy 2 — Sparse Upcycling (Dense → MoE)
Convert dense FFN layers into Mixture-of-Experts by duplicating weights
and adding a lightweight router.

Only top-K experts fire per token, so inference FLOPs ≈ original.

FLOW:
─────────────────────────────────────────────────────────────────────────
Step 1: Take pre-trained dense model
Step 2: For each FFN layer → duplicate weights N times (N experts)
Step 3: Add a Router (tiny linear layer) before the experts
Step 4: Fine-tune router + lightly tune experts; freeze attention
Step 5: Deploy — only top-K experts fire → same compute as original
─────────────────────────────────────────────────────────────────────────

DATA FLOW (per layer, per token):

    x  (B, S, H)                    ← input hidden states
    │
    ├──────────────────┐
    │                  │
    ▼                  ▼
  ┌─────────┐    ┌───────────────────────────────────────────┐
  │ Frozen  │    │  MoE FFN (replaces original dense FFN)    │
  │ Attn    │    │                                           │
  │ + Norm  │    │  x ──► Router (H → num_experts)           │
  └────┬────┘    │         │                                 │
       │         │    logits (B, S, E)                       │
       │         │         │                                 │
       │         │    Top-K selection                        │
       │         │         │                                 │
       │         │    ┌────┼─────┬─────┬─── ... ──┐         │
       │         │    ▼    ▼     ▼     ▼          ▼         │
       │         │  [FFN₀][FFN₁][FFN₂][FFN₃]  [FFNₑ₋₁]    │
       │         │  (copy) (copy)(copy)(copy)   (copy)      │
       │         │    │    │     │     │          │          │
       │         │    └────┼─────┼─────┼────...───┘          │
       │         │         ▼                                 │
       │         │    Weighted sum of top-K expert outputs   │
       │         └──────────────┬────────────────────────────┘
       │                        │
       ▼                        ▼
    attn_out (B,S,H)     moe_out (B,S,H)
       │                        │
       └────────── + ───────────┘
                   │
                   ▼
             out (B, S, H)

    E  = num_experts (e.g., 4 or 8)
    K  = top_k active experts (e.g., 2)
    Inference FLOPs ≈ K/E × dense_FFN_FLOPs (K≪E → big savings)

CONVERSION FLOW:

    Dense Model                    Upcycled MoE Model
    ───────────                    ──────────────────
    Layer i:                       Layer i:
      [Attention]  (frozen)          [Attention]  (frozen)
      [FFN]        (1 copy)    →     [Router]     (NEW, trainable)
                                     [FFN₀]       (copy of FFN, tunable)
                                     [FFN₁]       (copy of FFN, tunable)
                                     [FFN₂]       (copy of FFN, tunable)
                                     [FFN₃]       (copy of FFN, tunable)
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import copy


class Router(nn.Module):
    """Learned gating network that picks top-K experts per token."""

    def __init__(self, hidden_size: int, num_experts: int, top_k: int = 2):
        super().__init__()
        self.gate = nn.Linear(hidden_size, num_experts, bias=False)
        self.top_k = top_k

    def forward(self, x: torch.Tensor):
        logits = self.gate(x)                          # (batch, seq, num_experts)
        weights, indices = torch.topk(logits, self.top_k, dim=-1)
        weights = F.softmax(weights, dim=-1)           # normalize selected expert weights
        return weights, indices


class MoEFFN(nn.Module):
    """
    Mixture-of-Experts FFN built from copies of an existing dense FFN.
    Each expert starts as an identical copy (warm start), then diverges during fine-tuning.
    """

    def __init__(self, original_ffn: nn.Module, num_experts: int = 4, top_k: int = 2):
        super().__init__()
        self.num_experts = num_experts
        self.top_k = top_k

        self.experts = nn.ModuleList([copy.deepcopy(original_ffn) for _ in range(num_experts)])

        first_param = next(original_ffn.parameters())
        hidden_size = first_param.shape[-1] if first_param.dim() > 1 else first_param.shape[0]
        self.router = Router(hidden_size, num_experts, top_k)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Flow: x → Router → select top-K → run K experts → weighted sum
              (B,S,H)  (B,S,E)   (B,S,K)    K FFN passes    (B,S,H)
        """
        batch, seq_len, hidden = x.shape

        # Step 1: Router decides which experts to activate
        weights, indices = self.router(x)              # (B, S, K), (B, S, K)

        # Step 2: Flatten for per-token expert dispatch
        flat_x = x.view(-1, hidden)                    # (B*S, H)
        output = torch.zeros_like(flat_x)

        flat_indices = indices.view(-1, self.top_k)    # (B*S, K)
        flat_weights = weights.view(-1, self.top_k)    # (B*S, K)

        # Step 3: Run selected experts and combine outputs
        for k in range(self.top_k):
            expert_idx = flat_indices[:, k]            # (B*S,)
            expert_wt = flat_weights[:, k].unsqueeze(-1)

            for e in range(self.num_experts):
                mask = expert_idx == e
                if mask.any():
                    expert_input = flat_x[mask]
                    expert_out = self.experts[e](expert_input)
                    output[mask] += expert_wt[mask] * expert_out

        return output.view(batch, seq_len, hidden)


# ---------------------------------------------------------------------------
# Standalone demo with a toy transformer
# ---------------------------------------------------------------------------
class ToyFFN(nn.Module):
    def __init__(self, hidden: int, intermediate: int):
        super().__init__()
        self.up = nn.Linear(hidden, intermediate)
        self.act = nn.GELU()
        self.down = nn.Linear(intermediate, hidden)

    def forward(self, x):
        return self.down(self.act(self.up(x)))


class ToyTransformerLayer(nn.Module):
    def __init__(self, hidden: int, intermediate: int):
        super().__init__()
        self.attn = nn.MultiheadAttention(hidden, num_heads=4, batch_first=True)
        self.ffn = ToyFFN(hidden, intermediate)
        self.norm1 = nn.LayerNorm(hidden)
        self.norm2 = nn.LayerNorm(hidden)

    def forward(self, x):
        x = x + self.attn(self.norm1(x), self.norm1(x), self.norm1(x))[0]
        x = x + self.ffn(self.norm2(x))
        return x


def upcycle_layer(layer: ToyTransformerLayer, num_experts: int = 4, top_k: int = 2):
    """Replace the dense FFN in a layer with MoE FFN."""
    moe = MoEFFN(layer.ffn, num_experts=num_experts, top_k=top_k)
    layer.ffn = moe

    for p in layer.attn.parameters():
        p.requires_grad = False
    for p in layer.norm1.parameters():
        p.requires_grad = False
    for p in layer.norm2.parameters():
        p.requires_grad = False

    return layer


if __name__ == "__main__":
    HIDDEN, INTER, SEQ, BATCH = 256, 512, 32, 2
    NUM_EXPERTS, TOP_K = 4, 2

    layer = ToyTransformerLayer(HIDDEN, INTER)
    total_before = sum(p.numel() for p in layer.parameters())

    layer = upcycle_layer(layer, num_experts=NUM_EXPERTS, top_k=TOP_K)

    total_after = sum(p.numel() for p in layer.parameters())
    trainable = sum(p.numel() for p in layer.parameters() if p.requires_grad)

    print(f"Dense  params : {total_before:,}")
    print(f"MoE    params : {total_after:,}")
    print(f"Trainable     : {trainable:,}")
    print(f"Experts: {NUM_EXPERTS}, Top-K: {TOP_K}")
    print(f"Inference FLOPs ≈ original (only {TOP_K} of {NUM_EXPERTS} experts fire)\n")

    x = torch.randn(BATCH, SEQ, HIDDEN)
    out = layer(x)
    print(f"Input  shape: {x.shape}")
    print(f"Output shape: {out.shape}")
