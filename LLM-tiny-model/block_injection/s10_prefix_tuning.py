"""
Strategy 10 — Prefix / Prompt Tuning Blocks
Inject learnable "soft tokens" into every layer's key-value cache.
Extremely parameter-efficient: ~0.1% of model params.

Prefix tokens are continuous vectors (not real words).
Trained via backprop while the entire model stays frozen.

FLOW:
─────────────────────────────────────────────────────────────────────────
Step 1: Freeze entire model (embedding, attention, FFN, LM head)
Step 2: Add learnable prefix_keys + prefix_values at EVERY layer
Step 3: Reparameterize prefix through small MLP for training stability
Step 4: Train only prefix params via backprop on task data
Step 5: Deploy — swap prefix to switch tasks instantly
─────────────────────────────────────────────────────────────────────────

DATA FLOW (per attention layer):

    x (B, S, H)                          Learned Prefix (trainable)
        │                                     │
        ▼                                     ▼
    [QKV Projection] (frozen)           ┌──────────────────────────┐
        │                               │  prefix_keys  (1, P, H)  │
        ├── Q (B, heads, S, d)          │  prefix_values (1, P, H) │
        ├── K (B, heads, S, d)          │       │                  │
        └── V (B, heads, S, d)          │       ▼                  │
             │                          │  [Key MLP]   H→H/2→H    │
             │                          │  [Value MLP] H→H/2→H    │
             │                          │       │                  │
             │                          │       ▼                  │
             │                          │  PK (B, heads, P, d)     │
             │                          │  PV (B, heads, P, d)     │
             │                          └──────┬───────────────────┘
             │                                 │
             ▼                                 ▼
    ┌────────────────────────────────────────────────────────────┐
    │           CONCATENATE prefix into K, V                     │
    │                                                            │
    │   K_full = [PK ‖ K]    →  (B, heads, P+S, d)              │
    │   V_full = [PV ‖ V]    →  (B, heads, P+S, d)              │
    │                                                            │
    │   Attention = softmax( Q × K_fullᵀ / √d ) × V_full        │
    │               ─────────────────────────────                │
    │   Q attends to:                                            │
    │     [prefix₁, prefix₂, ... prefixₚ, token₁, token₂, ...] │
    │      ▲──── learned virtual tokens ──▲  ▲── real input ──▲ │
    └────────────────────────────────────────────────────────────┘
             │
             ▼
    attention output (B, S, H)   ← output has SAME shape as input
             │                     (prefix only in K/V, not Q)
             ▼
    + residual → out (B, S, H)

ATTENTION MATRIX VISUALIZATION (P=3 prefix, S=4 tokens):

    Queries:           Keys:
    (only real tokens) (prefix + real tokens)

                  p₁  p₂  p₃  t₁  t₂  t₃  t₄
              ┌────┬────┬────┬────┬────┬────┬────┐
    token₁    │ ■  │ ■  │ ■  │ ■  │ ■  │ ■  │ ■  │
    token₂    │ ■  │ ■  │ ■  │ ■  │ ■  │ ■  │ ■  │
    token₃    │ ■  │ ■  │ ■  │ ■  │ ■  │ ■  │ ■  │
    token₄    │ ■  │ ■  │ ■  │ ■  │ ■  │ ■  │ ■  │
              └────┴────┴────┴────┴────┴────┴────┘
               ▲────learned────▲  ▲──real tokens──▲

    ★ Every real token can attend to prefix → prefix guides behavior
    ★ Different prefix = different task behavior (same frozen model)

MULTI-TASK SWITCHING:

    Task A (sentiment):     Task B (translation):   Task C (QA):
    ┌────────────────┐     ┌────────────────┐      ┌────────────────┐
    │ prefix_A_keys  │     │ prefix_B_keys  │      │ prefix_C_keys  │
    │ prefix_A_values│     │ prefix_B_values│      │ prefix_C_values│
    └───────┬────────┘     └───────┬────────┘      └───────┬────────┘
            │                      │                       │
            └──────────┬───────────┴───────────────────────┘
                       │
                       ▼
              SAME frozen model
              (swap prefix to switch task in milliseconds)
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import math


class PrefixTuningWrapper(nn.Module):
    """
    Wraps an attention layer to prepend learned prefix key-value pairs.
    The prefix is re-parameterized through a small MLP for stability.
    """

    def __init__(self, hidden_size: int, num_heads: int, prefix_len: int = 20):
        super().__init__()
        self.prefix_len = prefix_len
        self.num_heads = num_heads
        self.head_dim = hidden_size // num_heads

        self.prefix_keys = nn.Parameter(torch.randn(1, prefix_len, hidden_size) * 0.01)
        self.prefix_values = nn.Parameter(torch.randn(1, prefix_len, hidden_size) * 0.01)

        self.key_reparam = nn.Sequential(
            nn.Linear(hidden_size, hidden_size // 2),
            nn.Tanh(),
            nn.Linear(hidden_size // 2, hidden_size),
        )
        self.value_reparam = nn.Sequential(
            nn.Linear(hidden_size, hidden_size // 2),
            nn.Tanh(),
            nn.Linear(hidden_size // 2, hidden_size),
        )

    def get_prefix_kv(self, batch_size: int):
        pk = self.key_reparam(self.prefix_keys).expand(batch_size, -1, -1)
        pv = self.value_reparam(self.prefix_values).expand(batch_size, -1, -1)
        return pk, pv


class PrefixTunedAttention(nn.Module):
    """Self-attention with learned prefix prepended to keys and values."""

    def __init__(self, hidden_size: int, num_heads: int = 4, prefix_len: int = 20):
        super().__init__()
        self.num_heads = num_heads
        self.head_dim = hidden_size // num_heads

        self.qkv = nn.Linear(hidden_size, hidden_size * 3, bias=False)
        self.out_proj = nn.Linear(hidden_size, hidden_size, bias=False)
        self.norm = nn.LayerNorm(hidden_size)

        self.prefix = PrefixTuningWrapper(hidden_size, num_heads, prefix_len)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Flow: x → norm → QKV (frozen) → prepend prefix to K,V → attention → out_proj + residual
              K = [prefix_keys ‖ real_keys]     (P+S keys)
              V = [prefix_values ‖ real_values] (P+S values)
              Q = real queries only             (S queries)
        """
        residual = x
        x = self.norm(x)                                     # (B,S,H)
        B, L, D = x.shape
        H, HD = self.num_heads, self.head_dim

        qkv = self.qkv(x).reshape(B, L, 3, H, HD).permute(2, 0, 3, 1, 4)
        q, k, v = qkv[0], qkv[1], qkv[2]                   # frozen Q, K, V

        pk, pv = self.prefix.get_prefix_kv(B)                # learned prefix (trainable)
        pk = pk.view(B, -1, H, HD).transpose(1, 2)           # (B, heads, P, d)
        pv = pv.view(B, -1, H, HD).transpose(1, 2)           # (B, heads, P, d)

        k = torch.cat([pk, k], dim=2)                        # K = [prefix ‖ real] (P+S)
        v = torch.cat([pv, v], dim=2)                        # V = [prefix ‖ real] (P+S)

        attn = torch.matmul(q, k.transpose(-2, -1)) / math.sqrt(HD)  # Q×K: (S, P+S)
        attn = F.softmax(attn, dim=-1)
        out = torch.matmul(attn, v)                          # weighted V: (B,heads,S,d)

        out = out.transpose(1, 2).reshape(B, L, D)
        return residual + self.out_proj(out)

    def freeze_attention_train_prefix(self):
        for p in self.qkv.parameters():
            p.requires_grad = False
        for p in self.out_proj.parameters():
            p.requires_grad = False
        for p in self.norm.parameters():
            p.requires_grad = False


class PrefixTunedModel(nn.Module):
    """Full model with prefix tuning at every layer."""

    def __init__(
        self,
        vocab_size: int,
        hidden: int,
        num_layers: int = 6,
        nhead: int = 4,
        prefix_len: int = 20,
    ):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, hidden)
        self.layers = nn.ModuleList([
            PrefixTunedAttention(hidden, nhead, prefix_len)
            for _ in range(num_layers)
        ])
        self.ffns = nn.ModuleList([
            nn.Sequential(
                nn.LayerNorm(hidden),
                nn.Linear(hidden, hidden * 4),
                nn.GELU(),
                nn.Linear(hidden * 4, hidden),
            )
            for _ in range(num_layers)
        ])
        self.norm = nn.LayerNorm(hidden)
        self.head = nn.Linear(hidden, vocab_size, bias=False)
        self.prefix_len = prefix_len

    def freeze_base_train_prefix(self):
        for p in self.parameters():
            p.requires_grad = False
        for layer in self.layers:
            for p in layer.prefix.parameters():
                p.requires_grad = True

    def forward(self, input_ids: torch.Tensor):
        x = self.embedding(input_ids)
        for attn, ffn in zip(self.layers, self.ffns):
            x = attn(x)
            x = x + ffn(x)
        return self.head(self.norm(x))


if __name__ == "__main__":
    VOCAB, HIDDEN, LAYERS, PREFIX_LEN = 500, 128, 6, 20

    model = PrefixTunedModel(VOCAB, HIDDEN, LAYERS, prefix_len=PREFIX_LEN)
    model.freeze_base_train_prefix()

    total = sum(p.numel() for p in model.parameters())
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Total params:   {total:,}")
    print(f"Prefix params:  {trainable:,} ({100 * trainable / total:.2f}%)")
    print(f"Prefix length:  {PREFIX_LEN} soft tokens per layer")
    print(f"Layers:         {LAYERS}\n")

    data = torch.randint(0, VOCAB, (2, 32))
    logits = model(data)
    print(f"Input:  {data.shape}")
    print(f"Output: {logits.shape}")

    print(f"\nTo adapt to a new task:")
    print(f"  1. Keep all {total - trainable:,} base params frozen")
    print(f"  2. Train only {trainable:,} prefix params")
    print(f"  3. Swap prefix weights to switch tasks instantly")
