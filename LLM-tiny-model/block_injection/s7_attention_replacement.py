"""
Strategy 7 — Attention Pattern Replacement
Replace standard O(n²) full attention with more efficient patterns:
  - Sliding Window Attention (local context only)
  - Linear Attention (kernel-based O(n))

The replacement heads are trained via distillation from the originals.

FLOW:
─────────────────────────────────────────────────────────────────────────
Step 1: Profile which layers need global vs local attention patterns
Step 2: Replace selected full-attention layers with efficient variants
Step 3: Distill: train replacement to match original attention output
Step 4: Validate accuracy drop < 1%, then deploy
─────────────────────────────────────────────────────────────────────────

FULL ATTENTION DATA FLOW (original, O(n²)):

    x (B, S, H)
    │
    ▼
  [QKV projection]  Linear(H → 3H)
    │
    ├── Q (B,heads,S,d)    query
    ├── K (B,heads,S,d)    key
    └── V (B,heads,S,d)    value
         │
         ▼
    Attn = Q × Kᵀ / √d        ← O(S²) computation
         │
         ▼
    ┌───┬───┬───┬───┬───┐
    │ ■ │ ■ │ ■ │ ■ │ ■ │     every token attends
    │ ■ │ ■ │ ■ │ ■ │ ■ │     to every other token
    │ ■ │ ■ │ ■ │ ■ │ ■ │     S×S attention matrix
    │ ■ │ ■ │ ■ │ ■ │ ■ │
    │ ■ │ ■ │ ■ │ ■ │ ■ │
    └───┴───┴───┴───┴───┘
         │
         ▼
    softmax(Attn) × V → output (B,heads,S,d) → project → (B,S,H)


SLIDING WINDOW ATTENTION (replacement, O(n×W)):

    Same Q, K, V projections, but mask restricts attention:

    ┌───┬───┬───┬───┬───┐
    │ ■ │ ■ │ ■ │   │   │     each token attends only
    │ ■ │ ■ │ ■ │ ■ │   │     to W nearest neighbors
    │   │ ■ │ ■ │ ■ │ ■ │     (W = window_size)
    │   │   │ ■ │ ■ │ ■ │
    │   │   │   │ ■ │ ■ │     ■ = attended,   = masked
    └───┴───┴───┴───┴───┘

    Complexity: O(S × W)  where W ≪ S


LINEAR ATTENTION (replacement, O(n)):

    x (B, S, H)
    │
    ▼
  [QKV projection]  Linear(H → 3H)
    │
    ├── Q (B,heads,S,d)
    ├── K (B,heads,S,d)
    └── V (B,heads,S,d)
         │
         ▼
    φ(Q), φ(K)                 ← feature map: elu(x) + 1
         │
         ▼
    KV = φ(K)ᵀ × V            ← O(S × d²)  matrix, computed ONCE
         │
         ▼
    out = φ(Q) × KV            ← O(S × d²)  per query
         │
         ▼
    normalize by Σφ(K)         ← O(S × d)
         │
         ▼
    output (B,heads,S,d) → project → (B,S,H)

    Total: O(S × d²) instead of O(S²× d)
    When S ≫ d (long sequences), this is dramatically faster

BENCHMARK COMPARISON (expected):

    Seq Length    Full Attn    Sliding (W=32)    Linear
    ──────────    ─────────    ──────────────    ──────
       64          1.0×          ~1.0×           ~1.0×
      256          1.0×          ~1.2×           ~1.5×
      512          1.0×          ~1.5×           ~2.0×
     1024          1.0×          ~2.0×           ~3.0×
     4096          1.0×          ~4.0×           ~8.0×
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import math


class SlidingWindowAttention(nn.Module):
    """
    Each token attends only to its W nearest neighbors.
    Complexity: O(n × W) instead of O(n²).
    """

    def __init__(self, hidden_size: int, num_heads: int = 4, window_size: int = 64):
        super().__init__()
        self.num_heads = num_heads
        self.head_dim = hidden_size // num_heads
        self.window_size = window_size
        self.qkv = nn.Linear(hidden_size, hidden_size * 3, bias=False)
        self.out_proj = nn.Linear(hidden_size, hidden_size, bias=False)
        self.norm = nn.LayerNorm(hidden_size)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Flow: x → norm → QKV → compute attn scores → apply window mask
              → softmax → weighted V → out_proj + residual
              Only tokens within window W attend to each other → O(n×W)
        """
        residual = x
        x = self.norm(x)                                    # (B,S,H)
        B, L, D = x.shape
        H, HD = self.num_heads, self.head_dim

        qkv = self.qkv(x).reshape(B, L, 3, H, HD).permute(2, 0, 3, 1, 4)
        q, k, v = qkv[0], qkv[1], qkv[2]                   # (B,heads,S,d)

        attn = torch.matmul(q, k.transpose(-2, -1)) / math.sqrt(HD)  # (B,heads,S,S)

        # Build window mask: only allow attention within W neighbors
        mask = torch.ones(L, L, device=x.device, dtype=torch.bool)
        for i in range(L):
            start = max(0, i - self.window_size // 2)
            end = min(L, i + self.window_size // 2 + 1)
            mask[i, start:end] = False                       # False = attend
        attn = attn.masked_fill(mask.unsqueeze(0).unsqueeze(0), float("-inf"))

        attn = F.softmax(attn, dim=-1)                       # local attention only
        out = torch.matmul(attn, v)                          # (B,heads,S,d)
        out = out.transpose(1, 2).reshape(B, L, D)
        return residual + self.out_proj(out)


class LinearAttention(nn.Module):
    """
    Kernel-based linear attention: O(n) time and memory.
    Uses feature map φ(x) = elu(x) + 1 to avoid softmax.
    """

    def __init__(self, hidden_size: int, num_heads: int = 4):
        super().__init__()
        self.num_heads = num_heads
        self.head_dim = hidden_size // num_heads
        self.qkv = nn.Linear(hidden_size, hidden_size * 3, bias=False)
        self.out_proj = nn.Linear(hidden_size, hidden_size, bias=False)
        self.norm = nn.LayerNorm(hidden_size)

    @staticmethod
    def _feature_map(x: torch.Tensor) -> torch.Tensor:
        return F.elu(x) + 1

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Flow: x → norm → QKV → feature_map φ(Q), φ(K)
              → KV = φ(K)ᵀ × V   (compute once, O(S·d²))
              → out = φ(Q) × KV   (per query, O(d²))
              → normalize → out_proj + residual
              Total: O(S·d²) instead of O(S²·d)
        """
        residual = x
        x = self.norm(x)                                    # (B,S,H)
        B, L, D = x.shape
        H, HD = self.num_heads, self.head_dim

        qkv = self.qkv(x).reshape(B, L, 3, H, HD).permute(2, 0, 3, 1, 4)
        q, k, v = qkv[0], qkv[1], qkv[2]                   # (B,heads,S,d)

        q = self._feature_map(q)                             # φ(Q) = elu(Q) + 1
        k = self._feature_map(k)                             # φ(K) = elu(K) + 1

        kv = torch.matmul(k.transpose(-2, -1), v)           # φ(K)ᵀ×V: (B,H,d,d) — O(S·d²)
        qkv_out = torch.matmul(q, kv)                       # φ(Q)×KV: (B,H,S,d) — O(S·d²)
        denom = torch.matmul(q, k.sum(dim=-2, keepdim=True).transpose(-2, -1)) + 1e-6
        out = qkv_out / denom                                # normalize

        out = out.transpose(1, 2).reshape(B, L, D)
        return residual + self.out_proj(out)


class FullAttention(nn.Module):
    """Standard O(n²) attention for comparison."""

    def __init__(self, hidden_size: int, num_heads: int = 4):
        super().__init__()
        self.attn = nn.MultiheadAttention(hidden_size, num_heads, batch_first=True)
        self.norm = nn.LayerNorm(hidden_size)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        normed = self.norm(x)
        return x + self.attn(normed, normed, normed)[0]


def benchmark_attention(attn_module, x, label, warmup=5, runs=50):
    """Measure average forward pass time."""
    import time
    for _ in range(warmup):
        attn_module(x)
    t0 = time.perf_counter()
    for _ in range(runs):
        with torch.no_grad():
            attn_module(x)
    elapsed = (time.perf_counter() - t0) / runs
    print(f"  {label:30s}  {elapsed * 1000:.2f} ms")
    return elapsed


if __name__ == "__main__":
    HIDDEN, HEADS = 128, 4

    full_attn = FullAttention(HIDDEN, HEADS)
    sliding = SlidingWindowAttention(HIDDEN, HEADS, window_size=32)
    linear = LinearAttention(HIDDEN, HEADS)

    print("Parameter counts:")
    for name, m in [("Full Attention", full_attn), ("Sliding Window", sliding), ("Linear Attention", linear)]:
        p = sum(p.numel() for p in m.parameters())
        print(f"  {name:30s}  {p:,}")

    for seq_len in [64, 256, 512, 1024]:
        print(f"\nSequence length: {seq_len}")
        x = torch.randn(2, seq_len, HIDDEN)
        t_full = benchmark_attention(full_attn, x, "Full Attention O(n²)")
        t_slide = benchmark_attention(sliding, x, f"Sliding Window (W=32)")
        t_lin = benchmark_attention(linear, x, "Linear Attention O(n)")
        print(f"  Sliding speedup: {t_full / t_slide:.2f}×")
        print(f"  Linear  speedup: {t_full / t_lin:.2f}×")
