"""
Strategy 3 — SSM / Mamba Block Replacement
Replace selected attention layers with a simplified Mamba-like SSM block
for O(n) inference instead of O(n²) attention.

The SSM blocks are trained via layer-wise distillation from the
original attention layers they replace.

FLOW:
─────────────────────────────────────────────────────────────────────────
Step 1: Identify which attention layers to replace (e.g., alternating)
Step 2: Create SSM (Mamba) blocks with matching hidden_size
Step 3: Distill: train each SSM to mimic its replaced attention layer
           minimize MSE(attention_output, ssm_output)
Step 4: Swap in the trained SSM blocks, discard old attention layers
Step 5: Deploy — hybrid model with O(n) SSM + O(n²) attention mix
─────────────────────────────────────────────────────────────────────────

LAYER REPLACEMENT FLOW:

    Before (all attention):             After (hybrid):
    ──────────────────────              ───────────────
    Layer 0: [Attention] O(n²)          Layer 0: [SSM]       O(n)  ← replaced
    Layer 1: [Attention] O(n²)          Layer 1: [Attention]  O(n²) ← kept
    Layer 2: [Attention] O(n²)          Layer 2: [SSM]       O(n)  ← replaced
    Layer 3: [Attention] O(n²)          Layer 3: [Attention]  O(n²) ← kept

SSM BLOCK DATA FLOW (replaces one attention layer):

    x  (B, S, H)                       ← input hidden states
    │
    ├── residual ──────────────────────────────┐
    │                                          │
    ▼                                          │
  [LayerNorm]                                  │
    │                                          │
    ▼                                          │
  [in_proj]  Linear (H → 2·inner)              │
    │                                          │
    ├──── x_branch (B,S,inner) ──┐             │
    │                            │             │
    │                       z (B,S,inner)      │
    ▼                            │             │
  [Conv1d]  (k=3, groups)       │             │
    │                            │             │
    ▼                            │             │
  SiLU activation                │             │
    │                            │             │
    ├─► [dt_proj] → Δt           │             │
    ├─► [B_proj]  → B            │             │
    ├─► [C_proj]  → C            │             │
    │                            │             │
    ▼                            │             │
  ┌─────────────────────────┐   │             │
  │  Selective Scan  O(n)   │   │             │
  │                         │   │             │
  │  for t = 0..L:          │   │             │
  │    Ā = exp(Δt · A)      │   │             │
  │    B̄ = Δt · B           │   │             │
  │    h = Ā·h + B̄·x[t]    │   │ state: O(1) │
  │    y[t] = (h · C).sum() │   │             │
  └──────────┬──────────────┘   │             │
             │                   │             │
             ▼                   │             │
       y + D·x_branch           │             │
             │                   │             │
             ▼                   ▼             │
          y * SiLU(z)     (gating)             │
             │                                 │
             ▼                                 │
          [out_proj]  Linear (inner → H)       │
             │                                 │
             ▼                                 ▼
          residual  +  ssm_output  ──────►  out (B,S,H)

DISTILLATION FLOW (per replaced layer):

    Training data (random/real sequences)
         │
         ├───────────────────────────────────┐
         ▼                                   ▼
    ┌──────────────┐                 ┌──────────────┐
    │  Attention   │  (frozen)       │  SSM Block   │  (trainable)
    │  (teacher)   │                 │  (student)   │
    └──────┬───────┘                 └──────┬───────┘
           │                                │
           ▼                                ▼
    teacher_out (B,S,H)            student_out (B,S,H)
           │                                │
           └─────── MSE Loss ───────────────┘
                       │
                 Backprop into SSM only
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class SimpleSSMBlock(nn.Module):
    """
    Simplified selective state-space block inspired by Mamba.
    Processes sequences in O(n) time with a recurrent scan.

    This is a teaching implementation — real Mamba uses CUDA kernels for speed.
    """

    def __init__(self, hidden_size: int, state_size: int = 16, expand: int = 2):
        super().__init__()
        inner = hidden_size * expand

        self.in_proj = nn.Linear(hidden_size, inner * 2, bias=False)
        self.conv1d = nn.Conv1d(inner, inner, kernel_size=3, padding=1, groups=inner)

        self.dt_proj = nn.Linear(inner, inner, bias=True)
        self.A_log = nn.Parameter(torch.randn(inner, state_size))
        self.D = nn.Parameter(torch.ones(inner))

        self.B_proj = nn.Linear(inner, state_size, bias=False)
        self.C_proj = nn.Linear(inner, state_size, bias=False)

        self.out_proj = nn.Linear(inner, hidden_size, bias=False)
        self.norm = nn.LayerNorm(hidden_size)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Flow: x → norm → in_proj → split(x_branch, z)
              x_branch → conv1d → SiLU → compute dt,B,C → selective_scan → y
              y = (y + D·x_branch) * SiLU(z) → out_proj → + residual
        """
        residual = x
        x = self.norm(x)                                    # (B,L,H) normalize
        B, L, D = x.shape

        xz = self.in_proj(x)                                # (B,L,H) → (B,L,2·inner)
        x_branch, z = xz.chunk(2, dim=-1)                   # split into x_branch & gate

        x_branch = x_branch.transpose(1, 2)                 # (B,inner,L) for conv1d
        x_branch = self.conv1d(x_branch).transpose(1, 2)    # local context mixing
        x_branch = F.silu(x_branch)                          # activation

        dt = F.softplus(self.dt_proj(x_branch))              # step size Δt
        A = -torch.exp(self.A_log)                           # state decay matrix
        B_input = self.B_proj(x_branch)                      # input-dependent B
        C_input = self.C_proj(x_branch)                      # input-dependent C

        y = self._selective_scan(x_branch, dt, A, B_input, C_input)  # O(n) recurrence
        y = y + self.D.unsqueeze(0).unsqueeze(0) * x_branch  # skip connection

        y = y * F.silu(z)                                    # gating with z branch
        return residual + self.out_proj(y)                   # project back + residual

    def _selective_scan(self, x, dt, A, B, C):
        """Simplified sequential scan (O(n) per sequence step)."""
        B_size, L, D = x.shape
        N = A.shape[1]

        h = torch.zeros(B_size, D, N, device=x.device, dtype=x.dtype)
        outputs = []

        for t in range(L):
            dt_t = dt[:, t, :].unsqueeze(-1)           # (B, D, 1)
            A_bar = torch.exp(dt_t * A.unsqueeze(0))    # (B, D, N)
            B_bar = dt_t * B[:, t, :].unsqueeze(1)      # (B, D, N)

            h = A_bar * h + B_bar * x[:, t, :].unsqueeze(-1)
            y_t = (h * C[:, t, :].unsqueeze(1)).sum(dim=-1)
            outputs.append(y_t)

        return torch.stack(outputs, dim=1)


class HybridTransformerSSM(nn.Module):
    """
    Toy model demonstrating attention/SSM alternation.
    Even-indexed layers use SSM, odd-indexed use attention.
    """

    def __init__(self, hidden: int, num_layers: int = 4, nhead: int = 4, state_size: int = 16):
        super().__init__()
        self.layers = nn.ModuleList()
        for i in range(num_layers):
            if i % 2 == 0:
                self.layers.append(SimpleSSMBlock(hidden, state_size=state_size))
            else:
                attn = nn.TransformerEncoderLayer(
                    d_model=hidden, nhead=nhead, dim_feedforward=hidden * 4, batch_first=True
                )
                self.layers.append(attn)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        for layer in self.layers:
            x = layer(x)
        return x


def distill_attention_to_ssm(
    teacher_attn: nn.Module,
    ssm_block: SimpleSSMBlock,
    data_loader,
    epochs: int = 3,
    lr: float = 1e-3,
):
    """Train SSM block to mimic attention layer output (layer-wise distillation)."""
    optimizer = torch.optim.Adam(ssm_block.parameters(), lr=lr)
    teacher_attn.eval()

    for epoch in range(epochs):
        total_loss = 0
        for batch in data_loader:
            with torch.no_grad():
                teacher_out = teacher_attn(batch)
                if isinstance(teacher_out, tuple):
                    teacher_out = teacher_out[0]
            student_out = ssm_block(batch)
            loss = F.mse_loss(student_out, teacher_out)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
        print(f"  Epoch {epoch + 1}/{epochs}  Loss: {total_loss / len(data_loader):.6f}")


if __name__ == "__main__":
    HIDDEN, LAYERS, SEQ, BATCH = 128, 4, 64, 4

    model = HybridTransformerSSM(HIDDEN, num_layers=LAYERS)
    print("Hybrid Transformer + SSM model:")
    for i, layer in enumerate(model.layers):
        kind = "SSM (O(n))" if isinstance(layer, SimpleSSMBlock) else "Attention (O(n²))"
        print(f"  Layer {i}: {kind}")

    x = torch.randn(BATCH, SEQ, HIDDEN)
    out = model(x)
    print(f"\nInput:  {x.shape}")
    print(f"Output: {out.shape}")

    total = sum(p.numel() for p in model.parameters())
    print(f"Total params: {total:,}")
