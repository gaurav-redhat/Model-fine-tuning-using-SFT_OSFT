"""
Strategy 4 — Early Exit Layers
Add lightweight classifier heads at intermediate layers so easy inputs
can exit early without traversing the full network.

FLOW:
─────────────────────────────────────────────────────────────────────────
Step 1: Take pre-trained model (freeze all layers)
Step 2: Attach small exit heads (LayerNorm + Linear) at chosen layers
Step 3: Train ONLY exit heads to predict final output from intermediate hidden states
Step 4: Deploy — at inference, exit early when confidence exceeds threshold
─────────────────────────────────────────────────────────────────────────

DATA FLOW (inference):

    input_ids (B, S)
        │
        ▼
    [Embedding]  (frozen)
        │
        ▼
    ┌─────────────────┐
    │  Layer 0         │  (frozen)
    └────────┬────────┘
             │
             ▼
    ┌─────────────────┐
    │  Layer 1         │  (frozen)
    └────────┬────────┘
             │
             ▼
    ┌─────────────────┐
    │  Layer 2         │  (frozen)
    └────────┬────────┘
             │
             ▼
    ┌─────────────────┐
    │  Layer 3         │  (frozen)
    └────────┬────────┘
             │
             ▼
    ╔═══════════════════════════╗
    ║  EXIT HEAD @ Layer 3      ║  ← NEW (trainable)
    ║                           ║
    ║  hidden (B,S,H)           ║
    ║     │                     ║
    ║     ▼                     ║
    ║  [LayerNorm]              ║
    ║     │                     ║
    ║     ▼                     ║
    ║  [Linear] (H → Vocab)     ║
    ║     │                     ║
    ║     ▼                     ║
    ║  logits (B,S,V)           ║
    ║     │                     ║
    ║     ▼                     ║
    ║  confidence = max(softmax)║
    ║     │                     ║
    ║     ├── conf > threshold? ║
    ║     │      │              ║
    ║     │     YES → EXIT ✓    ║──► return logits (skipped layers 4..N)
    ║     │      │              ║
    ║     │     NO → continue ↓ ║
    ╚═══════════════════════════╝
             │
             ▼
    ┌─────────────────┐
    │  Layer 4         │  ... continues
    └────────┬────────┘
             │
            ...
             │
             ▼
    ┌─────────────────┐
    │  Layer 7         │  (frozen)
    └────────┬────────┘
             │
             ▼
    ╔═══════════════════════════╗
    ║  EXIT HEAD @ Layer 7      ║  ← another exit checkpoint
    ║  (same logic as above)    ║
    ╚═══════════════════════════╝
             │
            ...
             │
             ▼
    ┌─────────────────┐
    │  Layer N (final) │  (frozen)
    └────────┬────────┘
             │
             ▼
    [Final LM Head]  (frozen)
             │
             ▼
    logits (B, S, V)           ← only reached if NO exit triggered

SPEEDUP LOGIC:

    Easy input ("The capital of France is"):
        Layer 0 → 1 → 2 → 3 → EXIT ✓  (used 4 of 12 layers = 3× faster)

    Hard input ("Prove that √2 is irrational"):
        Layer 0 → 1 → ... → 11 → Final Head  (used all 12 layers)

    Average: ~1.5–3× faster depending on input difficulty mix
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class EarlyExitHead(nn.Module):
    """Small projection head that predicts output + confidence score."""

    def __init__(self, hidden_size: int, vocab_size: int):
        super().__init__()
        self.norm = nn.LayerNorm(hidden_size)
        self.lm_head = nn.Linear(hidden_size, vocab_size, bias=False)

    def forward(self, hidden: torch.Tensor):
        logits = self.lm_head(self.norm(hidden))
        confidence = torch.max(F.softmax(logits, dim=-1), dim=-1).values.mean()
        return logits, confidence


class TransformerWithEarlyExit(nn.Module):
    """
    Toy transformer with early-exit heads at configurable layer positions.
    During inference, exits as soon as confidence exceeds threshold.
    """

    def __init__(
        self,
        hidden: int,
        vocab_size: int,
        num_layers: int = 12,
        nhead: int = 4,
        exit_layers: list | None = None,
        confidence_threshold: float = 0.8,
    ):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, hidden)
        self.layers = nn.ModuleList([
            nn.TransformerEncoderLayer(
                d_model=hidden, nhead=nhead, dim_feedforward=hidden * 4, batch_first=True
            )
            for _ in range(num_layers)
        ])
        self.final_norm = nn.LayerNorm(hidden)
        self.final_head = nn.Linear(hidden, vocab_size, bias=False)

        self.exit_layers = exit_layers or [num_layers // 3, 2 * num_layers // 3]
        self.exit_heads = nn.ModuleDict({
            str(i): EarlyExitHead(hidden, vocab_size) for i in self.exit_layers
        })
        self.threshold = confidence_threshold

    def forward(self, input_ids: torch.Tensor, allow_early_exit: bool = True):
        """
        Flow: embed → layer_0 → layer_1 → ... → [exit check] → ... → final_head
              At each exit point: compute logits + confidence
              If confidence > threshold → return early (skip remaining layers)
        """
        x = self.embedding(input_ids)                        # (B,S) → (B,S,H)
        exit_layer = len(self.layers)

        for i, layer in enumerate(self.layers):
            x = layer(x)                                     # frozen transformer layer

            if allow_early_exit and str(i) in self.exit_heads:
                logits, conf = self.exit_heads[str(i)](x)    # exit head: norm → linear
                if conf > self.threshold:                    # confident enough?
                    exit_layer = i                           # YES → exit early!
                    return logits, exit_layer

        logits = self.final_head(self.final_norm(x))         # reached final layer
        return logits, exit_layer

    def train_exit_heads_only(self):
        """Freeze everything except exit heads."""
        for p in self.parameters():
            p.requires_grad = False
        for head in self.exit_heads.values():
            for p in head.parameters():
                p.requires_grad = True


def compute_speedup(model, data, num_runs: int = 100):
    """Estimate average speedup from early exits."""
    import time

    model.eval()
    with torch.no_grad():
        t0 = time.perf_counter()
        for _ in range(num_runs):
            model(data, allow_early_exit=False)
        full_time = time.perf_counter() - t0

        t0 = time.perf_counter()
        exit_counts = {}
        for _ in range(num_runs):
            _, exit_layer = model(data, allow_early_exit=True)
            exit_counts[exit_layer] = exit_counts.get(exit_layer, 0) + 1
        early_time = time.perf_counter() - t0

    return full_time / early_time, exit_counts


if __name__ == "__main__":
    HIDDEN, VOCAB, LAYERS, SEQ, BATCH = 128, 1000, 12, 32, 4

    model = TransformerWithEarlyExit(
        hidden=HIDDEN,
        vocab_size=VOCAB,
        num_layers=LAYERS,
        exit_layers=[3, 7],
        confidence_threshold=0.5,
    )

    total = sum(p.numel() for p in model.parameters())
    exit_params = sum(p.numel() for h in model.exit_heads.values() for p in h.parameters())
    print(f"Total params: {total:,}")
    print(f"Exit head params: {exit_params:,} ({100 * exit_params / total:.2f}%)")
    print(f"Exit points at layers: {model.exit_layers}")

    model.train_exit_heads_only()
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Trainable (exit heads only): {trainable:,}\n")

    data = torch.randint(0, VOCAB, (BATCH, SEQ))
    logits, exited_at = model(data, allow_early_exit=True)
    print(f"Input shape:   {data.shape}")
    print(f"Output shape:  {logits.shape}")
    print(f"Exited at layer: {exited_at} / {LAYERS}")
