"""
Strategy 6 — Dynamic Layer Skipping
Add a tiny skip predictor before each layer that decides whether
to skip the layer for the current input.

Easy inputs skip many layers → fast. Hard inputs use all → accurate.

FLOW:
─────────────────────────────────────────────────────────────────────────
Step 1: Freeze all transformer layers
Step 2: Attach a tiny skip predictor (MLP: H→32→1) before each layer
Step 3: Train predictors with loss = distill_loss + λ × CE_loss
           (balance accuracy preservation vs compute savings)
Step 4: Deploy — each input dynamically chooses which layers to use
─────────────────────────────────────────────────────────────────────────

DATA FLOW (inference):

    input_ids (B, S)
        │
        ▼
    [Embedding]  (frozen)
        │
        ▼
    x (B, S, H)
        │
        ▼
    ╔═════════════════════════════════════╗
    ║  Skip Predictor for Layer 0        ║  ← TRAINABLE
    ║                                    ║
    ║  x.mean(dim=1) → (B, H)           ║  pool over sequence
    ║       │                            ║
    ║       ▼                            ║
    ║  [Linear H→32] → ReLU → [32→1]    ║
    ║       │                            ║
    ║       ▼                            ║
    ║  sigmoid → skip_prob (B,)          ║
    ║       │                            ║
    ║  skip_prob > threshold?            ║
    ║       │           │                ║
    ║      YES         NO               ║
    ╚═══════╪═══════════╪════════════════╝
            │           │
            ▼           ▼
    ┌──── SKIP ──┐  ┌── EXECUTE ──────────┐
    │ x passes   │  │ x = Layer_0(x)      │
    │ unchanged  │  │ (frozen transformer │
    │            │  │  layer runs)        │
    └─────┬──────┘  └──────┬──────────────┘
          │                │
          └──────┬─────────┘
                 │
                 ▼
    ╔═════════════════════════════════════╗
    ║  Skip Predictor for Layer 1        ║
    ║  (same logic)                      ║
    ╚═════════════════════════════════════╝
                 │
                 ▼
               ...  (repeat for all N layers)
                 │
                 ▼
    [LayerNorm + LM Head]  (frozen)
                 │
                 ▼
    logits (B, S, V)

EXAMPLE TRACES:

    Easy input: "Hi, how are you?"
    ─────────────────────────────────
    Layer 0: EXEC     Layer 4: SKIP     Layer  8: SKIP
    Layer 1: SKIP     Layer 5: SKIP     Layer  9: SKIP
    Layer 2: EXEC     Layer 6: EXEC     Layer 10: SKIP
    Layer 3: SKIP     Layer 7: SKIP     Layer 11: EXEC
    → Used 4 of 12 layers = 3× faster

    Hard input: "Derive the quadratic formula"
    ─────────────────────────────────────────
    Layer 0: EXEC     Layer 4: EXEC     Layer  8: EXEC
    Layer 1: EXEC     Layer 5: EXEC     Layer  9: EXEC
    Layer 2: EXEC     Layer 6: EXEC     Layer 10: EXEC
    Layer 3: EXEC     Layer 7: EXEC     Layer 11: EXEC
    → Used 12 of 12 layers = full accuracy

TRAINING LOSS:

    loss = KL_div(logits_skip, logits_full)   ← match full-model output
         + λ × CE(logits_skip, target)         ← maintain task accuracy
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import time


class SkipPredictor(nn.Module):
    """Binary classifier: should this layer be skipped for this input?"""

    def __init__(self, hidden_size: int):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(hidden_size, 32),
            nn.ReLU(),
            nn.Linear(32, 1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        pooled = x.mean(dim=1)
        return torch.sigmoid(self.net(pooled)).squeeze(-1)


class SkippableTransformer(nn.Module):
    """
    Transformer where each layer has a skip predictor.
    During inference, layers with skip probability > threshold are skipped.
    """

    def __init__(
        self,
        vocab_size: int,
        hidden: int,
        num_layers: int = 12,
        nhead: int = 4,
        skip_threshold: float = 0.5,
    ):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, hidden)
        self.layers = nn.ModuleList([
            nn.TransformerEncoderLayer(
                d_model=hidden, nhead=nhead, dim_feedforward=hidden * 4, batch_first=True
            )
            for _ in range(num_layers)
        ])
        self.skip_predictors = nn.ModuleList([
            SkipPredictor(hidden) for _ in range(num_layers)
        ])
        self.norm = nn.LayerNorm(hidden)
        self.head = nn.Linear(hidden, vocab_size, bias=False)
        self.skip_threshold = skip_threshold

    def forward(self, input_ids: torch.Tensor, enable_skipping: bool = True):
        """
        Flow per layer:
          x → skip_predictor → skip_prob
          if skip_prob > threshold: SKIP (x unchanged)
          else: x = transformer_layer(x)  (EXEC)
        """
        x = self.embedding(input_ids)                        # (B,S) → (B,S,H)
        layers_used = []

        for i, (layer, skip_pred) in enumerate(zip(self.layers, self.skip_predictors)):
            if enable_skipping:
                skip_prob = skip_pred(x.detach())            # pool → MLP → sigmoid
                should_skip = (skip_prob > self.skip_threshold).all()
                if should_skip:
                    layers_used.append(("SKIP", i))          # skip: x passes through
                    continue

            x = layer(x)                                     # exec: frozen transformer
            layers_used.append(("EXEC", i))

        return self.head(self.norm(x)), layers_used

    def freeze_base_train_skippers(self):
        """Freeze transformer layers, train only skip predictors."""
        for p in self.parameters():
            p.requires_grad = False
        for sp in self.skip_predictors:
            for p in sp.parameters():
                p.requires_grad = True

    def compute_skip_loss(self, logits_full, logits_skip, target, lam: float = 0.5):
        """
        Combined loss:  accuracy_loss + λ * compute_cost
        Encourages skipping while maintaining accuracy.
        """
        ce_loss = F.cross_entropy(logits_skip.view(-1, logits_skip.size(-1)), target.view(-1))
        distill_loss = F.kl_div(
            F.log_softmax(logits_skip, dim=-1),
            F.softmax(logits_full.detach(), dim=-1),
            reduction="batchmean",
        )
        return distill_loss + lam * ce_loss


if __name__ == "__main__":
    VOCAB, HIDDEN, LAYERS = 500, 128, 12

    model = SkippableTransformer(VOCAB, HIDDEN, LAYERS, skip_threshold=0.5)
    model.freeze_base_train_skippers()

    total = sum(p.numel() for p in model.parameters())
    skipper_params = sum(p.numel() for sp in model.skip_predictors for p in sp.parameters())
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Total params:   {total:,}")
    print(f"Skipper params: {skipper_params:,} ({100 * skipper_params / total:.3f}%)")
    print(f"Trainable:      {trainable:,}\n")

    data = torch.randint(0, VOCAB, (2, 32))

    logits_full, trace_full = model(data, enable_skipping=False)
    print("Full execution:")
    print(f"  Layers used: {len(trace_full)} / {LAYERS}\n")

    logits_skip, trace_skip = model(data, enable_skipping=True)
    executed = sum(1 for kind, _ in trace_skip if kind == "EXEC")
    skipped = sum(1 for kind, _ in trace_skip if kind == "SKIP")
    print("With skipping:")
    print(f"  Executed: {executed}, Skipped: {skipped}")
    for kind, idx in trace_skip:
        print(f"    Layer {idx:2d}: {kind}")
