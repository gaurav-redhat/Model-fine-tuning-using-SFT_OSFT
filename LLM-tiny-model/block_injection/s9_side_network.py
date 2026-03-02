"""
Strategy 9 — Side Networks / Ladder Side-Tuning
Attach a small parallel network alongside the frozen main model.
The side network injects additive corrections at each layer.

FLOW:
─────────────────────────────────────────────────────────────────────────
Step 1: Freeze entire main model
Step 2: Build a side network (~10% of main model size)
           side_hidden = main_hidden / 4  (e.g., 256 → 64)
Step 3: At each layer, side network reads main hidden state, computes
           a correction, and adds it back
Step 4: Train ONLY side network; remove it to revert to original model
─────────────────────────────────────────────────────────────────────────

DATA FLOW (per layer i):

    Main path (frozen):                Side path (trainable):

    main_hidden (B,S,H)               side_hidden (B,S,h) or None
         │                                   │
         ▼                                   │
    ┌──────────────────┐                    │
    │  Main Layer i    │ (frozen)            │
    │  (full-size      │                    │
    │   transformer)   │                    │
    └────────┬─────────┘                    │
             │                               │
             ▼                               ▼
    main_out (B,S,H) ────────────► ┌──────────────────────────┐
             │                     │  Side Layer i (trainable) │
             │                     │                          │
             │                     │  [Down-project] H → h    │
             │                     │       │                  │
             │                     │       ▼                  │
             │                     │  side = prev_side + down │
             │                     │       │                  │
             │                     │       ▼                  │
             │                     │  [Small Transformer]     │
             │                     │  (h-dim, 2 heads)        │
             │                     │       │                  │
             │                     │       ▼                  │
             │                     │  [Up-project] h → H      │
             │                     │       │                  │
             │                     │       ▼                  │
             │                     │  correction = tanh(gate) │
             │                     │              × up_output │
             │                     └──────┬───────────────────┘
             │                            │
             │          correction (B,S,H)│  new_side (B,S,h)
             │                ┌───────────┘        │
             │                │                    │
             ▼                ▼                    ▼
    main_out + correction = enhanced_out     → next Side Layer
             │
             ▼
       (to next Main Layer)

FULL MODEL FLOW:

    Main Network (FROZEN, H=256)        Side Network (TRAINABLE, h=64)
    ════════════════════════════        ═══════════════════════════════

    [Embedding] (B,S) → (B,S,256)
         │
         ▼                               (side_hidden = None initially)
    [Main Layer 0] ───── read ──────► [Side Layer 0]
         │ ◄──────── + correction ──────── │
         ▼                                 ▼
    [Main Layer 1] ───── read ──────► [Side Layer 1]
         │ ◄──────── + correction ──────── │
         ▼                                 ▼
    [Main Layer 2] ───── read ──────► [Side Layer 2]
         │ ◄──────── + correction ──────── │
         ▼                                 ▼
        ...                              ...
         │                                 │
    [Main Layer 7] ───── read ──────► [Side Layer 7]
         │ ◄──────── + correction ──────── │
         ▼
    [Norm + Head] (frozen)
         │
         ▼
    logits (B, S, V)

    ★ Gate initialized at 0 → corrections start near-zero
    ★ Gradually learns task-specific corrections during training
    ★ Remove side network → exact original model behavior
"""

import torch
import torch.nn as nn


class SideLayer(nn.Module):
    """One layer of the side network: smaller transformer + projection."""

    def __init__(self, main_hidden: int, side_hidden: int, nhead: int = 2):
        super().__init__()
        self.down = nn.Linear(main_hidden, side_hidden)
        self.transformer = nn.TransformerEncoderLayer(
            d_model=side_hidden, nhead=nhead, dim_feedforward=side_hidden * 2, batch_first=True
        )
        self.up = nn.Linear(side_hidden, main_hidden)
        self.gate = nn.Parameter(torch.zeros(1))

    def forward(self, main_hidden: torch.Tensor, side_hidden: torch.Tensor | None = None):
        if side_hidden is None:
            side_hidden = self.down(main_hidden)
        else:
            side_hidden = side_hidden + self.down(main_hidden)
        side_hidden = self.transformer(side_hidden)
        correction = torch.tanh(self.gate) * self.up(side_hidden)
        return correction, side_hidden


class ModelWithSideNetwork(nn.Module):
    """
    Frozen main transformer + trainable side network.
    Side network is ~10% the size of the main model.
    """

    def __init__(
        self,
        vocab_size: int,
        main_hidden: int,
        side_hidden: int,
        num_layers: int = 8,
        nhead_main: int = 4,
        nhead_side: int = 2,
    ):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, main_hidden)
        self.main_layers = nn.ModuleList([
            nn.TransformerEncoderLayer(
                d_model=main_hidden, nhead=nhead_main,
                dim_feedforward=main_hidden * 4, batch_first=True,
            )
            for _ in range(num_layers)
        ])
        self.norm = nn.LayerNorm(main_hidden)
        self.head = nn.Linear(vocab_size, main_hidden, bias=False)

        self.side_layers = nn.ModuleList([
            SideLayer(main_hidden, side_hidden, nhead_side)
            for _ in range(num_layers)
        ])

    def freeze_main(self):
        for p in self.embedding.parameters():
            p.requires_grad = False
        for layer in self.main_layers:
            for p in layer.parameters():
                p.requires_grad = False
        for p in self.norm.parameters():
            p.requires_grad = False
        for p in self.head.parameters():
            p.requires_grad = False

    def forward(self, input_ids: torch.Tensor, use_side: bool = True):
        """
        Flow per layer:
          main: x = main_layer(x)                      ← frozen
          side: correction, side_h = side_layer(x, side_h) ← trainable
          merge: x = x + correction
        """
        x = self.embedding(input_ids)                        # (B,S) → (B,S,H_main)
        side_h = None                                        # side state starts empty

        for i, main_layer in enumerate(self.main_layers):
            x = main_layer(x)                                # frozen transformer layer
            if use_side:
                correction, side_h = self.side_layers[i](x, side_h)  # side: down→TF→up
                x = x + correction                           # additive correction

        return self.head(self.norm(x))                       # (B,S,V) logits


if __name__ == "__main__":
    VOCAB, MAIN_H, SIDE_H, LAYERS = 500, 256, 64, 8

    model = ModelWithSideNetwork(VOCAB, MAIN_H, SIDE_H, LAYERS)
    model.freeze_main()

    total = sum(p.numel() for p in model.parameters())
    main_params = sum(
        p.numel()
        for name, p in model.named_parameters()
        if not name.startswith("side_")
    )
    side_params = sum(
        p.numel()
        for name, p in model.named_parameters()
        if name.startswith("side_")
    )
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)

    print(f"Main model params:  {main_params:,}")
    print(f"Side network params: {side_params:,} ({100 * side_params / main_params:.1f}% of main)")
    print(f"Trainable:           {trainable:,}\n")

    data = torch.randint(0, VOCAB, (2, 32))

    out_base = model(data, use_side=False)
    print(f"Without side network: {out_base.shape}")

    out_side = model(data, use_side=True)
    print(f"With    side network: {out_side.shape}")

    diff = (out_side - out_base).abs().mean().item()
    print(f"Output difference:    {diff:.6f}")
    print(f"(Small diff = gate initialized near 0; grows after training)")
