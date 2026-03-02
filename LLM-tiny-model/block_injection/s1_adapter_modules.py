"""
Strategy 1 — Adapter Modules
Inject small bottleneck layers between frozen transformer layers.

Only the adapter parameters are trained (~0.5–3% of total).

FLOW:
─────────────────────────────────────────────────────────────────────────
Step 1: Load pre-trained model (all weights frozen)
Step 2: For EACH transformer layer, wrap it with an AdapterBlock
Step 3: Train ONLY adapter weights on downstream task
Step 4: Deploy — original model intact + tiny adapter add-ons
─────────────────────────────────────────────────────────────────────────

DATA FLOW (per layer):

    x  (B, S, H)              ← input hidden states
    │
    ▼
  ┌───────────────────────┐
  │  Original Transformer │   ← FROZEN (attn + ffn + norms)
  │  Layer (unchanged)    │
  └───────────┬───────────┘
              │
              ▼
    h  (B, S, H)              ← layer output
    │
    ▼
  ┌───────────────────────┐
  │  ADAPTER BLOCK (NEW)  │   ← TRAINABLE
  │                       │
  │  h ──► Down (H→r)     │   Linear: (B,S,H) → (B,S,r)    r≪H
  │         │              │
  │         ▼              │
  │       ReLU             │   Nonlinearity
  │         │              │
  │         ▼              │
  │       Up (r→H)         │   Linear: (B,S,r) → (B,S,H)
  │         │              │
  │         ▼              │
  │    x + Up(ReLU(Down))  │   Residual connection
  └───────────┬───────────┘
              │
              ▼
    out (B, S, H)             ← adapter-enhanced output

    H = hidden_size (e.g., 768)
    r = bottleneck  (e.g., 64)
    Adapter params per layer = 2 × H × r ≈ 98K  (vs ~7M in full layer)

FULL MODEL FLOW:

    Input IDs
      │
      ▼
    [Embedding]                     ← frozen
      │
      ▼
    [Layer 0 (frozen)] → [Adapter 0 (trainable)]
      │
      ▼
    [Layer 1 (frozen)] → [Adapter 1 (trainable)]
      │
      ▼
    ...
      │
      ▼
    [Layer N (frozen)] → [Adapter N (trainable)]
      │
      ▼
    [LM Head]                       ← frozen
      │
      ▼
    Logits
"""

import torch
import torch.nn as nn
from transformers import AutoModelForCausalLM, AutoTokenizer


class AdapterBlock(nn.Module):
    """Bottleneck adapter: down-project → ReLU → up-project + residual."""

    def __init__(self, hidden_size: int, bottleneck_size: int = 64):
        super().__init__()
        self.down = nn.Linear(hidden_size, bottleneck_size)
        self.act = nn.ReLU()
        self.up = nn.Linear(bottleneck_size, hidden_size)
        nn.init.zeros_(self.up.weight)
        nn.init.zeros_(self.up.bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x (B,S,H) → down (B,S,r) → ReLU → up (B,S,H) → + residual
        return x + self.up(self.act(self.down(x)))


class TransformerLayerWithAdapter(nn.Module):
    """Wraps a frozen transformer layer and injects an adapter after it."""

    def __init__(self, original_layer: nn.Module, hidden_size: int, bottleneck: int = 64):
        super().__init__()
        self.original_layer = original_layer
        self.adapter = AdapterBlock(hidden_size, bottleneck)

        for p in self.original_layer.parameters():
            p.requires_grad = False

    def forward(self, *args, **kwargs):
        # Flow: input → frozen_layer → adapter → output
        out = self.original_layer(*args, **kwargs)           # frozen forward
        hidden = out[0] if isinstance(out, tuple) else out   # extract hidden
        hidden = self.adapter(hidden)                        # adapter: down→ReLU→up+res
        return (hidden, *out[1:]) if isinstance(out, tuple) else hidden


def inject_adapters(model, bottleneck: int = 64):
    """Inject adapter blocks into every transformer layer of a HuggingFace model."""
    config = model.config
    hidden_size = config.hidden_size

    layers = None
    if hasattr(model.model, "layers"):
        layers = model.model.layers
    elif hasattr(model.transformer, "h"):
        layers = model.transformer.h

    if layers is None:
        raise ValueError("Cannot locate transformer layers in this model architecture")

    for i in range(len(layers)):
        layers[i] = TransformerLayerWithAdapter(layers[i], hidden_size, bottleneck)

    return model


def count_params(model):
    total = sum(p.numel() for p in model.parameters())
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    return total, trainable


# ---------------------------------------------------------------------------
# Demo
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    MODEL_NAME = "microsoft/phi-2"  # swap for any HF model

    print(f"Loading {MODEL_NAME}...")
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME, trust_remote_code=True)
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_NAME, torch_dtype=torch.float16, device_map="auto", trust_remote_code=True
    )

    total_before, train_before = count_params(model)
    print(f"Before adapters — Total: {total_before:,}  Trainable: {train_before:,}")

    model = inject_adapters(model, bottleneck=64)

    total_after, train_after = count_params(model)
    print(f"After  adapters — Total: {total_after:,}  Trainable: {train_after:,}")
    print(f"Adapter params: {train_after:,} ({100 * train_after / total_after:.2f}%)")

    prompt = "The capital of France is"
    inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
    with torch.no_grad():
        out = model.generate(**inputs, max_new_tokens=20)
    print(f"\nPrompt: {prompt}")
    print(f"Output: {tokenizer.decode(out[0], skip_special_tokens=True)}")
