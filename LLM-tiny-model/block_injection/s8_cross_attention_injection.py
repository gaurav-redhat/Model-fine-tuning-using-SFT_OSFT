"""
Strategy 8 — Cross-Attention Injection (Multimodal)
Inject cross-attention blocks so a text-only LLM can attend to
a new modality (vision, audio) without retraining the base LLM.

FLOW:
─────────────────────────────────────────────────────────────────────────
Step 1: Freeze all text LLM layers
Step 2: Add a vision encoder (patchify → project → positional embed)
Step 3: Inject cross-attention blocks at selected layers (e.g., 1,3,5)
Step 4: Train ONLY vision encoder + cross-attention; text LLM stays frozen
Step 5: Deploy — same text model, now understands images too
─────────────────────────────────────────────────────────────────────────

FULL MODEL DATA FLOW:

    Image (B, 3, 64, 64)                     Text input_ids (B, S)
         │                                          │
         ▼                                          ▼
    ┌─────────────────────────────┐          [Embedding] (frozen)
    │  Vision Encoder (TRAINABLE) │                 │
    │                             │                 ▼
    │  Patchify: 64×64 → 8×8     │          x (B, S, H)
    │  patches of 8×8 pixels     │                 │
    │       │                     │                 │
    │       ▼                     │                 │
    │  Flatten: 64 patches ×      │                 │
    │         (3×8×8)=192 dims   │                 │
    │       │                     │                 │
    │       ▼                     │                 │
    │  [Linear] 192 → H          │                 │
    │       │                     │                 │
    │       ▼                     │                 │
    │  + positional embedding     │                 │
    │       │                     │                 │
    │       ▼                     │                 │
    │  visual (B, 64, H)          │                 │
    └────────────┬────────────────┘                 │
                 │                                  │
                 │              ┌────────────────────┤
                 │              │                    │
                 │              ▼                    │
                 │    ┌──────────────────┐           │
                 │    │  Text Layer 0    │ (frozen)  │
                 │    └────────┬─────────┘           │
                 │             │                     │
                 │             ▼                     │
                 │    ┌──────────────────┐           │
                 │    │  Text Layer 1    │ (frozen)  │
                 │    └────────┬─────────┘           │
                 │             │                     │
                 │             ▼                     │
                 │    ╔══════════════════════════╗   │
                 ├───►║  Cross-Attn @ Layer 1   ║   │  ← TRAINABLE
                 │    ║                          ║   │
                 │    ║  Q = text_hidden          ║   │
                 │    ║  K = visual_features     ║   │
                 │    ║  V = visual_features     ║   │
                 │    ║       │                  ║   │
                 │    ║       ▼                  ║   │
                 │    ║  Attn = softmax(QKᵀ/√d) ║   │
                 │    ║       │                  ║   │
                 │    ║       ▼                  ║   │
                 │    ║  out = Attn × V          ║   │
                 │    ║       │                  ║   │
                 │    ║       ▼                  ║   │
                 │    ║  x += tanh(gate) × out   ║   │
                 │    ║  (gate starts at 0 →     ║   │
                 │    ║   gradual integration)   ║   │
                 │    ╚════════════╤═════════════╝   │
                 │                │                  │
                 │                ▼                  │
                 │    ┌──────────────────┐           │
                 │    │  Text Layer 2    │ (frozen)  │
                 │    └────────┬─────────┘           │
                 │             │                     │
                 │             ▼                     │
                 │    ┌──────────────────┐           │
                 │    │  Text Layer 3    │ (frozen)  │
                 │    └────────┬─────────┘           │
                 │             │                     │
                 │             ▼                     │
                 ├───►╔══════════════════════════╗   │
                 │    ║  Cross-Attn @ Layer 3   ║   │  ← TRAINABLE
                 │    ╚════════════╤═════════════╝   │
                 │                │                  │
                 │               ...                 │
                 │                │                  │
                 ├───►╔══════════════════════════╗   │
                      ║  Cross-Attn @ Layer 5   ║   │  ← TRAINABLE
                      ╚════════════╤═════════════╝   │
                                   │                 │
                                   ▼                 │
                      [LM Head] (frozen)             │
                                   │                 │
                                   ▼                 │
                      logits (B, S, V)    ← multimodal output
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import math


class CrossAttentionBlock(nn.Module):
    """Injected cross-attention: text queries attend to visual key-values."""

    def __init__(self, hidden_size: int, num_heads: int = 4):
        super().__init__()
        self.num_heads = num_heads
        self.head_dim = hidden_size // num_heads

        self.q_proj = nn.Linear(hidden_size, hidden_size, bias=False)
        self.k_proj = nn.Linear(hidden_size, hidden_size, bias=False)
        self.v_proj = nn.Linear(hidden_size, hidden_size, bias=False)
        self.out_proj = nn.Linear(hidden_size, hidden_size, bias=False)
        self.norm_text = nn.LayerNorm(hidden_size)
        self.norm_visual = nn.LayerNorm(hidden_size)
        self.gate = nn.Parameter(torch.zeros(1))

    def forward(self, text_hidden: torch.Tensor, visual_features: torch.Tensor) -> torch.Tensor:
        B, L_t, D = text_hidden.shape
        H, HD = self.num_heads, self.head_dim

        text_normed = self.norm_text(text_hidden)
        vis_normed = self.norm_visual(visual_features)

        q = self.q_proj(text_normed).view(B, L_t, H, HD).transpose(1, 2)
        k = self.k_proj(vis_normed).view(B, -1, H, HD).transpose(1, 2)
        v = self.v_proj(vis_normed).view(B, -1, H, HD).transpose(1, 2)

        attn = torch.matmul(q, k.transpose(-2, -1)) / math.sqrt(HD)
        attn = F.softmax(attn, dim=-1)
        out = torch.matmul(attn, v)
        out = out.transpose(1, 2).reshape(B, L_t, D)

        return text_hidden + torch.tanh(self.gate) * self.out_proj(out)


class SimpleVisionEncoder(nn.Module):
    """Toy vision encoder: patchify + linear projection + positional encoding."""

    def __init__(self, image_size: int = 64, patch_size: int = 8, hidden_size: int = 128):
        super().__init__()
        self.patch_size = patch_size
        num_patches = (image_size // patch_size) ** 2
        patch_dim = 3 * patch_size * patch_size

        self.proj = nn.Linear(patch_dim, hidden_size)
        self.pos_emb = nn.Embedding(num_patches, hidden_size)

    def forward(self, images: torch.Tensor) -> torch.Tensor:
        B, C, H, W = images.shape
        P = self.patch_size
        patches = images.unfold(2, P, P).unfold(3, P, P)
        patches = patches.contiguous().view(B, -1, C * P * P)
        N = patches.shape[1]

        pos = torch.arange(N, device=images.device).unsqueeze(0).expand(B, -1)
        return self.proj(patches) + self.pos_emb(pos)


class MultimodalLM(nn.Module):
    """
    Text LLM with injected cross-attention for vision.
    Text layers are frozen; only cross-attention + vision encoder are trained.
    """

    def __init__(
        self,
        vocab_size: int,
        hidden: int,
        num_layers: int = 6,
        nhead: int = 4,
        inject_at: list | None = None,
    ):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, hidden)

        self.text_layers = nn.ModuleList([
            nn.TransformerEncoderLayer(
                d_model=hidden, nhead=nhead, dim_feedforward=hidden * 4, batch_first=True
            )
            for _ in range(num_layers)
        ])

        self.inject_at = inject_at or [1, 3, 5]
        self.cross_attns = nn.ModuleDict({
            str(i): CrossAttentionBlock(hidden, nhead) for i in self.inject_at
        })

        self.vision_encoder = SimpleVisionEncoder(image_size=64, patch_size=8, hidden_size=hidden)
        self.norm = nn.LayerNorm(hidden)
        self.head = nn.Linear(hidden, vocab_size, bias=False)

    def freeze_text_layers(self):
        for p in self.embedding.parameters():
            p.requires_grad = False
        for layer in self.text_layers:
            for p in layer.parameters():
                p.requires_grad = False
        for p in self.head.parameters():
            p.requires_grad = False

    def forward(self, input_ids: torch.Tensor, images: torch.Tensor | None = None):
        """
        Flow: text_embed → [text_layer → cross_attn?]×N → LM_head
              images → vision_encoder → visual features (shared across cross-attn layers)
        """
        x = self.embedding(input_ids)                        # (B,S) → (B,S,H)
        visual = self.vision_encoder(images) if images is not None else None  # (B,P,H)

        for i, layer in enumerate(self.text_layers):
            x = layer(x)                                     # frozen text self-attn + FFN
            if visual is not None and str(i) in self.cross_attns:
                x = self.cross_attns[str(i)](x, visual)      # text attends to visual features

        return self.head(self.norm(x))                       # (B,S,V) logits


if __name__ == "__main__":
    VOCAB, HIDDEN, LAYERS = 500, 128, 6

    model = MultimodalLM(VOCAB, HIDDEN, LAYERS, inject_at=[1, 3, 5])
    model.freeze_text_layers()

    total = sum(p.numel() for p in model.parameters())
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Total params:     {total:,}")
    print(f"Trainable (new):  {trainable:,} ({100 * trainable / total:.1f}%)")
    print(f"Cross-attn at layers: {model.inject_at}\n")

    text = torch.randint(0, VOCAB, (2, 32))
    image = torch.randn(2, 3, 64, 64)

    out_text = model(text, images=None)
    print(f"Text-only  output: {out_text.shape}")

    out_multi = model(text, images=image)
    print(f"Multimodal output: {out_multi.shape}")
    print(f"(Same LLM, new visual understanding via injected cross-attention)")
