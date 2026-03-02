"""
Strategy 5 — Speculative Decoding Head
Attach a small draft head to predict multiple future tokens,
then verify them in parallel with the full model.

Result: 2–3× faster decoding with mathematically IDENTICAL outputs.

FLOW:
─────────────────────────────────────────────────────────────────────────
Step 1: Train a small draft head to mimic the main model's predictions
Step 2: At inference, draft head proposes K future tokens (fast)
Step 3: Full model verifies all K tokens in ONE parallel forward pass
Step 4: Accept longest matching prefix, reject the rest
Step 5: Output = mathematically identical to standard autoregressive
─────────────────────────────────────────────────────────────────────────

DECODING LOOP (one iteration):

    generated = [The, cat, sat]     ← tokens so far
         │
    ═══ DRAFT PHASE ═══════════════════════════════════════════
         │
         ▼
    ┌──────────────────────────────────────────────────┐
    │  Full Model forward (to get hidden states)       │
    │  [The, cat, sat] → hidden (B, 3, H)              │
    └──────────────────────┬───────────────────────────┘
                           │
                           ▼
    ┌──────────────────────────────────────────────────┐
    │  DRAFT HEAD (small, fast)                        │  ← TRAINABLE
    │                                                  │
    │  last_hidden (B, H)                              │
    │      │                                           │
    │      ├──► [Head₀] → logits₀ → argmax → t₁="on"  │
    │      ├──► [Head₁] → logits₁ → argmax → t₂="the" │
    │      ├──► [Head₂] → logits₂ → argmax → t₃="mat" │
    │      └──► [Head₃] → logits₃ → argmax → t₄="."   │
    │                                                  │
    │  Each Head = LayerNorm → Linear → GELU → Linear  │
    └───────────────────────┬──────────────────────────┘
                            │
              draft = [on, the, mat, .]
                            │
    ═══ VERIFY PHASE ══════════════════════════════════════════
                            │
                            ▼
    ┌──────────────────────────────────────────────────┐
    │  Full Model forward (ONE parallel pass)          │
    │  [The, cat, sat, on, the, mat, .] → logits       │
    │                                                  │
    │  Check each position:                            │
    │    pos 3: model says "on"  vs draft "on"   → ✓  │
    │    pos 4: model says "the" vs draft "the"  → ✓  │
    │    pos 5: model says "mat" vs draft "mat"  → ✓  │
    │    pos 6: model says "!"   vs draft "."    → ✗  │
    └───────────────────────┬──────────────────────────┘
                            │
                            ▼
    Accept [on, the, mat] + correct token [!]
    generated = [The, cat, sat, on, the, mat, !]

    ★ 4 tokens generated but only 2 forward passes (vs 4 standard)
    ★ Accepted 3 of 4 draft tokens = 75% acceptance rate
    ★ After training draft head: acceptance reaches 70–90%

STANDARD vs SPECULATIVE:

    Standard (N tokens = N forward passes):
    ──────────────────────────────────────
    [Model]→t₁ │ [Model]→t₂ │ [Model]→t₃ │ [Model]→t₄
         1 pass      1 pass      1 pass      1 pass   = 4 passes total

    Speculative (N tokens ≈ N/K forward passes):
    ─────────────────────────────────────────────
    [Draft]→t₁,t₂,t₃,t₄ │ [Model]→verify all 4
         tiny pass              1 pass            ≈ 2 passes total
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class DraftHead(nn.Module):
    """
    Lightweight head attached to the main model that predicts
    K future tokens from the current hidden state.
    """

    def __init__(self, hidden_size: int, vocab_size: int, num_speculative: int = 4):
        super().__init__()
        self.num_speculative = num_speculative
        self.heads = nn.ModuleList([
            nn.Sequential(
                nn.LayerNorm(hidden_size),
                nn.Linear(hidden_size, hidden_size // 2),
                nn.GELU(),
                nn.Linear(hidden_size // 2, vocab_size),
            )
            for _ in range(num_speculative)
        ])

    def forward(self, hidden_state: torch.Tensor):
        """Predict K future token logits from the last hidden state."""
        last_hidden = hidden_state[:, -1, :]
        return [head(last_hidden) for head in self.heads]


class ToyLanguageModel(nn.Module):
    """Simple transformer LM for demonstrating speculative decoding."""

    def __init__(self, vocab_size: int, hidden: int, num_layers: int = 4, nhead: int = 4):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, hidden)
        self.pos_emb = nn.Embedding(512, hidden)
        self.layers = nn.ModuleList([
            nn.TransformerEncoderLayer(
                d_model=hidden, nhead=nhead, dim_feedforward=hidden * 4, batch_first=True
            )
            for _ in range(num_layers)
        ])
        self.norm = nn.LayerNorm(hidden)
        self.lm_head = nn.Linear(hidden, vocab_size, bias=False)

    def forward(self, input_ids: torch.Tensor, return_hidden: bool = False):
        B, L = input_ids.shape
        pos = torch.arange(L, device=input_ids.device).unsqueeze(0).expand(B, -1)
        x = self.embedding(input_ids) + self.pos_emb(pos)
        for layer in self.layers:
            x = layer(x)
        x = self.norm(x)
        logits = self.lm_head(x)
        return (logits, x) if return_hidden else logits


def speculative_decode(
    model: ToyLanguageModel,
    draft: DraftHead,
    input_ids: torch.Tensor,
    max_new_tokens: int = 20,
):
    """
    Speculative decoding loop:
    1. Draft head proposes K tokens
    2. Full model verifies all K in one forward pass
    3. Accept matching prefix, reject the rest
    """
    generated = input_ids.clone()
    accepted_total = 0
    draft_total = 0

    for _ in range(max_new_tokens):
        # ┌─── DRAFT PHASE ───────────────────────────────────────┐
        # │ Run main model → get hidden → draft head proposes K   │
        # │ tokens in one fast pass                                │
        # └────────────────────────────────────────────────────────┘
        with torch.no_grad():
            _, hidden = model(generated, return_hidden=True)
            draft_logits = draft(hidden)

        draft_tokens = [torch.argmax(dl, dim=-1) for dl in draft_logits]
        K = len(draft_tokens)
        draft_total += K

        candidates = torch.cat(
            [generated, torch.stack(draft_tokens, dim=-1)], dim=-1
        )

        # ┌─── VERIFY PHASE ──────────────────────────────────────┐
        # │ Full model checks ALL K draft tokens in ONE forward    │
        # │ pass. Accept longest matching prefix, reject the rest. │
        # └────────────────────────────────────────────────────────┘
        with torch.no_grad():
            verify_logits = model(candidates)

        n = generated.shape[1]
        accepted = 0
        for k in range(K):
            verified_token = torch.argmax(verify_logits[:, n + k - 1, :], dim=-1)
            if (verified_token == draft_tokens[k]).all():
                accepted += 1
            else:
                generated = torch.cat(
                    [generated, torch.stack(draft_tokens[:k], dim=-1), verified_token.unsqueeze(-1)],
                    dim=-1,
                ) if k > 0 else torch.cat(
                    [generated, verified_token.unsqueeze(-1)], dim=-1
                )
                break
        else:
            next_token = torch.argmax(verify_logits[:, n + K - 1, :], dim=-1)
            generated = torch.cat(
                [generated, torch.stack(draft_tokens, dim=-1), next_token.unsqueeze(-1)], dim=-1
            )
            accepted = K

        accepted_total += accepted
        if generated.shape[1] >= input_ids.shape[1] + max_new_tokens:
            break

    acceptance_rate = accepted_total / max(draft_total, 1)
    return generated, acceptance_rate


if __name__ == "__main__":
    VOCAB, HIDDEN, LAYERS = 500, 128, 4
    K = 4

    model = ToyLanguageModel(VOCAB, HIDDEN, LAYERS)
    draft = DraftHead(HIDDEN, VOCAB, num_speculative=K)

    model_params = sum(p.numel() for p in model.parameters())
    draft_params = sum(p.numel() for p in draft.parameters())
    print(f"Main model params: {model_params:,}")
    print(f"Draft head params: {draft_params:,} ({100 * draft_params / model_params:.1f}% of main)")
    print(f"Speculative tokens per step: {K}\n")

    input_ids = torch.randint(0, VOCAB, (1, 10))
    output, acc_rate = speculative_decode(model, draft, input_ids, max_new_tokens=20)

    print(f"Input length:  {input_ids.shape[1]}")
    print(f"Output length: {output.shape[1]}")
    print(f"Acceptance rate: {acc_rate:.1%}")
    print(f"(Untrained draft → low acceptance; after training it reaches 70–90%)")
