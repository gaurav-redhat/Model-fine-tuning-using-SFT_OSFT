# Injecting New Blocks into Pre-Trained LLMs

> Improve accuracy and reduce inference time — without training from scratch

---

## Table of Contents

- [Why Block Injection?](#why-block-injection)
- [Overview of Strategies](#overview-of-strategies)
- [Strategy 1 — Adapter Modules](#strategy-1--adapter-modules)
- [Strategy 2 — Sparse Upcycling (Dense → MoE)](#strategy-2--sparse-upcycling-dense--moe)
- [Strategy 3 — SSM / Mamba Block Replacement](#strategy-3--ssm--mamba-block-replacement)
- [Strategy 4 — Early Exit Layers](#strategy-4--early-exit-layers)
- [Strategy 5 — Speculative Decoding Head](#strategy-5--speculative-decoding-head)
- [Strategy 6 — Dynamic Layer Skipping](#strategy-6--dynamic-layer-skipping)
- [Strategy 7 — Attention Pattern Replacement](#strategy-7--attention-pattern-replacement)
- [Strategy 8 — Cross-Attention Injection (Multimodal)](#strategy-8--cross-attention-injection-multimodal)
- [Strategy 9 — Side Networks / Ladder Side-Tuning](#strategy-9--side-networks--ladder-side-tuning)
- [Strategy 10 — Prefix / Prompt Tuning Blocks](#strategy-10--prefix--prompt-tuning-blocks)
- [Strategy Comparison Matrix](#strategy-comparison-matrix)
- [Recommended Injection Pipelines](#recommended-injection-pipelines)
- [Paper References](#paper-references)

---

## Why Block Injection?

Training an LLM from scratch costs millions of dollars and weeks of GPU time. **Block injection** lets you surgically insert new architectural components into a pre-trained model, then train *only* those new components while freezing the original weights.

```
  Traditional approach:
    Design architecture → Train from scratch (weeks, $$$) → Deploy

  Block injection approach:
    Take pre-trained model → Inject new blocks → Train ONLY new blocks (hours, $) → Deploy
```

**Three goals of block injection:**

| Goal | How |
|---|---|
| **Improve accuracy** | Inject capacity where the model is weak (adapters, experts, retrieval heads) |
| **Reduce inference time** | Inject shortcuts that skip unnecessary computation (early exit, layer skipping) |
| **Reduce complexity** | Replace expensive blocks with efficient ones (attention → SSM, full → linear) |

---

## Overview of Strategies

```
┌─────────────────────────────────────────────────────────────────────────────────┐
│                    BLOCK INJECTION STRATEGIES                                    │
│                                                                                 │
│   ┌──────────────────────────────────────────────────────────────────────────┐  │
│   │  ACCURACY-FOCUSED                                                        │  │
│   │                                                                          │  │
│   │  ┌───────────┐  ┌─────────────────┐  ┌────────────────┐  ┌──────────┐   │  │
│   │  │ Adapter   │  │ Sparse Upcycle  │  │ Cross-Attn     │  │ Side     │   │  │
│   │  │ Modules   │  │ (Dense→MoE)     │  │ Injection      │  │ Networks │   │  │
│   │  │           │  │                 │  │ (Multimodal)   │  │          │   │  │
│   │  └───────────┘  └─────────────────┘  └────────────────┘  └──────────┘   │  │
│   └──────────────────────────────────────────────────────────────────────────┘  │
│                                                                                 │
│   ┌──────────────────────────────────────────────────────────────────────────┐  │
│   │  SPEED-FOCUSED                                                           │  │
│   │                                                                          │  │
│   │  ┌───────────┐  ┌─────────────────┐  ┌────────────────┐  ┌──────────┐   │  │
│   │  │ Early     │  │ Speculative     │  │ Dynamic Layer  │  │ Attn     │   │  │
│   │  │ Exit      │  │ Decoding Head   │  │ Skipping       │  │ Replace  │   │  │
│   │  │ Layers    │  │                 │  │                │  │ (→SSM)   │   │  │
│   │  └───────────┘  └─────────────────┘  └────────────────┘  └──────────┘   │  │
│   └──────────────────────────────────────────────────────────────────────────┘  │
│                                                                                 │
│   ┌──────────────────────────────────────────────────────────────────────────┐  │
│   │  BOTH (ACCURACY + SPEED)                                                 │  │
│   │                                                                          │  │
│   │  ┌─────────────────┐  ┌────────────────┐  ┌──────────────────────────┐  │  │
│   │  │ SSM / Mamba     │  │ Sparse Upcycle │  │ Prefix / Prompt Tuning   │  │  │
│   │  │ Block Replace   │  │ (Dense→MoE)    │  │ Blocks                   │  │  │
│   │  └─────────────────┘  └────────────────┘  └──────────────────────────┘  │  │
│   └──────────────────────────────────────────────────────────────────────────┘  │
└─────────────────────────────────────────────────────────────────────────────────┘
```

---

## Strategy 1 — Adapter Modules

> **Goal**: Insert small trainable bottleneck layers between existing frozen transformer layers.

```
  Original Transformer Layer:

    Input
      │
      ▼
    [Multi-Head Attention]  ← frozen
      │
      ▼
    [Feed-Forward Network]  ← frozen
      │
      ▼
    Output


  With Adapter Injection:

    Input
      │
      ▼
    [Multi-Head Attention]  ← frozen
      │
      ▼
    ┌─────────────────────┐
    │   ADAPTER BLOCK     │  ← NEW, trainable
    │                     │
    │   Down-project (d→r)│  d=768, r=64
    │   Nonlinearity      │
    │   Up-project  (r→d) │
    │   + Residual        │
    └─────────────────────┘
      │
      ▼
    [Feed-Forward Network]  ← frozen
      │
      ▼
    ┌─────────────────────┐
    │   ADAPTER BLOCK     │  ← NEW, trainable
    └─────────────────────┘
      │
      ▼
    Output
```

| Property | Detail |
|---|---|
| **What's injected** | Bottleneck layers (down-project → nonlinearity → up-project) |
| **Where** | After attention and/or after FFN, in every layer |
| **Trainable params** | ~0.5–3% of total model params |
| **Original weights** | Completely frozen |
| **Accuracy impact** | +2–8% on downstream tasks |
| **Speed impact** | Slight increase in latency (~5–10%) due to added layers |
| **Key advantage** | Task-specific adaptation with minimal parameters |

### Adapter Variants

| Variant | Difference | Paper |
|---|---|---|
| **Houlsby Adapter** | Two adapters per layer (after attention + after FFN) | [arXiv:1902.00751](https://arxiv.org/abs/1902.00751) |
| **Pfeiffer Adapter** | One adapter per layer (after FFN only) — faster | [arXiv:2005.00247](https://arxiv.org/abs/2005.00247) |
| **AdapterFusion** | Combine multiple task-specific adapters via attention | [arXiv:2005.00247](https://arxiv.org/abs/2005.00247) |
| **Compacter** | Kronecker product + low-rank for even smaller adapters | [arXiv:2106.04647](https://arxiv.org/abs/2106.04647) |
| **IA³** | Learned rescaling vectors (even fewer params than LoRA) | [arXiv:2205.05638](https://arxiv.org/abs/2205.05638) |

### When to Use

- You want per-task customization without modifying original weights
- You have limited GPU memory (adapters are tiny)
- You need multiple task variants from the same base model

---

## Strategy 2 — Sparse Upcycling (Dense → MoE)

> **Goal**: Convert a dense pre-trained model into a Mixture-of-Experts model by duplicating FFN layers and adding a learned router — more capacity with the same inference cost.

```
  Dense Model (original):

    Input → [Attention] → [FFN] → Output
                           │
                      All parameters
                      always active


  After Sparse Upcycling:

    Input → [Attention] → [Router] → Output
                             │
                    ┌────────┼────────┐
                    │        │        │
                 [FFN-1]  [FFN-2]  [FFN-3]  ...  [FFN-N]
                 (copy)   (copy)   (copy)        (copy)
                    │        │        │
                    └────── Top-K ────┘
                      (only K of N
                       experts active)

    ★ Each FFN-i starts as a COPY of the original FFN
    ★ Router is the ONLY new block — lightweight gating network
    ★ At inference: only K experts fire → same FLOPs as original
    ★ Total capacity: N× larger, but compute stays constant
```

| Property | Detail |
|---|---|
| **What's injected** | Router network + duplicated FFN layers (experts) |
| **Where** | Replaces each dense FFN layer |
| **New params** | Router is tiny; experts are copies of existing FFN (initialized from pre-trained weights) |
| **Training** | Fine-tune router + lightly tune experts; original attention frozen |
| **Accuracy impact** | +3–10% — more capacity, better specialization |
| **Speed impact** | **Same or faster** — only top-K experts compute per token |
| **Key advantage** | Massive capacity increase without increasing inference FLOPs |

### How It Works Step-by-Step

```
Step 1: Take pre-trained dense model
          │
Step 2: For each FFN layer:
          ├── Duplicate the FFN weights N times → N experts
          ├── Add a Router (small linear layer + softmax)
          └── Router learns to pick top-K experts per token
          │
Step 3: Fine-tune on moderate data (~50B tokens is enough)
          ├── Router: trained from scratch
          ├── Experts: fine-tuned from copies (warm start)
          └── Attention layers: frozen or lightly tuned
          │
Step 4: Deploy — inference cost ≈ original dense model
```

### Real-World Examples

| Model | Source | Experts | Active | Speedup | Paper |
|---|---|---|---|---|---|
| **Sparse Upcycled T5** | T5-Base | 32 | 2 | Comparable FLOP | [arXiv:2212.05055](https://arxiv.org/abs/2212.05055) |
| **Branch-Train-MiX** | LLaMA-7B | 8 | 2 | Same compute | [arXiv:2403.07816](https://arxiv.org/abs/2403.07816) |
| **LLaMA-MoE** | LLaMA-2-7B | 8 | 2 | 2× throughput | [arXiv:2406.16554](https://arxiv.org/abs/2406.16554) |

### When to Use

- You want to increase model capacity without increasing inference cost
- The dense model hits a performance ceiling and needs more "brain power"
- You have moderate compute for fine-tuning (not scratch training)

---

## Strategy 3 — SSM / Mamba Block Replacement

> **Goal**: Replace some attention layers with State-Space Model (Mamba) blocks to get O(1) memory and linear-time inference instead of O(n²) attention.

```
  Standard Transformer (all attention):

    Layer 1:  [Attention]  O(n²)
    Layer 2:  [Attention]  O(n²)
    Layer 3:  [Attention]  O(n²)
    Layer 4:  [Attention]  O(n²)
    ...
    Layer N:  [Attention]  O(n²)


  Hybrid after Mamba Injection:

    Layer 1:  [Mamba]      O(n)   ← REPLACED, trainable
    Layer 2:  [Attention]  O(n²)  ← kept frozen
    Layer 3:  [Mamba]      O(n)   ← REPLACED, trainable
    Layer 4:  [Attention]  O(n²)  ← kept frozen
    ...
    Layer N:  [Mamba]      O(n)   ← REPLACED, trainable

    ★ Alternate Mamba/Attention or use a learned ratio
    ★ Keep some attention for global context recall
    ★ Mamba handles local/sequential patterns efficiently
```

| Property | Detail |
|---|---|
| **What's injected** | Mamba / S4 / Linear Attention blocks |
| **Where** | Replace selected attention layers (typically alternating) |
| **Training** | Train new Mamba blocks via distillation from the replaced attention layers |
| **Accuracy impact** | -0.5 to +2% (Mamba is competitive; hybrid often matches or exceeds pure attention) |
| **Speed impact** | **2–3× faster** for long sequences; O(n) vs O(n²) |
| **Memory impact** | **O(1) per step** vs O(n) KV-cache — huge win for long context |

### Distillation-Based Replacement

You don't randomly initialize the Mamba blocks — you **distill** the behavior of the attention layer into the new block:

```
  For each replaced layer:

    Frozen attention layer (teacher)
          │
          │  layer-wise knowledge distillation
          │  minimize: MSE(attention_output, mamba_output)
          │
          ▼
    New Mamba block (student)
          │
    After convergence: remove attention layer, keep Mamba block
```

### Proven Hybrid Architectures

| Model | Approach | Result | Paper |
|---|---|---|---|
| **Hymba (1.5B)** | Mamba + Attention heads in parallel | Best sub-2B model | [arXiv:2411.13676](https://arxiv.org/abs/2411.13676) |
| **Samba (3.8B)** | Mamba + Sliding Window Attention | 3.73× throughput | [arXiv:2406.07522](https://arxiv.org/abs/2406.07522) |
| **Zamba (7B)** | Mamba + Shared Transformer block | 2× throughput | [arXiv:2405.18712](https://arxiv.org/abs/2405.18712) |
| **Jamba (52B MoE)** | Mamba + Transformer + MoE | Longest context | [arXiv:2403.19887](https://arxiv.org/abs/2403.19887) |
| **Mamba-2 in Transformers** | Drop-in replacement study | -0.3% accuracy, 2× speed | [arXiv:2405.21060](https://arxiv.org/abs/2405.21060) |

### When to Use

- Long-context workloads (the O(n²) → O(n) savings compound rapidly)
- Deployment on memory-constrained devices (constant KV-cache)
- Streaming / real-time applications where latency matters

---

## Strategy 4 — Early Exit Layers

> **Goal**: Add lightweight classifier heads at intermediate layers so easy inputs exit early without going through the full model.

```
  Standard inference (all inputs go through all layers):

    Input → Layer 1 → Layer 2 → ... → Layer N → Output
                                                   │
                                              Always N layers
                                              (even for "The capital of France is ___")


  With Early Exit:

    Input → Layer 1 → Layer 2 → ... → Layer k → EXIT ✓
                │          │               │
           [Exit Head] [Exit Head]    [Exit Head]
           confidence  confidence     confidence
           too low     too low        HIGH → output early!
                                      (skipped layers k+1..N)

    ★ Easy tokens exit after a few layers
    ★ Hard tokens go through all layers
    ★ Average inference time drops significantly
```

| Property | Detail |
|---|---|
| **What's injected** | Small classifier/LM heads at intermediate layers + confidence estimator |
| **Where** | After every K-th layer (e.g., every 4th layer) |
| **New params** | Very small — each exit head is just a linear projection |
| **Training** | Train only the exit heads; base model frozen |
| **Accuracy impact** | ~0% loss on hard inputs; slight degradation on easy ones (configurable threshold) |
| **Speed impact** | **1.5–3× faster** on average (input-dependent) |

### Confidence Estimation Methods

| Method | How It Works |
|---|---|
| **Softmax entropy** | Exit if top prediction has high confidence (low entropy) |
| **Learned threshold** | Train a small network to predict "should I exit?" |
| **Patience-based** | Exit if the prediction hasn't changed for K consecutive layers |
| **Token-level** | Different tokens in the same sequence exit at different layers |

### Research

| Work | Method | Speedup | Accuracy Loss | Paper |
|---|---|---|---|---|
| **CALM** | Confidence-based early exit for LLMs | 2–3× | <1% | [arXiv:2207.07061](https://arxiv.org/abs/2207.07061) |
| **FREE** | Fast & Robust Early Exiting | 2× | <0.5% | [arXiv:2310.01811](https://arxiv.org/abs/2310.01811) |
| **SkipDecode** | Token-level early exit for batched inference | 2–5× | <1% | [arXiv:2307.02628](https://arxiv.org/abs/2307.02628) |
| **LayerSkip** | Early exit + self-speculative decoding | 1.8× | ~0% | [arXiv:2404.16710](https://arxiv.org/abs/2404.16710) |

### When to Use

- Mixed-difficulty workloads (easy queries are common)
- Latency-sensitive applications (chatbots, real-time)
- You need adaptive compute per input without changing the model architecture

---

## Strategy 5 — Speculative Decoding Head

> **Goal**: Attach a small, fast "draft" head to the model that generates candidate tokens quickly, then the full model verifies them in parallel — same output quality, much faster.

```
  Standard autoregressive decoding:

    [Model] → token₁ → [Model] → token₂ → [Model] → token₃ → ...
                (slow)            (slow)            (slow)
    Total: N sequential forward passes for N tokens


  With Speculative Decoding Head:

    [Draft Head] → t₁,t₂,t₃,t₄,t₅  (fast, 5 guesses in 1 pass)
         │
         ▼
    [Full Model] → verify all 5 in ONE parallel forward pass
         │
         ├── t₁ ✓  t₂ ✓  t₃ ✓  t₄ ✗  (accept first 3, reject t₄)
         │
         ▼
    Output: t₁ t₂ t₃ [correct_t₄]

    ★ Accepted tokens = FREE (no extra cost)
    ★ Rejection = only 1 wasted draft step
    ★ Mathematically IDENTICAL output to standard decoding
```

| Property | Detail |
|---|---|
| **What's injected** | Lightweight draft model head (1–2 transformer layers or a simple MLP) |
| **Where** | Attached at an intermediate layer or as a separate small head on top |
| **New params** | Very small (the draft head is ~5–10% the size of the main model) |
| **Training** | Train draft head to mimic the full model's next-token distribution |
| **Accuracy impact** | **Zero** — speculative decoding produces mathematically identical outputs |
| **Speed impact** | **2–3× faster** decoding (depends on acceptance rate) |

### Variants

| Variant | Description | Paper |
|---|---|---|
| **Medusa** | Multiple parallel heads predict future tokens simultaneously | [arXiv:2401.10774](https://arxiv.org/abs/2401.10774) |
| **EAGLE** | Extrapolation-based draft using hidden states | [arXiv:2401.15077](https://arxiv.org/abs/2401.15077) |
| **Self-Speculative** | Use early layers of the SAME model as the draft | [arXiv:2404.16710](https://arxiv.org/abs/2404.16710) |
| **Staged Speculative** | Multiple draft stages with increasing accuracy | [arXiv:2308.04623](https://arxiv.org/abs/2308.04623) |
| **Lookahead Decoding** | N-gram draft from Jacobi iteration | [arXiv:2402.02057](https://arxiv.org/abs/2402.02057) |

### When to Use

- You need faster generation with **zero accuracy loss**
- Latency-bound applications (real-time chat, interactive coding)
- The acceptance rate is high (draft model closely matches the main model)

---

## Strategy 6 — Dynamic Layer Skipping

> **Goal**: Add a small "skip predictor" that decides, per input, which layers to skip — reducing average compute without losing accuracy on hard inputs.

```
  Standard (all layers always execute):

    Input → L1 → L2 → L3 → L4 → L5 → L6 → ... → L24 → Output
            ──────────────────────────────────────────────
            All 24 layers, always


  With Dynamic Layer Skipping:

    Input → L1 → [Skip?] → L2 → [Skip?] → SKIP → [Skip?] → L4 → ...
                   │                │        L3       │
                  No               No       Yes!     No
                                    │
            Only ~16 of 24 layers execute on average

    ★ Skip predictor = tiny binary classifier per layer
    ★ Easy inputs skip many layers → fast
    ★ Hard inputs skip few layers → accurate
```

| Property | Detail |
|---|---|
| **What's injected** | Binary skip predictor (small MLP) at each layer |
| **Where** | Before each transformer layer |
| **New params** | Negligible (~0.01% of model) |
| **Training** | Train skip predictors with a compute-accuracy trade-off loss |
| **Accuracy impact** | -0.5 to -2% at aggressive skipping; ~0% at conservative |
| **Speed impact** | **1.5–2.5× faster** on average |

### Research

| Work | Approach | Result | Paper |
|---|---|---|---|
| **LayerSkip (Meta)** | Learned layer dropout + early exit | 1.8× speedup, ~0% loss | [arXiv:2404.16710](https://arxiv.org/abs/2404.16710) |
| **AdaInfer** | Predict which layers to skip based on input | 1.5–2× speedup | [arXiv:2310.10072](https://arxiv.org/abs/2310.10072) |
| **Dynamic Depth** | Token-level depth routing | 2× speedup on easy tokens | Various |

### When to Use

- You want a single model that's fast on easy inputs and thorough on hard inputs
- Can tolerate slight accuracy trade-off for significant speed gains
- Edge deployment where average latency matters more than worst-case

---

## Strategy 7 — Attention Pattern Replacement

> **Goal**: Swap standard O(n²) full attention with more efficient attention mechanisms — without retraining the whole model.

```
  Full Attention (standard):

    Every token attends to every other token
    Complexity: O(n²) time and memory

    ┌───┬───┬───┬───┬───┐
    │ ■ │ ■ │ ■ │ ■ │ ■ │  ← token 1 attends to all
    │ ■ │ ■ │ ■ │ ■ │ ■ │  ← token 2 attends to all
    │ ■ │ ■ │ ■ │ ■ │ ■ │  ...
    │ ■ │ ■ │ ■ │ ■ │ ■ │
    │ ■ │ ■ │ ■ │ ■ │ ■ │
    └───┴───┴───┴───┴───┘


  Sliding Window Attention (replaced):

    Each token attends only to W neighbors
    Complexity: O(n × W)

    ┌───┬───┬───┬───┬───┐
    │ ■ │ ■ │ ■ │   │   │  ← token 1 attends to window
    │ ■ │ ■ │ ■ │ ■ │   │
    │   │ ■ │ ■ │ ■ │ ■ │
    │   │   │ ■ │ ■ │ ■ │
    │   │   │   │ ■ │ ■ │
    └───┴───┴───┴───┴───┘
```

### Replacement Options

| Original | Replacement | Complexity | Accuracy | Paper |
|---|---|---|---|---|
| Full Attention | **Sliding Window** | O(n×W) | ~Same for local tasks | [Mistral](https://arxiv.org/abs/2310.06825) |
| Full Attention | **Linear Attention** | O(n) | -1–3% | [arXiv:2006.16236](https://arxiv.org/abs/2006.16236) |
| Full Attention | **Multi-head Latent Attention (MLA)** | O(n×d_c) | ~Same | [DeepSeek-V2](https://arxiv.org/abs/2405.04434) |
| Full Attention | **Grouped Query Attention (GQA)** | O(n²) but less KV memory | ~Same | [arXiv:2305.13245](https://arxiv.org/abs/2305.13245) |
| Full Attention | **Flash Attention** (kernel-level) | O(n²) compute but O(n) memory | Same (exact) | [arXiv:2205.14135](https://arxiv.org/abs/2205.14135) |

### Post-Hoc Conversion Approach

```
Step 1: Identify which layers use full attention
          │
Step 2: For long-range layers → keep full attention (or MLA)
        For local-pattern layers → replace with sliding window
          │
Step 3: Distill: train replacement heads to match original attention output
          │
Step 4: Validate on held-out data — ensure <1% accuracy drop
```

### When to Use

- Long-context inference is too slow or OOM
- KV-cache memory is the bottleneck (especially on-device)
- You can identify which layers need global vs. local attention patterns

---

## Strategy 8 — Cross-Attention Injection (Multimodal)

> **Goal**: Inject cross-attention blocks to add a new modality (vision, audio, structured data) into a text-only LLM without retraining the language model.

```
  Text-only LLM (original):

    Text tokens → [Self-Attn] → [FFN] → ... → Text output


  After Cross-Attention Injection:

    Image/Audio ──► [Encoder] ──► encoded features
                                       │
    Text tokens → [Self-Attn] → [Cross-Attn] → [FFN] → ... → Multimodal output
                    (frozen)     (NEW block)    (frozen)
                                  │
                           Text attends to
                           image/audio features
```

| Property | Detail |
|---|---|
| **What's injected** | Cross-attention layers + modality encoder |
| **Where** | Between self-attention and FFN in selected layers |
| **Training** | Train cross-attention layers + encoder; freeze LLM |
| **Accuracy impact** | Enables entirely new capabilities (vision, audio) |
| **Speed impact** | Adds ~10–20% latency for multimodal inputs |
| **Key advantage** | Reuse powerful text LLM for multimodal tasks |

### Real-World Examples

| Model | Approach | Paper |
|---|---|---|
| **Flamingo** | Interleaved cross-attention for vision+text | [arXiv:2204.14198](https://arxiv.org/abs/2204.14198) |
| **LLaVA** | Visual tokens projected into LLM input space | [arXiv:2304.08485](https://arxiv.org/abs/2304.08485) |
| **Gemma-3 (4B)** | Vision encoder + cross-attention in decoder | [arXiv:2503.19786](https://arxiv.org/abs/2503.19786) |
| **Qwen-VL** | Visual encoder with cross-attention injection | [arXiv:2308.12966](https://arxiv.org/abs/2308.12966) |

### When to Use

- You want to add vision/audio/structured-data understanding to a text LLM
- The text model is already strong and you don't want to retrain it
- Multimodal tasks (image captioning, VQA, document understanding)

---

## Strategy 9 — Side Networks / Ladder Side-Tuning

> **Goal**: Attach a small parallel "side" network that runs alongside the frozen main model, injecting corrections at each layer.

```
  Main Model (frozen):          Side Network (trainable):

    Input                          Input (downsampled)
      │                               │
    [Layer 1]  ◄── add ──────── [Side Layer 1]
      │                               │
    [Layer 2]  ◄── add ──────── [Side Layer 2]
      │                               │
    [Layer 3]  ◄── add ──────── [Side Layer 3]
      │                               │
    ...                              ...
      │                               │
    Output                         (corrections)

    ★ Side network is much smaller (e.g., 10% of main model)
    ★ Main model is completely frozen
    ★ Side network learns task-specific corrections
```

| Property | Detail |
|---|---|
| **What's injected** | A parallel small network with additive connections |
| **Where** | Runs alongside the main model, merges at each layer |
| **New params** | 5–15% of main model (configurable) |
| **Training** | Train only the side network |
| **Accuracy impact** | +2–5% on downstream tasks |
| **Speed impact** | ~10–15% slower (side network adds overhead) |
| **Key advantage** | More expressive than adapters; can be discarded to revert to base model |

### Research

| Work | Approach | Paper |
|---|---|---|
| **Ladder Side-Tuning** | Side network with shortcut connections | [arXiv:2206.06522](https://arxiv.org/abs/2206.06522) |
| **Side-Tuning** | Additive side network for continual learning | [arXiv:1912.13503](https://arxiv.org/abs/1912.13503) |

### When to Use

- You need more capacity than LoRA/adapters but can't modify main weights
- Continual learning — add new knowledge without catastrophic forgetting
- You want a "reversible" modification (just remove the side network)

---

## Strategy 10 — Prefix / Prompt Tuning Blocks

> **Goal**: Inject learnable "soft tokens" into the model's input or internal layers — extremely parameter-efficient.

```
  Standard input:

    [user tokens] → [Model] → Output


  With Prefix Tuning:

    [LEARNABLE PREFIX TOKENS] + [user tokens] → [Model] → Output
     ▲                                            │
     │     These virtual tokens are learned        │
     │     continuous vectors (not real words)      │
     │     prepended at EVERY layer's KV-cache      │
     └─────────────────────────────────────────────┘

  Prefix length: typically 10–100 virtual tokens
  Trainable params: ~0.1% of model
```

| Variant | What's Tuned | Params | Paper |
|---|---|---|---|
| **Prompt Tuning** | Soft tokens at input embedding only | ~0.01% | [arXiv:2104.08691](https://arxiv.org/abs/2104.08691) |
| **Prefix Tuning** | Soft tokens at every layer's key/value | ~0.1% | [arXiv:2101.00190](https://arxiv.org/abs/2101.00190) |
| **P-Tuning v2** | Prefix at every layer + task-specific head | ~0.1–1% | [arXiv:2110.07602](https://arxiv.org/abs/2110.07602) |

| Property | Detail |
|---|---|
| **What's injected** | Continuous embedding vectors ("soft prompts") |
| **Where** | Prepended to key/value pairs at each layer |
| **New params** | 0.01–0.1% of model (extremely efficient) |
| **Training** | Backprop through soft tokens only; model frozen |
| **Accuracy impact** | +1–5% on task (closes gap with full fine-tuning at scale) |
| **Speed impact** | Negligible — a few extra tokens in the context window |

### When to Use

- Extreme parameter efficiency needed
- You want many task-specific "modes" for one model (just swap prefix)
- As a complement to other injection strategies

---

## Strategy Comparison Matrix

| Strategy | Accuracy Δ | Speed Δ | New Params | Training Cost | Complexity | Best For |
|---|---|---|---|---|---|---|
| **Adapter Modules** | +2–8% | -5–10% slower | 0.5–3% | Low | Low | Task adaptation |
| **Sparse Upcycling (→MoE)** | +3–10% | Same or faster | Router only (experts are copies) | Medium | Medium | Capacity scaling |
| **SSM/Mamba Injection** | -0.5 to +2% | **2–3× faster** | Replacement blocks | Medium | High | Long context |
| **Early Exit** | ~0% | **1.5–3× faster** | <0.1% | Low | Low | Mixed workloads |
| **Speculative Head** | **0%** | **2–3× faster** | 5–10% | Low | Medium | Latency-critical |
| **Dynamic Layer Skip** | -0.5–2% | **1.5–2.5× faster** | <0.01% | Low | Medium | Adaptive compute |
| **Attention Replacement** | -1–3% | **1.5–2× faster** (long ctx) | Replacement blocks | Medium | High | Long context |
| **Cross-Attention** | New capability | -10–20% | 10–20% | Medium-High | High | Multimodal |
| **Side Networks** | +2–5% | -10–15% | 5–15% | Medium | Medium | Continual learning |
| **Prefix Tuning** | +1–5% | Negligible | 0.01–0.1% | Very Low | Very Low | Multi-task |

---

## Recommended Injection Pipelines

### Pipeline A: Maximum Accuracy (accept slight latency increase)

```
Pre-trained Model
  │
  ▼
Step 1: Sparse Upcycling (Dense → MoE, 8 experts, top-2)
  │       ↳ +5–10% accuracy, same inference FLOPs
  │
  ▼
Step 2: Adapter Modules on attention layers
  │       ↳ +2–5% task-specific boost
  │
  ▼
Step 3: Prefix Tuning for final task alignment
  │       ↳ +1–2%, near-zero params
  │
  ▼
Deploy with Speculative Decoding Head
          ↳ 2× faster generation, ZERO accuracy loss

Expected: +8–15% accuracy, 2× faster generation
```

### Pipeline B: Maximum Speed (maintain accuracy)

```
Pre-trained Model
  │
  ▼
Step 1: Replace 50% of attention layers with Mamba blocks
  │       ↳ 2× faster long-sequence inference
  │
  ▼
Step 2: Add Early Exit heads at layers 8, 16, 24 (of 32)
  │       ↳ 1.5–2× speedup on easy inputs
  │
  ▼
Step 3: Attach Speculative Decoding Head
  │       ↳ Additional 2× decoding speedup
  │
  ▼
Deploy

Expected: ~0% accuracy loss, 3–5× total speedup
```

### Pipeline C: Balanced (practical recommendation for 1B models)

```
Pre-trained Phi-4-Mini (1B)
  │
  ▼
Step 1: LoRA Fine-Tuning (from README Block 2)
  │       ↳ Task alignment, +5–10% on target benchmarks
  │
  ▼
Step 2: Add Early Exit Layers (after layers 6, 12, 18 of 24)
  │       ↳ Easy queries exit 2× faster
  │
  ▼
Step 3: Attach Medusa Speculative Head
  │       ↳ 2× faster decoding
  │
  ▼
Step 4: AWQ 4-bit Quantization (from README Block 3)
  │       ↳ 3–4× memory reduction
  │
  ▼
Deploy on edge device

Expected: +5–10% accuracy, 2–3× faster, 4× smaller
```

---

## Paper References

### Adapter & Parameter-Efficient Methods

| # | Topic | Paper | Link |
|---|---|---|---|
| 1 | Houlsby Adapters | *Parameter-Efficient Transfer Learning for NLP* | [arXiv:1902.00751](https://arxiv.org/abs/1902.00751) |
| 2 | Pfeiffer Adapters | *AdapterHub: A Framework for Adapting Transformers* | [arXiv:2007.07779](https://arxiv.org/abs/2007.07779) |
| 3 | Compacter | *Compacter: Efficient Low-Rank Hypercomplex Adapter Layers* | [arXiv:2106.04647](https://arxiv.org/abs/2106.04647) |
| 4 | IA³ | *Few-Shot Parameter-Efficient Fine-Tuning* | [arXiv:2205.05638](https://arxiv.org/abs/2205.05638) |
| 5 | Prefix Tuning | *Prefix-Tuning: Optimizing Continuous Prompts* | [arXiv:2101.00190](https://arxiv.org/abs/2101.00190) |
| 6 | Prompt Tuning | *The Power of Scale for Parameter-Efficient Prompt Tuning* | [arXiv:2104.08691](https://arxiv.org/abs/2104.08691) |
| 7 | P-Tuning v2 | *P-Tuning v2: Prompt Tuning Can Be Comparable to Fine-Tuning* | [arXiv:2110.07602](https://arxiv.org/abs/2110.07602) |

### Sparse Upcycling & MoE

| # | Topic | Paper | Link |
|---|---|---|---|
| 8 | Sparse Upcycling | *Sparse Upcycling: Training MoE from Dense Checkpoints* | [arXiv:2212.05055](https://arxiv.org/abs/2212.05055) |
| 9 | Branch-Train-MiX | *Branch-Train-MiX: Mixing Expert LLMs into a MoE* | [arXiv:2403.07816](https://arxiv.org/abs/2403.07816) |
| 10 | LLaMA-MoE | *LLaMA-MoE: Building MoE from LLaMA with Continual Pre-Training* | [arXiv:2406.16554](https://arxiv.org/abs/2406.16554) |

### Early Exit & Layer Skipping

| # | Topic | Paper | Link |
|---|---|---|---|
| 11 | CALM | *Confident Adaptive Language Modeling* | [arXiv:2207.07061](https://arxiv.org/abs/2207.07061) |
| 12 | LayerSkip | *LayerSkip: Enabling Early-Exit and Self-Speculative Decoding* | [arXiv:2404.16710](https://arxiv.org/abs/2404.16710) |
| 13 | SkipDecode | *SkipDecode: Autoregressive Skip Decoding* | [arXiv:2307.02628](https://arxiv.org/abs/2307.02628) |
| 14 | FREE | *Fast and Robust Early Exiting for LLMs* | [arXiv:2310.01811](https://arxiv.org/abs/2310.01811) |
| 15 | AdaInfer | *AdaInfer: Adaptive Inference for LLMs* | [arXiv:2310.10072](https://arxiv.org/abs/2310.10072) |

### Speculative Decoding

| # | Topic | Paper | Link |
|---|---|---|---|
| 16 | Speculative Decoding | *Fast Inference from Transformers via Speculative Decoding* | [arXiv:2211.17192](https://arxiv.org/abs/2211.17192) |
| 17 | Medusa | *Medusa: Simple LLM Inference Acceleration with Multiple Heads* | [arXiv:2401.10774](https://arxiv.org/abs/2401.10774) |
| 18 | EAGLE | *EAGLE: Speculative Sampling Requires Rethinking Feature Uncertainty* | [arXiv:2401.15077](https://arxiv.org/abs/2401.15077) |
| 19 | Lookahead | *Break the Sequential Dependency of LLM Inference* | [arXiv:2402.02057](https://arxiv.org/abs/2402.02057) |

### SSM / Hybrid Architecture Injection

| # | Topic | Paper | Link |
|---|---|---|---|
| 20 | Mamba | *Mamba: Linear-Time Sequence Modeling with Selective SSMs* | [arXiv:2312.00752](https://arxiv.org/abs/2312.00752) |
| 21 | Mamba-2 | *Transformers are SSMs: Generalized Models and Efficient Algorithms* | [arXiv:2405.21060](https://arxiv.org/abs/2405.21060) |
| 22 | Hymba | *Hymba: A Hybrid-head Architecture for Small Language Models* | [arXiv:2411.13676](https://arxiv.org/abs/2411.13676) |
| 23 | Samba | *Samba: Simple Hybrid State Space Models for Efficient Unlimited Context* | [arXiv:2406.07522](https://arxiv.org/abs/2406.07522) |
| 24 | Jamba | *Jamba: A Hybrid Transformer-Mamba Language Model* | [arXiv:2403.19887](https://arxiv.org/abs/2403.19887) |

### Attention Mechanism Replacements

| # | Topic | Paper | Link |
|---|---|---|---|
| 25 | Flash Attention | *FlashAttention: Fast and Memory-Efficient Exact Attention* | [arXiv:2205.14135](https://arxiv.org/abs/2205.14135) |
| 26 | Linear Attention | *Transformers are RNNs: Fast Autoregressive Transformers with Linear Attention* | [arXiv:2006.16236](https://arxiv.org/abs/2006.16236) |
| 27 | GQA | *GQA: Training Generalized Multi-Query Transformer Models* | [arXiv:2305.13245](https://arxiv.org/abs/2305.13245) |
| 28 | MLA (DeepSeek) | *DeepSeek-V2: A Strong, Economical, and Efficient MoE Model* | [arXiv:2405.04434](https://arxiv.org/abs/2405.04434) |

### Multimodal Injection

| # | Topic | Paper | Link |
|---|---|---|---|
| 29 | Flamingo | *Flamingo: A Visual Language Model for Few-Shot Learning* | [arXiv:2204.14198](https://arxiv.org/abs/2204.14198) |
| 30 | LLaVA | *Visual Instruction Tuning* | [arXiv:2304.08485](https://arxiv.org/abs/2304.08485) |

### Side Networks

| # | Topic | Paper | Link |
|---|---|---|---|
| 31 | Side-Tuning | *Side-Tuning: A Baseline for Network Adaptation via Additive Side Networks* | [arXiv:1912.13503](https://arxiv.org/abs/1912.13503) |
| 32 | Ladder Side-Tuning | *LST: Ladder Side-Tuning for Parameter and Memory Efficient Transfer Learning* | [arXiv:2206.06522](https://arxiv.org/abs/2206.06522) |

---

*Last updated: February 2026*
