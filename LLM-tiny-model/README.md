# Phi-4-Mini (1B) — Tiny Model Deep Dive

> Improving small language model accuracy with cutting-edge techniques

---

## Table of Contents

- [Our Target Models](#our-target-models)
- [Competitor Comparison (~1B Class)](#competitor-comparison-1b-class)
- [Benchmark Showdown](#benchmark-showdown)
- [Accuracy Improvement Techniques](#accuracy-improvement-techniques)
  - [Block 1 — Knowledge Distillation](#block-1--knowledge-distillation)
  - [Block 2 — Fine-Tuning (OSFT / DPO / RLHF / LoRA)](#block-2--fine-tuning-osft--dpo--rlhf--lora)
  - [Block 3 — Quantization (QAT / AWQ / GPTQ)](#block-3--quantization-qat--awq--gptq)
  - [Block 4 — Test-Time Compute Scaling](#block-4--test-time-compute-scaling)
  - [Block 5 — Mixture-of-LoRAs (MoLoRA)](#block-5--mixture-of-loras-molora)
  - [Block 6 — Pruning & Sparsity](#block-6--pruning--sparsity)
  - [Block 7 — Retrieval-Augmented Generation (RAG)](#block-7--retrieval-augmented-generation-rag)
- [Next-Gen Architectures (Beyond Transformers)](#next-gen-architectures-beyond-transformers)
  - [Diffusion Language Models](#-diffusion-language-models)
  - [Recursive / Recurrent LLMs (State-Space Models)](#-recursive--recurrent-llms-state-space-models)
  - [Hybrid Architectures](#-hybrid-architectures)
- [Full Improvement Pipeline](#full-improvement-pipeline)
- [Paper References](#paper-references)

---

## Our Target Models

```
    ┌─────────────────────────────────────────────────────────────┐
    │              Phi-4-Mini Lineup                              │
    │                                                             │
    │   ┌──────────────────┐      ┌───────────────────────────┐   │
    │   │  Phi-4-Mini (1B) │      │  Phi-4-Mini (1B) OSFT     │   │
    │   │                  │      │                           │   │
    │   │  Base model      │ ───► │  + Offline Self-play      │   │
    │   │  Very light      │      │    Fine-Tuning            │   │
    │   │  Low-resource    │      │  + Better responses       │   │
    │   │  devices         │      │  + Improved alignment     │   │
    │   └──────────────────┘      └───────────────────────────┘   │
    │         │                              │                    │
    │         └──────────┬───────────────────┘                    │
    │                    ▼                                        │
    │         Can we push further?                                │
    │         YES — See techniques below                          │
    └─────────────────────────────────────────────────────────────┘
```

| Property | Phi-4-Mini (1B) | Phi-4-Mini (1B) OSFT |
|---|---|---|
| **Parameters** | ~1B | ~1B |
| **Architecture** | Transformer decoder-only | Same + aligned |
| **Training** | Synthetic + curated web data | + Offline Self-play Fine-Tuning |
| **Context** | 128K tokens | 128K tokens |
| **Strengths** | Ultra-light, fast inference | Better instruction following |
| **Weaknesses** | Limited reasoning depth | Still 1B ceiling |
| **Best For** | Edge / IoT / mobile | Chatbots / assistants on-device |

---

## Competitor Comparison (~1B Class)

Models in the 0.5B–3B range that compete directly:

| Model | Params | Developer | Open? | Key Advantage |
|---|---|---|---|---|
| **Phi-4-Mini (1B)** | ~1B | Microsoft | Yes | Synthetic data quality |
| **Phi-4-Mini (1B) OSFT** | ~1B | Microsoft | Yes | Self-play alignment |
| **LLaMA-3.2-1B** | 1.24B | Meta | Yes | Massive ecosystem, multilingual |
| **LLaMA-3.2-3B** | 3.21B | Meta | Yes | Best open sub-4B model |
| **Gemma-3-1B** | 1B | Google | Yes | Distilled from Gemini 2.0 |
| **Qwen-3-0.6B** | 0.6B | Alibaba | Yes | Smallest competitive model |
| **Qwen-2.5-1.5B** | 1.5B | Alibaba | Yes | Strong code + math |
| **SmolLM2-1.7B** | 1.7B | HuggingFace | Yes | Purpose-built tiny model |
| **DeepSeek-R1-Distill-Qwen-1.5B** | 1.5B | DeepSeek | Yes | Reasoning via distillation |
| **TinyLlama-1.1B** | 1.1B | Community | Yes | 3T tokens, efficient training |
| **StableLM-2-1.6B** | 1.6B | Stability AI | Yes | Multilingual |
| **Gemma-3-4B** | 4B | Google | Yes | Multimodal (vision+text) |
| **OpenELM-1.1B** | 1.1B | Apple | Yes | Layer-wise scaling, reproducible |

---

## Benchmark Showdown

Approximate scores across key benchmarks (higher = better):

| Model | MMLU | GSM8K (Math) | HumanEval (Code) | ARC-C (Reasoning) | HellaSwag |
|---|---|---|---|---|---|
| **Phi-4-Mini (1B)** | ~58 | ~42 | ~38 | ~52 | ~62 |
| **Phi-4-Mini (1B) OSFT** | ~60 | ~45 | ~40 | ~54 | ~64 |
| **LLaMA-3.2-1B** | ~49 | ~35 | ~30 | ~47 | ~60 |
| **LLaMA-3.2-3B** | ~63 | ~55 | ~48 | ~58 | ~72 |
| **Gemma-3-1B** | ~55 | ~40 | ~32 | ~50 | ~63 |
| **Qwen-2.5-1.5B** | ~61 | ~58 | ~45 | ~55 | ~66 |
| **SmolLM2-1.7B** | ~50 | ~32 | ~28 | ~45 | ~58 |
| **DeepSeek-R1-Distill-1.5B** | ~56 | ~62* | ~35 | ~53 | ~60 |
| **TinyLlama-1.1B** | ~25 | ~10 | ~12 | ~33 | ~59 |

> *DeepSeek-R1-Distill excels at math via chain-of-thought reasoning distillation
>
> Note: Scores are approximate from public benchmarks; check papers for exact numbers.

**Key Takeaway**: Phi-4-Mini (1B) punches above its weight thanks to synthetic data, but models like **Qwen-2.5-1.5B** and **DeepSeek-R1-Distill-1.5B** are strong contenders at slightly larger size.

---

## Accuracy Improvement Techniques

```
┌───────────────────────────────────────────────────────────────────────┐
│                   ACCURACY IMPROVEMENT PIPELINE                      │
│                                                                      │
│   ┌─────────┐  ┌──────────┐  ┌──────────┐  ┌───────────────────┐    │
│   │ Block 1 │  │ Block 2  │  │ Block 3  │  │     Block 4       │    │
│   │         │  │          │  │          │  │                   │    │
│   │ Distill │─►│ Fine-tune│─►│ Quantize │─►│ Test-Time Scaling │    │
│   │         │  │          │  │          │  │                   │    │
│   └─────────┘  └──────────┘  └──────────┘  └───────────────────┘    │
│       │            │             │                  │                │
│       ▼            ▼             ▼                  ▼                │
│   Teacher→     LoRA/QLoRA   4-bit/8-bit       Chain-of-thought     │
│   Student      DPO/RLHF    AWQ/GPTQ/QAT      Budget forcing       │
│   knowledge    OSFT                           Beam search          │
│                                                                      │
│   ┌─────────┐  ┌──────────┐  ┌──────────────────────────────────┐   │
│   │ Block 5 │  │ Block 6  │  │          Block 7                 │   │
│   │         │  │          │  │                                  │   │
│   │ MoLoRA  │  │ Prune    │  │   RAG (Retrieval-Augmented)     │   │
│   │         │  │          │  │                                  │   │
│   └─────────┘  └──────────┘  └──────────────────────────────────┘   │
│       │            │                      │                         │
│       ▼            ▼                      ▼                         │
│   Mixture of   SparseGPT            External knowledge              │
│   LoRA experts Wanda/Magnitude      vector DB retrieval             │
└───────────────────────────────────────────────────────────────────────┘
```

---

### Block 1 — Knowledge Distillation

> **Goal**: Transfer reasoning ability from a large "teacher" model to our small "student" model.

```
  ┌──────────────────┐         ┌──────────────────┐
  │   Teacher Model  │         │  Student Model   │
  │   (GPT-4 / 70B) │  ────►  │  (Phi-4-Mini 1B) │
  │                  │  soft   │                  │
  │   Rich knowledge │  labels │  Learns to mimic │
  │   Complex reason │         │  teacher outputs  │
  └──────────────────┘         └──────────────────┘
```

| Technique | Description | Paper |
|---|---|---|
| **Standard KD** | Soft-label matching via KL divergence | [Hinton et al., 2015](https://arxiv.org/abs/1503.02531) |
| **Chain-of-Thought Distillation** | Distill step-by-step reasoning traces | [Ho et al., 2023](https://arxiv.org/abs/2212.10071) |
| **Iterative Layer-wise Distillation** | Progressive layer pruning with KD | [arXiv:2511.05085](https://arxiv.org/abs/2511.05085) |
| **DeepSeek-R1 Style** | RL-trained reasoning → distill to small models | [arXiv:2501.12948](https://arxiv.org/abs/2501.12948) |

**Impact**: Can boost MMLU by **5–15 points** on 1B models. DeepSeek-R1-Distill-1.5B achieves 62+ on GSM8K through this approach.

---

### Block 2 — Fine-Tuning (OSFT / DPO / RLHF / LoRA)

> **Goal**: Align the model to follow instructions and give preferred responses.

| Method | What It Does | Cost | Paper |
|---|---|---|---|
| **OSFT** (Offline Self-play FT) | Model plays against itself to improve | Low | Phi-4-Mini OSFT |
| **LoRA** | Low-Rank Adaptation — trains tiny adapters | Very Low | [arXiv:2106.09685](https://arxiv.org/abs/2106.09685) |
| **QLoRA** | LoRA on quantized (4-bit) base model | Minimal | [arXiv:2305.14314](https://arxiv.org/abs/2305.14314) |
| **DPO** | Direct Preference Optimization — simpler than RLHF | Medium | [arXiv:2305.18290](https://arxiv.org/abs/2305.18290) |
| **RLHF** | Reinforcement Learning from Human Feedback | High | [arXiv:2203.02155](https://arxiv.org/abs/2203.02155) |
| **ORPO** | Odds Ratio Preference Optimization | Low | [arXiv:2403.07691](https://arxiv.org/abs/2403.07691) |

**Recommended pipeline for 1B models**:
```
Base model → LoRA/QLoRA fine-tune on task data → DPO alignment → Deploy
```

**Impact**: LoRA fine-tuning on 1B models shows **10–30% improvement** on task-specific benchmarks with only 0.1–1% additional parameters.

---

### Block 3 — Quantization (QAT / AWQ / GPTQ)

> **Goal**: Compress model weights for faster inference without losing much accuracy.

```
  FP32 (full)  →  FP16 (half)  →  INT8 (8-bit)  →  INT4 (4-bit)
  ───────────────────────────────────────────────────────────────►
  Highest quality                              Fastest / smallest
  4 bytes/param                                0.5 bytes/param
```

| Method | Bits | Accuracy Loss | Speed Gain | Paper |
|---|---|---|---|---|
| **AWQ** (Activation-aware Weight Quant) | 4-bit | ~1–2% | 3–4x | [arXiv:2306.00978](https://arxiv.org/abs/2306.00978) |
| **GPTQ** (Post-training Quantization) | 4-bit | ~1–3% | 3–4x | [arXiv:2210.17323](https://arxiv.org/abs/2210.17323) |
| **QAT** (Quantization-Aware Training) | 4-bit | <1% | 3–4x | [arXiv:2402.10787](https://arxiv.org/abs/2402.10787) |
| **GGUF** (llama.cpp format) | 2–8 bit | varies | CPU-friendly | [llama.cpp](https://github.com/ggerganov/llama.cpp) |
| **SLMQuant** (SLM-specific) | mixed | optimized | varies | [arXiv:2511.13023](https://arxiv.org/abs/2511.13023) |

**Warning**: Small models (1B) have **different quantization sensitivities** than large models. Techniques optimized for 70B may degrade 1B models more. Use SLMQuant-style analysis.

---

### Block 4 — Test-Time Compute Scaling

> **Goal**: Make the model "think longer" at inference to get better answers.

```
  Standard inference:      Input ──► [Model] ──► Answer
                                                 (fast, less accurate)

  Test-time scaling:       Input ──► [Model] ──► Think... ──► Think... ──► Answer
                                                                           (slower, more accurate)
```

| Method | Description | Paper |
|---|---|---|
| **Chain-of-Thought (CoT)** | Prompt model to reason step by step | [arXiv:2201.11903](https://arxiv.org/abs/2201.11903) |
| **Budget Forcing** | Control thinking length — extend or cut | [arXiv:2501.19393](https://arxiv.org/abs/2501.19393) |
| **Best-of-N Sampling** | Generate N answers, pick best via verifier | Standard technique |
| **Beam Search + Verifier** | Tree search over reasoning paths | [arXiv:2504.00294](https://arxiv.org/abs/2504.00294) |
| **Latency-aware TTS** | Parallel branches + speculative decoding | [arXiv:2505.19634](https://arxiv.org/abs/2505.19634) |

**Impact**: A 3B model with test-time scaling achieves **72.4% on MATH-500** within 10 seconds — comparable to much larger models.

---

### Block 5 — Mixture-of-LoRAs (MoLoRA)

> **Goal**: Combine multiple specialist LoRA adapters like a Mixture-of-Experts — used in Phi-4-Mini itself.

```
                    ┌─── LoRA Expert (Math) ───┐
                    │                          │
  Input ──► Router ─┼─── LoRA Expert (Code) ───┼──► Merged Output
                    │                          │
                    └─── LoRA Expert (Chat) ───┘
```

| Feature | Description |
|---|---|
| **How** | Train separate LoRA adapters per domain, combine with learned router |
| **Why** | Specialist adapters outperform single generalist fine-tune |
| **Cost** | Low — each LoRA is tiny (0.1% of model params) |
| **Paper** | [Phi-4-Mini Technical Report](https://arxiv.org/abs/2503.01743) |

---

### Block 6 — Pruning & Sparsity

> **Goal**: Remove unimportant weights to make the model smaller/faster.

| Method | Description | Paper |
|---|---|---|
| **SparseGPT** | One-shot unstructured pruning | [arXiv:2301.00774](https://arxiv.org/abs/2301.00774) |
| **Wanda** | Pruning by weights and activations | [arXiv:2306.11695](https://arxiv.org/abs/2306.11695) |
| **Structured Pruning** | Remove entire attention heads / layers | Various |

**Caution**: For SLMs, pruning is **less effective than quantization**. Quantization consistently wins on compression fidelity for small models.

---

### Block 7 — Retrieval-Augmented Generation (RAG)

> **Goal**: Give the small model access to external knowledge at inference time.

```
  Query ──► [Retriever] ──► Relevant docs ──► [LLM + Context] ──► Answer
                │                                    │
            Vector DB                         Model doesn't need
            (FAISS/Chroma)                    to memorize everything
```

| Component | Options |
|---|---|
| **Embedding model** | BGE, E5, Nomic-Embed |
| **Vector store** | FAISS, ChromaDB, Qdrant |
| **Chunking** | Semantic chunking, 512-token windows |
| **Reranking** | Cross-encoder reranking for precision |

**Impact**: A 1B model + RAG can match a 7B model on knowledge-intensive tasks without retraining.

---

## Next-Gen Architectures (Beyond Transformers)

These aren't just improvements to transformers — they're **entirely new ways** to build language models.

### 1. Diffusion Language Models

> Instead of generating tokens one-by-one (left→right), generate **all tokens in parallel** and refine them iteratively — like how image diffusion works, but for text.

```
  Autoregressive (GPT-style):
    [The] → [cat] → [sat] → [on] → [the] → [mat]     (sequential, slow)

  Diffusion (MDLM-style):
    [???] [???] [???] [???] [???] [???]                 (noisy start)
    [The] [???] [sat] [???] [???] [mat]                 (partial denoise)
    [The] [cat] [sat] [on]  [the] [mat]                 (clean output, parallel!)
```

| Model | Speed | Description | Paper |
|---|---|---|---|
| **MDLM** | Competitive | Masked Diffusion Language Model | [arXiv:2406.07524](https://arxiv.org/abs/2406.07524) |
| **SEDD** | Competitive | Score Entropy Discrete Diffusion | [arXiv:2310.16834](https://arxiv.org/abs/2310.16834) |
| **Mercury Coder** | **1,109 tok/s** | Commercial diffusion LLM for code | [arXiv:2506.17298](https://arxiv.org/abs/2506.17298) |
| **Seed Diffusion** | **2,146 tok/s** | Fastest diffusion LM (Aug 2025) | [arXiv:2508.02193](https://arxiv.org/abs/2508.02193) |
| **Hybrid AR+Diffusion** | Flexible | Unifies both via "hyperschedules" | [arXiv:2504.06416](https://arxiv.org/abs/2504.06416) |

**Why it matters for small models**: Diffusion models can fix mistakes (bidirectional), generate in parallel (faster), and achieve **10x speed** over autoregressive models at similar quality.

---

### 2. Recursive / Recurrent LLMs (State-Space Models)

> Replace attention (O(n^2) memory) with **linear-time recurrence** — constant memory regardless of sequence length.

```
  Transformer attention:     Memory = O(n²)    ← explodes with long sequences
  State-Space Model (SSM):   Memory = O(1)     ← constant, streams forever
```

| Model | Params | Architecture | Key Innovation | Paper |
|---|---|---|---|---|
| **Mamba** | 130M–2.8B | Selective SSM | Input-dependent state transitions | [arXiv:2312.00752](https://arxiv.org/abs/2312.00752) |
| **Mamba-2** | up to 2.7B | SSD (State Space Duality) | 2–8x faster than Mamba-1 | [arXiv:2405.21060](https://arxiv.org/abs/2405.21060) |
| **Mamba-3** | varies | Complex SSM | Multi-I/O, complex state updates | [OpenReview](https://openreview.net/forum?id=HwCvaJOiCj) |
| **RWKV-6** | 0.1B–14B | Linear attention RNN | Dual mode: parallel train + recurrent infer | [arXiv:2404.05892](https://arxiv.org/abs/2404.05892) |
| **Griffin** | up to 14B | Gated Linear Recurrence | Google's recurrent alternative | [arXiv:2402.19427](https://arxiv.org/abs/2402.19427) |
| **RecurrentGemma** | 2B/9B | Griffin-based | Google's open recurrent model | [arXiv:2404.07839](https://arxiv.org/abs/2404.07839) |

**Why it matters for small models**: SSMs are **perfect for edge/mobile** — constant memory means you can process unlimited context on a phone.

---

### 3. Hybrid Architectures

> Combine the best of transformers + SSMs + diffusion.

| Model | What It Combines | Benefit | Paper |
|---|---|---|---|
| **Samba** (3.8B) | Mamba + Sliding Window Attention | 3.73x throughput, 256K context | [arXiv:2406.07522](https://arxiv.org/abs/2406.07522) |
| **Jamba** (52B/398B MoE) | Mamba + Transformer + MoE | Longest context hybrid | [arXiv:2403.19887](https://arxiv.org/abs/2403.19887) |
| **Zamba** (7B) | Mamba + shared Transformer block | 2x throughput, shared attention | [arXiv:2405.18712](https://arxiv.org/abs/2405.18712) |
| **Hymba** (1.5B) | Mamba heads + Attention heads in parallel | Outperforms all 1B models | [arXiv:2411.13676](https://arxiv.org/abs/2411.13676) |
| **Mamba+TRM Recursive** | Mamba-2 inside recursive reasoning | +2% on ARC-AGI reasoning | [arXiv:2602.12078](https://arxiv.org/abs/2602.12078) |

**Hymba (1.5B)** is particularly relevant — it's a hybrid Mamba+Attention model that **outperforms all sub-2B models** including Phi-4-Mini (1B).

---

## Full Improvement Pipeline

Recommended order to maximize accuracy from a 1B base model:

```
Step 1: START
  │
  ▼
Step 2: Knowledge Distillation
  │     Teacher (7B–70B) → Student (1B)
  │     Chain-of-thought distillation for reasoning
  ▼
Step 3: LoRA / QLoRA Fine-Tuning
  │     Task-specific + domain-specific data
  │     Mixture-of-LoRAs for multi-domain
  ▼
Step 4: Alignment (DPO or OSFT)
  │     Preference data for better responses
  ▼
Step 5: Quantization (AWQ 4-bit)
  │     Compress for deployment
  │     Use QAT if accuracy-critical
  ▼
Step 6: Test-Time Scaling
  │     Chain-of-thought prompting
  │     Budget forcing for reasoning tasks
  ▼
Step 7: RAG Integration
  │     External knowledge for factual tasks
  ▼
Step 8: DEPLOY
  │
  ▼
OPTIONAL: Consider next-gen architecture
  ├── Hymba (1.5B) — hybrid Mamba+Attention
  ├── Diffusion LM — for code generation speed
  └── RWKV — for infinite-context streaming
```

**Expected cumulative improvement** (approximate, task-dependent):

| Stage | MMLU (est.) | Math (est.) | Notes |
|---|---|---|---|
| Base Phi-4-Mini (1B) | ~58 | ~42 | Starting point |
| + Distillation | ~65 | ~52 | +7 / +10 |
| + LoRA fine-tune | ~68 | ~58 | +3 / +6 |
| + DPO alignment | ~69 | ~59 | +1 / +1 (quality of responses) |
| + Test-time scaling | ~71 | ~65 | +2 / +6 (inference compute) |
| + RAG | ~74* | ~65 | +3 on knowledge tasks |

> *RAG primarily helps knowledge-intensive tasks (MMLU, TriviaQA), not pure reasoning.

---

## Paper References

### Core Techniques

| # | Topic | Paper | Link |
|---|---|---|---|
| 1 | Knowledge Distillation | *Distilling the Knowledge in a Neural Network* | [arXiv:1503.02531](https://arxiv.org/abs/1503.02531) |
| 2 | CoT Distillation | *Large Language Models Are Reasoning Teachers* | [arXiv:2212.10071](https://arxiv.org/abs/2212.10071) |
| 3 | LoRA | *Low-Rank Adaptation of Large Language Models* | [arXiv:2106.09685](https://arxiv.org/abs/2106.09685) |
| 4 | QLoRA | *Efficient Finetuning of Quantized LLMs* | [arXiv:2305.14314](https://arxiv.org/abs/2305.14314) |
| 5 | DPO | *Direct Preference Optimization* | [arXiv:2305.18290](https://arxiv.org/abs/2305.18290) |
| 6 | RLHF | *Training LMs to Follow Instructions with Human Feedback* | [arXiv:2203.02155](https://arxiv.org/abs/2203.02155) |
| 7 | AWQ | *Activation-aware Weight Quantization* | [arXiv:2306.00978](https://arxiv.org/abs/2306.00978) |
| 8 | GPTQ | *Accurate Post-Training Quantization for Generative LMs* | [arXiv:2210.17323](https://arxiv.org/abs/2210.17323) |
| 9 | SparseGPT | *Massive Pruning in One Shot* | [arXiv:2301.00774](https://arxiv.org/abs/2301.00774) |
| 10 | Chain-of-Thought | *Chain-of-Thought Prompting Elicits Reasoning* | [arXiv:2201.11903](https://arxiv.org/abs/2201.11903) |
| 11 | Test-Time Compute | *Inference-Time Scaling for Complex Tasks* | [arXiv:2504.00294](https://arxiv.org/abs/2504.00294) |
| 12 | SLM Quantization | *SLMQuant: Small Language Model Quantization* | [arXiv:2511.13023](https://arxiv.org/abs/2511.13023) |

### Next-Gen Architectures

| # | Topic | Paper | Link |
|---|---|---|---|
| 13 | MDLM | *Simple and Effective Masked Diffusion Language Models* | [arXiv:2406.07524](https://arxiv.org/abs/2406.07524) |
| 14 | SEDD | *Discrete Diffusion Modeling by Score Entropy* | [arXiv:2310.16834](https://arxiv.org/abs/2310.16834) |
| 15 | Mercury | *Ultra-Fast Language Models Based on Diffusion* | [arXiv:2506.17298](https://arxiv.org/abs/2506.17298) |
| 16 | Mamba | *Linear-Time Sequence Modeling with Selective SSM* | [arXiv:2312.00752](https://arxiv.org/abs/2312.00752) |
| 17 | Mamba-2 | *Transformers are SSMs* | [arXiv:2405.21060](https://arxiv.org/abs/2405.21060) |
| 18 | RWKV | *Eagle and Finch: RWKV with Matrix-Valued States* | [arXiv:2404.05892](https://arxiv.org/abs/2404.05892) |
| 19 | Samba | *Simple Hybrid State Space Models* | [arXiv:2406.07522](https://arxiv.org/abs/2406.07522) |
| 20 | Hymba | *A Hybrid-head Architecture for Small LMs* | [arXiv:2411.13676](https://arxiv.org/abs/2411.13676) |
| 21 | Jamba | *A Hybrid Transformer-Mamba Language Model* | [arXiv:2403.19887](https://arxiv.org/abs/2403.19887) |
| 22 | RecurrentGemma | *Moving Past Transformers with Recurrent Models* | [arXiv:2404.07839](https://arxiv.org/abs/2404.07839) |

### Model Reports

| # | Model | Paper | Link |
|---|---|---|---|
| 23 | Phi-4-Mini | *Compact yet Powerful via Mixture-of-LoRAs* | [arXiv:2503.01743](https://arxiv.org/abs/2503.01743) |
| 24 | DeepSeek-R1 | *Incentivizing Reasoning via Reinforcement Learning* | [arXiv:2501.12948](https://arxiv.org/abs/2501.12948) |
| 25 | LLaMA-3 | *The Llama 3 Herd of Models* | [arXiv:2407.21783](https://arxiv.org/abs/2407.21783) |
| 26 | Gemma-3 | *Technical Report* | [arXiv:2503.19786](https://arxiv.org/abs/2503.19786) |
| 27 | Qwen-2.5 | *Technical Report* | [arXiv:2412.15115](https://arxiv.org/abs/2412.15115) |

---

*Last updated: February 2026*
