# LLM Model Family Tree & Summary

## Family Tree Diagram

```
                          ┌─────────────────────────────────────────────────┐
                          │             LLM  FAMILY  TREE                   │
                          └─────────────────────────────────────────────────┘


═══════════════════════════════════════════════════════════════════════════════
  MICROSOFT  PHI  FAMILY                          
═══════════════════════════════════════════════════════════════════════════════

  Phi-1 (1.3B, Jun 2023)
    │   └── "Textbooks Are All You Need"
    │
    ├── Phi-1.5 (1.3B, Sep 2023)
    │       └── Improved reasoning with synthetic data
    │
    ├── Phi-2 (2.7B, Dec 2023)
    │       └── Scaled up, strong benchmark performance
    │
    ├── Phi-3 Family (Apr 2024)
    │   ├── Phi-3-Mini (3.8B)
    │   ├── Phi-3-Small (7B)
    │   └── Phi-3-Medium (14B)
    │
    └── Phi-4 Family (Dec 2024 – Feb 2025)
        ├── Phi-4 (14B)
        │       └── Synthetic data + curated web data
        │
        └── Phi-4-Mini (3.8B) ◄── (Feb 2025)
            │   └── Compact model, strong reasoning
            │
            ├───────────────────────────────────┐
            │                                   │
       ★ Phi-4-Mini (1B)               ★ Phi-4-Mini (1B) OSFT
         "Very light,                    "Fine-tuned variant,
          low-resource devices"           better responses"


═══════════════════════════════════════════════════════════════════════════════
  ALIBABA  QWEN  FAMILY
═══════════════════════════════════════════════════════════════════════════════

  Qwen (7B, Aug 2023)
    │   └── Alibaba Cloud's first open LLM
    │
    ├── Qwen-1.5 (Feb 2024)
    │   ├── 0.5B / 1.8B / 4B / 7B / 14B / 32B / 72B / 110B
    │   └── Improved multilingual + longer context
    │
    └── Qwen-2 Family (Jun 2024)
        ├── Qwen-2 (0.5B – 72B)
        │   └── GQA, dual-SWA, multilingual (29 langs)
        │
        └── Qwen-2.5 Family (Sep 2024)
            ├── Qwen-2.5 (0.5B – 72B)
            ├── Qwen-2.5-Coder
            ├── Qwen-2.5-Math
            │
            └── ★ Qwen-2.5 (7B) ◄── Our target
                  "Balanced code + text,
                   mid-range performance"


═══════════════════════════════════════════════════════════════════════════════
  OPENAI  GPT  FAMILY
═══════════════════════════════════════════════════════════════════════════════

  GPT-1 (117M, Jun 2018)
    │   └── Generative Pre-Training on books corpus
    │
    ├── GPT-2 (1.5B, Feb 2019)
    │       └── Zero-shot multitask, WebText dataset
    │
    ├── GPT-3 (175B, May 2020)
    │   │   └── Few-shot learning, scaling laws
    │   │
    │   └── GPT-3.5 / InstructGPT (2022)
    │       └── RLHF alignment, ChatGPT backbone
    │
    ├── GPT-4 (Mar 2023)
    │   │   └── Multimodal (text + vision), massive scale
    │   │
    │   ├── GPT-4 Turbo (Nov 2023)
    │   │       └── 128K context, cheaper, faster
    │   │
    │   └── GPT-4o (May 2024)
    │       │   └── Omni: native audio/image/text
    │       │
    │       └── GPT-4o-mini (Jul 2024)
    │               └── Cost-efficient smaller variant
    │
    └── ★ GPT-4.1 (Apr 2025) ◄── Our target
          "Top performance,
           complex reasoning & high quality"


═══════════════════════════════════════════════════════════════════════════════
  META  LLaMA  FAMILY
═══════════════════════════════════════════════════════════════════════════════

  LLaMA-1 (7B–65B, Feb 2023)
    │   └── Open-weight foundation model, research-only license
    │
    ├── LLaMA-2 (7B–70B, Jul 2023)
    │   │   └── Commercial license, RLHF chat variants
    │   │
    │   └── Code Llama (7B–34B, Aug 2023)
    │           └── Code-specialized fine-tune
    │
    ├── LLaMA-3 Family (Apr 2024)
    │   ├── LLaMA-3-8B
    │   ├── LLaMA-3-70B
    │   │   └── 15T tokens, GQA, 8K context
    │   │
    │   └── LLaMA-3.1 (Jul 2024)
    │       ├── LLaMA-3.1-8B
    │       ├── LLaMA-3.1-70B
    │       └── LLaMA-3.1-405B ◄── Largest open model
    │               └── 128K context, tool use, multilingual
    │
    ├── LLaMA-3.2 (Sep 2024)
    │   ├── LLaMA-3.2-1B  ◄── Lightweight / on-device
    │   ├── LLaMA-3.2-3B  ◄── Lightweight / on-device
    │   ├── LLaMA-3.2-11B-Vision
    │   └── LLaMA-3.2-90B-Vision
    │           └── Multimodal (text + image)
    │
    └── LLaMA-4 (Apr 2025)
        ├── LLaMA-4-Scout (17B active / 109B total, 16 experts)
        │       └── 10M token context, MoE architecture
        └── LLaMA-4-Maverick (17B active / 400B total, 128 experts)
                └── Top open-weight model, MoE architecture


═══════════════════════════════════════════════════════════════════════════════
  GOOGLE  GEMMA  /  GEMINI  FAMILY
═══════════════════════════════════════════════════════════════════════════════

  Gemini 1.0 (Dec 2023)
    │   ├── Gemini Ultra / Pro / Nano
    │   └── Multimodal from the ground up
    │
    ├── Gemini 1.5 (Feb 2024)
    │   ├── Gemini 1.5 Pro (MoE)
    │   │       └── 1M–2M token context window
    │   └── Gemini 1.5 Flash
    │           └── Lightweight, fast inference
    │
    ├── Gemini 2.0 (Dec 2024)
    │   └── Gemini 2.0 Flash
    │           └── Agentic capabilities, tool use
    │
    └── Gemini 2.5 (Mar 2025)
        └── Gemini 2.5 Pro
                └── Thinking model, hybrid reasoning

  Gemma (open-weight, derived from Gemini research):

  Gemma-1 (Feb 2024)
    │   ├── Gemma-2B
    │   └── Gemma-7B
    │       └── Lightweight open models from Google DeepMind
    │
    ├── Gemma-2 (Jun 2024)
    │   ├── Gemma-2-2B
    │   ├── Gemma-2-9B
    │   └── Gemma-2-27B
    │       └── Knowledge distillation, sliding window attention
    │
    └── Gemma-3 (Mar 2025)
        ├── Gemma-3-1B
        ├── Gemma-3-4B
        ├── Gemma-3-12B
        └── Gemma-3-27B
            └── Multimodal (vision+text), 128K context


═══════════════════════════════════════════════════════════════════════════════
  MISTRAL  AI  FAMILY
═══════════════════════════════════════════════════════════════════════════════

  Mistral-7B (Sep 2023)
    │   └── Sliding window attention, GQA, open-weight
    │
    ├── Mixtral-8x7B (Dec 2023)
    │       └── Sparse MoE (12.9B active / 46.7B total)
    │
    ├── Mixtral-8x22B (Apr 2024)
    │       └── Larger MoE (39B active / 141B total)
    │
    ├── Mistral Small (Sep 2024 → v25.01)
    │       └── 22B–24B, cost-efficient workhorse
    │
    ├── Mistral NeMo (Jul 2024)
    │       └── 12B, with NVIDIA, Tekken tokenizer, 128K ctx
    │
    ├── Mistral Large (Feb 2024 → v25.01)
    │       └── 123B, flagship reasoning model
    │
    ├── Codestral (May 2024)
    │       └── 22B, code-specialized (80+ languages)
    │
    └── Pixtral (Sep 2024)
        ├── Pixtral-12B  ◄── Multimodal (vision)
        └── Pixtral Large (124B)


═══════════════════════════════════════════════════════════════════════════════
  DEEPSEEK  FAMILY  (China)
═══════════════════════════════════════════════════════════════════════════════

  DeepSeek LLM (Nov 2023)
    │   ├── DeepSeek-7B
    │   └── DeepSeek-67B
    │       └── 2T tokens, open-weight foundation
    │
    ├── DeepSeek-Coder (Nov 2023)
    │       └── 1.3B / 6.7B / 33B, code-specialized
    │
    ├── DeepSeek-V2 (May 2024)
    │       └── 236B (21B active), MoE + Multi-head Latent Attention (MLA)
    │
    ├── DeepSeek-V2.5 (Sep 2024)
    │       └── Merged general + code capabilities
    │
    ├── DeepSeek-V3 (Dec 2024)
    │   │   └── 671B (37B active), MoE + MLA, FP8 training
    │   │       trained for $5.6M — extremely cost-efficient
    │   │
    │   └── DeepSeek-V3-0324 (Mar 2025)
    │           └── Improved reasoning checkpoint
    │
    └── DeepSeek-R1 (Jan 2025)
        │   └── 671B (37B active), reasoning via long chain-of-thought
        │       RL-trained "thinking" model
        │
        └── DeepSeek-R1-Distill
            ├── R1-Distill-Qwen-1.5B
            ├── R1-Distill-Qwen-7B
            ├── R1-Distill-Qwen-14B
            ├── R1-Distill-Qwen-32B
            ├── R1-Distill-Llama-8B
            └── R1-Distill-Llama-70B
                └── Reasoning distilled into smaller open models


═══════════════════════════════════════════════════════════════════════════════
  ANTHROPIC  CLAUDE  FAMILY
═══════════════════════════════════════════════════════════════════════════════

  Claude 1 (Mar 2023)
    │   ├── Claude / Claude Instant
    │   └── Constitutional AI (RLHF + RLAIF)
    │
    ├── Claude 2 (Jul 2023)
    │       └── 100K context, improved coding & math
    │
    ├── Claude 3 Family (Mar 2024)
    │   ├── Claude 3 Haiku    ◄── Fast & light
    │   ├── Claude 3 Sonnet   ◄── Balanced
    │   └── Claude 3 Opus     ◄── Top intelligence
    │       └── Multimodal (vision), 200K context
    │
    ├── Claude 3.5 Family (Jun–Oct 2024)
    │   ├── Claude 3.5 Sonnet  ◄── Surpassed Opus
    │   └── Claude 3.5 Haiku
    │       └── Computer use, agentic tools
    │
    └── Claude 4 Family (2025)
        ├── Claude 4 Haiku
        ├── Claude 4 Sonnet
        └── Claude 4 Opus
            └── Extended thinking, parallel tool use


═══════════════════════════════════════════════════════════════════════════════
  COHERE  COMMAND  FAMILY
═══════════════════════════════════════════════════════════════════════════════

  Command (2023)
    │   └── Enterprise-focused generative model
    │
    ├── Command R (Mar 2024)
    │       └── 35B, RAG-optimized, 128K context
    │
    ├── Command R+ (Apr 2024)
    │       └── 104B, top-tier RAG & multilingual
    │
    └── Command A (Mar 2025)
            └── Agentic + RAG, "best for enterprise"


═══════════════════════════════════════════════════════════════════════════════
  01.AI  YI  FAMILY  (China)
═══════════════════════════════════════════════════════════════════════════════

  Yi-1 (Nov 2023)
    │   ├── Yi-6B / Yi-34B
    │   └── Bilingual (EN/ZH), 200K context for 6B
    │
    ├── Yi-1.5 (May 2024)
    │   ├── Yi-1.5-6B / 9B / 34B
    │   └── 3.6T tokens, strong chat & code
    │
    └── Yi-Lightning (2024)
            └── Fast inference variant


═══════════════════════════════════════════════════════════════════════════════
  xAI  GROK  FAMILY
═══════════════════════════════════════════════════════════════════════════════

  Grok-1 (Nov 2023, open-weight Mar 2024)
    │   └── 314B MoE (86B active), open under Apache 2.0
    │
    ├── Grok-2 (Aug 2024)
    │       └── Improved reasoning, multimodal
    │
    └── Grok-3 (Feb 2025)
            └── Trained on Colossus (200K H100s), "thinking" mode


═══════════════════════════════════════════════════════════════════════════════
  APPLE  ON-DEVICE  FAMILY
═══════════════════════════════════════════════════════════════════════════════

  OpenELM (Apr 2024)
    │   ├── 270M / 450M / 1.1B / 3B
    │   └── Layer-wise scaling, fully reproducible open-source
    │
    └── Apple Foundation Models — AFM (Jun 2024)
        ├── AFM-on-device (~3B)
        │       └── Runs on iPhone/iPad/Mac
        └── AFM-server
                └── Private Cloud Compute


═══════════════════════════════════════════════════════════════════════════════
  STABILITY  AI  FAMILY
═══════════════════════════════════════════════════════════════════════════════

  StableLM (Apr 2023)
    │   ├── StableLM-3B / 7B
    │   └── Based on EleutherAI's GPT-NeoX
    │
    ├── StableLM-2 (Jan 2024)
    │   ├── StableLM-2-1.6B
    │   └── StableLM-2-12B
    │       └── Multilingual, 4K–16K context
    │
    └── Stable Code (Aug 2023)
            └── 3B, code-specialized


═══════════════════════════════════════════════════════════════════════════════
```

---

## Our Model Lineup

| Model | Params | Strength | Best For |
|---|---|---|---|
| **Phi-4Mini (1B)** | ~1B | Very light | Low-resource devices, trivial tasks |
| **Phi-4Mini (1B) OSFT** | ~1B | Fine-tuned | Better small model responses |
| **Qwen-2.5 (7B)** | ~7B | Mid-range | Balanced performance (code/text) |
| **GPT-4.1** | Proprietary (large) | Top performance | Complex reasoning & high quality |

---

## Paper Links & References

### Microsoft Phi Family

| Paper | Year | Link |
|---|---|---|
| **Phi-1**: *Textbooks Are All You Need* | 2023 | [arXiv:2306.11644](https://arxiv.org/abs/2306.11644) |
| **Phi-1.5**: *Textbooks Are All You Need II* | 2023 | [arXiv:2309.05463](https://arxiv.org/abs/2309.05463) |
| **Phi-2**: *The Surprising Power of Small Language Models* | 2023 | [Microsoft Blog](https://www.microsoft.com/en-us/research/blog/phi-2-the-surprising-power-of-small-language-models/) |
| **Phi-3**: *Phi-3 Technical Report* | 2024 | [arXiv:2404.14219](https://arxiv.org/abs/2404.14219) |
| **Phi-4**: *Phi-4 Technical Report* | 2024 | [arXiv:2412.08905](https://arxiv.org/abs/2412.08905) |
| **Phi-4-Mini**: *Phi-4-Mini Technical Report* | 2025 | [arXiv:2503.01743](https://arxiv.org/abs/2503.01743) |

### Alibaba Qwen Family

| Paper | Year | Link |
|---|---|---|
| **Qwen**: *Qwen Technical Report* | 2023 | [arXiv:2309.16609](https://arxiv.org/abs/2309.16609) |
| **Qwen-2**: *Qwen2 Technical Report* | 2024 | [arXiv:2407.10671](https://arxiv.org/abs/2407.10671) |
| **Qwen-2.5**: *Qwen2.5 Technical Report* | 2024 | [arXiv:2412.15115](https://arxiv.org/abs/2412.15115) |
| **Qwen-2.5-Coder**: *Qwen2.5-Coder Technical Report* | 2024 | [arXiv:2409.12186](https://arxiv.org/abs/2409.12186) |

### OpenAI GPT Family

| Paper | Year | Link |
|---|---|---|
| **GPT-1**: *Improving Language Understanding by Generative Pre-Training* | 2018 | [OpenAI Paper](https://cdn.openai.com/research-covers/language-unsupervised/language_understanding_paper.pdf) |
| **GPT-2**: *Language Models are Unsupervised Multitask Learners* | 2019 | [OpenAI Paper](https://cdn.openai.com/better-language-models/language_models_are_unsupervised_multitask_learners.pdf) |
| **GPT-3**: *Language Models are Few-Shot Learners* | 2020 | [arXiv:2005.14165](https://arxiv.org/abs/2005.14165) |
| **InstructGPT**: *Training LMs to Follow Instructions with Human Feedback* | 2022 | [arXiv:2203.02155](https://arxiv.org/abs/2203.02155) |
| **GPT-4**: *GPT-4 Technical Report* | 2023 | [arXiv:2303.08774](https://arxiv.org/abs/2303.08774) |
| **GPT-4.1**: *System Card* | 2025 | [OpenAI GPT-4.1 Page](https://openai.com/index/gpt-4-1/) |

### Meta LLaMA Family

| Paper | Year | Link |
|---|---|---|
| **LLaMA**: *Open and Efficient Foundation Language Models* | 2023 | [arXiv:2302.13971](https://arxiv.org/abs/2302.13971) |
| **LLaMA-2**: *Open Foundation and Fine-Tuned Chat Models* | 2023 | [arXiv:2307.09288](https://arxiv.org/abs/2307.09288) |
| **Code Llama**: *Open Foundation Models for Code* | 2023 | [arXiv:2308.12950](https://arxiv.org/abs/2308.12950) |
| **LLaMA-3**: *The Llama 3 Herd of Models* | 2024 | [arXiv:2407.21783](https://arxiv.org/abs/2407.21783) |
| **LLaMA-4**: *The Llama 4 Herd of Models* | 2025 | [Meta AI Blog](https://ai.meta.com/blog/llama-4-multimodal-intelligence/) |

### Google Gemini / Gemma Family

| Paper | Year | Link |
|---|---|---|
| **Gemini 1.0**: *A Family of Highly Capable Multimodal Models* | 2023 | [arXiv:2312.11805](https://arxiv.org/abs/2312.11805) |
| **Gemini 1.5**: *Unlocking Multimodal Understanding Across Millions of Tokens* | 2024 | [arXiv:2403.05530](https://arxiv.org/abs/2403.05530) |
| **Gemma**: *Open Models Based on Gemini Research* | 2024 | [arXiv:2403.08295](https://arxiv.org/abs/2403.08295) |
| **Gemma-2**: *Improving Open Language Models at a Practical Size* | 2024 | [arXiv:2408.00118](https://arxiv.org/abs/2408.00118) |
| **Gemma-3**: *Technical Report* | 2025 | [arXiv:2503.19786](https://arxiv.org/abs/2503.19786) |

### Mistral AI Family

| Paper | Year | Link |
|---|---|---|
| **Mistral-7B**: *The Best 7B Model to Date* | 2023 | [arXiv:2310.06825](https://arxiv.org/abs/2310.06825) |
| **Mixtral-8x7B**: *Mixtral of Experts* | 2024 | [arXiv:2401.04088](https://arxiv.org/abs/2401.04088) |
| **Mistral Large / NeMo / Small** | 2024 | [Mistral Docs](https://docs.mistral.ai/) |

### DeepSeek Family

| Paper | Year | Link |
|---|---|---|
| **DeepSeek LLM**: *Scaling Open-Source Language Models with Longtermism* | 2024 | [arXiv:2401.02954](https://arxiv.org/abs/2401.02954) |
| **DeepSeek-Coder**: *When the Large Language Model Meets Programming* | 2024 | [arXiv:2401.14196](https://arxiv.org/abs/2401.14196) |
| **DeepSeek-V2**: *A Strong, Economical, and Efficient MoE Model* | 2024 | [arXiv:2405.04434](https://arxiv.org/abs/2405.04434) |
| **DeepSeek-V3**: *Technical Report* | 2024 | [arXiv:2412.19437](https://arxiv.org/abs/2412.19437) |
| **DeepSeek-R1**: *Incentivizing Reasoning in LLMs via RL* | 2025 | [arXiv:2501.12948](https://arxiv.org/abs/2501.12948) |

### Anthropic Claude Family

| Paper | Year | Link |
|---|---|---|
| **Constitutional AI**: *Harmlessness from AI Feedback* | 2022 | [arXiv:2212.08073](https://arxiv.org/abs/2212.08073) |
| **Claude 3**: *Model Card and Evaluations* | 2024 | [Anthropic Blog](https://www.anthropic.com/news/claude-3-family) |
| **Claude 3.5 Sonnet**: *Model Card* | 2024 | [Anthropic Blog](https://www.anthropic.com/news/claude-3-5-sonnet) |
| **Claude 4 / Opus**: *System Card* | 2025 | [Anthropic Blog](https://www.anthropic.com/news/claude-4) |

### Cohere Command Family

| Paper | Year | Link |
|---|---|---|
| **Command R**: *Scalable RAG-Optimized Model* | 2024 | [Cohere Blog](https://cohere.com/blog/command-r) |
| **Command R+**: *Enterprise RAG & Multilingual* | 2024 | [Cohere Blog](https://cohere.com/blog/command-r-plus-microsoft-azure) |
| **Command A**: *Agentic Enterprise Model* | 2025 | [Cohere Blog](https://cohere.com/blog/command-a) |

### 01.AI Yi Family

| Paper | Year | Link |
|---|---|---|
| **Yi**: *Open Foundation Models by 01.AI* | 2024 | [arXiv:2403.04652](https://arxiv.org/abs/2403.04652) |
| **Yi-1.5**: *Technical Report* | 2024 | [arXiv:2412.01253](https://arxiv.org/abs/2412.01253) |

### xAI Grok Family

| Paper | Year | Link |
|---|---|---|
| **Grok-1**: *Open Release* | 2024 | [GitHub](https://github.com/xai-org/grok-1) |
| **Grok-3**: *Announcement* | 2025 | [xAI Blog](https://x.ai/blog/grok-3) |

### Apple On-Device Family

| Paper | Year | Link |
|---|---|---|
| **OpenELM**: *An Efficient Language Model Family with Open Training and Inference* | 2024 | [arXiv:2404.14619](https://arxiv.org/abs/2404.14619) |
| **Apple Intelligence**: *Foundation Language Models (AFM)* | 2024 | [Apple ML Research](https://machinelearning.apple.com/research/introducing-apple-foundation-models) |

### Stability AI Family

| Paper | Year | Link |
|---|---|---|
| **StableLM**: *Stability AI Language Models* | 2023 | [GitHub](https://github.com/Stability-AI/StableLM) |
| **StableLM-2**: *1.6B Technical Report* | 2024 | [arXiv:2402.17834](https://arxiv.org/abs/2402.17834) |

---

## HuggingFace Model Cards

| Model | HuggingFace Link |
|---|---|
| Phi-4-Mini | [microsoft/Phi-4-mini-instruct](https://huggingface.co/microsoft/Phi-4-mini-instruct) |
| Qwen-2.5-7B | [Qwen/Qwen2.5-7B](https://huggingface.co/Qwen/Qwen2.5-7B) |
| Qwen-2.5-7B-Instruct | [Qwen/Qwen2.5-7B-Instruct](https://huggingface.co/Qwen/Qwen2.5-7B-Instruct) |
| LLaMA-3.1-8B | [meta-llama/Llama-3.1-8B](https://huggingface.co/meta-llama/Llama-3.1-8B) |
| LLaMA-3.2-1B | [meta-llama/Llama-3.2-1B](https://huggingface.co/meta-llama/Llama-3.2-1B) |
| Gemma-3-4B | [google/gemma-3-4b-it](https://huggingface.co/google/gemma-3-4b-it) |
| Mistral-7B | [mistralai/Mistral-7B-v0.3](https://huggingface.co/mistralai/Mistral-7B-v0.3) |
| DeepSeek-R1 | [deepseek-ai/DeepSeek-R1](https://huggingface.co/deepseek-ai/DeepSeek-R1) |
| DeepSeek-R1-Distill-Qwen-7B | [deepseek-ai/DeepSeek-R1-Distill-Qwen-7B](https://huggingface.co/deepseek-ai/DeepSeek-R1-Distill-Qwen-7B) |
| Yi-1.5-9B | [01-ai/Yi-1.5-9B](https://huggingface.co/01-ai/Yi-1.5-9B) |
| StableLM-2-1.6B | [stabilityai/stablelm-2-1_6b](https://huggingface.co/stabilityai/stablelm-2-1_6b) |
| Grok-1 | [xai-org/grok-1](https://huggingface.co/xai-org/grok-1) |

---

## Key Architectural Notes

### Phi-4-Mini (1B)
- **Architecture**: Transformer decoder-only
- **Training data**: Heavy use of **synthetic data** generated by larger models + curated textbook-quality web data
- **Key innovation**: Data quality over data quantity — small models can punch above their weight

### Qwen-2.5 (7B)
- **Architecture**: Transformer decoder-only with **Grouped Query Attention (GQA)** and **dual Sliding Window Attention (SWA)**
- **Training data**: 18T+ tokens, 29 languages
- **Key innovation**: Strong code and math specialization via dedicated Coder/Math variants

### GPT-4.1
- **Architecture**: Proprietary (believed to be Mixture-of-Experts)
- **Context window**: 1M tokens
- **Key innovation**: Instruction following, long-context, coding — significant upgrade over GPT-4o

### LLaMA-3 / 3.1 / 3.2
- **Architecture**: Transformer decoder-only with **GQA**, RoPE
- **Training data**: 15T+ tokens (3.1: multilingual 8 langs)
- **Key innovation**: Largest open-weight model at 405B; 3.2 brought small on-device models (1B/3B) and vision

### LLaMA-4
- **Architecture**: **Mixture-of-Experts** (Scout: 16 experts, Maverick: 128 experts)
- **Context window**: 10M tokens (Scout)
- **Key innovation**: Native multimodal, MoE for efficiency, early fusion architecture

### Gemma-3
- **Architecture**: Transformer decoder with **local + global attention** interleaving
- **Training data**: Distilled from Gemini 2.0
- **Key innovation**: Multimodal at small sizes (1B–27B), 128K context, runs on single GPU

### Mistral-7B / Mixtral
- **Architecture**: Transformer + **Sliding Window Attention** (SWA) + **GQA**; Mixtral adds **Sparse MoE**
- **Key innovation**: SWA for efficient long context; MoE for scaling without linear compute cost

### DeepSeek-V3 / R1
- **Architecture**: **MoE + Multi-head Latent Attention (MLA)** — KV-cache compression
- **Training data**: 14.8T tokens; V3 trained for only **$5.6M** (FP8 mixed precision)
- **Key innovation**: MLA drastically reduces memory; R1 pioneered RL-based reasoning without supervised fine-tuning

### Claude 4
- **Architecture**: Proprietary (Constitutional AI alignment)
- **Context window**: 200K tokens
- **Key innovation**: Extended thinking (chain-of-thought), computer use, agentic tool orchestration
