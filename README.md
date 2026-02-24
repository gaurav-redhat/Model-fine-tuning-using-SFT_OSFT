# LoRA Fine-Tuning Framework

A from-scratch implementation of **LoRA (Low-Rank Adaptation)** for parameter-efficient fine-tuning of transformer models, with support for **Adaptive LoRA** (AdaLoRA) rank pruning.

## Project Structure

```
├── model/
│   ├── base.py            # LoRALayerBase — shared hyperparameters & scaling
│   ├── LinearLora.py      # LoRALinear — core LoRA on nn.Linear
│   ├── EmbeddingLora.py   # LoRAEmbedding — LoRA on nn.Embedding
│   ├── ConvLora.py        # LoRAConv2d — LoRA on nn.Conv2d
│   ├── adaptive.py        # AdaptiveLoRALinear — gated rank pruning
│   └── wrapper.py         # LoraModel — automatic injection wrapper + LoraConfig
├── config.py              # Training configuration dataclass
├── dataset.py             # Dataset utilities (HuggingFace / CSV loading)
├── loss.py                # Label smoothing & composite loss with gate regularisation
└── finetune.py            # Training script
```

## Features

- **Layer support** — Linear, Conv2d, and Embedding LoRA variants
- **Adaptive LoRA** — learned per-rank importance gates with pruning
- **Rank-stabilized scaling** (rsLoRA) — `α/√r` instead of `α/r`
- **Automatic injection** — recursively replaces target layers by name
- **Weight merging** — fold LoRA into base weights for zero-overhead inference
- **Adapter-only saving** — save just the tiny LoRA parameters via safetensors

## Quick Start

```bash
# Install dependencies
pip install torch transformers datasets safetensors

# Standard LoRA fine-tuning on IMDB
python finetune.py --no_adaptive --epochs 3

# Adaptive LoRA with rank pruning
python finetune.py --adaptive --rank 16 --prune_at_epoch 2
```


## References

- [LoRA: Low-Rank Adaptation of Large Language Models](https://arxiv.org/abs/2106.09685) (Hu et al., 2021)
- [AdaLoRA: Adaptive Budget Allocation for Parameter-Efficient Fine-Tuning](https://arxiv.org/abs/2303.10512) (Zhang et al., 2023)
- [Training Hub](https://github.com/Red-Hat-AI-Innovation-Team/training_hub) — Algorithm-focused interface for LLM training, continual learning, and reinforcement learning (Red Hat AI Innovation Team)
- [SDG Hub](https://github.com/Red-Hat-AI-Innovation-Team/sdg_hub) — Synthetic Data Generation Toolkit for LLMs (Red Hat AI Innovation Team)
