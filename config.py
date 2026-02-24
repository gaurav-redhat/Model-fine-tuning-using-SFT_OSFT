"""
Training configuration dataclass.

All hyperparameters live here so they can be serialised, logged, and
overridden from the CLI.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import List, Optional


@dataclass
class LoRAConfig:

    # --- Model ---
    model_name: str = "bert-base-uncased"
    num_labels: int = 2
    target_modules: List[str] = field(
        default_factory=lambda: ["query", "value"]
    )

    # --- LoRA ---
    rank: int = 8
    alpha: float = 16.0
    lora_dropout: float = 0.05
    adaptive: bool = True
    gate_init: float = 5.0

    # --- Dataset ---
    dataset_name: str = "imdb"
    text_column: str = "text"
    label_column: str = "label"
    max_length: int = 256

    # --- Training ---
    epochs: int = 3
    batch_size: int = 16
    learning_rate: float = 3e-4
    weight_decay: float = 0.01
    warmup_ratio: float = 0.06
    max_grad_norm: float = 1.0
    label_smoothing: float = 0.0

    # --- Adaptive LoRA ---
    gate_lambda: float = 1e-4
    prune_threshold: float = 0.1
    prune_at_epoch: Optional[int] = 2

    # --- Infrastructure ---
    output_dir: str = "checkpoints"
    seed: int = 42
    fp16: bool = True
    num_workers: int = 2
    log_every: int = 50
