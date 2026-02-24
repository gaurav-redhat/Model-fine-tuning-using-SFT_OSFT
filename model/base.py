"""
LoRALayerBase — The Foundation.

Every LoRA layer (Linear, Embedding, Conv2d) shares the same hyperparameters:
rank, alpha, scaling, and dropout. We extract these into a base class using
Python's multiple inheritance.

"""

import torch.nn as nn


class LoRALayerBase:
    """
    Base class for all LoRA layers.

    Stores shared hyperparameters and computes the scaling factor.

    Args:
        rank:         Low-rank dimension r
        lora_alpha:   Scaling constant α
        lora_dropout: Dropout probability on the LoRA path
        use_rslora:   If True, use rank-stabilized scaling α/√r
                      instead of standard α/r
    """

    def __init__(
        self,
        rank=8,
        lora_alpha=8,
        lora_dropout=0.0,
        use_rslora=True,
    ):
        self.rank = rank
        self.lora_alpha = lora_alpha

        # Standard LoRA: scale = α/r  → output variance decreases with rank
        # rsLoRA:        scale = α/√r → output variance stays constant
        if use_rslora:
            self.scaling = self.lora_alpha / (self.rank ** 0.5)
        else:
            self.scaling = self.lora_alpha / self.rank

        # Dropout on the LoRA path only — base model path is unaffected
        if lora_dropout > 0:
            self.lora_dropout = nn.Dropout(lora_dropout)
        else:
            self.lora_dropout = lambda x: x

    def _load_pretrained_weights(self, state_dict):
        """Copy pre-trained weights from the original layer into this LoRA layer."""
        self.weight.data = state_dict["weight"]
        if "bias" in state_dict:
            self.bias.data = state_dict["bias"]
