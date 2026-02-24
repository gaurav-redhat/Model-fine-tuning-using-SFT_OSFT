"""
LoRALinear — Core Implementation.

This is the heart of LoRA — wrapping nn.Linear to inject a trainable
low-rank branch while keeping the original weights frozen.

Forward: h = xW^T + (α/r) · x @ A @ B
Merge:   W' = W + (α/r) · (AB)^T  →  zero-overhead nn.Linear

"""

import math
import torch
import torch.nn as nn
import torch.nn.functional as F

from .base import LoRALayerBase


class LoRALinear(nn.Linear, LoRALayerBase):
    """
    LoRA wrapper around nn.Linear.

    Forward: h = xW^T + (α/r) · x @ A @ B
    The original weight W is frozen; only A and B are trained.
    """

    def __init__(
        self,
        in_features,
        out_features,
        bias=True,
        rank=8,
        lora_alpha=8,
        lora_dropout=0.0,
        use_rslora=True,
        **kwargs,
    ):
        # Step 1: Initialize the standard nn.Linear (creates self.weight)
        nn.Linear.__init__(
            self,
            in_features,
            out_features,
            bias=bias,
            **kwargs,
        )

        # Step 2: Initialize LoRA hyperparameters (scaling, dropout, etc.)
        LoRALayerBase.__init__(
            self,
            rank=rank,
            lora_alpha=lora_alpha,
            lora_dropout=lora_dropout,
            use_rslora=use_rslora,
        )

        assert rank > 0, "If Rank is 0, Why are you doing LoRA?"

        # Step 3: Freeze the original weight — this is the whole point of LoRA
        self.weight.requires_grad = False

        # Step 4: Define low-rank matrices A and B as raw Parameters
        # Shape: A is (in_features, rank), B is (rank, out_features)
        # Note: nn.Linear stores weight as (out, in), but we define A,B
        # in the natural (in→rank→out) direction to avoid extra transposes.
        self.lora_A = nn.Parameter(torch.zeros(in_features, rank))
        self.lora_B = nn.Parameter(torch.zeros(rank, out_features))

        # Step 5: Initialize A with Kaiming uniform, B with zeros
        # This guarantees ΔW = AB = 0 at initialization:
        # the model starts exactly at the pre-trained solution.
        nn.init.kaiming_uniform_(self.lora_A, a=math.sqrt(5))

    def _merge_weights(self):
        """
        Merge LoRA into the base weight: W' = W + (α/r) · (AB)^T
        Returns a new nn.Linear with merged weights (no LoRA layers).
        """
        lora_update = self.lora_A @ self.lora_B
        lora_update_scaled = lora_update.T * self.scaling
        merged_weight = self.weight.data + lora_update_scaled

        has_bias = self.bias is not None

        state_dict = {"weight": merged_weight}
        if has_bias:
            state_dict["bias"] = self.bias

        merged_linear = nn.Linear(
            self.in_features,
            self.out_features,
            bias=has_bias,
        )
        merged_linear.load_state_dict(state_dict)
        return merged_linear

    def forward(self, x):
        # Path 1: Frozen pre-trained weights (standard F.linear)
        base_output = F.linear(x, self.weight, bias=self.bias)

        # Path 2: Low-rank adaptation
        # x @ A @ B computes the rank-r update
        # dropout is only on this path, not the frozen path
        lora_product = self.lora_A @ self.lora_B
        lora_scaled = lora_product * self.scaling

        dropped_x = self.lora_dropout(x)
        lora_output = dropped_x @ lora_scaled

        return base_output + lora_output

    def extra_repr(self):
        info = (
            f"in_features={self.in_features}, "
            f"out_features={self.out_features}, "
            f"rank={self.rank}, "
            f"alpha={self.lora_alpha}, "
            f"scaling={self.scaling:.4f}"
        )
        return info
