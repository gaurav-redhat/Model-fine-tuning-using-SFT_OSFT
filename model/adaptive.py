"""
Adaptive LoRA layer.

Extends the base LoRALinear by dynamically adjusting the effective rank per
layer based on learned importance scores. A gate vector g ∈ R^r produces
soft masks via sigmoid; low-importance dimensions can be pruned.

Reference: AdaLoRA (Zhang et al., 2023) — "Adaptive Budget Allocation for
Parameter-Efficient Fine-Tuning"
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

from .LinearLora import LoRALinear


class AdaptiveLoRALinear(LoRALinear):
    """
    LoRA layer with per-rank importance gating.

    A learnable gate vector g ∈ R^r produces soft masks via sigmoid.
    During training the effective contribution of rank-i is scaled by g_i.
    At pruning time, ranks below a threshold are zeroed out.
    """

    def __init__(
        self,
        in_features,
        out_features,
        bias=True,
        rank=16,
        lora_alpha=16.0,
        lora_dropout=0.0,
        use_rslora=True,
        gate_init=5.0,
    ):
        super().__init__(
            in_features=in_features,
            out_features=out_features,
            bias=bias,
            rank=rank,
            lora_alpha=lora_alpha,
            lora_dropout=lora_dropout,
            use_rslora=use_rslora,
        )

        self.gate = nn.Parameter(
            torch.full((rank,), gate_init)
        )

    def _gate_mask(self):
        """Soft mask from sigmoid of gate logits — shape (rank,)."""
        return torch.sigmoid(self.gate)

    def forward(self, x):
        # Path 1: Frozen pre-trained weights
        base_output = F.linear(x, self.weight, bias=self.bias)

        # Path 2: Adaptive low-rank branch
        dropped_x = self.lora_dropout(x)

        # Down-project through A
        h = dropped_x @ self.lora_A

        # Gate each rank dimension with learned importance
        gate_mask = self._gate_mask()
        h = h * gate_mask

        # Up-project through B and scale
        lora_output = h @ self.lora_B
        lora_output = lora_output * self.scaling

        return base_output + lora_output

    # ------------------------------------------------------------------
    # Rank pruning utilities
    # ------------------------------------------------------------------

    def importance_scores(self):
        """Return per-rank importance as sigmoid(gate) — detached."""
        scores = self._gate_mask()
        return scores.detach()

    def prune(self, threshold=0.1):
        """
        Zero-out rank dimensions whose importance falls below *threshold*.
        Returns the number of ranks pruned.
        """
        mask = self._gate_mask()
        prune_mask = (mask < threshold)
        pruned_count = prune_mask.sum().item()

        with torch.no_grad():
            self.lora_A.data[:, prune_mask] = 0.0
            self.lora_B.data[prune_mask, :] = 0.0
            self.gate.data[prune_mask] = -10.0

        return int(pruned_count)

    def active_rank(self, threshold=0.1):
        """Number of rank dimensions still active (above threshold)."""
        mask = self._gate_mask()
        active_count = (mask >= threshold).sum().item()
        return int(active_count)

    def gate_regularization(self):
        """L1 penalty on gate activations to encourage rank sparsity."""
        mask = self._gate_mask()
        return mask.sum()

    def extra_repr(self):
        active = self.active_rank()
        info = (
            f"in_features={self.in_features}, "
            f"out_features={self.out_features}, "
            f"rank={self.rank} (active={active}), "
            f"alpha={self.lora_alpha}"
        )
        return info
