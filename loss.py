"""
Loss functions for LoRA fine-tuning.

Provides:
  - LabelSmoothingLoss : smoothed cross-entropy for better generalisation
  - LoRAFinetuneLoss   : composite loss that combines task loss with
                         optional gate-sparsity regularisation (for Adaptive LoRA)
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class LabelSmoothingLoss(nn.Module):
    """Cross-entropy with uniform label smoothing."""

    def __init__(self, num_classes, smoothing=0.1):
        super().__init__()
        assert 0.0 <= smoothing < 1.0
        self.num_classes = num_classes
        self.smoothing = smoothing
        self.confidence = 1.0 - smoothing

    def forward(self, logits, targets):
        log_probs = F.log_softmax(logits, dim=-1)

        # Standard negative log-likelihood per sample
        target_indices = targets.unsqueeze(-1)
        gathered = log_probs.gather(dim=-1, index=target_indices)
        nll_loss = -gathered.squeeze(-1)

        # Uniform smoothing term
        smooth_loss = -log_probs.mean(dim=-1)

        # Weighted combination
        loss = self.confidence * nll_loss + self.smoothing * smooth_loss
        return loss.mean()


class LoRAFinetuneLoss(nn.Module):
    """
    Combines a task-level loss with optional Adaptive-LoRA gate regularisation.

    total = task_loss + gate_lambda * gate_reg
    """

    def __init__(self, task_loss_fn=None, gate_lambda=1e-4):
        super().__init__()
        if task_loss_fn is not None:
            self.task_loss_fn = task_loss_fn
        else:
            self.task_loss_fn = nn.CrossEntropyLoss()
        self.gate_lambda = gate_lambda

    def forward(self, logits, targets, gate_reg=None):
        task_loss = self.task_loss_fn(logits, targets)
        total = task_loss.clone()

        out = {
            "task_loss": task_loss,
        }

        if gate_reg is not None and self.gate_lambda > 0:
            reg = self.gate_lambda * gate_reg
            total = total + reg
            out["gate_reg"] = reg

        out["loss"] = total
        return out
