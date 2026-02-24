"""
LoRAConv2d — Vision Model Adaptation.

LoRA extends to convolutional layers for vision models (ViT, ResNet, UNet).

A is a Conv2d with `rank` output channels (compresses spatially).
B is a (rank, out_channels) matrix that expands back (NOT a convolution).

Forward:
    base      = Conv2d(x, W, bias)                          # frozen
    z         = Conv2d(x, A)                                 # (B, rank, H', W')
    Δy        = permute(z) @ B → permute back                # (B, C_out, H', W')
    output    = base + (α/r) · Δy

Merge:
    K_merged = K_0 + (α/r) · reshape(B^T @ flatten(A))

Parameter savings (C_in=512, C_out=512, k=3, r=8):
    original = 512×512×3×3 = 2.36M
    LoRA     = 8×(512×9 + 512) = 40K  →  59× compression
"""

import math
import torch
import torch.nn as nn
import torch.nn.functional as F

from .base import LoRALayerBase


class LoRAConv2d(nn.Conv2d, LoRALayerBase):
    """
    LoRA wrapper around nn.Conv2d.

    A is a Conv2d with `rank` output channels (compresses spatially).
    B is a (rank, out_channels) matrix that expands back.
    """

    def __init__(
        self,
        in_channels,
        out_channels,
        kernel_size,
        stride=1,
        padding=0,
        bias=True,
        rank=8,
        lora_alpha=8,
        lora_dropout=0.0,
        use_rslora=True,
        **kwargs,
    ):
        nn.Conv2d.__init__(
            self,
            in_channels=in_channels,
            out_channels=out_channels,
            kernel_size=kernel_size,
            stride=stride,
            padding=padding,
            bias=bias,
            **kwargs,
        )
        LoRALayerBase.__init__(
            self,
            rank=rank,
            lora_alpha=lora_alpha,
            lora_dropout=lora_dropout,
            use_rslora=use_rslora,
        )

        assert rank > 0, "If Rank is 0, Why are you doing LoRA?"

        self.weight.requires_grad = False

        # A: a convolution that reduces channels from in_channels → rank
        # Shape: (rank, in_channels, kernel_h, kernel_w)
        self.lora_A = nn.Parameter(
            torch.zeros(rank, self.in_channels, *self.kernel_size)
        )

        # B: a simple matrix that expands rank → out_channels
        # Shape: (rank, out_channels) — NOT a convolution
        self.lora_B = nn.Parameter(
            torch.zeros(rank, self.out_channels)
        )

        nn.init.kaiming_uniform_(self.lora_A, a=math.sqrt(5))

    def _merge_weights(self):
        """Merge conv weights: W' = W + (α/r) · B^T @ flatten(A)"""
        # (rank, C_in * k_h * k_w)
        lora_A_flat = self.lora_A.flatten(1)

        # (out_channels, rank) @ (rank, C_in*k_h*k_w) → (out_ch, C_in*k_h*k_w)
        lora_mult = self.lora_B.T @ lora_A_flat
        lora_mult = lora_mult * self.scaling

        # Reshape back to conv weight shape
        lora_mult = lora_mult.reshape(
            self.out_channels,
            self.in_channels,
            *self.kernel_size,
        )

        merged_weight = self.weight.data + lora_mult

        has_bias = self.bias is not None

        state_dict = {"weight": merged_weight}
        if has_bias:
            state_dict["bias"] = self.bias

        merged_conv = nn.Conv2d(
            self.in_channels,
            self.out_channels,
            kernel_size=self.kernel_size,
            stride=self.stride,
            padding=self.padding,
            bias=has_bias,
        )
        merged_conv.load_state_dict(state_dict)
        return merged_conv

    def forward(self, x):
        # Path 1: Frozen convolution
        base_output = F.conv2d(
            input=x,
            weight=self.weight,
            bias=self.bias,
            stride=self.stride,
            padding=self.padding,
        )

        # Path 2: LoRA — convolve with A (produces rank channels)
        lora_A_output = F.conv2d(
            input=x,
            weight=self.lora_A,
            bias=None,
            stride=self.stride,
            padding=self.padding,
        )  # → (B, rank, H', W')

        # Permute to put rank last: (B, H', W', rank)
        lora_A_output = lora_A_output.permute(0, 2, 3, 1)

        # Apply dropout, then multiply by B
        # (B, H', W', rank) @ (rank, out_ch) → (B, H', W', out_ch)
        dropped = self.lora_dropout(lora_A_output)
        lora_output = dropped @ self.lora_B
        lora_output = lora_output * self.scaling

        # Permute back to image format: (B, out_ch, H', W')
        lora_output = lora_output.permute(0, 3, 1, 2)

        return base_output + lora_output

    def extra_repr(self):
        info = (
            f"in_channels={self.in_channels}, "
            f"out_channels={self.out_channels}, "
            f"kernel_size={self.kernel_size}, "
            f"stride={self.stride}, "
            f"padding={self.padding}, "
            f"rank={self.rank}, "
            f"alpha={self.lora_alpha}, "
            f"scaling={self.scaling:.4f}"
        )
        return info
