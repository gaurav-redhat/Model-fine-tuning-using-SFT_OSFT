"""
LoraModel — Automatic Injection Wrapper.

Wraps any nn.Module, recursively finds target layers, and replaces them
with their LoRA equivalents. Handles freezing, bias control, weight merging,
and adapter-only saving.

Usage:
    from model import LoraModel, LoraConfig

    base = AutoModelForCausalLM.from_pretrained("meta-llama/Llama-2-7b-hf")
    config = LoraConfig(rank=16, target_modules=["q_proj", "v_proj"])
    model = LoraModel(base, config)

"""

from __future__ import annotations

from dataclasses import dataclass
from typing import List, Literal, Optional, Union

import torch
import torch.nn as nn
from safetensors.torch import save_file

from .LinearLora import LoRALinear
from .EmbeddingLora import LoRAEmbedding
from .ConvLora import LoRAConv2d
from .adaptive import AdaptiveLoRALinear


# ------------------------------------------------------------------
# 4.5  LoraConfig — Configuration Dataclass
# ------------------------------------------------------------------

@dataclass
class LoraConfig:
    """All hyperparameters for a LoRA adaptation."""

    rank: int = 8
    target_modules: Optional[Union[List[str], str]] = None
    exclude_modules: Optional[Union[List[str], str]] = None
    lora_alpha: float = 8.0
    lora_dropout: float = 0.0
    bias: Literal["none", "all", "lora_only"] = "none"
    use_rslora: bool = True
    adaptive: bool = False
    gate_init: float = 5.0


# ------------------------------------------------------------------
# 4.6  LoraModel — Automatic Injection Wrapper
# ------------------------------------------------------------------

class LoraModel(nn.Module):
    """
    Wraps an arbitrary nn.Module and injects LoRA into specified layers.

    Handles:
      - Recursive layer replacement (Linear, Embedding, Conv2d)
      - Freezing all non-LoRA parameters
      - Bias gradient control
      - Weight merging for deployment
      - Adapter-only saving
    """

    def __init__(self, model, config):
        super(LoraModel, self).__init__()

        self.lora_model = model
        self.config = config

        # Normalize target_modules to a list
        if self.config.target_modules is None:
            self.config.target_modules = []
        elif isinstance(self.config.target_modules, str):
            self.config.target_modules = [self.config.target_modules]

        # Normalize exclude_modules to a list
        if self.config.exclude_modules is None:
            self.config.exclude_modules = []
        elif isinstance(self.config.exclude_modules, str):
            self.config.exclude_modules = [self.config.exclude_modules]

        # Count original trainable params (for the summary printout)
        orig_trainable_params = self._compute_trainable_parameters()

        # Step 1: Freeze everything
        self._disable_all_grads()

        # Step 2: Replace target layers with LoRA versions
        self._apply_lora(self.lora_model)

        # Step 3: Toggle bias gradients based on config
        self._toggle_bias_grad()

        # Print summary
        lora_trainable_params = self._compute_trainable_parameters()
        trainable_pct = round(
            lora_trainable_params * 100 / orig_trainable_params, 2
        )
        print(
            f"Initial Parameters : {orig_trainable_params} || "
            f"LoRA Parameters : {lora_trainable_params} || "
            f"Trainable Proportion : {trainable_pct}%"
        )

    def forward(self, *inputs, **kwargs):
        """Pass-through — all inputs go directly to the wrapped model."""
        return self.lora_model(*inputs, **kwargs)

    # ------------------------------------------------------------------
    # The Recursive Replacement Engine
    # ------------------------------------------------------------------

    def _exclude_module_name_check(self, name):
        """Returns True if this module name matches any exclusion pattern."""
        for exclude_pattern in self.config.exclude_modules:
            if exclude_pattern in name:
                return True
        return False

    def _target_module_name_check(self, name):
        """Returns True if this module name matches any target pattern."""
        for target_pattern in self.config.target_modules:
            if target_pattern in name:
                return True
        return False

    def _apply_lora(self, module):
        """
        Recursively walk the model tree.
        For each child whose name matches a target:
          - If it's nn.Linear    → replace with LoRALinear
          - If it's nn.Conv2d    → replace with LoRAConv2d
          - If it's nn.Embedding → replace with LoRAEmbedding
        Copy pre-trained weights into the new LoRA layer.
        """
        for name, child in module.named_children():

            if self._target_module_name_check(name):

                # --- nn.Linear replacement ---
                if isinstance(child, nn.Linear):
                    if self.config.adaptive:
                        lora_layer_class = AdaptiveLoRALinear
                    else:
                        lora_layer_class = LoRALinear

                    has_bias = child.bias is not None

                    # Build kwargs for the constructor
                    layer_kwargs = {
                        "in_features": child.in_features,
                        "out_features": child.out_features,
                        "bias": has_bias,
                        "rank": self.config.rank,
                        "lora_alpha": self.config.lora_alpha,
                        "lora_dropout": self.config.lora_dropout,
                        "use_rslora": self.config.use_rslora,
                    }
                    if self.config.adaptive:
                        layer_kwargs["gate_init"] = self.config.gate_init

                    new_layer = lora_layer_class(**layer_kwargs)
                    new_layer._load_pretrained_weights(child.state_dict())
                    setattr(module, name, new_layer)

                # --- nn.Conv2d replacement ---
                elif isinstance(child, nn.Conv2d):
                    has_bias = child.bias is not None

                    new_layer = LoRAConv2d(
                        in_channels=child.in_channels,
                        out_channels=child.out_channels,
                        kernel_size=child.kernel_size,
                        stride=child.stride,
                        padding=child.padding,
                        bias=has_bias,
                        rank=self.config.rank,
                        lora_alpha=self.config.lora_alpha,
                        lora_dropout=self.config.lora_dropout,
                        use_rslora=self.config.use_rslora,
                    )
                    new_layer._load_pretrained_weights(child.state_dict())
                    setattr(module, name, new_layer)

                # --- nn.Embedding replacement ---
                elif isinstance(child, nn.Embedding):
                    new_layer = LoRAEmbedding(
                        num_embeddings=child.num_embeddings,
                        embedding_dim=child.embedding_dim,
                        rank=self.config.rank,
                        lora_alpha=self.config.lora_alpha,
                        lora_dropout=self.config.lora_dropout,
                        use_rslora=self.config.use_rslora,
                    )
                    new_layer._load_pretrained_weights(child.state_dict())
                    setattr(module, name, new_layer)

            # Recurse deeper if the child has its own children
            # (and isn't in the exclusion list)
            has_children = len(list(child.children())) > 0
            is_excluded = self._exclude_module_name_check(name)

            if has_children and not is_excluded:
                self._apply_lora(child)

    # ------------------------------------------------------------------
    # Gradient Control
    # ------------------------------------------------------------------

    def _disable_all_grads(self):
        """Freeze every parameter in the model (except excluded modules)."""
        for name, param in self.lora_model.named_parameters():
            if not self._exclude_module_name_check(name):
                param.requires_grad = False

    def _toggle_bias_grad(self):
        """
        Control which biases are trainable based on config.bias:
          - "none":      all biases frozen
          - "all":       all biases trainable
          - "lora_only": only biases in LoRA-targeted layers are trainable
        """
        for name, param in self.lora_model.named_parameters():
            if self._exclude_module_name_check(name):
                continue

            if ".bias" not in name:
                continue

            if self.config.bias == "none":
                param.requires_grad = False

            elif self.config.bias == "all":
                param.requires_grad = True

            elif self.config.bias == "lora_only":
                if self._target_module_name_check(name):
                    param.requires_grad = True

    def _compute_trainable_parameters(self):
        """Count parameters with requires_grad == True."""
        total = 0
        for param in self.lora_model.parameters():
            if param.requires_grad:
                total += param.numel()
        return total

    # ------------------------------------------------------------------
    # Weight Merging & Saving
    # ------------------------------------------------------------------

    def _merge_weights(self, module):
        """Recursively merge LoRA weights into base weights."""
        for name, child in module.named_children():
            is_lora = isinstance(
                child, (LoRALinear, LoRAEmbedding, LoRAConv2d)
            )

            if is_lora:
                merged_layer = child._merge_weights()
                setattr(module, name, merged_layer)
            else:
                has_children = len(list(child.children())) > 0
                if has_children:
                    self._merge_weights(child)

    def save_model(self, path, merge_weights=False):
        """
        Save model to disk using safetensors format.

        merge_weights=True:  Merge LoRA into base weights → save full model.
                             The saved file can be loaded into the ORIGINAL
                             model class (no LoRA wrapper needed).

        merge_weights=False: Save ONLY the trainable parameters (A, B, bias).
                             Tiny file (~10 MB). To load, the model must be
                             re-wrapped in LoraModel first.
        """
        if merge_weights:
            self._merge_weights(self.lora_model)

            # Remove "lora_model." prefix so weights map to the original model
            state_dict = {}
            for name, param in self.named_parameters():
                clean_name = name.replace("lora_model.", "")
                state_dict[clean_name] = param.detach().cpu()

        else:
            # Only save trainable params (lora_A, lora_B, and any trained biases)
            state_dict = {}
            for name, param in self.named_parameters():
                if param.requires_grad:
                    state_dict[name] = param.detach().cpu()

        save_file(state_dict, path)

    # ------------------------------------------------------------------
    # Adaptive LoRA utilities
    # ------------------------------------------------------------------

    def get_adaptive_layers(self):
        """Return all AdaptiveLoRALinear layers with their names."""
        layers = []
        for name, module in self.lora_model.named_modules():
            if isinstance(module, AdaptiveLoRALinear):
                layers.append((name, module))
        return layers

    def prune_adaptive(self, threshold=0.1):
        """Prune low-importance ranks across all adaptive layers."""
        results = {}
        for name, layer in self.get_adaptive_layers():
            pruned = layer.prune(threshold)
            results[name] = pruned
        return results

    def gate_regularization_loss(self):
        """Aggregate gate regularization across all adaptive layers."""
        device = next(self.parameters()).device
        total = torch.tensor(0.0, device=device)

        for _, layer in self.get_adaptive_layers():
            total = total + layer.gate_regularization()

        return total
