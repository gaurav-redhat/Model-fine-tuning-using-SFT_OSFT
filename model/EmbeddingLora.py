"""
LoRAEmbedding — Vocabulary Adaptation.

LoRA on embedding tables. The key difference from LoRALinear: A is looked up
via F.embedding() (not multiplied), then the result is projected through B.

Forward: h = Embed(x) + (α/r) · Embed_A(x) @ B
Merge:   E' = E + (α/r) · A_E @ B_E

Parameter savings (V=32768, d=4096, r=16):
    original = V×d = 134M  →  LoRA = r(V+d) = 0.59M  →  227× compression

"""

import math
import torch
import torch.nn as nn
import torch.nn.functional as F

from .base import LoRALayerBase


class LoRAEmbedding(nn.Embedding, LoRALayerBase):
    """
    LoRA wrapper around nn.Embedding.

    Forward: h = Embed(x) + (α/r) · Embed_A(x) @ B
    """

    def __init__(
        self,
        num_embeddings,
        embedding_dim,
        rank=8,
        lora_alpha=8,
        lora_dropout=0.0,
        use_rslora=True,
        **kwargs,
    ):
        nn.Embedding.__init__(
            self,
            num_embeddings,
            embedding_dim,
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

        # Freeze the original embedding table
        self.weight.requires_grad = False

        # A is an embedding table of shape (vocab_size, rank)
        # B is a projection matrix of shape (rank, embedding_dim)
        self.lora_A = nn.Parameter(
            torch.zeros(self.num_embeddings, rank)
        )
        self.lora_B = nn.Parameter(
            torch.zeros(rank, self.embedding_dim)
        )

        nn.init.kaiming_uniform_(self.lora_A, a=math.sqrt(5))

    def _merge_weights(self):
        """Merge: W' = W + (α/r) · AB"""
        lora_update = self.lora_A @ self.lora_B
        lora_update_scaled = lora_update * self.scaling
        merged_weights = self.weight.data + lora_update_scaled

        state_dict = {"weight": merged_weights}

        merged_emb = nn.Embedding(
            self.num_embeddings,
            self.embedding_dim,
        )
        merged_emb.load_state_dict(state_dict)
        return merged_emb

    def forward(self, x):
        # Path 1: Frozen embedding lookup
        base_output = F.embedding(
            input=x,
            weight=self.weight,
            padding_idx=self.padding_idx,
            max_norm=self.max_norm,
            norm_type=self.norm_type,
            scale_grad_by_freq=self.scale_grad_by_freq,
            sparse=self.sparse,
        )

        # Path 2: LoRA branch — look up in A, then project through B
        lora_A_output = F.embedding(
            input=x,
            weight=self.lora_A,
            padding_idx=self.padding_idx,
            max_norm=self.max_norm,
            norm_type=self.norm_type,
            scale_grad_by_freq=self.scale_grad_by_freq,
            sparse=self.sparse,
        )

        # (B, S, rank) @ (rank, embedding_dim) → (B, S, embedding_dim)
        lora_output = lora_A_output @ self.lora_B
        lora_output = lora_output * self.scaling

        return base_output + lora_output

    def extra_repr(self):
        info = (
            f"num_embeddings={self.num_embeddings}, "
            f"embedding_dim={self.embedding_dim}, "
            f"rank={self.rank}, "
            f"alpha={self.lora_alpha}, "
            f"scaling={self.scaling:.4f}"
        )
        return info
