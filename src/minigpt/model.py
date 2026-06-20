"""A tiny GPT, built from scratch in PyTorch (nanoGPT-style).

The whole point of this file is to make the transformer *legible* — every piece
is written out by hand, especially self-attention, rather than calling
``nn.MultiheadAttention``.

Self-attention, without the word "magic"
----------------------------------------
For each token we compute three vectors by linear projection:

  * **query (q)**  - "what am I looking for?"
  * **key (k)**    - "what do I offer?"
  * **value (v)**  - "what will I contribute if attended to?"

The attention score from token i to token j is the dot product ``q_i . k_j``:
how well i's query matches j's key. We scale by ``1/sqrt(head_dim)`` (so the
scores don't blow up with dimension), apply a **causal mask** (token i may only
look at j <= i, never the future), softmax across j to get weights that sum to 1,
then take the weighted sum of the values. That weighted sum is the token's new
representation: each token mixes in information from the earlier tokens it finds
relevant. Stack several such "heads" and several layers, and that is a GPT.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import torch
import torch.nn as nn
from torch.nn import functional as F


@dataclass
class GPTConfig:
    vocab_size: int = 65
    block_size: int = 128      # max context length (tokens the model can see)
    n_layer: int = 4
    n_head: int = 4
    n_embd: int = 128
    dropout: float = 0.1


class CausalSelfAttention(nn.Module):
    """Multi-head masked self-attention, implemented explicitly."""

    def __init__(self, cfg: GPTConfig) -> None:
        super().__init__()
        assert cfg.n_embd % cfg.n_head == 0, "n_embd must be divisible by n_head"
        self.n_head = cfg.n_head
        self.head_dim = cfg.n_embd // cfg.n_head

        # One linear produces q, k, v for all heads at once (then we split).
        self.qkv = nn.Linear(cfg.n_embd, 3 * cfg.n_embd)
        self.proj = nn.Linear(cfg.n_embd, cfg.n_embd)  # output projection
        self.attn_dropout = nn.Dropout(cfg.dropout)
        self.resid_dropout = nn.Dropout(cfg.dropout)

        # Lower-triangular causal mask (1 where attention is allowed). Registered
        # as a buffer so it moves with the module but isn't a learned parameter.
        mask = torch.tril(torch.ones(cfg.block_size, cfg.block_size))
        self.register_buffer("mask", mask.view(1, 1, cfg.block_size, cfg.block_size))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, T, C = x.shape  # batch, time (tokens), channels (n_embd)

        # Project to q, k, v and reshape into (B, n_head, T, head_dim).
        q, k, v = self.qkv(x).split(C, dim=2)
        q = q.view(B, T, self.n_head, self.head_dim).transpose(1, 2)
        k = k.view(B, T, self.n_head, self.head_dim).transpose(1, 2)
        v = v.view(B, T, self.n_head, self.head_dim).transpose(1, 2)

        # Scaled dot-product attention scores: (B, n_head, T, T).
        att = (q @ k.transpose(-2, -1)) / math.sqrt(self.head_dim)

        # Causal mask: forbid attending to the future by setting those scores to
        # -inf, so softmax gives them zero weight.
        att = att.masked_fill(self.mask[:, :, :T, :T] == 0, float("-inf"))
        att = F.softmax(att, dim=-1)        # weights over past tokens, sum to 1
        att = self.attn_dropout(att)

        y = att @ v                         # weighted sum of values
        y = y.transpose(1, 2).contiguous().view(B, T, C)  # recombine heads
        return self.resid_dropout(self.proj(y))


class MLP(nn.Module):
    """Position-wise feed-forward network (the 'thinking' between attentions)."""

    def __init__(self, cfg: GPTConfig) -> None:
        super().__init__()
        self.fc = nn.Linear(cfg.n_embd, 4 * cfg.n_embd)
        self.proj = nn.Linear(4 * cfg.n_embd, cfg.n_embd)
        self.dropout = nn.Dropout(cfg.dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.dropout(self.proj(F.gelu(self.fc(x))))


class Block(nn.Module):
    """A transformer block: pre-norm attention + pre-norm MLP, both residual."""

    def __init__(self, cfg: GPTConfig) -> None:
        super().__init__()
        self.ln1 = nn.LayerNorm(cfg.n_embd)
        self.attn = CausalSelfAttention(cfg)
        self.ln2 = nn.LayerNorm(cfg.n_embd)
        self.mlp = MLP(cfg)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Residual connections let gradients flow and let each block *refine*
        # the representation rather than replace it.
        x = x + self.attn(self.ln1(x))
        x = x + self.mlp(self.ln2(x))
        return x


class GPT(nn.Module):
    """The full model: token + positional embeddings -> blocks -> LM head."""

    def __init__(self, cfg: GPTConfig) -> None:
        super().__init__()
        self.cfg = cfg
        self.tok_emb = nn.Embedding(cfg.vocab_size, cfg.n_embd)
        self.pos_emb = nn.Embedding(cfg.block_size, cfg.n_embd)  # learned positions
        self.drop = nn.Dropout(cfg.dropout)
        self.blocks = nn.ModuleList([Block(cfg) for _ in range(cfg.n_layer)])
        self.ln_f = nn.LayerNorm(cfg.n_embd)
        self.head = nn.Linear(cfg.n_embd, cfg.vocab_size, bias=False)

        # Weight tying: the input embedding and output projection share weights,
        # a standard trick that saves parameters and helps generalisation.
        self.head.weight = self.tok_emb.weight
        self.apply(self._init_weights)

    @staticmethod
    def _init_weights(module: nn.Module) -> None:
        if isinstance(module, (nn.Linear, nn.Embedding)):
            nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if isinstance(module, nn.Linear) and module.bias is not None:
                nn.init.zeros_(module.bias)

    def forward(self, idx: torch.Tensor, targets: torch.Tensor | None = None):
        B, T = idx.shape
        assert T <= self.cfg.block_size, "sequence longer than block_size"
        pos = torch.arange(T, device=idx.device)

        # Why positional embeddings? Attention is permutation-invariant — without
        # position info it can't tell "dog bites man" from "man bites dog". We add
        # a learned vector per position so order is encoded.
        x = self.drop(self.tok_emb(idx) + self.pos_emb(pos))
        for block in self.blocks:
            x = block(x)
        x = self.ln_f(x)
        logits = self.head(x)  # (B, T, vocab_size)

        loss = None
        if targets is not None:
            loss = F.cross_entropy(
                logits.view(-1, logits.size(-1)), targets.view(-1)
            )
        return logits, loss

    @torch.no_grad()
    def generate(
        self, idx: torch.Tensor, max_new_tokens: int,
        temperature: float = 1.0, top_k: int | None = None,
    ) -> torch.Tensor:
        """Autoregressively sample ``max_new_tokens`` continuations of ``idx``."""
        for _ in range(max_new_tokens):
            # Crop context to the last block_size tokens (the model's window).
            idx_cond = idx[:, -self.cfg.block_size:]
            logits, _ = self(idx_cond)
            logits = logits[:, -1, :] / temperature  # last step's distribution
            if top_k is not None:
                v, _ = torch.topk(logits, min(top_k, logits.size(-1)))
                logits[logits < v[:, [-1]]] = float("-inf")
            probs = F.softmax(logits, dim=-1)
            next_id = torch.multinomial(probs, num_samples=1)
            idx = torch.cat([idx, next_id], dim=1)
        return idx

    def num_params(self) -> int:
        return sum(p.numel() for p in self.parameters())
