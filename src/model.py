"""GPT-2-style model, built from scratch.

Phase 1: GPTConfig, sinusoidal_encoding, Embeddings.
Phase 2: LayerNorm, CausalSelfAttention, MLP, Block.
Phase 3 (next): the full GPT module.

Shape notation used in comments:
    B = batch size, T = sequence length (<= block_size), C = n_embd, V = vocab_size
    nh = n_head, hs = head_size = C / nh
"""
from __future__ import annotations

import math
from dataclasses import dataclass

import torch
import torch.nn as nn
import torch.nn.functional as F


@dataclass
class GPTConfig:
    vocab_size: int = 65          # set from the tokenizer at runtime
    block_size: int = 256         # max context length T
    n_layer: int = 6
    n_head: int = 6
    n_embd: int = 384             # C
    dropout: float = 0.2
    pos_encoding: str = "learned"  # "learned" | "sinusoidal"
    norm_position: str = "pre"     # "pre" | "post"        (used in Phase 2)
    tie_weights: bool = True       # (used in Phase 3)

    def __post_init__(self) -> None:
        assert self.pos_encoding in ("learned", "sinusoidal"), self.pos_encoding
        assert self.norm_position in ("pre", "post"), self.norm_position
        assert self.n_embd % self.n_head == 0, "n_embd must be divisible by n_head"


def sinusoidal_encoding(block_size: int, n_embd: int) -> torch.Tensor:
    """Fixed sinusoidal positional encoding from Vaswani et al. (2017), Sec. 3.5.

        PE[pos, 2i]   = sin(pos / 10000^(2i / C))
        PE[pos, 2i+1] = cos(pos / 10000^(2i / C))

    Returns a (block_size, n_embd) tensor. Each pair of dims is a sine/cosine of a different
    frequency: low dims oscillate fast (resolve nearby positions), high dims slowly.
    """
    assert n_embd % 2 == 0, "sinusoidal encoding needs an even n_embd"
    pos = torch.arange(block_size, dtype=torch.float32).unsqueeze(1)             # (T, 1)
    # 1 / 10000^(2i/C) computed as exp(-(2i/C) * ln 10000) for numerical stability
    two_i = torch.arange(0, n_embd, 2, dtype=torch.float32)                      # (C/2,)
    inv_freq = torch.exp(-two_i / n_embd * math.log(10000.0))                    # (C/2,)
    angles = pos * inv_freq                                                      # (T, C/2) via broadcasting
    pe = torch.zeros(block_size, n_embd)                                         # (T, C)
    pe[:, 0::2] = torch.sin(angles)                                              # even dims
    pe[:, 1::2] = torch.cos(angles)                                              # odd dims
    return pe


class Embeddings(nn.Module):
    """Token embedding + positional encoding (+ dropout): (B, T) int ids -> (B, T, C) floats.

    token embedding : nn.Embedding(V, C)   -- a learned lookup table, row i = vector for token i
    positional enc. : "learned"    -> nn.Embedding(block_size, C)   (GPT-2)
                      "sinusoidal" -> fixed buffer, no parameters   (Vaswani et al.)
    """

    def __init__(self, cfg: GPTConfig) -> None:
        super().__init__()
        self.cfg = cfg
        self.tok_emb = nn.Embedding(cfg.vocab_size, cfg.n_embd)  # weight: (V, C)
        if cfg.pos_encoding == "learned":
            self.pos_emb = nn.Embedding(cfg.block_size, cfg.n_embd)  # weight: (block_size, C)
            # Deviation note: GPT-2 init -> N(0, 0.02) for both tables (nn.Embedding defaults to N(0,1)).
            nn.init.normal_(self.tok_emb.weight, mean=0.0, std=0.02)
            nn.init.normal_(self.pos_emb.weight, mean=0.0, std=0.02)
            self.emb_scale = 1.0
        else:
            # register_buffer: moves with .to(device), saved in state_dict, but NOT a Parameter.
            self.register_buffer("pe", sinusoidal_encoding(cfg.block_size, cfg.n_embd), persistent=False)  # (block_size, C)
            nn.init.normal_(self.tok_emb.weight, mean=0.0, std=0.02)
            # Deviation from GPT-2 (follows Vaswani): scale token embeddings by sqrt(C).
            # The sinusoid has entries in [-1, 1] while token vectors start at std 0.02, so
            # without scaling the position signal would drown out the token identity.
            self.emb_scale = math.sqrt(cfg.n_embd)
        self.drop = nn.Dropout(cfg.dropout)

    def forward(self, idx: torch.Tensor) -> torch.Tensor:
        B, T = idx.shape                                          # idx: (B, T) integer token ids
        assert T <= self.cfg.block_size, f"sequence length {T} > block_size {self.cfg.block_size}"
        tok = self.tok_emb(idx) * self.emb_scale                  # (B, T, C)
        if self.cfg.pos_encoding == "learned":
            positions = torch.arange(T, device=idx.device)        # (T,)  = [0, 1, ..., T-1]
            pos = self.pos_emb(positions)                         # (T, C)
        else:
            pos = self.pe[:T]                                     # (T, C)
        x = tok + pos                                             # (B, T, C) + (T, C) -> broadcast over B
        return self.drop(x)                                       # (B, T, C)


# ======================================================================================
# Phase 2: Transformer building blocks
# ======================================================================================
class LayerNorm(nn.Module):
    """Layer normalization over the last dim, written from scratch.

        y = gamma * (x - mean) / sqrt(var + eps) + beta

    mean/var are computed PER TOKEN over the C features (not over the batch, unlike BatchNorm).
    var is the biased estimate (divide by C, not C-1), matching nn.LayerNorm.
    """

    def __init__(self, n_embd: int, eps: float = 1e-5) -> None:
        super().__init__()
        self.eps = eps
        self.gamma = nn.Parameter(torch.ones(n_embd))   # (C,) learnable scale
        self.beta = nn.Parameter(torch.zeros(n_embd))   # (C,) learnable shift

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        mean = x.mean(dim=-1, keepdim=True)                              # (B, T, 1)
        var = x.var(dim=-1, keepdim=True, unbiased=False)                # (B, T, 1)
        x_hat = (x - mean) / torch.sqrt(var + self.eps)                  # (B, T, C)
        return self.gamma * x_hat + self.beta                            # (B, T, C), (C,) broadcasts


class CausalSelfAttention(nn.Module):
    """Multi-head causal (masked) self-attention.

    For each head:  Attention(Q, K, V) = softmax(Q K^T / sqrt(d_k) + mask) V
    where d_k = head_size. All heads are computed in parallel via an extra tensor dimension.
    """

    def __init__(self, cfg: GPTConfig) -> None:
        super().__init__()
        self.n_head = cfg.n_head
        self.n_embd = cfg.n_embd
        self.head_size = cfg.n_embd // cfg.n_head
        # Q, K, V projections fused into ONE Linear(C, 3C) (as in GPT-2): mathematically identical
        # to three separate Linear(C, C) layers, but a single bigger matmul is faster.
        self.c_attn = nn.Linear(cfg.n_embd, 3 * cfg.n_embd)
        self.c_proj = nn.Linear(cfg.n_embd, cfg.n_embd)       # output projection W_O
        self.c_proj.is_residual_proj = True                   # tag: Phase 3 init scales this by 1/sqrt(2*n_layer)
        self.attn_drop = nn.Dropout(cfg.dropout)
        self.resid_drop = nn.Dropout(cfg.dropout)
        # Lower-triangular matrix of ones: mask[i, j] = 1 iff j <= i (token i may look at j).
        # A buffer: moves with .to(device) but is not trained.
        mask = torch.tril(torch.ones(cfg.block_size, cfg.block_size))
        self.register_buffer("mask", mask.view(1, 1, cfg.block_size, cfg.block_size), persistent=False)

    def forward(self, x: torch.Tensor, return_attn: bool = False):
        """x: (B, T, C) -> (y: (B, T, C), attn: (B, nh, T, T) or None)."""
        B, T, C = x.shape
        qkv = self.c_attn(x)                                              # (B, T, 3C)
        q, k, v = qkv.split(self.n_embd, dim=2)                           # each (B, T, C)
        # Split C into nh heads of size hs, then move heads next to batch so matmul batches over them.
        q = q.view(B, T, self.n_head, self.head_size).transpose(1, 2)     # (B, nh, T, hs)
        k = k.view(B, T, self.n_head, self.head_size).transpose(1, 2)     # (B, nh, T, hs)
        v = v.view(B, T, self.n_head, self.head_size).transpose(1, 2)     # (B, nh, T, hs)

        # scores[b,h,i,j] = <q_i, k_j> / sqrt(hs): how much token i wants to read from token j
        scores = (q @ k.transpose(-2, -1)) / math.sqrt(self.head_size)    # (B, nh, T, hs) @ (B, nh, hs, T) -> (B, nh, T, T)
        # Causal mask: forbid j > i by setting those scores to -inf (softmax -> exactly 0).
        scores = scores.masked_fill(self.mask[:, :, :T, :T] == 0, float("-inf"))  # (B, nh, T, T)
        attn = F.softmax(scores, dim=-1)                                  # (B, nh, T, T), each row sums to 1
        attn_d = self.attn_drop(attn)                                     # (B, nh, T, T) dropout on the weights
        y = attn_d @ v                                                    # (B, nh, T, T) @ (B, nh, T, hs) -> (B, nh, T, hs)
        y = y.transpose(1, 2).contiguous().view(B, T, C)                  # (B, T, nh, hs) -> (B, T, C): concat heads
        y = self.resid_drop(self.c_proj(y))                               # (B, T, C)
        return y, (attn if return_attn else None)


class MLP(nn.Module):
    """Position-wise feed-forward network: Linear(C, 4C) -> GELU -> Linear(4C, C) -> dropout."""

    def __init__(self, cfg: GPTConfig) -> None:
        super().__init__()
        self.c_fc = nn.Linear(cfg.n_embd, 4 * cfg.n_embd)
        self.act = nn.GELU(approximate="tanh")   # GPT-2 used the tanh approximation of GELU
        self.c_proj = nn.Linear(4 * cfg.n_embd, cfg.n_embd)
        self.c_proj.is_residual_proj = True
        self.drop = nn.Dropout(cfg.dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.act(self.c_fc(x))               # (B, T, C) -> (B, T, 4C)
        return self.drop(self.c_proj(x))         # (B, T, 4C) -> (B, T, C)


class Block(nn.Module):
    """One transformer block with residual connections.

    pre-norm  (GPT-2):             x = x + Attn(LN(x));   x = x + MLP(LN(x))
    post-norm (Vaswani et al.):    x = LN(x + Attn(x));   x = LN(x + MLP(x))
    Pre-norm keeps a clean identity path for gradients through all layers, so it trains stably
    without warmup tricks; post-norm normalizes the residual stream itself and is harder to train deep.
    """

    def __init__(self, cfg: GPTConfig) -> None:
        super().__init__()
        self.norm_position = cfg.norm_position
        self.ln1 = LayerNorm(cfg.n_embd)
        self.attn = CausalSelfAttention(cfg)
        self.ln2 = LayerNorm(cfg.n_embd)
        self.mlp = MLP(cfg)

    def forward(self, x: torch.Tensor, return_attn: bool = False):
        """x: (B, T, C) -> (x: (B, T, C), attn: (B, nh, T, T) or None)."""
        if self.norm_position == "pre":
            a, attn = self.attn(self.ln1(x), return_attn)   # (B, T, C)
            x = x + a                                       # residual
            x = x + self.mlp(self.ln2(x))                   # residual
        else:
            a, attn = self.attn(x, return_attn)
            x = self.ln1(x + a)
            x = self.ln2(x + self.mlp(x))
        return x, attn                                      # (B, T, C)
