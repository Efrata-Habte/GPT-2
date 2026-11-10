"""GPT-2-style model, built from scratch.

Phase 1 contains: GPTConfig, sinusoidal_encoding, and the input Embeddings module.
Attention / MLP / LayerNorm / Block / GPT are added in Phases 2-3.

Shape notation used in comments:
    B = batch size, T = sequence length (<= block_size), C = n_embd, V = vocab_size
"""
from __future__ import annotations

import math
from dataclasses import dataclass

import torch
import torch.nn as nn


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
