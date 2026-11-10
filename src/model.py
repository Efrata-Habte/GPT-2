"""GPT-2-style model, built from scratch.

Phase 1: GPTConfig, sinusoidal_encoding, Embeddings.
Phase 2: LayerNorm, CausalSelfAttention, MLP, Block.
Phase 3: GPT (embeddings, blocks, final LN, LM head), init, weight tying, param counts.

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


# ======================================================================================
# Phase 3: the full GPT model
# ======================================================================================
class GPT(nn.Module):
    """GPT-2-style decoder-only language model.

    idx (B, T) -> Embeddings (B, T, C) -> N x Block (B, T, C) -> final LayerNorm -> lm_head -> logits (B, T, V)
    """

    def __init__(self, cfg: GPTConfig) -> None:
        super().__init__()
        self.cfg = cfg
        self.embed = Embeddings(cfg)                                        # token + position + dropout
        self.blocks = nn.ModuleList([Block(cfg) for _ in range(cfg.n_layer)])
        # Final LayerNorm. Pre-norm blocks leave the residual stream un-normalized, so GPT-2 normalizes
        # it once before the head. Post-norm blocks already end with a LayerNorm, so we skip it
        # (the original decoder has no extra final norm either).
        self.ln_f = LayerNorm(cfg.n_embd) if cfg.norm_position == "pre" else nn.Identity()
        self.lm_head = nn.Linear(cfg.n_embd, cfg.vocab_size, bias=False)   # weight: (V, C)
        if cfg.tie_weights:
            # Weight tying: the SAME (V, C) matrix embeds tokens and un-embeds hidden states.
            # Saves V*C params and ties "what a token means going in" to "how it is scored coming out".
            self.lm_head.weight = self.embed.tok_emb.weight
        self.apply(self._init_weights)
        # GPT-2 scaled init: each block adds 2 residual contributions (attn + mlp), so 2*n_layer
        # in total. Scaling their output projections by 1/sqrt(2*n_layer) keeps the variance of
        # the residual stream from growing with depth.
        for module in self.modules():
            if getattr(module, "is_residual_proj", False):
                nn.init.normal_(module.weight, mean=0.0, std=0.02 / math.sqrt(2 * cfg.n_layer))

    @staticmethod
    def _init_weights(module: nn.Module) -> None:
        """GPT-2 init: N(0, 0.02) for Linear/Embedding weights, zeros for biases."""
        if isinstance(module, nn.Linear):
            nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if module.bias is not None:
                nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            nn.init.normal_(module.weight, mean=0.0, std=0.02)

    def forward(self, idx: torch.Tensor, targets: torch.Tensor | None = None, return_attn: bool = False):
        """idx: (B, T) token ids; targets: (B, T) next-token ids or None.

        Returns (logits, loss), or (logits, loss, attns) if return_attn=True, where
        logits: (B, T, V), loss: scalar or None, attns: list of n_layer tensors (B, nh, T, T).
        """
        x = self.embed(idx)                                                 # (B, T, C)
        attns = []
        for block in self.blocks:
            x, attn = block(x, return_attn)                                 # (B, T, C)
            attns.append(attn)
        x = self.ln_f(x)                                                    # (B, T, C)
        logits = self.lm_head(x)                                            # (B, T, V)

        loss = None
        if targets is not None:
            # cross_entropy wants (N, V) scores and (N,) class ids: flatten batch and time together.
            loss = F.cross_entropy(logits.view(-1, logits.size(-1)), targets.view(-1))  # scalar
        return (logits, loss, attns) if return_attn else (logits, loss)

    # ---- parameter accounting -------------------------------------------------------
    def param_breakdown(self) -> dict[str, int]:
        """Parameters per component. Shared (tied) tensors are counted once, where first seen."""
        seen: set[int] = set()

        def count(module: nn.Module) -> int:
            n = 0
            for p in module.parameters():
                if id(p) not in seen:
                    seen.add(id(p))
                    n += p.numel()
            return n

        out = {"token embedding": count(self.embed.tok_emb)}
        if self.cfg.pos_encoding == "learned":
            out["position embedding"] = count(self.embed.pos_emb)
        for i, blk in enumerate(self.blocks):
            out[f"block {i}: attention"] = count(blk.attn)
            out[f"block {i}: mlp"] = count(blk.mlp)
            out[f"block {i}: layernorms"] = count(blk.ln1) + count(blk.ln2)
        out["final layernorm"] = count(self.ln_f)
        out["lm_head (0 if tied)"] = count(self.lm_head)
        return out

    def num_params(self) -> int:
        return sum(self.param_breakdown().values())

    def print_param_breakdown(self) -> None:
        bd = self.param_breakdown()
        total = sum(bd.values())
        rows = {k: v for k, v in bd.items() if not k.startswith("block ")}
        for part in ("attention", "mlp", "layernorms"):
            rows[f"all blocks: {part}"] = sum(v for k, v in bd.items() if k.startswith("block ") and k.endswith(part))
        print(f"{'component':<28}{'params':>12}{'share':>9}")
        for k, v in rows.items():
            print(f"{k:<28}{v:>12,}{100 * v / total:>8.1f}%")
        print(f"{'TOTAL':<28}{total:>12,}")
        per_block = sum(v for k, v in bd.items() if k.startswith("block 0"))
        print(f"(one block = {per_block:,} params; n_layer = {self.cfg.n_layer}; tied = {self.cfg.tie_weights})")


if __name__ == "__main__":
    import argparse

    from src.utils import load_config

    ap = argparse.ArgumentParser(description="Print the parameter breakdown for a config.")
    ap.add_argument("--config", default="configs/small.yaml")
    ap.add_argument("--vocab_size", type=int, default=65)
    a = ap.parse_args()
    GPT(GPTConfig(vocab_size=a.vocab_size, **load_config(a.config)["model"])).print_param_breakdown()
