"""GPT-2-style language model, built from scratch.

Reading order (each part feeds the next):
    GPTConfig                    the model's settings
    TokenAndPositionEmbedding    token ids -> vectors that say WHAT each token is and WHERE it is
    LayerNorm                    keeps each token's numbers in a stable range
    CausalSelfAttention          each token gathers information from earlier tokens
    FeedForward                  each token processes what it gathered, on its own
    TransformerBlock             attention + feed-forward, each with a residual "+"
    GPT                          embedding -> N blocks -> final LayerNorm -> next-token scores

Shape names used in the comments:
    batch = number of sequences, seq_len = tokens per sequence (<= block_size),
    embed_dim = n_embd (384), vocab = vocab_size, heads = n_head, head_dim = embed_dim / heads
"""
from __future__ import annotations

import math
from dataclasses import dataclass

import torch
import torch.nn as nn
import torch.nn.functional as F


@dataclass
class GPTConfig:
    """The model's settings. The field names match the YAML configs and saved checkpoints."""
    vocab_size: int = 65            # number of different tokens (set from the tokenizer)
    block_size: int = 256           # context length: the most tokens the model can look at
    n_layer: int = 6                # number of Transformer blocks (depth)
    n_head: int = 6                 # attention heads per block (parallel patterns)
    n_embd: int = 384               # numbers per token vector (width)
    dropout: float = 0.2            # share of values switched off at random during training
    pos_encoding: str = "learned"   # "learned" (GPT-2) | "sinusoidal" (original Transformer)
    norm_position: str = "pre"      # "pre" (GPT-2) | "post" (original Transformer)
    tie_weights: bool = True        # output layer reuses the token-embedding table

    def __post_init__(self) -> None:
        assert self.pos_encoding in ("learned", "sinusoidal"), self.pos_encoding
        assert self.norm_position in ("pre", "post"), self.norm_position
        assert self.n_embd % self.n_head == 0, "n_embd must be divisible by n_head"


# ======================================================================================
# 1. Embedding: token ids -> vectors
# ======================================================================================
def sinusoidal_encoding(block_size: int, n_embd: int) -> torch.Tensor:
    """Fixed (not learned) position vectors from the original Transformer paper.

        position_vector[pos, 2i]   = sin(pos / 10000^(2i / embed_dim))
        position_vector[pos, 2i+1] = cos(pos / 10000^(2i / embed_dim))

    Each pair of numbers is a sine/cosine wave of a different speed, so every position gets a unique pattern.
    Returns (block_size, n_embd).
    """
    assert n_embd % 2 == 0, "sinusoidal encoding needs an even n_embd"
    positions = torch.arange(block_size, dtype=torch.float32).unsqueeze(1)          # (block_size, 1)
    even_dims = torch.arange(0, n_embd, 2, dtype=torch.float32)                     # (embed_dim/2,)
    wave_speeds = torch.exp(-even_dims / n_embd * math.log(10000.0))                # 1 / 10000^(2i/embed_dim)
    angles = positions * wave_speeds                                                # (block_size, embed_dim/2)
    position_vectors = torch.zeros(block_size, n_embd)
    position_vectors[:, 0::2] = torch.sin(angles)                                   # even dims
    position_vectors[:, 1::2] = torch.cos(angles)                                   # odd dims
    return position_vectors


class TokenAndPositionEmbedding(nn.Module):
    """token ids (batch, seq_len) -> vectors (batch, seq_len, embed_dim).

    vector = token_embedding[token]  (WHAT the token is)  +  position_vector[position]  (WHERE it is),
    then dropout.
    """

    def __init__(self, config: GPTConfig) -> None:
        super().__init__()
        self.config = config
        self.token_embedding = nn.Embedding(config.vocab_size, config.n_embd)           # one learned row per token
        if config.pos_encoding == "learned":
            self.position_embedding = nn.Embedding(config.block_size, config.n_embd)    # one learned row per position
            nn.init.normal_(self.token_embedding.weight, mean=0.0, std=0.02)            # GPT-2 init
            nn.init.normal_(self.position_embedding.weight, mean=0.0, std=0.02)
            self.token_scale = 1.0
        else:
            # A buffer: saved and moved with the model, but never trained.
            self.register_buffer("sinusoidal_positions", sinusoidal_encoding(config.block_size, config.n_embd),
                                 persistent=False)
            nn.init.normal_(self.token_embedding.weight, mean=0.0, std=0.02)
            # Sine values are in [-1, 1] but token vectors start near 0.02, so scale the tokens up by
            # sqrt(embed_dim) (as in the original paper); otherwise position would drown out the token.
            self.token_scale = math.sqrt(config.n_embd)
        self.dropout = nn.Dropout(config.dropout)

    def forward(self, token_ids: torch.Tensor, first_position: int = 0) -> torch.Tensor:
        """first_position: the true position of token_ids[:, 0]. It is 0 except during generation with the
        KV cache, where only the newest token is fed in but must still get its real position."""
        seq_len = token_ids.shape[1]
        assert first_position + seq_len <= self.config.block_size, \
            f"positions up to {first_position + seq_len} > block_size {self.config.block_size}"
        token_vectors = self.token_embedding(token_ids) * self.token_scale             # (batch, seq_len, embed_dim)
        if self.config.pos_encoding == "learned":
            positions = torch.arange(first_position, first_position + seq_len, device=token_ids.device)
            position_vectors = self.position_embedding(positions)                       # (seq_len, embed_dim)
        else:
            position_vectors = self.sinusoidal_positions[first_position:first_position + seq_len]
        return self.dropout(token_vectors + position_vectors)                          # same position vectors for every sequence


# ======================================================================================
# 2. LayerNorm
# ======================================================================================
class LayerNorm(nn.Module):
    """Rescale each token's numbers to average 0 and spread 1, then apply a learned scale and shift.

        output = scale * (x - mean) / sqrt(variance + eps) + shift          (scale, shift = gamma, beta in papers)

    mean and variance are taken over each token's own embed_dim numbers. The variance divides by embed_dim
    (not embed_dim - 1), exactly like nn.LayerNorm.
    """

    def __init__(self, n_embd: int, eps: float = 1e-5) -> None:
        super().__init__()
        self.eps = eps                                    # avoids dividing by zero
        self.scale = nn.Parameter(torch.ones(n_embd))     # learned, starts as "no change"
        self.shift = nn.Parameter(torch.zeros(n_embd))    # learned, starts as "no change"

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        mean = x.mean(dim=-1, keepdim=True)                                  # (batch, seq_len, 1)
        variance = x.var(dim=-1, keepdim=True, unbiased=False)               # (batch, seq_len, 1)
        normalized = (x - mean) / torch.sqrt(variance + self.eps)            # average 0, spread 1
        return self.scale * normalized + self.shift


# ======================================================================================
# 3. Causal self-attention
# ======================================================================================
class CausalSelfAttention(nn.Module):
    """Each token looks back at earlier tokens and takes a weighted mix of their information.

        weights = softmax( mask( queries @ keys^T / sqrt(head_dim) ) )
        output  = weights @ values

    All heads are computed at once by giving them their own tensor dimension.
    """

    def __init__(self, config: GPTConfig) -> None:
        super().__init__()
        self.num_heads = config.n_head
        self.embed_dim = config.n_embd
        self.head_dim = config.n_embd // config.n_head
        # One Linear makes queries, keys and values together (faster than three; same result).
        self.query_key_value = nn.Linear(config.n_embd, 3 * config.n_embd)
        self.output_projection = nn.Linear(config.n_embd, config.n_embd)    # mixes the heads back together
        self.output_projection.writes_to_residual_stream = True              # gets the smaller GPT-2 init
        self.attention_dropout = nn.Dropout(config.dropout)
        self.output_dropout = nn.Dropout(config.dropout)
        # causal_mask[i, j] = 1 when token i may look at token j, i.e. j <= i (lower triangle).
        causal_mask = torch.tril(torch.ones(config.block_size, config.block_size))
        self.register_buffer("causal_mask", causal_mask.view(1, 1, config.block_size, config.block_size),
                             persistent=False)

    def split_into_heads(self, x: torch.Tensor) -> torch.Tensor:
        """(batch, seq_len, embed_dim) -> (batch, heads, seq_len, head_dim)"""
        batch, seq_len, _ = x.shape
        return x.view(batch, seq_len, self.num_heads, self.head_dim).transpose(1, 2)

    def forward(self, x: torch.Tensor, return_attention: bool = False, cached_keys_values=None,
                use_cache: bool = False):
        """x: (batch, seq_len, embed_dim) -> (output, attention weights or None[, keys_values if use_cache]).

        KV cache: `cached_keys_values` = (keys, values) of earlier tokens; x then holds only the NEW tokens.
        """
        batch, seq_len, embed_dim = x.shape
        queries, keys, values = self.query_key_value(x).split(self.embed_dim, dim=2)
        queries, keys, values = self.split_into_heads(queries), self.split_into_heads(keys), self.split_into_heads(values)
        if cached_keys_values is not None:
            # Earlier tokens' keys and values never change, so reuse them instead of recomputing.
            keys = torch.cat([cached_keys_values[0], keys], dim=2)
            values = torch.cat([cached_keys_values[1], values], dim=2)
        num_visible = keys.size(2)                       # tokens that can be looked at (= seq_len without a cache)

        # scores[i, j]: how well token i's query matches token j's key, scaled so softmax stays smooth.
        scores = (queries @ keys.transpose(-2, -1)) / math.sqrt(self.head_dim)    # (batch, heads, seq_len, num_visible)
        # Block the future: -inf becomes exactly 0 after softmax. The new tokens are the LAST seq_len rows.
        mask = self.causal_mask[:, :, num_visible - seq_len:num_visible, :num_visible]
        scores = scores.masked_fill(mask == 0, float("-inf"))
        attention_weights = F.softmax(scores, dim=-1)                              # each row sums to 1
        mixed = self.attention_dropout(attention_weights) @ values                 # (batch, heads, seq_len, head_dim)
        mixed = mixed.transpose(1, 2).contiguous().view(batch, seq_len, embed_dim)  # join the heads back together
        output = self.output_dropout(self.output_projection(mixed))

        attention_weights = attention_weights if return_attention else None
        return (output, attention_weights, (keys, values)) if use_cache else (output, attention_weights)


# ======================================================================================
# 4. Feed-forward network (also called the MLP)
# ======================================================================================
class FeedForward(nn.Module):
    """Processes each token on its own: Linear(embed_dim -> 4*embed_dim) -> GELU -> Linear(back) -> dropout."""

    def __init__(self, config: GPTConfig) -> None:
        super().__init__()
        self.expand = nn.Linear(config.n_embd, 4 * config.n_embd)          # widen: more room to compute
        self.gelu = nn.GELU(approximate="tanh")                             # GPT-2's (tanh) version of GELU
        self.output_projection = nn.Linear(4 * config.n_embd, config.n_embd)   # narrow back to embed_dim
        self.output_projection.writes_to_residual_stream = True
        self.dropout = nn.Dropout(config.dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.dropout(self.output_projection(self.gelu(self.expand(x))))


# ======================================================================================
# 5. Transformer block
# ======================================================================================
class TransformerBlock(nn.Module):
    """Attention, then feed-forward, each added onto x with a residual "+".

    pre-norm  (GPT-2):                x = x + attention(norm(x));   x = x + feed_forward(norm(x))
    post-norm (original Transformer): x = norm(x + attention(x));   x = norm(x + feed_forward(x))
    Pre-norm keeps the "+" shortcut free of LayerNorms, so gradients flow cleanly and training is stable.
    """

    def __init__(self, config: GPTConfig) -> None:
        super().__init__()
        self.norm_position = config.norm_position
        self.norm_before_attention = LayerNorm(config.n_embd)
        self.attention = CausalSelfAttention(config)
        self.norm_before_feed_forward = LayerNorm(config.n_embd)
        self.feed_forward = FeedForward(config)

    def forward(self, x: torch.Tensor, return_attention: bool = False, cached_keys_values=None,
                use_cache: bool = False):
        """x: (batch, seq_len, embed_dim) -> (x, attention weights or None[, keys_values if use_cache])."""
        if self.norm_position == "pre":
            attended, attention_weights, *keys_values = self.attention(
                self.norm_before_attention(x), return_attention, cached_keys_values, use_cache)
            x = x + attended                                                       # residual
            x = x + self.feed_forward(self.norm_before_feed_forward(x))             # residual
        else:
            attended, attention_weights, *keys_values = self.attention(
                x, return_attention, cached_keys_values, use_cache)
            x = self.norm_before_attention(x + attended)
            x = self.norm_before_feed_forward(x + self.feed_forward(x))
        return (x, attention_weights, *keys_values)


# ======================================================================================
# 6. The full GPT model
# ======================================================================================
# Older checkpoints (the trained models) used GPT-2's short names; this maps them to the names above.
OLD_TO_NEW_PARAMETER_NAMES = {
    "embed": "embedding", "tok_emb": "token_embedding", "pos_emb": "position_embedding",
    "ln1": "norm_before_attention", "ln2": "norm_before_feed_forward", "attn": "attention", "mlp": "feed_forward",
    "c_attn": "query_key_value", "c_proj": "output_projection", "c_fc": "expand",
    "ln_f": "final_norm", "lm_head": "output_head", "gamma": "scale", "beta": "shift",
}


def rename_old_parameter_names(state_dict: dict) -> dict:
    """'blocks.0.attn.c_attn.weight' -> 'blocks.0.attention.query_key_value.weight' (new names pass unchanged)."""
    return {".".join(OLD_TO_NEW_PARAMETER_NAMES.get(part, part) for part in name.split(".")): tensor
            for name, tensor in state_dict.items()}


class GPT(nn.Module):
    """token ids -> embedding -> n_layer TransformerBlocks -> final LayerNorm -> output head -> next-token scores"""

    def __init__(self, config: GPTConfig) -> None:
        super().__init__()
        self.config = config
        self.embedding = TokenAndPositionEmbedding(config)
        self.blocks = nn.ModuleList([TransformerBlock(config) for _ in range(config.n_layer)])
        # Pre-norm blocks never normalise the main path, so normalise it once at the end.
        # (Post-norm blocks already end with a LayerNorm, so none is needed.)
        self.final_norm = LayerNorm(config.n_embd) if config.norm_position == "pre" else nn.Identity()
        self.output_head = nn.Linear(config.n_embd, config.vocab_size, bias=False)   # one score per token
        if config.tie_weights:
            # Weight tying: the output head reuses the token-embedding table (saves vocab x embed_dim parameters).
            self.output_head.weight = self.embedding.token_embedding.weight
        self.apply(self.gpt2_init)
        # The 2 * n_layer layers that write onto the residual stream start smaller, so the running sum
        # does not grow with depth.
        for module in self.modules():
            if getattr(module, "writes_to_residual_stream", False):
                nn.init.normal_(module.weight, mean=0.0, std=0.02 / math.sqrt(2 * config.n_layer))

    @staticmethod
    def gpt2_init(module: nn.Module) -> None:
        """GPT-2 init: weights from N(0, 0.02), biases 0."""
        if isinstance(module, nn.Linear):
            nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if module.bias is not None:
                nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            nn.init.normal_(module.weight, mean=0.0, std=0.02)

    def load_state_dict(self, state_dict, strict: bool = True, assign: bool = False):
        """Also accepts checkpoints saved with the old GPT-2-style names."""
        return super().load_state_dict(rename_old_parameter_names(state_dict), strict=strict, assign=assign)

    def forward(self, token_ids: torch.Tensor, targets: torch.Tensor | None = None, return_attention: bool = False,
                cached_keys_values: list | None = None, use_cache: bool = False):
        """token_ids: (batch, seq_len); targets: the true next tokens (batch, seq_len), or None.

        Returns (logits, loss), plus, only when asked for:
            attention weights of every layer (return_attention=True)
            keys/values of every layer (use_cache=True) -> pass back as `cached_keys_values` for the next step
        logits: (batch, seq_len, vocab) scores for the next token; loss: a number, or None without targets.
        """
        first_position = 0 if cached_keys_values is None else cached_keys_values[0][0].size(2)
        x = self.embedding(token_ids, first_position)
        all_attention_weights, all_keys_values = [], []
        for layer, block in enumerate(self.blocks):
            layer_cache = None if cached_keys_values is None else cached_keys_values[layer]
            x, attention_weights, *keys_values = block(x, return_attention, layer_cache, use_cache)
            all_attention_weights.append(attention_weights)
            all_keys_values.extend(keys_values)
        logits = self.output_head(self.final_norm(x))                       # (batch, seq_len, vocab)

        loss = None
        if targets is not None:
            # Cross-entropy at every position: how surprised was the model by the real next token?
            loss = F.cross_entropy(logits.view(-1, logits.size(-1)), targets.view(-1))
        result = (logits, loss)
        if return_attention:
            result += (all_attention_weights,)
        if use_cache:
            result += (all_keys_values,)
        return result

    # ---- parameter counting ------------------------------------------------------------
    def param_breakdown(self) -> dict[str, int]:
        """Parameters per component. A shared (tied) tensor is counted once, where it is first seen."""
        already_counted: set[int] = set()

        def count(module: nn.Module) -> int:
            total = 0
            for parameter in module.parameters():
                if id(parameter) not in already_counted:
                    already_counted.add(id(parameter))
                    total += parameter.numel()
            return total

        breakdown = {"token embedding": count(self.embedding.token_embedding)}
        if self.config.pos_encoding == "learned":
            breakdown["position embedding"] = count(self.embedding.position_embedding)
        for i, block in enumerate(self.blocks):
            breakdown[f"block {i}: attention"] = count(block.attention)
            breakdown[f"block {i}: feed-forward"] = count(block.feed_forward)
            breakdown[f"block {i}: layernorms"] = count(block.norm_before_attention) + count(block.norm_before_feed_forward)
        breakdown["final layernorm"] = count(self.final_norm)
        breakdown["output head (0 if tied)"] = count(self.output_head)
        return breakdown

    def num_params(self) -> int:
        return sum(self.param_breakdown().values())

    def print_param_breakdown(self) -> None:
        breakdown = self.param_breakdown()
        total = sum(breakdown.values())
        rows = {name: n for name, n in breakdown.items() if not name.startswith("block ")}
        for part in ("attention", "feed-forward", "layernorms"):
            rows[f"all blocks: {part}"] = sum(n for name, n in breakdown.items()
                                               if name.startswith("block ") and name.endswith(part))
        print(f"{'component':<28}{'params':>12}{'share':>9}")
        for name, n in rows.items():
            print(f"{name:<28}{n:>12,}{100 * n / total:>8.1f}%")
        print(f"{'TOTAL':<28}{total:>12,}")
        one_block = sum(n for name, n in breakdown.items() if name.startswith("block 0"))
        print(f"(one block = {one_block:,} params; n_layer = {self.config.n_layer}; tied = {self.config.tie_weights})")


if __name__ == "__main__":
    import argparse

    from src.utils import load_config

    parser = argparse.ArgumentParser(description="Print the parameter breakdown for a config.")
    parser.add_argument("--config", default="configs/small.yaml")
    parser.add_argument("--vocab_size", type=int, default=65)
    args = parser.parse_args()
    GPT(GPTConfig(vocab_size=args.vocab_size, **load_config(args.config)["model"])).print_param_breakdown()
