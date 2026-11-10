"""Autoregressive text generation: greedy, temperature, top-k, top-p, with an optional KV cache.

    python -m src.generate --ckpt checkpoints/small/ckpt_best.pt --prompt "ROMEO:" \
        --temperature 0.8 --max_new_tokens 300
    python -m src.generate --ckpt checkpoints/small/ckpt_best.pt --benchmark
"""
from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import torch
import torch.nn.functional as F

from src.model import GPT, GPTConfig
from src.tokenizer import load_tokenizer, tokenizer_from_dict


# --------------------------------------------------------------------------------------
# Sampling
# --------------------------------------------------------------------------------------
def next_token_probs(
    logits: torch.Tensor, temperature: float = 1.0, top_k: int | None = None, top_p: float | None = None
) -> torch.Tensor:
    """The distribution the sampler draws from. logits: (B, V) -> probabilities (B, V).

    Order: temperature -> top-k filter -> top-p filter -> softmax.
      temperature: logits / T. T<1 sharpens the distribution, T>1 flattens it, T->0 is greedy.
      top_k:       keep only the k highest-logit tokens.
      top_p:       keep the smallest set of top tokens whose probability mass reaches p (nucleus).
    Removed tokens get probability exactly 0.
    """
    logits = logits / max(temperature, 1e-8)                                    # (B, V)
    if top_k is not None:
        k = min(top_k, logits.size(-1))
        kth_best = torch.topk(logits, k, dim=-1).values[:, -1:]                 # (B, 1) k-th largest logit
        logits = logits.masked_fill(logits < kth_best, float("-inf"))           # (B, V)
    if top_p is not None and top_p < 1.0:
        sorted_logits, sorted_idx = torch.sort(logits, dim=-1, descending=True)  # (B, V)
        sorted_probs = F.softmax(sorted_logits, dim=-1)                          # (B, V)
        cum = sorted_probs.cumsum(dim=-1)                                        # (B, V)
        # Drop a token if the mass BEFORE it already reached p (so the token that crosses p is kept,
        # and the single most likely token is always kept).
        drop_sorted = (cum - sorted_probs) >= top_p                              # (B, V)
        drop = torch.zeros_like(drop_sorted).scatter(-1, sorted_idx, drop_sorted)  # undo the sort
        logits = logits.masked_fill(drop, float("-inf"))
    return F.softmax(logits, dim=-1)                                             # (B, V)


def sample_next(
    logits: torch.Tensor,
    temperature: float = 1.0,
    top_k: int | None = None,
    top_p: float | None = None,
    greedy: bool = False,
    generator: torch.Generator | None = None,
) -> torch.Tensor:
    """Pick the next token from next-token logits. logits: (B, V) -> (B, 1) token ids."""
    if greedy or temperature == 0:
        return logits.argmax(dim=-1, keepdim=True)                              # (B, 1)
    probs = next_token_probs(logits, temperature, top_k, top_p)                  # (B, V)
    return torch.multinomial(probs, num_samples=1, generator=generator)         # (B, 1)


# --------------------------------------------------------------------------------------
# Generation loop
# --------------------------------------------------------------------------------------
@torch.no_grad()
def generate_stream(
    model: GPT,
    idx: torch.Tensor,
    max_new_tokens: int,
    temperature: float = 1.0,
    top_k: int | None = None,
    top_p: float | None = None,
    greedy: bool = False,
    use_cache: bool = True,
    generator: torch.Generator | None = None,
):
    """Generator: yields the next token id as a (B, 1) tensor, one decoding step at a time.

    Without cache: every step re-runs the model on the last block_size tokens -> O(T) work per
    new token (O(T^2) with attention). With cache: the K/V of earlier tokens are stored, and each
    step feeds only the single newest token.

    Limitation (true of GPT-2 as well): with ABSOLUTE position embeddings, once the text is longer
    than block_size the window slides and every token's position changes, which invalidates all
    cached K/V. We then rebuild the cache from the window each step, so the speed-up only exists
    while prompt + generated tokens <= block_size. Results are identical either way.
    """
    was_training = model.training
    model.eval()
    block_size = model.cfg.block_size
    past = None
    try:
        for _ in range(max_new_tokens):
            if not use_cache:
                logits, _ = model(idx[:, -block_size:])                              # (B, T', V)
            elif past is None or past[0][0].size(2) >= block_size:
                # (Re)build the cache from the current window ("prefill").
                logits, _, past = model(idx[:, -block_size:], use_cache=True)        # (B, T', V)
            else:
                # Decode: feed ONLY the newest token; its position = number of cached tokens.
                logits, _, past = model(idx[:, -1:], past_kvs=past, use_cache=True)  # (B, 1, V)
            next_id = sample_next(logits[:, -1, :], temperature, top_k, top_p, greedy, generator)  # (B, 1)
            idx = torch.cat([idx, next_id], dim=1)                                   # (B, T+1)
            yield next_id
    finally:
        model.train(was_training)


@torch.no_grad()
def generate(
    model: GPT,
    idx: torch.Tensor,
    max_new_tokens: int,
    temperature: float = 1.0,
    top_k: int | None = None,
    top_p: float | None = None,
    greedy: bool = False,
    use_cache: bool = True,
    generator: torch.Generator | None = None,
) -> torch.Tensor:
    """Extend `idx` (B, T) by `max_new_tokens` tokens. Returns (B, T + max_new_tokens)."""
    new = list(generate_stream(model, idx, max_new_tokens, temperature, top_k, top_p, greedy, use_cache, generator))
    return torch.cat([idx, *new], dim=1) if new else idx


@torch.no_grad()
def last_token_logits(model: GPT, idx: torch.Tensor) -> torch.Tensor:
    """Logits for the token that follows `idx` (1, T): (V,). Context is cropped to block_size."""
    was_training = model.training
    model.eval()
    logits, _ = model(idx[:, -model.cfg.block_size:])
    model.train(was_training)
    return logits[0, -1]


# --------------------------------------------------------------------------------------
# Loading + CLI
# --------------------------------------------------------------------------------------
def load_model(ckpt_path: str | Path, device: str = "cpu") -> tuple[GPT, dict]:
    """Rebuild a model from a checkpoint (architecture comes from the checkpoint itself)."""
    ck = torch.load(ckpt_path, map_location=device, weights_only=False)
    model = GPT(GPTConfig(**ck["model_config"])).to(device)
    model.load_state_dict(ck["model"])
    model.eval()
    return model, ck


def tokenizer_for_checkpoint(ck: dict, ckpt_path: str | Path):
    """The tokenizer stored inside the checkpoint; falls back to tokenizer.json beside it (older checkpoints)."""
    if ck.get("tokenizer"):
        return tokenizer_from_dict(ck["tokenizer"])
    p = Path(ckpt_path).parent / "tokenizer.json"
    if not p.exists():
        raise SystemExit(f"checkpoint has no embedded tokenizer and {p} does not exist")
    return load_tokenizer(p)


def encode_prompt(tok, prompt: str, device: str) -> torch.Tensor:
    """(1, T) tensor. An empty prompt starts from token 0 (a newline in char-level Shakespeare)."""
    try:
        ids = tok.encode(prompt) if prompt else [0]
    except KeyError as e:
        raise SystemExit(f"prompt contains a character the tokenizer has never seen: {e}")
    return torch.tensor([ids], dtype=torch.long, device=device)


def benchmark(model: GPT, prompt_len: int, new_tokens: int, device: str) -> dict:
    """Tokens/sec with vs. without the KV cache (greedy), and check the outputs are identical."""
    V = model.cfg.vocab_size
    g = torch.Generator().manual_seed(0)
    prompt = torch.randint(0, V, (1, prompt_len), generator=g).to(device)
    res = {}
    for name, cache in (("no cache", False), ("kv cache", True)):
        generate(model, prompt, 4, greedy=True, use_cache=cache)  # warm-up
        if device.startswith("cuda"):
            torch.cuda.synchronize()
        t0 = time.time()
        out = generate(model, prompt, new_tokens, greedy=True, use_cache=cache)
        if device.startswith("cuda"):
            torch.cuda.synchronize()
        dt = time.time() - t0
        res[name] = {"seconds": dt, "tokens_per_sec": new_tokens / dt, "out": out}
    same = torch.equal(res["no cache"]["out"], res["kv cache"]["out"])
    return {
        "prompt_len": prompt_len, "new_tokens": new_tokens, "identical_output": same,
        "no_cache_tps": res["no cache"]["tokens_per_sec"], "cache_tps": res["kv cache"]["tokens_per_sec"],
        "speedup": res["kv cache"]["tokens_per_sec"] / res["no cache"]["tokens_per_sec"],
    }


def main() -> None:
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8")
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", help="checkpoint (.pt); contains its own tokenizer")
    ap.add_argument("--prompt", default="")
    ap.add_argument("--max_new_tokens", type=int, default=300)
    ap.add_argument("--temperature", type=float, default=0.8)
    ap.add_argument("--top_k", type=int, default=None)
    ap.add_argument("--top_p", type=float, default=None)
    ap.add_argument("--greedy", action="store_true")
    ap.add_argument("--no_cache", action="store_true", help="disable the KV cache")
    ap.add_argument("--num_samples", type=int, default=1)
    ap.add_argument("--seed", type=int, default=1337)
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    ap.add_argument("--benchmark", action="store_true", help="compare tokens/sec with vs without KV cache")
    ap.add_argument("--config", default="configs/small.yaml", help="benchmark only: random-weights model if no --ckpt")
    ap.add_argument("--out", default="results/kv_cache_benchmark.md")
    a = ap.parse_args()

    if a.benchmark:
        if a.ckpt:
            model, _ = load_model(a.ckpt, a.device)
        else:
            from src.utils import load_config
            model = GPT(GPTConfig(vocab_size=65, **load_config(a.config)["model"])).to(a.device).eval()
        bs = model.cfg.block_size
        r = benchmark(model, prompt_len=16, new_tokens=bs - 16, device=a.device)  # stays inside block_size
        lines = [
            f"KV-cache benchmark ({a.device}, {model.num_params():,} params, block_size={bs}, "
            f"prompt={r['prompt_len']}, new={r['new_tokens']}, greedy)", "",
            "| mode | tokens/sec |", "|---|---|",
            f"| no cache | {r['no_cache_tps']:.1f} |", f"| kv cache | {r['cache_tps']:.1f} |", "",
            f"speed-up: {r['speedup']:.2f}x; identical greedy output: {r['identical_output']}",
        ]
        print("\n".join(lines))
        Path(a.out).parent.mkdir(parents=True, exist_ok=True)
        Path(a.out).write_text("\n".join(lines) + "\n", encoding="utf-8")
        return

    if not a.ckpt:
        raise SystemExit("--ckpt is required")
    model, ck = load_model(a.ckpt, a.device)
    tok = tokenizer_for_checkpoint(ck, a.ckpt)
    gen = torch.Generator(device="cpu").manual_seed(a.seed)
    idx = encode_prompt(tok, a.prompt, a.device).repeat(a.num_samples, 1)
    out = generate(model, idx, a.max_new_tokens, a.temperature, a.top_k, a.top_p, a.greedy,
                   use_cache=not a.no_cache, generator=gen)
    for i, row in enumerate(out.tolist()):
        if a.num_samples > 1:
            print(f"--- sample {i + 1} ---")
        print(tok.decode(row))


if __name__ == "__main__":
    main()
