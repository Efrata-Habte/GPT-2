"""Training loop: AdamW, linear warmup + cosine decay, grad clipping, optional fp16 autocast,
periodic train/val evaluation, CSV logging, checkpoint/resume.

    python -m src.train --config configs/tiny.yaml
    python -m src.train --config configs/small.yaml --set train.max_iters=2000 model.n_layer=4
    python -m src.train --config configs/small.yaml --resume checkpoints/small/ckpt_last.pt

A "step" always means "number of completed optimizer updates". An eval/checkpoint at step s
therefore happens BEFORE update s+1, which makes resuming exact.
"""
from __future__ import annotations

import argparse
import contextlib
import csv
import json
import math
import shutil
import time
from dataclasses import asdict
from pathlib import Path

import torch

from src.data import default_data_dir, get_batch, load_data, load_meta, prepare_dataset
from src.model import GPT, GPTConfig
from src.utils import apply_overrides, load_config, plot_loss_curves, set_seed

LOG_FIELDS = ["step", "lr", "train_loss", "tokens_per_sec", "eval_train_loss", "eval_val_loss", "elapsed_s"]

TRAIN_DEFAULTS = dict(
    batch_size=64, max_iters=3000, eval_interval=250, eval_iters=50, log_interval=10,
    lr=1e-3, min_lr=1e-4, warmup_iters=100, weight_decay=0.1, beta1=0.9, beta2=0.95,
    grad_clip=1.0, mixed_precision=True, seed=1337, out_dir="checkpoints/run", stop_iter=None,
)


def get_lr(step: int, tc: dict) -> float:
    """Linear warmup to `lr`, then cosine decay to `min_lr` at max_iters."""
    if step < tc["warmup_iters"]:
        return tc["lr"] * (step + 1) / tc["warmup_iters"]        # 0 < lr_0 <= ... <= lr
    if step >= tc["max_iters"]:
        return tc["min_lr"]
    progress = (step - tc["warmup_iters"]) / (tc["max_iters"] - tc["warmup_iters"])   # 0 -> 1
    coeff = 0.5 * (1.0 + math.cos(math.pi * progress))                                # 1 -> 0
    return tc["min_lr"] + coeff * (tc["lr"] - tc["min_lr"])


def make_optimizer(model: GPT, tc: dict, device_type: str) -> torch.optim.AdamW:
    """AdamW with weight decay only on matrices (Linear/Embedding weights), not on biases/LayerNorm."""
    params = [p for p in model.parameters() if p.requires_grad]   # tied weights appear once
    decay = [p for p in params if p.dim() >= 2]
    no_decay = [p for p in params if p.dim() < 2]
    groups = [{"params": decay, "weight_decay": tc["weight_decay"]}, {"params": no_decay, "weight_decay": 0.0}]
    return torch.optim.AdamW(groups, lr=tc["lr"], betas=(tc["beta1"], tc["beta2"]),
                             fused=(device_type == "cuda"))


@torch.no_grad()
def estimate_loss(model: GPT, data: dict, tc: dict, block_size: int, device: str, amp_ctx) -> dict[str, float]:
    """Mean loss over `eval_iters` batches for train AND val.

    Uses a freshly seeded generator, so every evaluation sees the SAME batches: differences
    between evals reflect the model, not batch sampling noise.
    """
    model.eval()
    out = {}
    for split in ("train", "val"):
        gen = torch.Generator().manual_seed(12345)
        losses = torch.zeros(tc["eval_iters"])
        for i in range(tc["eval_iters"]):
            x, y = get_batch(data[split], tc["batch_size"], block_size, gen, device)
            with amp_ctx:
                _, loss = model(x, y)
            losses[i] = loss.item()
        out[split] = losses.mean().item()
    model.train()
    return out


def save_checkpoint(path: Path, model, opt, scaler, cfg: dict, step: int, best_val: float, gen) -> None:
    torch.save({
        "model": model.state_dict(), "optimizer": opt.state_dict(),
        "scaler": scaler.state_dict() if scaler is not None else None,
        "config": cfg, "model_config": asdict(model.cfg), "step": step, "best_val_loss": best_val,
        "batch_gen_state": gen.get_state(), "torch_rng_state": torch.get_rng_state(),
    }, path)


def train(cfg: dict, resume: str | None = None) -> dict:
    """Run training from a config dict (see configs/*.yaml). Returns a summary dict."""
    tc = {**TRAIN_DEFAULTS, **cfg.get("train", {})}
    dc = cfg["data"]
    set_seed(tc["seed"])
    device = "cuda" if torch.cuda.is_available() else "cpu"

    # ---- data (prepared on the fly if missing) ----
    data_dir = Path(dc.get("dir") or default_data_dir(dc["path"], dc["tokenizer"], dc.get("bpe_vocab_size", 512)))
    if not (data_dir / "train.bin").exists():
        print(f"preparing dataset {data_dir} from {dc['path']} ...")
        prepare_dataset(Path(dc["path"]).read_text(encoding="utf-8"), data_dir, dc["tokenizer"],
                        dc.get("bpe_vocab_size", 512))
    meta, data = load_meta(data_dir), load_data(data_dir)

    # ---- model ----
    model_cfg = GPTConfig(vocab_size=meta["vocab_size"], **cfg["model"])
    model = GPT(model_cfg).to(device)
    model.print_param_breakdown()
    opt = make_optimizer(model, tc, device)
    use_amp = bool(tc["mixed_precision"]) and device == "cuda"
    scaler = torch.amp.GradScaler("cuda") if use_amp else None   # fp16 needs loss scaling (T4 has no bf16)
    amp_ctx = torch.autocast("cuda", dtype=torch.float16) if use_amp else contextlib.nullcontext()

    gen = torch.Generator().manual_seed(tc["seed"])   # dedicated RNG for batch sampling
    out_dir = Path(tc["out_dir"])
    out_dir.mkdir(parents=True, exist_ok=True)
    for name in ("tokenizer.json", "meta.json"):   # keep the checkpoint folder self-contained for generate.py
        shutil.copy(data_dir / name, out_dir / name)
    start_step, best_val, resumed = 0, float("inf"), False
    if resume:
        ck = torch.load(resume, map_location=device, weights_only=False)
        model.load_state_dict(ck["model"])
        opt.load_state_dict(ck["optimizer"])
        if scaler is not None and ck["scaler"] is not None:
            scaler.load_state_dict(ck["scaler"])
        gen.set_state(ck["batch_gen_state"])
        torch.set_rng_state(ck["torch_rng_state"].cpu())
        start_step, best_val, resumed = ck["step"], ck["best_val_loss"], True
        print(f"resumed from {resume} at step {start_step} (best val {best_val:.4f})")

    log_path = out_dir / "log.csv"
    new_log = not (resumed and log_path.exists())
    log_f = open(log_path, "w" if new_log else "a", newline="", encoding="utf-8")
    log = csv.DictWriter(log_f, fieldnames=LOG_FIELDS)
    if new_log:
        log.writeheader()

    B, T = tc["batch_size"], model_cfg.block_size
    print(f"device={device} amp={use_amp} tokens/step={B * T:,} data={data_dir} vocab={meta['vocab_size']}")
    t_start = time.time() - (0.0)
    t_last, running, n_running = time.time(), torch.zeros((), device=device), 0
    tps_hist: list[float] = []

    model.train()
    # stop_iter lets you stop early (e.g. a Colab session limit) while keeping the LR schedule of max_iters.
    last_step = min(tc["max_iters"], tc["stop_iter"] or tc["max_iters"])
    for step in range(start_step, last_step + 1):
        # ---------------- periodic evaluation + checkpointing ----------------
        is_eval = step % tc["eval_interval"] == 0 or step == last_step
        if is_eval and not (resumed and step == start_step):
            losses = estimate_loss(model, data, tc, T, device, amp_ctx)
            lr_now = get_lr(step, tc)
            print(f"step {step:>6} | train {losses['train']:.4f} | val {losses['val']:.4f} | lr {lr_now:.2e}")
            log.writerow({"step": step, "lr": f"{lr_now:.6e}", "eval_train_loss": f"{losses['train']:.6f}",
                          "eval_val_loss": f"{losses['val']:.6f}", "elapsed_s": f"{time.time() - t_start:.1f}"})
            log_f.flush()
            if losses["val"] < best_val:
                best_val = losses["val"]
                save_checkpoint(out_dir / "ckpt_best.pt", model, opt, scaler, cfg, step, best_val, gen)
            save_checkpoint(out_dir / "ckpt_last.pt", model, opt, scaler, cfg, step, best_val, gen)
            t_last = time.time()   # don't count eval time in tokens/sec
        if step == last_step:
            break

        # ---------------- one optimizer step ----------------
        lr = get_lr(step, tc)
        for g in opt.param_groups:
            g["lr"] = lr
        x, y = get_batch(data["train"], B, T, gen, device)       # both (B, T)
        with amp_ctx:
            _, loss = model(x, y)
        opt.zero_grad(set_to_none=True)
        if scaler is not None:
            scaler.scale(loss).backward()
            scaler.unscale_(opt)                                 # so clipping sees true gradient norms
            torch.nn.utils.clip_grad_norm_(model.parameters(), tc["grad_clip"])
            scaler.step(opt)
            scaler.update()
        else:
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), tc["grad_clip"])
            opt.step()
        running += loss.detach()
        n_running += 1

        if (step + 1) % tc["log_interval"] == 0:
            avg = (running / n_running).item()                   # .item() syncs the GPU -> timing is honest
            now = time.time()
            tps = B * T * n_running / (now - t_last)
            tps_hist.append(tps)
            if not math.isfinite(avg):
                raise RuntimeError(f"loss became {avg} at step {step + 1}")
            log.writerow({"step": step + 1, "lr": f"{lr:.6e}", "train_loss": f"{avg:.6f}",
                          "tokens_per_sec": f"{tps:.0f}", "elapsed_s": f"{now - t_start:.1f}"})
            running.zero_()
            n_running, t_last = 0, now
    log_f.close()

    elapsed = time.time() - t_start
    final = estimate_loss(model, data, tc, T, device, amp_ctx)
    # bits per character: loss is nats/token; tokens-per-char converts to nats/char; /ln2 -> bits
    bpc = final["val"] * meta["val_tokens"] / meta["val_chars"] / math.log(2)
    summary = {
        "params": model.num_params(), "best_val_loss": best_val, "final_val_loss": final["val"],
        "final_train_loss": final["train"], "val_bits_per_char": bpc, "train_seconds": elapsed,
        "mean_tokens_per_sec": sum(tps_hist) / max(1, len(tps_hist)), "steps": last_step,
        "vocab_size": meta["vocab_size"], "device": device,
    }
    (out_dir / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    plot_loss_curves(log_path, out_dir / "loss_curve.png", title=out_dir.name)
    print(json.dumps(summary, indent=2))
    return summary


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    ap.add_argument("--resume", default=None, help="path to ckpt_last.pt to resume from")
    ap.add_argument("--set", nargs="*", default=[], metavar="KEY=VAL", help="override config, e.g. train.max_iters=200")
    args = ap.parse_args()
    cfg = apply_overrides(load_config(args.config), args.set)
    train(cfg, resume=args.resume)


if __name__ == "__main__":
    main()
