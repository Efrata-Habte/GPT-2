import csv
import math
from pathlib import Path

import numpy as np
import pytest
import torch

from src.data import get_batch, load_data, load_meta, prepare_dataset
from src.train import get_lr, train
from src.utils import apply_overrides

TEXT = ("the quick brown fox jumps over the lazy dog. " * 400) + ("pack my box with five dozen liquor jugs. " * 400)


@pytest.fixture(scope="module")
def data_dir(tmp_path_factory):
    d = tmp_path_factory.mktemp("ds")
    prepare_dataset(TEXT, d, "char")
    return d


def test_split_is_90_10_and_files_match_meta(data_dir):
    meta, data = load_meta(data_dir), load_data(data_dir)
    assert len(data["train"]) == meta["train_tokens"] and len(data["val"]) == meta["val_tokens"]
    assert meta["train_chars"] == int(0.9 * len(TEXT))
    assert meta["train_tokens"] == meta["train_chars"]  # char level: 1 token per char


def test_y_is_x_shifted_by_one(data_dir):
    data = load_data(data_dir)["train"]
    x, y = get_batch(data, 8, 16, torch.Generator().manual_seed(0))
    assert x.shape == y.shape == (8, 16) and x.dtype == torch.int64
    assert torch.equal(x[:, 1:], y[:, :-1])
    # each row is a real contiguous window of the data
    row = x[0].numpy()
    starts = [i for i in range(len(data) - 17) if np.array_equal(data[i:i + 16], row)]
    assert starts and np.array_equal(data[starts[0] + 1:starts[0] + 17], y[0].numpy())


def test_get_batch_is_seeded(data_dir):
    d = load_data(data_dir)["train"]
    a = get_batch(d, 4, 8, torch.Generator().manual_seed(1))
    b = get_batch(d, 4, 8, torch.Generator().manual_seed(1))
    assert torch.equal(a[0], b[0])


def test_lr_schedule():
    tc = dict(lr=1e-3, min_lr=1e-4, warmup_iters=10, max_iters=110)
    assert 0 < get_lr(0, tc) < get_lr(5, tc) < get_lr(9, tc) == pytest.approx(1e-3)   # linear warmup
    assert get_lr(10, tc) == pytest.approx(1e-3)                                      # peak
    assert get_lr(60, tc) == pytest.approx((1e-3 + 1e-4) / 2)                         # cosine midpoint
    assert get_lr(110, tc) == pytest.approx(1e-4) and get_lr(500, tc) == pytest.approx(1e-4)
    lrs = [get_lr(s, tc) for s in range(10, 111)]
    assert all(a >= b - 1e-12 for a, b in zip(lrs, lrs[1:]))                          # monotone decay


def make_cfg(data_dir: Path, out_dir: Path, max_iters: int) -> dict:
    return {
        "model": dict(n_layer=2, n_head=2, n_embd=32, block_size=32, dropout=0.0),
        "data": dict(path="unused.txt", tokenizer="char", dir=str(data_dir)),
        "train": dict(batch_size=16, max_iters=max_iters, eval_interval=20, eval_iters=5, log_interval=5,
                      lr=3e-3, min_lr=3e-4, warmup_iters=5, mixed_precision=False, out_dir=str(out_dir)),
    }


def test_training_reduces_loss_and_writes_artifacts(data_dir, tmp_path):
    out = tmp_path / "run"
    s = train(make_cfg(data_dir, out, 60))
    rows = list(csv.DictReader(open(out / "log.csv")))
    evals = [float(r["eval_val_loss"]) for r in rows if r["eval_val_loss"]]
    assert evals[0] == pytest.approx(math.log(load_meta(data_dir)["vocab_size"]), abs=0.15)  # starts at ln V
    assert evals[-1] < evals[0] - 0.5
    assert s["best_val_loss"] <= min(evals) + 1e-5   # CSV rounds to 6 digits
    for f in ["ckpt_best.pt", "ckpt_last.pt", "loss_curve.png", "summary.json"]:
        assert (out / f).exists()
    assert any(r["tokens_per_sec"] for r in rows)


def test_resume_matches_uninterrupted_run(data_dir, tmp_path):
    """Train 40 straight vs train 20 + resume to 40: same final weights."""
    full = tmp_path / "full"
    train(make_cfg(data_dir, full, 40))
    part = tmp_path / "part"
    first = make_cfg(data_dir, part, 40)
    first["train"]["stop_iter"] = 20          # same LR schedule, just stop halfway
    train(first)
    train(make_cfg(data_dir, part, 40), resume=str(part / "ckpt_last.pt"))
    a = torch.load(full / "ckpt_last.pt", weights_only=False)["model"]
    b = torch.load(part / "ckpt_last.pt", weights_only=False)["model"]
    assert torch.load(part / "ckpt_last.pt", weights_only=False)["step"] == 40
    for k in a:
        assert torch.allclose(a[k], b[k], atol=1e-6), k


def test_apply_overrides():
    cfg = apply_overrides({"train": {"lr": 1}, "model": {}}, ["train.max_iters=200", "model.dropout=0.5", "train.mixed_precision=false"])
    assert cfg["train"]["max_iters"] == 200 and cfg["model"]["dropout"] == 0.5 and cfg["train"]["mixed_precision"] is False
