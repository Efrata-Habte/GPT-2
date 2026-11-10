import pytest
import torch

from src.data import prepare_dataset
from src.generate import generate, load_model
from src.model import GPT, GPTConfig
from src.train import make_optimizer, save_checkpoint, train

VARIANTS = [dict(), dict(pos_encoding="sinusoidal"), dict(norm_position="post"), dict(tie_weights=False)]


@pytest.mark.parametrize("kw", VARIANTS)
def test_save_then_load_gives_identical_outputs(kw, tmp_path):
    torch.manual_seed(0)
    cfg = GPTConfig(vocab_size=40, block_size=16, n_layer=2, n_head=2, n_embd=32, dropout=0.1, **kw)
    model = GPT(cfg)
    # take a few optimizer steps so weights are not just the init
    tc = dict(lr=1e-3, weight_decay=0.1, beta1=0.9, beta2=0.95)
    opt = make_optimizer(model, tc, "cpu")
    x = torch.randint(0, 40, (4, 16))
    for _ in range(3):
        _, loss = model(x, x)
        opt.zero_grad()
        loss.backward()
        opt.step()
    model.eval()

    path = tmp_path / "ckpt.pt"
    save_checkpoint(path, model, opt, None, {"model": {}}, step=3, best_val=1.23, gen=torch.Generator())
    loaded, ck = load_model(path)

    assert ck["step"] == 3 and ck["best_val_loss"] == 1.23
    assert loaded.cfg == cfg                                   # architecture restored from the file
    if cfg.tie_weights:
        assert loaded.lm_head.weight is loaded.embed.tok_emb.weight   # tying survives the round trip
    for (n1, p1), (n2, p2) in zip(model.state_dict().items(), loaded.state_dict().items()):
        assert n1 == n2 and torch.equal(p1, p2), n1
    logits_a, _ = model(x)
    logits_b, _ = loaded(x)
    assert torch.equal(logits_a, logits_b)                     # bit-identical, not just close
    prompt = x[:, :3]
    assert torch.equal(generate(model, prompt, 20, greedy=True), generate(loaded, prompt, 20, greedy=True))


def test_checkpoint_from_training_loads_and_matches_best_val(tmp_path):
    """End to end: train -> load ckpt_best -> its val loss equals the one recorded at save time."""
    text = "the quick brown fox jumps over the lazy dog. " * 300
    ddir = tmp_path / "ds"
    prepare_dataset(text, ddir, "char")
    out = tmp_path / "run"
    cfg = {
        "model": dict(n_layer=1, n_head=2, n_embd=32, block_size=16, dropout=0.0),
        "data": dict(path="unused", tokenizer="char", dir=str(ddir)),
        "train": dict(batch_size=8, max_iters=30, eval_interval=10, eval_iters=4, log_interval=5,
                      mixed_precision=False, out_dir=str(out)),
    }
    train(cfg)
    assert (out / "tokenizer.json").exists()                   # folder is self-contained for generate.py
    model, ck = load_model(out / "ckpt_best.pt")

    from src.data import get_batch, load_data
    data = load_data(ddir)
    losses = []
    gen = torch.Generator().manual_seed(12345)   # same batches estimate_loss used
    with torch.no_grad():
        for _ in range(4):
            xb, yb = get_batch(data["val"], 8, 16, gen)
            losses.append(model(xb, yb)[1].item())
    assert sum(losses) / 4 == pytest.approx(ck["best_val_loss"], abs=1e-5)
