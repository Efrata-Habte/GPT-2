"""Interactive demo for the from-scratch GPT (English + Amharic).

    streamlit run app/app.py

Everything here uses only this repository's own code (src/model.py, src/tokenizer.py, src/generate.py). The
UI lives in this file; the logic is in app/lib.py; core model code is untouched.
"""
from __future__ import annotations

import html
import json
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
# `streamlit run app/app.py` puts app/ itself on sys.path, so `import app` would find THIS file instead of the
# app/ package. Put the repo root first and drop the script's own folder so `from app import lib` is unambiguous.
_here = Path(__file__).resolve().parent
sys.path[:] = [str(ROOT)] + [p for p in sys.path if Path(p or ".").resolve() not in (_here, ROOT)]
if "app" in sys.modules and not hasattr(sys.modules["app"], "__path__"):
    del sys.modules["app"]                       # a stale module object for app.py (not the package)

import altair as alt
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd
import streamlit as st

from app import lib
from src.utils import setup_ethiopic_font

plt.style.use("dark_background")                  # matplotlib figures (attention heatmaps) match the dark theme
plt.rcParams.update({"figure.facecolor": "#0E1117", "axes.facecolor": "#0E1117", "savefig.facecolor": "#0E1117"})
setup_ethiopic_font()
FONT = lib.ETHIOPIC_CSS_FONTS

st.set_page_config(page_title="GPT from scratch", layout="wide")
ACCENT = "#6366F1"
ACCENT_TEXT = "#A5B4FC"
PALETTE = ["#818CF8", "#F59E0B", "#34D399", "#F472B6", "#22D3EE", "#FACC15"]
BG, SURFACE, SURFACE_2, BORDER, MUTED = "#0E1117", "#161A26", "#1E2333", "#2A3045", "#9AA1BD"
st.markdown(
    f"""
    <style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700&family=Noto+Sans+Ethiopic:wght@400;600&display=swap');
    @font-face {{ font-family: 'Noto Sans Ethiopic Local'; src: url('app/static/NotoSansEthiopic-Regular.ttf') format('truetype'); }}
    html, body, .stApp, textarea, input, button, .stMarkdown, .stTextArea textarea {{
        font-family: 'Inter', {FONT}, 'Noto Sans Ethiopic Local', sans-serif; }}
    .block-container {{ padding-top: 3.2rem; max-width: 1280px; }}
    h1, h2, h3, h4 {{ font-weight: 650; letter-spacing: -0.01em; }}

    /* header banner */
    .hero {{ background: linear-gradient(120deg, #312E81 0%, #4F46E5 55%, #7C3AED 100%); color: #fff;
             border-radius: 16px; padding: 1.6rem 1.9rem 1.4rem; margin-bottom: 1.2rem;
             box-shadow: 0 10px 30px rgba(0,0,0,0.45); }}
    .hero h1 {{ color: #fff; font-size: 1.9rem; margin: 0 0 .35rem; padding: 0; }}
    .hero p {{ color: rgba(255,255,255,.85); margin: 0 0 1rem; font-size: 1rem; }}
    .chips {{ display: flex; flex-wrap: wrap; gap: .45rem; }}
    .chip {{ background: rgba(255,255,255,.14); border: 1px solid rgba(255,255,255,.25); color: #fff;
             border-radius: 999px; padding: .22rem .75rem; font-size: .83rem; white-space: nowrap; }}

    /* tabs as pills (Streamlit 1.66 uses react-aria tabs) */
    .stTabs [role="tablist"] {{ gap: .35rem; flex-wrap: wrap; border-bottom: none; overflow: visible; }}
    .stTabs [role="tab"] {{ background: {SURFACE}; border: 1px solid {BORDER}; border-radius: 999px; padding: .38rem 1rem;
                            height: auto; margin: 0; transition: background .15s; }}
    .stTabs [role="tab"]:hover {{ background: {SURFACE_2}; }}
    .stTabs [role="tab"] p {{ font-size: .9rem; font-weight: 500; margin: 0; }}
    .stTabs [role="tab"][aria-selected="true"] {{ background: {ACCENT}; border-color: {ACCENT}; }}
    .stTabs [role="tab"][aria-selected="true"] p {{ color: #fff; }}
    .stTabs .react-aria-SelectionIndicator {{ display: none; }}
    .stTabs [role="tabpanel"] {{ padding-top: 1.2rem; }}

    /* cards */
    .out-card {{ background: {SURFACE}; border: 1px solid {BORDER}; border-left: 4px solid {ACCENT}; border-radius: 12px;
                 padding: 1rem 1.2rem; white-space: pre-wrap; line-height: 1.75; }}
    .out-title {{ font-weight: 650; color: {ACCENT_TEXT}; margin-bottom: .35rem; font-size: .85rem; text-transform: uppercase; letter-spacing: .05em; }}
    [data-testid="stMetric"] {{ background: {SURFACE}; border: 1px solid {BORDER}; border-radius: 10px; padding: .5rem .75rem; }}
    [data-testid="stMetricValue"] {{ font-size: 1.45rem; font-weight: 600; }}
    .side-card {{ background: {SURFACE_2}; border: 1px solid {BORDER}; border-radius: 12px; padding: .75rem .95rem; font-size: .86rem;
                  display: grid; grid-template-columns: auto 1fr; gap: .38rem .8rem; }}
    .side-card .k {{ color: {MUTED}; }}
    .side-card .v {{ font-weight: 600; text-align: right; }}
    </style>
    """,
    unsafe_allow_html=True,
)

DEFAULT_PROMPT = {"English": "ROMEO:\n", "Amharic": "የኢትዮጵያ "}
DEFAULT_SENTENCE = {"English": "ROMEO:\nO, where art thou? Speak, speak!", "Amharic": "ሰላም ለዓለም። የኢትዮጵያ ሕዝብ ሰላም ነው።"}


# ----------------------------------------------------------------------------------------------
# helpers
# ----------------------------------------------------------------------------------------------
@st.cache_resource(show_spinner="Loading model...")
def get_model(key: str):
    path = lib.find_checkpoint(key)
    return lib.load_demo_model(path) if path else None


@st.cache_resource(show_spinner=False)
def get_playground(set_name: str, vocab_size: int):
    return lib.load_playground_tokenizers(set_name, vocab_size)


def show_text(text: str) -> None:
    """Plain text that keeps newlines and renders Ge'ez (st.code would use a monospace font without Ethiopic glyphs)."""
    st.markdown(f"<div class='out-card'>{html.escape(text)}</div>", unsafe_allow_html=True)


def altair_font(chart: alt.Chart) -> alt.Chart:
    """One look for every chart: the app's font (with Ge'ez fallbacks), its palette, light grid lines and no frame."""
    return (chart.configure(font="Inter, Noto Sans Ethiopic, Nyala, Ebrima, sans-serif", background="transparent")
            .configure_range(category=PALETTE)
            .configure_mark(color=PALETTE[0])
            .configure_view(strokeWidth=0)
            .configure_axis(gridColor="#252B3D", domainColor="#3A4160", labelColor="#C3C8DC", titleColor="#C3C8DC",
                            labelFontSize=12, titleFontSize=12)
            .configure_legend(labelFontSize=12, titleFontSize=12, labelColor="#C3C8DC", titleColor="#C3C8DC"))


def plot_attention(matrix, labels: list[str], title: str):
    n = len(labels)
    fig, ax = plt.subplots(figsize=(min(11, 3 + 0.28 * n), min(11, 3 + 0.28 * n)))
    im = ax.imshow(matrix, cmap="magma", vmin=0, vmax=float(matrix.max()) if float(matrix.max()) > 0 else 1)
    ax.set_xticks(range(n))
    ax.set_yticks(range(n))
    fs = 9 if n <= 24 else 7 if n <= 48 else 5
    ax.set_xticklabels(labels, rotation=90, fontsize=fs)
    ax.set_yticklabels(labels, fontsize=fs)
    ax.set_xlabel("key token (attended to)")
    ax.set_ylabel("query token")
    ax.set_title(title)
    fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    fig.tight_layout()
    return fig


# ----------------------------------------------------------------------------------------------
# sidebar: model selector
# ----------------------------------------------------------------------------------------------
st.sidebar.markdown("### Choose a model")
choice = st.sidebar.radio("Model", list(lib.MODELS), help="Each language has its own checkpoint and tokenizer.")
data_choice = st.sidebar.radio("Training data", list(lib.DATA_VARIANTS),
                               help="Same architecture and training steps; the full-data model was trained on ~3-4x more text.")
mkey = lib.MODELS[choice] + lib.DATA_VARIANTS[data_choice]          # e.g. english, english_full
bundle = get_model(mkey)
if bundle is None and mkey != lib.MODELS[choice]:
    st.sidebar.warning(f"No full-data {choice} model yet (train `configs/demo_{mkey}.yaml`); showing the small-data model.")
    mkey = lib.MODELS[choice]
    bundle = get_model(mkey)
if bundle is None:
    st.sidebar.error(f"No checkpoint found for the {choice} model.")
    st.error(
        f"No **{choice}** checkpoint found. Train one (`python -m src.train --config configs/demo_"
        f"{mkey}.yaml`) and export it with `python scripts/export_for_demo.py`, "
        "or put `model.pt` in `app/checkpoints/" + mkey + "/`."
    )
    st.stop()
model, tok, info = bundle
mc = info["model_config"]
tok_kind = {"char": "character-level", "byte": "byte-level BPE"}.get(info["tokenizer_type"], info["tokenizer_type"])
if getattr(tok, "level", None) == "char":
    tok_kind = "character-level BPE"
train_chars = f"{info['train_chars'] / 1e6:.1f}M characters" if info.get("train_chars") else "?"
best = f"{info['best_val_loss']:.3f}" if info.get("best_val_loss") else "?"
side_rows = [("Parameters", f"{info['params']:,}"), ("Layers / heads", f"{mc['n_layer']} / {mc['n_head']}"), ("Embedding size", mc["n_embd"]),
             ("Context", f"{mc['block_size']} tokens"), ("Tokenizer", tok_kind), ("Vocabulary", f"{info['vocab_size']} tokens"),
             ("Training steps", f"{info['step'] or '?'}"), ("Best val. loss", best), ("Training text", train_chars)]
st.sidebar.markdown("<div class='side-card'>" + "".join(f"<span class='k'>{k}</span><span class='v'>{html.escape(str(v))}</span>" for k, v in side_rows)
                    + "</div>", unsafe_allow_html=True)
st.sidebar.caption(f"Data: {lib.DATA_DESCRIPTIONS.get(mkey, mkey)}. Trained from random weights on a Colab T4 GPU, so expect "
                   "Shakespeare-like or news-like text without real meaning.")

chips = [f"<b>{choice}</b>", data_choice, f"{info['params'] / 1e6:.1f}M parameters", f"{mc['n_layer']} layers", f"{mc['n_head']} heads",
         f"{mc['block_size']}-token context", tok_kind]
st.markdown(
    "<div class='hero'><h1>A GPT-2-style language model, built from scratch</h1>"
    "<p>Tokenizer, attention, LayerNorm, Transformer blocks, training and sampling written in plain PyTorch, "
    "trained on English Shakespeare and Amharic text.</p>"
    "<div class='chips'>" + "".join(f"<span class='chip'>{c}</span>" for c in chips) + "</div></div>",
    unsafe_allow_html=True,
)
tabs = st.tabs(["Generate", "Next-token inspector", "Temperature comparison", "Small vs full data", "Attention explorer",
                "Tokenizer playground", "About"])

# ----------------------------------------------------------------------------------------------
# 1. Generate
# ----------------------------------------------------------------------------------------------
with tabs[0]:
    st.subheader("Generate text")
    prompt = st.text_area("Prompt", DEFAULT_PROMPT[choice], height=90, key=f"gen_prompt_{mkey}")
    c1, c2 = st.columns(2)
    method = c1.radio("Decoding method", ["greedy", "temperature", "top-k", "top-p"], index=1, horizontal=True, key="gen_method")
    temperature = c1.slider("Temperature", 0.1, 2.0, 0.8, 0.05, disabled=method == "greedy", key="gen_temp")
    seed = c1.number_input("Seed", value=1337, step=1, key="gen_seed")
    top_k = c2.slider("top-k", 1, min(200, info["vocab_size"]), min(40, info["vocab_size"]), disabled=method != "top-k", key="gen_topk")
    top_p = c2.slider("top-p (nucleus)", 0.05, 1.0, 0.9, 0.05, disabled=method != "top-p", key="gen_topp")
    max_new = c2.slider("Max new tokens", 10, 1000, 300, 10, key="gen_max")
    use_cache = c2.checkbox("Use KV cache", True, help="Same output, faster while prompt + output fits in the context window.", key="gen_cache")
    if st.button("Generate", type="primary", key="gen_go"):
        _, dropped = lib.encode_prompt_safe(tok, prompt)
        if dropped:
            st.warning(f"Skipped characters this tokenizer has never seen: {' '.join(dropped)}")
        t0, n_chars = time.time(), 0
        st.markdown("<div class='out-title'>Output</div>", unsafe_allow_html=True)
        box, out = st.empty(), ""
        head = f"<span style='opacity:.5'>{html.escape(prompt)}</span>"
        for i, chunk in enumerate(lib.stream_text(model, tok, prompt, max_new, method, temperature, top_k, top_p, int(seed), use_cache)):
            out += chunk
            if i % 4 == 0:                                   # redraw every few tokens; keeps newlines, unlike st.write_stream
                box.markdown(f"<div class='out-card'>{head}{html.escape(out)}</div>", unsafe_allow_html=True)
        box.markdown(f"<div class='out-card'>{head}{html.escape(out)}</div>", unsafe_allow_html=True)
        dt = time.time() - t0
        st.caption(f"{max_new} tokens in {dt:.1f}s ({max_new / dt:.0f} tokens/s, KV cache {'on' if use_cache else 'off'}). "
                   f"Same seed + settings always gives the same text.")

# ----------------------------------------------------------------------------------------------
# 2. Next-token inspector
# ----------------------------------------------------------------------------------------------
with tabs[1]:
    st.subheader("Next-token inspector")
    st.caption("The model outputs a probability for every token in its vocabulary. Click a token to append it and step forward manually.")
    ids_key, txt_key = f"inspect_ids_{mkey}", f"inspect_txt_{mkey}"
    if ids_key not in st.session_state:
        st.session_state[ids_key] = lib.encode_prompt_safe(tok, DEFAULT_PROMPT[choice])[0]
        st.session_state[txt_key] = tok.decode(st.session_state[ids_key])

    def _on_edit(ids_key=ids_key, txt_key=txt_key):
        st.session_state[ids_key] = lib.encode_prompt_safe(tok, st.session_state[txt_key])[0]

    def _append(token_id: int, ids_key=ids_key, txt_key=txt_key):
        st.session_state[ids_key] = st.session_state[ids_key] + [token_id]
        st.session_state[txt_key] = tok.decode(st.session_state[ids_key])

    def _undo(ids_key=ids_key, txt_key=txt_key):
        st.session_state[ids_key] = st.session_state[ids_key][:-1] or [0]
        st.session_state[txt_key] = tok.decode(st.session_state[ids_key])

    def _reset(ids_key=ids_key, txt_key=txt_key):
        st.session_state[ids_key] = lib.encode_prompt_safe(tok, DEFAULT_PROMPT[choice])[0]
        st.session_state[txt_key] = tok.decode(st.session_state[ids_key])

    st.text_area("Text so far (edit freely)", key=txt_key, on_change=_on_edit, height=110)
    ids = st.session_state[ids_key]
    with st.expander("Sampler settings (change what the bars show)"):
        e1, e2, e3 = st.columns(3)
        i_temp = e1.slider("Temperature", 0.1, 2.0, 1.0, 0.05, key="insp_temp")
        i_topk = e2.slider("top-k (0 = off)", 0, min(200, info["vocab_size"]), 0, key="insp_topk")
        i_topp = e3.slider("top-p (1.0 = off)", 0.05, 1.0, 1.0, 0.05, key="insp_topp")
    top = lib.top_next_tokens(model, tok, ids, 10, i_temp, i_topk or None, i_topp if i_topp < 1.0 else None)
    left, right = st.columns([3, 2])
    with left:
        df = pd.DataFrame({"token": [t["label"] for t in top], "probability": [t["prob"] for t in top]})
        chart = alt.Chart(df).mark_bar(cornerRadiusEnd=4).encode(
            x=alt.X("probability:Q", axis=alt.Axis(format="%")), y=alt.Y("token:N", sort=None, title=None),
            tooltip=["token", alt.Tooltip("probability:Q", format=".2%")]).properties(height=330)
        st.altair_chart(altair_font(chart), width="stretch")
        st.caption(f"Top-10 cover {sum(t['prob'] for t in top):.1%} of the probability mass. `·` = space, `↵` = newline, `<E1 88>` = part of a character.")
    with right:
        st.markdown("**Click a token to append it:**")
        cols = st.columns(2)
        for rank, t in enumerate(top):
            cols[rank % 2].button(f"{t['label']}  ·  {t['prob']:.1%}", key=f"pick_{mkey}_{rank}", on_click=_append, args=(t["id"],), width="stretch")
        b1, b2 = st.columns(2)
        b1.button("Undo last token", on_click=_undo, key=f"undo_{mkey}", width="stretch")
        b2.button("Reset", on_click=_reset, key=f"reset_{mkey}", width="stretch")
    st.markdown("**Current tokens:**")
    st.markdown(lib.tokens_to_html([lib.token_labels(tok, [i], visible=False)[0] for i in ids[-60:]]), unsafe_allow_html=True)
    st.caption(f"{len(ids)} tokens in context" + (f" (the model only looks at the last {mc['block_size']})" if len(ids) > mc["block_size"] else ""))

# ----------------------------------------------------------------------------------------------
# 3. Temperature comparison
# ----------------------------------------------------------------------------------------------
with tabs[2]:
    st.subheader("Same prompt, different temperatures")
    st.caption("Same seed everywhere, so the only difference is temperature. Low = safe and repetitive, high = diverse and error-prone.")
    t_prompt = st.text_area("Prompt", DEFAULT_PROMPT[choice], height=70, key=f"cmp_prompt_{mkey}")
    temps = st.multiselect("Temperatures", [0.1, 0.2, 0.5, 0.8, 1.0, 1.2, 1.5, 2.0], default=[0.2, 0.5, 0.8, 1.0, 1.5], key="cmp_temps")
    d1, d2, d3 = st.columns(3)
    t_max = d1.slider("Max new tokens", 20, 400, 150, 10, key="cmp_max")
    t_seed = d2.number_input("Seed", value=1337, step=1, key="cmp_seed")
    with_greedy = d3.checkbox("Include greedy", True, key="cmp_greedy")
    if st.button("Compare", type="primary", key="cmp_go"):
        settings = ([("greedy", {"greedy": True})] if with_greedy else []) + [(f"T = {t}", {"temperature": t}) for t in sorted(temps)]
        if not settings:
            st.info("Pick at least one setting.")
        else:
            cols = st.columns(len(settings))
            for col, (name, kw) in zip(cols, settings):
                text = lib.generate_text(model, tok, t_prompt, t_max, int(t_seed), **kw)
                with col:
                    st.markdown(f"<div class='out-title'>{name}</div>", unsafe_allow_html=True)
                    show_text(t_prompt + text)
                    st.caption(f"distinct char bigrams: {lib.distinct_n(text, 2):.2f}")

# ----------------------------------------------------------------------------------------------
# 4. Attention explorer
# ----------------------------------------------------------------------------------------------
with tabs[4]:
    st.subheader("Attention explorer")
    st.caption("Row = the token doing the looking, column = the token it looks at. Each row sums to 1; the upper triangle is zero (causal mask).")
    sentence = st.text_input("Sentence", DEFAULT_SENTENCE[choice], key=f"att_text_{mkey}")
    labels, attn, dropped = lib.attention_for_text(model, tok, sentence)
    if dropped:
        st.warning(f"Skipped characters this tokenizer has never seen: {' '.join(dropped)}")
    a1, a2 = st.columns(2)
    layer = a1.slider("Layer", 0, mc["n_layer"] - 1, 0, key="att_layer")
    head = a2.selectbox("Head", ["average"] + list(range(mc["n_head"])), key="att_head")
    mat = attn[layer].mean(0) if head == "average" else attn[layer][head]
    st.pyplot(plot_attention(mat.numpy(), labels, f"layer {layer}, " + ("mean over heads" if head == "average" else f"head {head}")), clear_figure=True)
    with st.expander("All heads of this layer"):
        n = mc["n_head"]
        fig, axes = plt.subplots(1, n, figsize=(2.6 * n, 2.8), squeeze=False)
        for h in range(n):
            axes[0][h].imshow(attn[layer][h].numpy(), cmap="magma", vmin=0, vmax=1)
            axes[0][h].set_title(f"head {h}", fontsize=8)
            axes[0][h].set_xticks([])
            axes[0][h].set_yticks([])
        fig.tight_layout()
        st.pyplot(fig, clear_figure=True)
    st.caption("Look for a bright diagonal just below the main diagonal (a previous-token head), a bright first column (attention sink), "
               "or vertical stripes on punctuation.")

# ----------------------------------------------------------------------------------------------
# 5. Tokenizer playground
# ----------------------------------------------------------------------------------------------
with tabs[5]:
    st.subheader("Tokenizer playground")
    p1, p2, p3 = st.columns([2, 2, 3])
    tok_set = p1.radio("Tokenizers trained on", ["English", "Amharic"], index=0 if choice == "English" else 1, horizontal=True, key="pg_set")
    vocab = p2.select_slider("BPE vocabulary size", [500, 1000, 4000], value=1000, key="pg_vocab")
    normalize = p3.checkbox("Apply Amharic homophone normalization (ሐ ኀ → ሀ, ሠ → ሰ, ዐ → አ, ፀ → ጸ)", False, key="pg_norm")
    raw_text = st.text_area("Text (English or Amharic)", DEFAULT_SENTENCE[tok_set], height=90, key=f"pg_text_{tok_set}")
    text = raw_text
    set_name = "english" if tok_set == "English" else "amharic"
    if normalize:
        text, n_changed = lib.apply_normalization(raw_text)
        st.markdown(f"**Normalization changed {n_changed} character(s)** (highlighted):")
        st.markdown(lib.normalization_diff_html(raw_text, text), unsafe_allow_html=True)
        if tok_set == "Amharic":
            set_name = "amharic_norm"
    toks = get_playground(set_name, vocab)
    if not toks:
        st.info("Tokenizer files not found in `assets/tokenizers/`. Build them with `python experiments/exp11_amharic_study.py`.")
    else:
        cols = st.columns(len(toks))
        summary = []
        for col, (name, t) in zip(cols, toks.items()):
            r = lib.tokenize_for_display(t, text)
            with col:
                st.markdown(f"#### {lib.TOKENIZER_TITLES[name]}")
                st.caption(f"vocab size {r['vocab_size']}")
                st.markdown(lib.tokens_to_html(r["labels"]), unsafe_allow_html=True)
                m1, m2 = st.columns(2)
                m1.metric("tokens", r["n_tokens"])
                m2.metric("bytes / token", f"{r['bytes_per_token']:.2f}")
                m1.metric("chars / token", f"{r['chars_per_token']:.2f}")
                m2.metric("tokens / word", f"{r['tokens_per_word']:.2f}")
                if r["fragments"]:
                    st.warning(f"{r['fragments']} of {r['n_tokens']} tokens are only part of a character (<hex bytes>).")
                if r["dropped"]:
                    st.caption(f"Characters not in this tokenizer's alphabet were skipped: {' '.join(r['dropped'])}")
                with st.expander("Token IDs"):
                    st.code(", ".join(map(str, r["ids"])) or "(empty)")
            summary.append({"tokenizer": lib.TOKENIZER_TITLES[name], "tokens": r["n_tokens"], "bytes": r["n_bytes"], "chars": r["n_chars"],
                            "chars/token": round(r["chars_per_token"], 2), "tokens/word": round(r["tokens_per_word"], 2)})
        st.dataframe(pd.DataFrame(summary), hide_index=True, width="stretch")
        st.caption("Byte-level BPE starts from UTF-8 bytes: every Ge'ez letter is 3 bytes, so it needs many merges before it makes whole letters; "
                   "character-level tokenizers start with one token per letter but need a large alphabet (and cannot encode unseen letters).")

# ----------------------------------------------------------------------------------------------
# 6. About
# ----------------------------------------------------------------------------------------------
with tabs[6]:
    st.subheader("About this model")
    a1, a2 = st.columns([2, 3])
    with a1:
        st.markdown(f"**{choice} model, {data_choice.lower()}**")
        st.json({**mc, "tokenizer": tok_kind, "parameters": info["params"], "trained_steps": info["step"], "checkpoint": Path(info["path"]).name})
        st.markdown("**Parameter breakdown**")
        st.dataframe(pd.DataFrame(lib.param_breakdown_rows(model), columns=["component", "parameters"]), hide_index=True, width="stretch")
    with a2:
        arch = ROOT / "report" / "architecture.png"
        if arch.exists():
            st.image(str(arch), caption="Data flow and tensor shapes")
    st.markdown(
        """
        **What is from scratch:** tokenizers (character-level, my own BPE over characters and over bytes), embeddings,
        multi-head causal self-attention (explicit `QKᵀ/√d`, mask, softmax), LayerNorm, MLP, residual blocks, GPT-2 init,
        weight tying, training loop, sampling (greedy / temperature / top-k / top-p) and a KV cache.
        No pretrained weights, no Hugging Face model code, no `nn.Transformer`, no `nn.MultiheadAttention`, no fused attention kernels.

        **What to expect:** all four models (English and Amharic, small and full data) have about 10.8M parameters and were trained from
        random weights for 3,000 steps (about 13 minutes each) on a free Colab T4. They write text in the right style and mostly real
        words, but without coherent meaning.
        """
    )


# ----------------------------------------------------------------------------------------------
# 4. Small vs full data
# ----------------------------------------------------------------------------------------------
with tabs[3]:
    st.subheader("Does more training data help?")
    st.caption("Two models with the same architecture, the same number of training steps and the same seed. The only difference is "
               "how much text they were trained on.")
    base = lib.MODELS[choice]
    small_b, full_b = get_model(base), get_model(base + "_full")
    if small_b is None or full_b is None:
        st.info(f"Both {choice} models are needed: train `configs/demo_{base}.yaml` and `configs/demo_{base}_full.yaml`, "
                "then run `python scripts/export_for_demo.py`.")
    else:
        variants = [("Small dataset", base, small_b), ("Full dataset", base + "_full", full_b)]
        cols = st.columns(2)
        for col, (label, key, (m, t, inf)) in zip(cols, variants):
            col.markdown(f"**{label}:** {lib.DATA_DESCRIPTIONS.get(key, key)}")
            col.caption(f"{inf['train_chars'] / 1e6:.1f}M training characters · {inf['params']:,} parameters · {inf['step']} steps"
                        if inf.get("train_chars") else f"{inf['params']:,} parameters · {inf['step']} steps")

        scores = ROOT / "results" / "data_scale" / "data_scale.json"
        if scores.exists():
            res = json.loads(scores.read_text(encoding="utf-8")).get(choice)
            if res:
                st.markdown("**Scored on the same unseen text** (bits per character, lower is better):")
                rows = []
                for name, d in res["models"].items():
                    for tn, r in d["tests"].items():
                        rows.append({"model": name, "test text": tn, "bits per character": round(r["bits_per_char"], 3)})
                df = pd.DataFrame(rows)
                chart = alt.Chart(df).mark_bar().encode(
                    x=alt.X("test text:N", title=None), xOffset="model:N", y=alt.Y("bits per character:Q"),
                    color=alt.Color("model:N", legend=alt.Legend(orient="bottom")), tooltip=["model", "test text", "bits per character"])
                st.altair_chart(altair_font(chart.properties(height=280)), width="stretch")
                names = list(res["models"])
                msgs = []
                for tn in res["tests"]:
                    a_, b_ = res["models"][names[0]]["tests"][tn]["bits_per_char"], res["models"][names[-1]]["tests"][tn]["bits_per_char"]
                    msgs.append(f"{tn}: {a_:.3f} → {b_:.3f} ({abs(100 * (a_ - b_) / a_):.1f}% {'better' if b_ < a_ else 'worse'})")
                st.caption(" · ".join(msgs) + ". Each model's own validation loss is on different text, so only this shared test is a fair comparison.")
                st.caption("Valid-word rate at T=0.8: " + ", ".join(f"{n}: {100 * d['valid_words']:.0f}%" for n, d in res["models"].items()))
        else:
            st.info("Run `python experiments/exp12_data_scale.py` to score both models on the shared held-out test text.")

        st.markdown("**Same prompt, same seed, both models:**")
        d_prompt = st.text_area("Prompt", DEFAULT_PROMPT[choice], height=70, key=f"data_prompt_{base}")
        e1, e2, e3 = st.columns(3)
        d_temp = e1.slider("Temperature", 0.1, 1.5, 0.8, 0.05, key="data_temp")
        d_max = e2.slider("Max new tokens", 20, 500, 250, 10, key="data_max")
        d_seed = e3.number_input("Seed", value=1337, step=1, key="data_seed")
        if st.button("Generate with both models", type="primary", key="data_go"):
            cols = st.columns(2)
            for col, (label, key, (m, t, inf)) in zip(cols, variants):
                with col:
                    st.markdown(f"<div class='out-title'>{label}</div>", unsafe_allow_html=True)
                    show_text(d_prompt + lib.generate_text(m, t, d_prompt, d_max, int(d_seed), temperature=d_temp))
