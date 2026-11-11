"""Build the "full data" Amharic corpus: Amharic Wikipedia + the MasakhaNEWS training text.

Outputs
    data/amharic/amharic_full.txt                 training corpus for the full-data model
    data/test_sets/amharic_news_test.txt          the last 10% of data/amharic/amharic.txt = the small model's own
                                                  validation split, unseen by BOTH models (not added to amharic_full.txt)
    data/test_sets/amharic_wiki_test.txt          5% of Wikipedia articles, unseen by BOTH models

The small-data corpus (news, 90%) is a subset of the full one, so the comparison is "same text + more text".
Ethiopic punctuation is kept. Wiki markup is removed with stdlib regexes (templates, tables, references, file/category
links, HTML); only articles that are mostly Ge'ez script are kept.

Source: https://dumps.wikimedia.org/amwiki/latest/ (CC BY-SA 4.0). Usage (from the repo root):
    python -I data/amharic/prepare_wikipedia.py
"""
from __future__ import annotations

import argparse
import bz2
import html
import random
import re
import urllib.request
import xml.etree.ElementTree as ET
from pathlib import Path

URL = "https://dumps.wikimedia.org/amwiki/latest/amwiki-latest-pages-articles.xml.bz2"
ETHIOPIC = re.compile(r"[ሀ-፿ᎀ-᎟ⶀ-⷟꬀-꬯]")
# characters we keep: Ge'ez blocks, ASCII printable, newline and a few common punctuation marks
ALLOWED = re.compile(r"[^ሀ-፿ᎀ-᎟ⶀ-⷟꬀-꬯\x20-\x7E\n«»“”‘’–—…]")


def download(url: str, dest: Path) -> None:
    """Download to `dest`. Wikimedia (and some other hosts) answer 403 to Python's default user agent, so send a
    descriptive one; write to a temporary file first so an interrupted download never leaves a broken file behind."""
    dest.parent.mkdir(parents=True, exist_ok=True)
    tmp = dest.with_name(dest.name + ".part")
    req = urllib.request.Request(url, headers={"User-Agent": "gpt2-from-scratch-data-prep/1.0 (educational project; Python urllib)"})
    with urllib.request.urlopen(req, timeout=120) as r, open(tmp, "wb") as f:
        while chunk := r.read(1 << 20):
            f.write(chunk)
    tmp.replace(dest)


def clean_wikitext(t: str) -> str:
    t = re.sub(r"<!--.*?-->", "", t, flags=re.S)
    t = re.sub(r"<ref[^>]*/>", "", t)
    t = re.sub(r"<ref[^>]*>.*?</ref>", "", t, flags=re.S)
    prev = None
    while prev != t:                                            # nested templates {{ ... {{ ... }} ... }}
        prev = t
        t = re.sub(r"\{\{[^{}]*\}\}", "", t)
    t = re.sub(r"\{\|.*?\|\}", "", t, flags=re.S)               # tables
    t = re.sub(r"\[\[[^\[\]|]*:[^\[\]]*\]\]", "", t)            # [[File:..]], [[Category:..]], interwiki (namespace prefix)
    t = re.sub(r"\[\[([^\[\]|]*)\|([^\[\]]*)\]\]", r"\2", t)    # [[target|label]] -> label
    t = re.sub(r"\[\[([^\[\]]*)\]\]", r"\1", t)                 # [[target]] -> target
    t = re.sub(r"\[https?://\S+\s([^\]]*)\]", r"\1", t)        # [url label] -> label
    t = re.sub(r"\[?https?://\S+\]?", "", t)
    t = re.sub(r"<[^>]+>", "", t)                               # remaining HTML tags
    t = re.sub(r"'{2,}", "", t)                                 # bold / italic
    t = re.sub(r"^=+\s*(.*?)\s*=+\s*$", r"\1", t, flags=re.M)   # == heading == -> heading
    t = re.sub(r"^[*#:;]+\s*", "", t, flags=re.M)               # list markers
    t = re.sub(r"^\s*\|.*$", "", t, flags=re.M)                 # left-over table/template rows
    t = html.unescape(t)
    t = ALLOWED.sub("", t)
    t = re.sub(r"[ \t]+", " ", t)
    lines = [ln.strip() for ln in t.splitlines()]
    t = "\n".join(ln for ln in lines if ln)
    return t.strip()


def is_amharic_article(text: str, min_ethiopic: int = 300, min_share: float = 0.7) -> bool:
    eth = len(ETHIOPIC.findall(text))
    letters = sum(c.isalpha() for c in text)
    return eth >= min_ethiopic and eth / max(1, letters) >= min_share


def iter_articles(dump: Path):
    """(title, wikitext) for main-namespace, non-redirect pages."""
    with bz2.open(dump, "rb") as f:
        for _, el in ET.iterparse(f, events=("end",)):
            if el.tag.endswith("}page"):
                ns = el.find("{*}ns")
                redirect = el.find("{*}redirect")
                text = el.find("{*}revision/{*}text")
                title = el.find("{*}title")
                if ns is not None and ns.text == "0" and redirect is None and text is not None and text.text:
                    yield (title.text if title is not None else ""), text.text
                el.clear()


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--dump", default="data/amharic_wiki_download/amwiki-latest-pages-articles.xml.bz2")
    ap.add_argument("--news", default="data/amharic/amharic.txt", help="the small-data corpus (built by data/amharic/prepare.py)")
    ap.add_argument("--out", default="data/amharic/amharic_full.txt")
    ap.add_argument("--test_dir", default="data/test_sets")
    ap.add_argument("--wiki_test_frac", type=float, default=0.05)
    ap.add_argument("--wiki_only", action="store_true", help="full corpus = Wikipedia only (default: Wikipedia + news training text)")
    ap.add_argument("--seed", type=int, default=0)
    a = ap.parse_args()

    dump = Path(a.dump)
    if not dump.exists():
        print(f"downloading {URL}")
        download(URL, dump)

    n_pages, articles = 0, []
    for title, wikitext in iter_articles(dump):
        n_pages += 1
        text = clean_wikitext(wikitext)
        if is_amharic_article(text):
            articles.append(f"{title}\n{text}")
    rng = random.Random(a.seed)
    rng.shuffle(articles)
    n_test = round(a.wiki_test_frac * len(articles))
    wiki_test, wiki_train = articles[:n_test], articles[n_test:]

    news = Path(a.news).read_text(encoding="utf-8")
    split = int(0.9 * len(news))                       # identical to prepare_dataset's split for the small model
    news_train, news_test = news[:split], news[split:]

    units = wiki_train + ([] if a.wiki_only else [u for u in news_train.split("\n\n") if u.strip()])
    rng.shuffle(units)                                 # mix news and wiki so the full model's own val split has both
    full = "\n\n".join(units) + "\n"
    test_dir = Path(a.test_dir)
    test_dir.mkdir(parents=True, exist_ok=True)
    Path(a.out).write_text(full, encoding="utf-8")
    (test_dir / "amharic_news_test.txt").write_text(news_test, encoding="utf-8")
    (test_dir / "amharic_wiki_test.txt").write_text("\n\n".join(wiki_test) + "\n", encoding="utf-8")
    wiki_chars = sum(map(len, wiki_train))
    print(f"{n_pages:,} main-namespace pages -> {len(articles):,} mostly-Ge'ez articles kept")
    print(f"wiki training text: {wiki_chars:,} chars; news training text: {len(news_train):,} chars")
    print(f"wrote {a.out}: {len(full):,} chars ({len(full) / len(news_train):.1f}x the small model's training text), {len(set(full))} distinct chars")
    print(f"wrote test sets: news {len(news_test):,} chars, wiki {sum(map(len, wiki_test)):,} chars ({len(wiki_test)} articles)")


if __name__ == "__main__":
    main()
