"""Data loading, segmentation, cleaning ladder and leakage-safe splits.

Corpus: isiXhosa Intellectual Traditions dataset (Schoots et al., 2023),
page-level OCR text for Imvo Zabantsundu (IMVO) and Isigidimi sama-Xosa (ISIG).
Files are named PAPER_YYYY-MM-DD_pNNN.txt.

Unit of analysis: fixed-length windows of WINDOW raw whitespace tokens taken
from each page. Windows are cut from the *raw* text once, and every cleaning
level is applied to the same window, so all levels share identical units,
labels and splits (only the text representation changes).
"""
from __future__ import annotations

import os
import re
from dataclasses import dataclass

import numpy as np
import pandas as pd

YEAR_MIN, YEAR_MAX = 1876, 1888      # 1876-Oct 1884 Isigidimi is used only as
                                      # an out-of-sample probe; the classification
                                      # window is Nov 1884-Dec 1888 (both papers)
WINDOW = 200                          # raw tokens per segment
MIN_TOKENS = 60                       # drop windows shorter than this (raw)

# --------------------------------------------------------------------------
# Loading
# --------------------------------------------------------------------------
FNAME = re.compile(r"^(IMVO|ISIG)_(\d{4})-(\d{2})-(\d{2})_p(\d+)\.txt$")


def load_pages(folder: str) -> pd.DataFrame:
    rows = []
    for f in sorted(os.listdir(folder)):
        m = FNAME.match(f)
        if not m:
            continue
        paper, y, mo, d, p = m.groups()
        y = int(y)
        if not (YEAR_MIN <= y <= YEAR_MAX):
            continue
        with open(os.path.join(folder, f), encoding="utf-8", errors="replace") as fh:
            txt = fh.read()
        rows.append(dict(file=f, paper=paper, year=y, month=int(mo),
                         date=f"{y}-{mo}-{d}", page=int(p),
                         issue=f"{paper}_{y}-{mo}-{d}", text=txt))
    return pd.DataFrame(rows)


# --------------------------------------------------------------------------
# Normalisation shared by every level
# --------------------------------------------------------------------------
QUOTES = str.maketrans({"’": "'", "‘": "'", "`": "'", "“": '"',
                        "”": '"', "—": " - ", "–": " - "})


def base_normalise(text: str) -> str:
    text = text.translate(QUOTES)
    # re-join words hyphenated across line breaks / OCR column breaks
    # e.g. "esi- nesitozela" -> "esinesitozela", "ku-\nngena" -> "kungena"
    text = re.sub(r"([A-Za-z])-\s+([a-z])", r"\1\2", text)
    return text


def segment(text: str, window: int = WINDOW) -> list[str]:
    toks = base_normalise(text).split()
    segs = []
    for i in range(0, len(toks), window):
        chunk = toks[i:i + window]
        if len(chunk) >= MIN_TOKENS:
            segs.append(" ".join(chunk))
    return segs


# --------------------------------------------------------------------------
# Cleaning ladder (applied to a raw window)
# --------------------------------------------------------------------------
# Tokens that directly identify the newspaper, its printing place, staff,
# or running page headers.  Matching is case-insensitive on stems.
SELF_REF = re.compile(
    r"(imvo|mvo\b|zabantsundu|isigidimi|sigidimi|gidimi|samaxosa|"
    r"sama-?xosa|native|opinion|lovedale|lovedali|king|william|qonce|"
    r"jabavu|tengo|gqoba|makiwane|weir|elliot|ofisi|office|lihlaulelwa|"
    r"ipepa|pepa|volume|vol\b)", re.I)
NONWORD = re.compile(r"[^A-Za-z']+")
SENT_END = re.compile(r"[.!?:;\"]$")


def level2(seg: str) -> str:
    """Remove newspaper self-references, digits and punctuation."""
    out = []
    for t in seg.split():
        if SELF_REF.search(t):
            continue
        t = NONWORD.sub(" ", t).strip()
        if t:
            out.extend(t.split())
    return " ".join(out)


def level3(seg: str) -> str:
    """Level 2 + mask capitalised tokens that are not sentence-initial
    (proxy for proper nouns: people, places, firms, English titles)."""
    out = []
    prev_end = True
    for raw in seg.split():
        is_end = bool(SENT_END.search(raw)) and not ABBREV.match(raw)
        if SELF_REF.search(raw):
            prev_end = is_end
            continue
        words = NONWORD.sub(" ", raw).split()
        for w in words:
            out.append("NAME" if (w[0].isupper() and not prev_end) else w)
            prev_end = False
        if words or is_end:
            prev_end = is_end
    return " ".join(out)


ABBREV = re.compile(r"^(Mr|Mrs|Messrs|Rev|Dr|St|Esq|Hon|Capt|Col|Co)\.$")


def level4_tokens(seg: str) -> list[str]:
    """Level 3, lower-cased, letters only, length>=2, with the two papers'
    orthographic conventions equalised: apostrophe elision removed
    (sel'esifundile -> selesifundile) and the digraphs <bh>, <xh>, which
    Imvo uses and Isigidimi does not, collapsed to <b>, <x>.
    The shared-vocabulary restriction (token frequent in BOTH papers'
    training data) is applied later because it must be fitted on train only."""
    toks = []
    for w in level3(seg).split():
        if w == "NAME":
            continue
        w = w.lower().replace("'", "").replace("bh", "b").replace("xh", "x")
        if len(w) >= 2 and w.isalpha():
            toks.append(w)
    return toks


# --------------------------------------------------------------------------
# Build segment table
# --------------------------------------------------------------------------
@dataclass
class Corpus:
    seg: pd.DataFrame          # one row per window, all text levels
    seg_full: pd.DataFrame     # windows from the bilingual (full) text: level 0


def build_corpus(data_dir: str) -> Corpus:
    xh = load_pages(os.path.join(data_dir, "XhTxt"))
    full = load_pages(os.path.join(data_dir, "TxtFiles"))

    def to_segs(pages):
        rows = []
        for r in pages.itertuples():
            for k, s in enumerate(segment(r.text)):
                rows.append(dict(paper=r.paper, year=r.year, month=r.month,
                                 issue=r.issue, page=r.page, file=r.file,
                                 widx=k, raw=s))
        return pd.DataFrame(rows)

    seg = to_segs(xh)
    seg["L2"] = seg.raw.map(level2)
    seg["L3"] = seg.raw.map(level3)
    seg["L4"] = seg.raw.map(lambda s: " ".join(level4_tokens(s)))
    seg = seg[seg.L4.str.split().str.len() >= 30].reset_index(drop=True)
    seg["y"] = (seg.paper == "IMVO").astype(int)

    sf = to_segs(full)
    sf["y"] = (sf.paper == "IMVO").astype(int)
    return Corpus(seg=seg, seg_full=sf)


# --------------------------------------------------------------------------
# Splits: grouped by issue (no issue in two splits), stratified by paper-year
# --------------------------------------------------------------------------
def issue_split(issues: pd.DataFrame, seed: int = 13,
                frac=(0.70, 0.15, 0.15)) -> dict[str, str]:
    """issues: DataFrame with columns issue, paper, year (one row per issue).
    Returns mapping issue -> split. Stratifies by (paper, year)."""
    rng = np.random.default_rng(seed)
    out = {}
    for _, g in issues.groupby(["paper", "year"]):
        ids = g.issue.tolist()
        rng.shuffle(ids)
        n = len(ids)
        n_te = max(1, int(round(frac[2] * n)))
        n_va = max(1, int(round(frac[1] * n)))
        for i, iss in enumerate(ids):
            out[iss] = "test" if i < n_te else ("val" if i < n_te + n_va else "train")
    return out


def temporal_split(issues: pd.DataFrame, seed: int = 13,
                   test_year: int = 1888, val_frac: float = 0.15) -> dict[str, str]:
    """Train on years < test_year (with a random grouped val), test on test_year."""
    rng = np.random.default_rng(seed)
    out = {}
    for _, g in issues.groupby(["paper", "year"]):
        ids = g.issue.tolist()
        if g.year.iloc[0] == test_year:
            for iss in ids:
                out[iss] = "test"
            continue
        rng.shuffle(ids)
        n_va = max(1, int(round(val_frac * len(ids))))
        for i, iss in enumerate(ids):
            out[iss] = "val" if i < n_va else "train"
    return out


# --------------------------------------------------------------------------
# Near-duplicate filter (recurring adverts / notices across issues)
# --------------------------------------------------------------------------
def shingles(text: str, n: int = 6) -> set[int]:
    t = text.lower().split()
    return {hash(" ".join(t[i:i + n])) for i in range(max(0, len(t) - n + 1))}


def duplicate_mask(train_texts, other_texts, n: int = 6, thr: float = 0.3):
    """True for segments in other_texts whose word n-gram overlap with the
    union of training n-grams exceeds thr (likely reprinted adverts, notices)."""
    bank = set()
    for t in train_texts:
        bank |= shingles(t, n)
    mask = []
    for t in other_texts:
        s = shingles(t, n)
        mask.append(len(s) > 0 and len(s & bank) / len(s) > thr)
    return np.array(mask)
