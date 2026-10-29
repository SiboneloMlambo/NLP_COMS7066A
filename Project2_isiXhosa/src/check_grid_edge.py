"""Grid-edge check: several models chose the largest C in their grid.
Re-score larger C values on the VALIDATION split only (seed 13) to confirm
the validation score has plateaued, i.e. the grid boundary does not limit
the reported results.

Usage: python src/check_grid_edge.py --data data --out outputs
"""
from __future__ import annotations

import argparse
import os
import sys

import pandas as pd
from sklearn.metrics import f1_score

sys.path.insert(0, os.path.dirname(__file__))
from data import build_corpus, issue_split  # noqa: E402
from models import char_lr, word_lr  # noqa: E402
from run_experiments import MAIN_START, prepare_split, shared_vocab, filt  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default="data")
    ap.add_argument("--out", default="outputs")
    a = ap.parse_args()
    C = build_corpus(a.data)
    seg = C.seg[C.seg.issue.str[5:12] >= MAIN_START].reset_index(drop=True)
    sf = C.seg_full[C.seg_full.issue.str[5:12] >= MAIN_START].reset_index(drop=True)
    smap = issue_split(seg[["issue", "paper", "year"]].drop_duplicates(), seed=13)
    seg, sf, _ = prepare_split(seg, sf, smap)
    tr = seg[seg.split == "train"]
    vocab, _ = shared_vocab([t.split() for t in tr.L4], tr.y.values)
    rows = []
    for level, col in (("L1", "raw"), ("L4", "L4")):
        X = {s: seg[(seg.split == s) & ~seg.dup] for s in ("train", "val")}
        X["train"] = seg[seg.split == "train"]
        txt = {s: (filt(d[col].tolist(), vocab) if level == "L4" else d[col].tolist())
               for s, d in X.items()}
        for name, fac, kw in (("char", char_lr, dict(ngram=(2, 4))),
                              ("word", word_lr, dict(ngram=(1, 1), min_df=2))):
            for Cc in (10, 30, 100, 300):
                m = fac(C=Cc, **kw).fit(txt["train"], X["train"].y.values)
                f = f1_score(X["val"].y.values, m.predict(txt["val"]), average="macro")
                rows.append(dict(level=level, model=name, C=Cc, val_macro_f1=round(f, 4)))
                print(rows[-1], flush=True)
    pd.DataFrame(rows).to_csv(f"{a.out}/grid_edge_check.csv", index=False)


if __name__ == "__main__":
    main()
