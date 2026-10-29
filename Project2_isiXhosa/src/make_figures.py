"""Build the figures and LaTeX table used in the extended abstract from outputs/.

Usage: python src/make_figures.py --out outputs
"""
from __future__ import annotations

import argparse
import glob
import json
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

BLUE, ORANGE, AQUA, YELLOW, MAGENTA, GREEN, VIOLET = (
    "#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4", "#008300", "#6250d6")
INK, MUTED, GRID = "#0b0b0b", "#52514e", "#e4e3df"
plt.rcParams.update({
    "font.family": "DejaVu Sans", "font.size": 7.5, "axes.titlesize": 8,
    "axes.labelsize": 7.5, "xtick.labelsize": 7, "ytick.labelsize": 7,
    "legend.fontsize": 6.5, "axes.edgecolor": MUTED, "axes.labelcolor": INK,
    "xtick.color": MUTED, "ytick.color": MUTED, "axes.spines.top": False,
    "axes.spines.right": False, "axes.grid": True, "grid.color": GRID,
    "grid.linewidth": 0.6, "axes.axisbelow": True, "lines.linewidth": 1.6,
    "savefig.dpi": 300, "savefig.bbox": "tight", "pdf.fonttype": 42})

MODEL_STYLE = {"word": (BLUE, "o", "word TF-IDF + LR"),
               "char": (ORANGE, "s", "char 2-5-gram + LR"),
               "nb": (AQUA, "^", "Complement NB"),
               "fasttext": (VIOLET, "D", "fastText (own) + LR")}
LEVEL_LABEL = {"L0": "L0\nfull text", "L1": "L1\nisiXhosa", "L2": "L2\n−self-ref.",
               "L3": "L3\n−names", "L4": "L4\nshared voc."}


def load(out):
    res = pd.read_csv(f"{out}/results.csv")
    extra = [pd.read_csv(f) for f in glob.glob(f"{out}/results_e1_*.csv")]
    if extra:
        res = pd.concat([res] + extra, ignore_index=True)
    return res


def fig_ladder(res, matched, out):
    ladder = res[res.tag.str.startswith("random_s")]
    agg = ladder.groupby(["level", "model"]).bal_acc.agg(["mean", "std", "count"]).reset_index()
    fam = res[res.tag.str.startswith("family")]
    fig, (ax, bx) = plt.subplots(1, 2, figsize=(7.0, 2.25),
                                 gridspec_kw=dict(width_ratios=[1.3, 1], wspace=0.62))
    xs = np.arange(5)
    for m, (c, mk, lab) in MODEL_STYLE.items():
        d = agg[agg.model == m].set_index("level").reindex(list(LEVEL_LABEL))
        if d["mean"].isna().all():
            continue
        off = {"word": -0.09, "char": -0.03, "nb": 0.03, "fasttext": 0.09}[m]
        ax.errorbar(xs + off, d["mean"], yerr=d["std"].fillna(0), color=c, marker=mk,
                    ms=4.5, capsize=2, lw=1.4, label=lab, mec="white", mew=0.6)
    ax.set_xticks(xs, [LEVEL_LABEL[k] for k in LEVEL_LABEL], fontsize=6.3)
    ax.set_ylabel("balanced accuracy (test)")
    ax.set_ylim(0.6, 1.0)
    ax.set_title("(a) Cleaning ladder: shortcuts removed step by step", loc="left")
    ax.legend(loc="lower left", frameon=False, ncol=2, handlelength=1.6)

    # (b) feature families, full test set vs non-content-matched subset
    rows = [("non-content\n(noise+orthog.)", "noncontent_gb"),
            ("orthography\nonly", "ortho_gb"), ("transcription\nnoise only", "surface_gb"),
            ("style: top-100\nwords (L4)", "L4-word[style]"),
            ("content: L4 minus\ntop-100", "L4-word[content]")]
    mt = matched.set_index("model")
    y = np.arange(len(rows))[::-1]
    for yi, (lab, key) in zip(y, rows):
        full, mat = mt.loc[key, "full_bal_acc"], mt.loc[key, "bal_acc"]
        bx.plot([full, mat], [yi, yi], color=GRID, lw=2.2, zorder=1)
        bx.scatter(full, yi, s=26, color=BLUE, zorder=3, edgecolor="white", lw=0.6)
        bx.scatter(mat, yi, s=26, color=ORANGE, marker="s", zorder=3, edgecolor="white", lw=0.6)
    bx.axvline(0.5, color=MUTED, lw=0.8, ls="--")
    bx.set_yticks(y, [r[0] for r in rows])
    bx.set_xlim(0.4, 1.0)
    bx.set_xlabel("balanced accuracy")
    bx.scatter([], [], color=BLUE, s=20, label="all test segments")
    bx.scatter([], [], color=ORANGE, marker="s", s=20, label="matched on non-content features")
    bx.set_ylim(-1.5, len(rows) - 0.5)
    bx.legend(loc="lower right", frameon=False, ncol=1, bbox_to_anchor=(1.02, -0.03))
    bx.set_title("(b) What carries the signal?", loc="left")
    bx.grid(axis="y", visible=False)
    fig.savefig(f"{out}/figures/fig1_ladder.pdf")
    fig.savefig(f"{out}/figures/fig1_ladder.png")
    plt.close(fig)


def fig_ladder_col(res, out):
    """Column-width version of panel (a) for the main text."""
    ladder = res[res.tag.str.startswith("random_s")]
    agg = ladder.groupby(["level", "model"]).bal_acc.agg(["mean", "std"]).reset_index()
    fig, ax = plt.subplots(figsize=(3.35, 1.75))
    xs = np.arange(5)
    for m, (c, mk, lab) in MODEL_STYLE.items():
        d = agg[agg.model == m].set_index("level").reindex(list(LEVEL_LABEL))
        off = {"word": -0.09, "char": -0.03, "nb": 0.03, "fasttext": 0.09}[m]
        ax.errorbar(xs + off, d["mean"], yerr=d["std"].fillna(0), color=c, marker=mk,
                    ms=4, capsize=2, lw=1.4, label=lab, mec="white", mew=0.6)
    ax.set_xticks(xs, [LEVEL_LABEL[k] for k in LEVEL_LABEL], fontsize=6.3)
    ax.set_ylabel("balanced accuracy (test)")
    ax.set_ylim(0.6, 1.0)
    ax.legend(loc="lower left", frameon=False, ncol=2, handlelength=1.5, columnspacing=0.8)
    fig.savefig(f"{out}/figures/fig1_ladder_col.pdf")
    fig.savefig(f"{out}/figures/fig1_ladder_col.png")
    plt.close(fig)


def fig_eras(out):
    lx = pd.read_csv(f"{out}/lexicon_by_issue.csv")
    pb = pd.read_csv(f"{out}/probe_by_issue.csv")
    fig, (ax, bx) = plt.subplots(1, 2, figsize=(7.0, 1.75), sharex=True,
                                 gridspec_kw=dict(wspace=0.22))
    for a in (ax, bx):
        a.axvspan(1881, 1884 + 10 / 12, color="#f3efe4", zorder=0)
        a.axvline(1884 + 10 / 12, color=MUTED, lw=0.8, ls=":")
    ax.text(1882.9, 0.23, "Jabavu edits\nIsigidimi", ha="center", va="bottom",
            fontsize=6.5, color=MUTED)
    bx.text(1884 + 9 / 12, 0.97, "Imvo founded", ha="right", va="top",
            fontsize=6.5, color=MUTED)

    def yearly(df, col):
        df = df.assign(month=df.issue.str[10:12].astype(int),
                       post=df.issue.str[5:12] >= "1884-11")
        df["x"] = df.year + (df.month - 0.5) / 12
        g = df.groupby(["paper", "year", "post"])
        return pd.DataFrame(dict(year=g.x.mean(), v=g[col].mean(), sd=g[col].sem(),
                                 n=g.size())).reset_index(drop=True).sort_values("year")

    for paper, c, mk, lab in (("ISIG", ORANGE, "s", "Isigidimi"), ("IMVO", BLUE, "o", "Imvo")):
        d = yearly(lx[lx.paper == paper], "pol_share")
        ax.errorbar(d.year, d.v, yerr=1.96 * d.sd.fillna(0), color=c, marker=mk,
                    ms=4, capsize=1.5, label=lab, mec="white", mew=0.5)
        e = yearly(pb[pb.paper == paper], "frac_imvo")
        bx.errorbar(e.year, e.v, yerr=1.96 * e.sd.fillna(0), color=c, marker=mk,
                    ms=4, capsize=1.5, label=lab, mec="white", mew=0.5)
    ax.set_ylabel("political / (political+religious)")
    ax.set_ylim(0.2, 1.0)
    ax.set_title("(a) Lexicon: political share of thematic terms", loc="left")
    ax.legend(loc="lower right", frameon=False)
    bx.axhline(0.5, color=MUTED, lw=0.8, ls="--")
    bx.set_ylim(0, 1)
    bx.set_ylabel("segments classified as Imvo")
    bx.set_title("(b) Content classifier applied to unseen years", loc="left")
    for a in (ax, bx):
        a.set_xticks(range(1876, 1890, 2))
        a.set_xlim(1875.8, 1889.2)
    fig.savefig(f"{out}/figures/fig2_eras.pdf")
    fig.savefig(f"{out}/figures/fig2_eras.png")
    plt.close(fig)


def fig_noise(out):
    nz = pd.read_csv(f"{out}/noise_injection.csv")
    fig, ax = plt.subplots(figsize=(3.3, 2.0))
    for key, c, mk in (("L1-word", BLUE, "o"), ("L1-char", ORANGE, "s"),
                       ("L4-word", AQUA, "^"), ("L4-char", VIOLET, "D")):
        d = nz[nz.model == key]
        ax.plot(100 * d.rate, d.frac_pred_noisy_paper, color=c, marker=mk, ms=4, label=key)
    ax.set_xlabel("tokens corrupted in Isigidimi test segments (%)")
    ax.set_ylabel("share classified as Imvo")
    ax.set_ylim(0, 1)
    ax.legend(frameon=False, ncol=2)
    fig.savefig(f"{out}/figures/figS1_noise.pdf")
    fig.savefig(f"{out}/figures/figS1_noise.png")
    plt.close(fig)


def fig_surface(out):
    sf = pd.read_csv(f"{out}/surface_features.csv")
    sf = sf[sf.split == "train"]
    cols = ["frac_vowelless_tok", "frac_mixed_alnum_tok", "frac_upper_chars",
            "frac_english_func", "frac_bh_tok", "frac_elision_tok"]
    fig, axs = plt.subplots(1, len(cols), figsize=(7.0, 1.6))
    for a, c in zip(axs, cols):
        data = [sf[sf.paper == p][c].values for p in ("IMVO", "ISIG")]
        bp = a.boxplot(data, widths=0.55, showfliers=False, patch_artist=True,
                       medianprops=dict(color=INK, lw=1))
        for patch, col in zip(bp["boxes"], (BLUE, ORANGE)):
            patch.set_facecolor(col); patch.set_alpha(0.75); patch.set_edgecolor("white")
        a.set_xticks([1, 2], ["Imvo", "Isig."])
        a.set_title(c.replace("frac_", "").replace("_", " "), fontsize=6.5)
        a.grid(axis="x", visible=False)
    fig.tight_layout()
    fig.savefig(f"{out}/figures/figS2_surface.pdf")
    fig.savefig(f"{out}/figures/figS2_surface.png")
    plt.close(fig)


def table_main(res, matched, out):
    """Compact results table (seed-13 split, cluster-bootstrap CIs)."""
    s13 = res[res.tag.isin(["random_s13", "family_s13"])]
    seeds = res[res.tag.str.startswith("random_s")].groupby(["level", "model"]).macro_f1.std()
    mt = matched.set_index("model")
    rows = [("L0", "char", "Full page text, char n-grams", "L0-char"),
            ("L1", "char", "isiXhosa only, char n-grams", "L1-char"),
            ("L1", "word", "isiXhosa only, words", "L1-word"),
            ("L4", "word", "Shared vocab., words (uni+bi)", "L4-word"),
            ("L4", "fasttext", "Shared vocab., fastText", "L4-fasttext"),
            ("L4", "word[content_noTop100]", "Content words only", "L4-word[content]"),
            ("L4", "word[style_top100]", "Top-100 (function) words only", "L4-word[style]"),
            ("raw", "surface_gb", "Transcription features (GB)", "surface_gb"),
            ("raw", "ortho_gb", "Orthography features (GB)", "ortho_gb"),
            ("raw", "noncontent_gb", "All non-content features (GB)", "noncontent_gb")]
    lines = []
    for lv, m, lab, mkey in rows:
        r = s13[(s13.level == lv) & (s13.model == m)]
        if r.empty:
            continue
        r = r.iloc[0]
        lo, hi = eval(r.macro_f1_ci) if isinstance(r.macro_f1_ci, str) else r.macro_f1_ci
        sd = seeds.get((lv, m), np.nan)
        sd_s = f"{sd:.3f}" if np.isfinite(sd) else "--"
        mb = mt.loc[mkey, "bal_acc"] if mkey in mt.index else np.nan
        mb_s = f"{mb:.2f}" if np.isfinite(mb) else "--"
        lines.append(f"{lab} & {r.macro_f1:.3f} [{lo:.2f}, {hi:.2f}] & {sd_s} & "
                     f"{r.bal_acc:.2f} & {mb_s} \\\\")
    with open(f"{out}/table_main.tex", "w") as fh:
        fh.write("\n".join(lines) + "\n")
    print("\n".join(lines))


def supp_tables(res, out):
    """LaTeX tables for the supplementary material."""
    L = []
    # S1: ladder, mean +- sd over splits
    lad = res[res.tag.str.startswith("random_s")]
    g = lad.groupby(["level", "model"])[["macro_f1", "bal_acc"]].agg(["mean", "std"])
    names = {"word": "word", "char": "char", "nb": "CNB", "fasttext": "fastText"}
    L.append(r"\begin{table}[h]\centering\small\caption{Cleaning ladder, test macro-F1 / balanced accuracy, mean $\pm$ s.d.\ over three issue-grouped splits (seeds 13, 21, 42).}")
    L.append(r"\begin{tabular}{@{}llcc@{}}\toprule Level & Model & Macro-F1 & Bal.\ acc. \\ \midrule")
    for (lv, m), r in g.iterrows():
        L.append(f"{lv} & {names[m]} & {r[('macro_f1','mean')]:.3f} $\\pm$ {r[('macro_f1','std')]:.3f} & "
                 f"{r[('bal_acc','mean')]:.3f} $\\pm$ {r[('bal_acc','std')]:.3f} \\\\")
    L.append(r"\bottomrule\end{tabular}\label{tab:s1}\end{table}")
    # S2: robustness runs + grid-edge check
    rb = res[res.tag.isin(["nodedup_s13", "temporal_1888"])]
    base = res[res.tag == "random_s13"].set_index(["level", "model"])
    L.append(r"\begin{table}[h]\centering\small\caption{Robustness (seed 13). \emph{No dedup}: near-duplicate filter switched off. \emph{Temporal}: train/validate on Nov 1884--1887, test on 1888. Reference: the main random split.}")
    L.append(r"\begin{tabular}{@{}lllcc@{}}\toprule Run & Level & Model & Macro-F1 & Ref.\ macro-F1 \\ \midrule")
    for r in rb.itertuples():
        tag = "No dedup" if r.tag.startswith("nodedup") else "Temporal"
        L.append(f"{tag} & {r.level} & {r.model} & {r.macro_f1:.3f} & {base.loc[(r.level, r.model), 'macro_f1']:.3f} \\\\")
    L.append(r"\bottomrule\end{tabular}\label{tab:s2}\end{table}")
    ge = f"{out}/grid_edge_check.csv"
    if os.path.exists(ge):
        d = pd.read_csv(ge)
        L.append(r"\begin{table}[h]\centering\small\caption{Grid-edge check (validation macro-F1, seed 13). Several selected $C$ values sat at the top of the grid; larger values do not improve validation scores.}")
        L.append(r"\begin{tabular}{@{}llcccc@{}}\toprule Level & Model & $C$=10 & 30 & 100 & 300 \\ \midrule")
        for (lv, m), x in d.groupby(["level", "model"]):
            vals = " & ".join(f"{v:.3f}" for v in x.sort_values("C").val_macro_f1)
            L.append(f"{lv} & {m} & {vals} \\\\")
        L.append(r"\bottomrule\end{tabular}\label{tab:s3}\end{table}")
    # S4: era summary
    st = json.load(open(f"{out}/stats.json"))
    order = ["ISIG 1876-80 (pre-Jabavu)", "ISIG 1881-Oct 1884 (Jabavu)",
             "ISIG Nov 1884-88 (post-Jabavu)", "IMVO Nov 1884-88"]
    L.append(r"\begin{table*}[h]\centering\small\caption{Editorial eras. Lexicon rates per 1{,}000 tokens (issue-level means, bootstrap 95\% CI over issues; all issues). Probe: mean P(Imvo) from the L4 content model and share of segments classified Imvo (post-1884 rows: held-out test issues only).}")
    L.append(r"\begin{tabular}{@{}lcccccc@{}}\toprule Group & Issues (lex.) & Political/1k & Religious/1k & Political share & Mean P(Imvo) & \% segs.\ Imvo \\ \midrule")
    for k in order:
        lx, pr = st["lexicon"][k], st["probe"][k]
        f = lambda v: f"{v[0]:.2f} [{v[1]:.2f}, {v[2]:.2f}]"
        L.append(f"{k.replace('ISIG', 'Isigidimi').replace('IMVO', 'Imvo')} & {lx['n_issues']} & {f(lx['political_per1k'])} & "
                 f"{f(lx['religious_per1k'])} & {f(lx['pol_share'])} & {pr['mean_p_imvo']:.2f} & "
                 f"{100 * pr['frac_segments_imvo']:.0f} \\\\")
    L.append(r"\bottomrule\end{tabular}\label{tab:s4}\end{table*}")
    # S5: top content features
    tf = pd.read_csv(f"{out}/top_content_features.csv")
    a = tf[tf.favours == "IMVO"].head(20).token.tolist()
    b = tf[tf.favours == "ISIG"].head(20).token.tolist()
    L.append(r"\begin{table}[h]\centering\small\caption{Twenty largest-weight features of the L4 content model, in order (seed 13). Glosses for selected items are given in the Results.}")
    L.append(r"\begin{tabular}{@{}p{0.45\linewidth}p{0.45\linewidth}@{}}\toprule Favour Imvo & Favour Isigidimi \\ \midrule")
    L.append("\\emph{" + ", ".join(a) + "} & \\emph{" + ", ".join(b) + "} \\\\")
    L.append(r"\bottomrule\end{tabular}\label{tab:s5}\end{table}")
    with open(f"{out}/supp_tables.tex", "w") as fh:
        fh.write("\n".join(L) + "\n")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="outputs")
    a = ap.parse_args()
    os.makedirs(f"{a.out}/figures", exist_ok=True)
    res = load(a.out)
    matched = pd.read_csv(f"{a.out}/matched_eval.csv")
    fig_ladder(res, matched, a.out)
    fig_ladder_col(res, a.out)
    fig_eras(a.out)
    fig_noise(a.out)
    fig_surface(a.out)
    table_main(res, matched, a.out)
    supp_tables(res, a.out)
    summ = res.groupby(["tag", "level", "model"])[["macro_f1", "bal_acc", "auroc"]].mean()
    summ.round(4).to_csv(f"{a.out}/results_summary.csv")
    seedagg = res[res.tag.str.startswith("random_s")].groupby(["level", "model"])[
        ["macro_f1", "bal_acc"]].agg(["mean", "std", "count"]).round(4)
    seedagg.to_csv(f"{a.out}/results_seed_mean_sd.csv")
    print(seedagg)


if __name__ == "__main__":
    main()
