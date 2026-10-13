"""Figure for the extended abstract, built only from saved results.

    python analyse_leakage.py --config configs/submission.yaml   (once)
    python make_figures.py
"""
from __future__ import annotations
import json
from pathlib import Path
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd

ROOT = Path("outputs/submission")
OUT = Path("report/figures")
OUT.mkdir(parents=True, exist_ok=True)
BASE, EXT, GREY = "#2f6db3", "#d1662b", "#7a7a7a"

plt.rcParams.update({"font.family": "serif", "font.size": 7, "axes.titlesize": 7.5,
                     "axes.labelsize": 7, "legend.fontsize": 6, "xtick.labelsize": 6.5,
                     "ytick.labelsize": 6.5, "axes.spines.top": False,
                     "axes.spines.right": False, "lines.linewidth": 1.2, "lines.markersize": 3})

res = pd.read_csv(ROOT / "all_results.csv")
leak = pd.read_csv(ROOT / "leakage_analysis.csv")
ref = json.loads((ROOT / "reference.json").read_text())
pilot = pd.read_csv("outputs/pilot_short_budget/all_results.csv")


def band(ax, df, col, color, label, scale=1.0, marker="o"):
    g = df.groupby("generation")[col].agg(["mean", "std"]) * scale
    ax.plot(g.index, g["mean"], marker=marker, color=color, label=label)
    ax.fill_between(g.index, g["mean"] - g["std"], g["mean"] + g["std"], color=color, alpha=0.2, lw=0)


fig, axes = plt.subplots(1, 4, figsize=(7.0, 1.68))

# (a) Training curves: the induction transition needs a larger budget.
ax = axes[0]
for n, color, label in [(50000, BASE, "50k examples"), (3000, GREY, "3k (pilot)")]:
    h = pd.read_csv(ROOT / f"tuning/history_base_learning_rate=0.001_train_samples={n}.csv")
    ax.plot(h["step"], h["val_first_value_accuracy"], marker="o", color=color, label=label)
ax.axhline(ref["baseline_most_frequent_context_value"], ls="--", lw=0.8, color=GREY)
ax.text(5400, ref["baseline_most_frequent_context_value"] - 0.075, "heuristic baseline",
        fontsize=5.5, color=GREY, ha="right")
ax.set(xlabel="Training step", ylabel="Val. first-value acc.", ylim=(0, 1.05),
       title="(a) Base, gen. 0 training")
ax.legend(frameon=False, loc="center right", bbox_to_anchor=(1.0, 0.68))

# (b) Extended task: errors are out-of-context symbols; leaked mass tracks them.
ax = axes[1]
ext = res[res.task == "extended"].copy()
ext["error"] = 1 - ext["sampled_pair_consistency"]
band(ax, ext, "error", EXT, "inconsistent", 100)
band(ax, leak, "leaked_mass", "#8a3ea8", "unseen-symbol mass", 100, marker="s")
ax.set(xlabel="Generation", ylabel="% of test examples / prob.",
       title="(b) Ext.: errors vs leaked mass")
ax.legend(frameon=False, loc="upper left")

# (c) Training-label symbol distribution skews towards particular symbols.
ax = axes[2]
band(ax, ext, "trainlabel_max_symbol_share", EXT, "extended", 100)
ax.axhline(12.5, ls="--", lw=0.8, color=GREY)
ax.text(5, 12.9, "uniform (1/8)", fontsize=5.5, color=GREY, ha="right")
ax.set(xlabel="Generation", ylabel="Top symbol share (%)",
       title="(c) Ext.: label skew")

# (d) Induction attention stays intact in both tasks.
ax = axes[3]
band(ax, res[res.task == "base"], "best_induction_score", BASE, "base")
band(ax, ext, "best_induction_score", EXT, "extended", marker="s")
pil = pilot.groupby("generation")["best_induction_score"].mean()
ax.plot(pil.index, pil.values, color=GREY, ls=":", marker="x", label="pilot (no transition)")
ax.axhline(ref["baseline_uniform_value_attention_induction_score"], ls="--", lw=0.8, color=GREY)
ax.set(xlabel="Generation", ylabel="Best induction score", ylim=(0, 1.05),
       title="(d) Induction attention")
ax.legend(frameon=False, loc="center right", bbox_to_anchor=(1.0, 0.62))

fig.tight_layout(pad=0.3, w_pad=0.8)
fig.savefig(OUT / "main_figure.pdf")
fig.savefig(OUT / "main_figure.png", dpi=220)
print("saved", OUT / "main_figure.pdf")
