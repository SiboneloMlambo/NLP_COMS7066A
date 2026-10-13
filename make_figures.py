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
band(ax, leak, "leaked_mass", "#8a3ea8", "unseen mass", 100, marker="s")
MIX = Path("outputs/mix50/all_results.csv")
if MIX.exists():
    mix = pd.read_csv(MIX)
    mix = mix[mix.task == "extended"].copy()
    mix["error"] = 1 - mix["sampled_pair_consistency"]
    band(ax, mix, "error", "#3a9a5b", "incons., 50% real", 100, marker="D")
ax.set(xlabel="Generation", ylabel="% of test examples / prob.", ylim=(0, 7.2),
       title="(b) Ext.: errors vs leaked mass")
ax.legend(frameon=False, loc="upper left")

# (c) Direct logit attribution: the shown-vs-unseen contrast shrinks, mostly in layer 2.
ax = axes[2]
dla = pd.read_csv(ROOT / "mechanism_dla.csv")
dla["group"] = dla.component.map(lambda c: "L1 heads" if c.startswith("L1H") else
                                 "L2 heads" if c.startswith("L2H") else "other")
agg = dla.groupby(["seed", "generation", "group"]).unseen_pref.sum().reset_index()
total = agg.groupby(["seed", "generation"]).unseen_pref.sum().reset_index()
for frame, color, label, marker in [
        (total, "black", "total", "o"),
        (agg[agg.group == "L2 heads"], EXT, "layer-2 heads", "s"),
        (agg[agg.group == "L1 heads"], BASE, "layer-1 heads", "^"),
        (agg[agg.group == "other"], GREY, "MLPs + emb.", "x")]:
    band(ax, frame, "unseen_pref", color, label, -1, marker=marker)
ax.set(xlabel="Generation", ylabel="Shown$-$unseen logit gap", ylim=(-0.2, 8.8),
       title="(c) Ext.: logit attribution")
ax.set_yticks([0, 2, 4, 6])
ax.legend(frameon=False, loc="upper center", ncol=2, columnspacing=0.6, handlelength=1.5,
          borderaxespad=0.1)

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
