from __future__ import annotations

import argparse
import json
from pathlib import Path
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import yaml

from run_experiment import choose_device, run_task

SUMMARY_METRICS = [
    "first_value_accuracy", "exact_accuracy", "loss", "perplexity",
    "greedy_first_value_accuracy", "sampled_first_value_accuracy",
    "sampled_symbol_in_context", "sampled_pair_consistency", "sampled_full_correct",
    "sampled_symbol_entropy", "sampled_max_symbol_share", "sampled_symbol_kl_to_uniform",
    "sampled_mean_relative_position", "trainlabel_first_value_accuracy",
    "trainlabel_pair_consistency", "trainlabel_symbol_entropy",
    "best_previous_token_score", "best_induction_score", "max_ablation_drop",
    "train_steps",
]


def study_plots(frame: pd.DataFrame, reference: dict, output_dir: Path) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    sns.set_theme(style="whitegrid", context="paper")
    reference = {**reference, "uniform_share": 1 / 8}
    panels = [
        ("first_value_accuracy", "First-value accuracy (clean test)", "Accuracy",
         [("baseline_most_frequent_context_value", "copy most frequent value")]),
        ("sampled_pair_consistency", "Extended: symbol-value consistency", "Fraction", []),
        ("best_induction_score", "Best induction attention score", "Attention mass",
         [("baseline_uniform_value_attention_induction_score", "uniform over values")]),
        ("max_ablation_drop", "Largest single-head ablation drop", "Accuracy drop", []),
        ("sampled_symbol_entropy", "Extended: generated-symbol entropy", "Bits",
         [("real_symbol_entropy", "real data")]),
        ("trainlabel_max_symbol_share", "Extended: most common symbol's share of training labels",
         "Share", [("uniform_share", "uniform (1/8)")]),
    ]
    fig, axes = plt.subplots(2, 3, figsize=(11, 6))
    for ax, (column, title, ylabel, refs) in zip(axes.flat, panels):
        subset = frame.dropna(subset=[column]) if column in frame else frame.iloc[0:0]
        if len(subset):
            sns.lineplot(data=subset, x="generation", y=column, hue="task", marker="o",
                         errorbar="sd", ax=ax, hue_order=["base", "extended"])
        for key, label in refs:
            if key in reference:
                ax.axhline(reference[key], ls="--", lw=1, color="grey")
                ax.text(0, reference[key], f" {label}", fontsize=7, va="bottom", color="grey")
        ax.set_title(title, fontsize=9)
        ax.set_ylabel(ylabel)
        ax.set_xlabel("Generation")
        if ax.legend_:
            ax.legend_.remove()
    handles, labels = axes.flat[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="upper center", ncol=2, frameon=False)
    fig.tight_layout(rect=(0, 0, 1, 0.95))
    fig.savefig(output_dir / "main_results.png", dpi=240, bbox_inches="tight")
    fig.savefig(output_dir / "main_results.pdf", bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/submission.yaml")
    args = parser.parse_args()
    with open(args.config, encoding="utf-8") as f:
        cfg = yaml.safe_load(f)
    root = Path(cfg["output_dir"])
    device = choose_device(cfg["device"])
    all_rows, reference = [], {}
    for seed in cfg["seeds"]:
        run_cfg = yaml.safe_load(yaml.safe_dump(cfg))
        run_cfg["seed"] = seed
        seed_dir = root / f"seed_{seed}"
        for task in cfg.get("tasks", ["base", "extended"]):
            rows = run_task(task, run_cfg, device, seed_dir)
            for row in rows:
                row["seed"] = seed
            all_rows.extend(rows)
            reference.update({k: v for k, v in json.loads(
                (seed_dir / task / "reference.json").read_text()).items()
                if not k.startswith("baseline") or task == "base"})
    frame = pd.DataFrame(all_rows)
    frame.to_csv(root / "all_results.csv", index=False)
    present = [m for m in SUMMARY_METRICS if m in frame]
    summary = frame.groupby(["task", "generation"])[present].agg(["mean", "std"])
    summary.columns = [f"{m}_{s}" for m, s in summary.columns]
    summary.reset_index().to_csv(root / "summary.csv", index=False)
    (root / "reference.json").write_text(json.dumps(reference, indent=2))
    study_plots(frame, reference, root)
    print(f"Study complete on {device}: {root.resolve()}")


if __name__ == "__main__":
    main()
