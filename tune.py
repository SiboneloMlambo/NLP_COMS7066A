"""Hyper-parameter selection for the generation-0 model.

Trains a generation-0 model on real data for every combination in the
`tuning` grid of the config and selects by validation first-value accuracy
(ties broken by validation loss). The test set is never used here.

    python tune.py --config configs/submission.yaml
"""
from __future__ import annotations

import argparse
import itertools
from pathlib import Path
import pandas as pd
import yaml

from run_experiment import build_model, choose_device, make_splits
from src.data import SequenceDataset
from src.train import evaluate, fit, set_seed


def steps_to_threshold(history: list[dict], threshold: float = 0.99) -> float:
    """First optimisation step at which validation first-value accuracy reached threshold."""
    for row in history:
        if row["val_first_value_accuracy"] >= threshold:
            return row["step"]
    return float("inf")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/submission.yaml")
    parser.add_argument("--task", default="base", choices=["base", "extended"])
    args = parser.parse_args()
    with open(args.config, encoding="utf-8") as f:
        cfg = yaml.safe_load(f)
    device = choose_device(cfg["device"])
    grid = cfg["tuning"]
    keys = list(grid)
    rows = []
    out = Path(cfg["output_dir"]) / "tuning"
    out.mkdir(parents=True, exist_ok=True)
    for values in itertools.product(*(grid[k] for k in keys)):
        run = yaml.safe_load(yaml.safe_dump(cfg))
        setting = dict(zip(keys, values))
        for key, value in setting.items():
            section, name = key.split(".")
            run[section][name] = value
        seed = run["seed"] + (0 if args.task == "base" else 10000)
        set_seed(seed)
        vocab, (tx, ty), (vx, vy), _ = make_splits(args.task, run, seed)
        target_length = 1 if args.task == "base" else 3
        model = build_model(run["model"], vocab.vocab_size, tx.shape[1] + target_length)
        history = fit(model, SequenceDataset(tx, ty), SequenceDataset(vx, vy), run["training"], device)
        val = evaluate(model, SequenceDataset(vx, vy), run["training"]["batch_size"], device)
        name = "_".join(f"{k.split('.')[1]}={v}" for k, v in setting.items())
        pd.DataFrame(history).to_csv(out / f"history_{args.task}_{name}.csv", index=False)
        row = {**setting, "val_first_value_accuracy": round(val["first_value_accuracy"], 3),
               "val_loss": val["loss"], "epochs": len(history), "steps": history[-1]["step"],
               "steps_to_99pct_val": steps_to_threshold(history)}
        print(row, flush=True)
        rows.append(row)
    # Select by validation accuracy; break ties by how quickly the induction
    # transition happens (fewer steps = cheaper per generation).
    frame = pd.DataFrame(rows).sort_values(["val_first_value_accuracy", "steps_to_99pct_val"],
                                           ascending=[False, True])
    frame.to_csv(out / f"tuning_{args.task}.csv", index=False)
    print("\nSelected (by validation):", frame.iloc[0].to_dict())


if __name__ == "__main__":
    main()
