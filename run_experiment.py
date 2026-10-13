from __future__ import annotations

import argparse
import json
from pathlib import Path
import numpy as np
import pandas as pd
import torch
import yaml

from src.analysis import (circuit_scores, generation_metrics, head_ablation, heuristic_baselines,
                          save_attention_plot, save_metric_plots)
from src.data import Vocabulary, SequenceDataset, generate_real_examples
from src.model import TinyTransformer
from src.train import evaluate, fit, generate_targets, set_seed


def choose_device(name: str) -> torch.device:
    if name == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return torch.device(name)


def build_model(cfg: dict, vocab_size: int, max_length: int) -> TinyTransformer:
    return TinyTransformer(vocab_size=vocab_size, max_length=max_length, **cfg)


def make_splits(task: str, cfg: dict, seed: int):
    dc = cfg["data"]
    vocab = Vocabulary(dc["num_symbols"])
    src = dc.get("next_symbol_source", "context")
    gen = lambda n, s: generate_real_examples(n, vocab, dc["sequence_pairs"], task, s, src)
    return vocab, gen(dc["train_samples"], seed), gen(dc["val_samples"], seed + 1), \
        gen(dc["test_samples"], seed + 2)


def run_task(task: str, cfg: dict, device: torch.device, output_dir: Path) -> list[dict]:
    seed = cfg["seed"] + (0 if task == "base" else 10000)
    set_seed(seed)
    tc, cc = cfg["training"], cfg["collapse"]
    vocab, (train_x, real_train_y), (val_x, val_y), (test_x, test_y) = make_splits(task, cfg, seed)
    target_length = 1 if task == "base" else 3
    val_data, test_data = SequenceDataset(val_x, val_y), SequenceDataset(test_x, test_y)
    task_dir = output_dir / task
    task_dir.mkdir(parents=True, exist_ok=True)

    reference = heuristic_baselines(test_x, test_y)
    reference.update(generation_metrics(test_y, test_x, test_y, vocab.num_symbols, task, "real"))
    (task_dir / "reference.json").write_text(json.dumps(reference, indent=2))

    current_y = real_train_y.copy()
    rows = []
    for generation in range(cc["generations"] + 1):
        set_seed(seed + generation)
        model = build_model(cfg["model"], vocab.vocab_size, train_x.shape[1] + target_length)
        # Distribution of the labels this generation is trained on (real at gen 0).
        label_stats = generation_metrics(current_y, train_x, real_train_y, vocab.num_symbols,
                                         task, "trainlabel")
        history = fit(model, SequenceDataset(train_x, current_y), val_data, tc, device)
        pd.DataFrame(history).to_csv(task_dir / f"history_generation_{generation}.csv", index=False)

        result = evaluate(model, test_data, tc["batch_size"], device)
        row = {k: v for k, v in result.items() if k not in {"predictions", "targets"}}
        row["epochs_trained"] = len(history)
        row["train_steps"] = history[-1]["step"]
        row.update(label_stats)
        greedy = generate_targets(model, test_x, target_length, tc["batch_size"], device, 0.0)
        sampled = generate_targets(model, test_x, target_length, tc["batch_size"], device,
                                   cc["temperature"])
        row.update(generation_metrics(greedy, test_x, test_y, vocab.num_symbols, task, "greedy"))
        row.update(generation_metrics(sampled, test_x, test_y, vocab.num_symbols, task, "sampled"))

        head_scores = circuit_scores(model, test_x, device)
        row["best_previous_token_score"] = head_scores.pop("best_previous_token_score")
        row["best_induction_score"] = head_scores.pop("best_induction_score")
        ablation = head_ablation(model, test_x, test_y, device)
        row["max_ablation_drop"] = ablation["max_ablation_drop"]
        row["max_ablation_head"] = ablation["max_ablation_head"]
        per_head = {**head_scores, **{k: v for k, v in ablation.items()
                                      if k.endswith("_ablation_drop") and not k.startswith("max")}}
        pd.DataFrame([per_head]).to_csv(task_dir / f"circuit_scores_generation_{generation}.csv",
                                        index=False)

        row.update({"task": task, "generation": generation})
        rows.append(row)
        torch.save(model.state_dict(), task_dir / f"model_generation_{generation}.pt")
        save_attention_plot(model, test_x[0], task_dir / f"attention_generation_{generation}.png",
                            device)
        print(json.dumps({k: (round(v, 4) if isinstance(v, float) else v) for k, v in row.items()}),
              flush=True)

        if generation < cc["generations"]:
            synthetic = generate_targets(model, train_x, target_length, tc["batch_size"], device,
                                         cc["temperature"])
            fraction = cc["synthetic_fraction"]
            use_synthetic = np.random.default_rng(seed + generation).random(len(train_x)) < fraction
            current_y = real_train_y.copy()
            current_y[use_synthetic] = synthetic[use_synthetic]
    return rows


def main():
    parser = argparse.ArgumentParser(description="Study induction heads under recursive model collapse")
    parser.add_argument("--config", default="configs/quick.yaml")
    parser.add_argument("--task", choices=["base", "extended", "both"], default="both")
    args = parser.parse_args()
    with open(args.config, encoding="utf-8") as f:
        cfg = yaml.safe_load(f)
    output_dir = Path(cfg["output_dir"])
    output_dir.mkdir(parents=True, exist_ok=True)
    device = choose_device(cfg["device"])
    tasks = ["base", "extended"] if args.task == "both" else [args.task]
    rows = []
    for task in tasks:
        rows.extend(run_task(task, cfg, device, output_dir))
    metrics = pd.DataFrame(rows)
    metrics.to_csv(output_dir / "generation_metrics.csv", index=False)
    save_metric_plots(metrics, output_dir / "plots")
    print(f"Finished on {device}. Results: {output_dir.resolve()}")


if __name__ == "__main__":
    main()
