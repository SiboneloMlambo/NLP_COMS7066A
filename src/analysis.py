from __future__ import annotations

from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import torch


def _entropy_bits(counts: np.ndarray) -> float:
    p = counts[counts > 0] / counts.sum()
    return float(-(p * np.log2(p)).sum())


def generation_metrics(samples: np.ndarray, contexts: np.ndarray, targets: np.ndarray,
                       num_symbols: int, task: str, prefix: str) -> dict:
    """Metrics on free-running generations (sampled or greedy), not teacher forcing.

    For the base task: first-value accuracy of the generated value.
    For the extended task, additionally:
      * symbol_valid: generated 2nd token is a symbol token
      * symbol_in_context: generated symbol appears in the context
      * pair_consistency: generated 3rd token is the value the context maps the
        generated symbol to (only defined when the symbol is in context;
        reported over all examples, so invalid/out-of-context symbols count as
        failures)
      * symbol_entropy (bits), max_symbol_share, symbol_kl_to_uniform: the
        marginal distribution of generated symbol identities. Real data is
        uniform over symbols (entropy log2(num_symbols)), so falling entropy
        means the model is favouring particular symbols.
      * mean_relative_position: where in the context (0 = first pair,
        1 = last pair) the generated symbol first appears. Compare with the
        same statistic on the real targets (prefix "real"); a drift towards 0
        or 1 indicates a positional (primacy/recency) bias being amplified.
    """
    out = {f"{prefix}_first_value_accuracy": float((samples[:, 0] == targets[:, 0]).mean())}
    if task != "extended":
        return out
    from .data import Vocabulary, context_mapping
    vocab = Vocabulary(num_symbols)
    sym = samples[:, 1]
    valid = (sym >= 0) & (sym < num_symbols)
    in_ctx = np.zeros(len(sym), bool)
    consistent = np.zeros(len(sym), bool)
    position = []
    for i, ctx in enumerate(contexts):
        mapping = context_mapping(ctx, vocab)
        s = int(sym[i])
        if valid[i] and s in mapping:
            in_ctx[i] = True
            consistent[i] = int(samples[i, 2]) == mapping[s]
            shown = list(ctx[1:-1:2])
            position.append(shown.index(s) / max(len(shown) - 1, 1))
    counts = np.bincount(sym[valid], minlength=num_symbols).astype(float) if valid.any() \
        else np.zeros(num_symbols)
    if counts.sum():
        p = counts / counts.sum()
        nz = p > 0
        kl = float((p[nz] * np.log2(p[nz] * num_symbols)).sum())
        ent, share = _entropy_bits(counts), float(p.max())
    else:
        kl, ent, share = float("nan"), 0.0, 1.0
    out.update({
        f"{prefix}_symbol_valid": float(valid.mean()),
        f"{prefix}_symbol_in_context": float(in_ctx.mean()),
        f"{prefix}_pair_consistency": float(consistent.mean()),
        f"{prefix}_full_correct": float(((samples[:, 0] == targets[:, 0]) & consistent).mean()),
        f"{prefix}_symbol_entropy": ent,
        f"{prefix}_max_symbol_share": share,
        f"{prefix}_symbol_kl_to_uniform": kl,
        f"{prefix}_mean_relative_position": float(np.mean(position)) if position else float("nan"),
    })
    return out


def heuristic_baselines(contexts: np.ndarray, targets: np.ndarray, seed: int = 0) -> dict:
    """Accuracy of simple non-induction strategies on the first value, and the
    induction score a head would get by attending uniformly to value tokens."""
    from collections import Counter
    rng = np.random.default_rng(seed)
    vals, syms, q, t = contexts[:, 2:-1:2], contexts[:, 1:-1:2], contexts[:, -1], targets[:, 0]
    return {
        "baseline_random_context_value": float(np.mean([rng.choice(v) == y for v, y in zip(vals, t)])),
        "baseline_most_frequent_context_value": float(np.mean(
            [Counter(v).most_common(1)[0][0] == y for v, y in zip(vals, t)])),
        "baseline_uniform_value_attention_induction_score": float(np.mean(
            [np.sum(s == qq) / len(s) for s, qq in zip(syms, q)])),
    }


@torch.no_grad()
def head_ablation(model, contexts: np.ndarray, targets: np.ndarray, device: torch.device,
                  max_examples: int = 1000) -> dict:
    """Zero each attention head's output in turn and measure the drop in
    teacher-forced first-value accuracy. A large drop is causal evidence that
    the head is needed for in-context retrieval."""
    model.eval()
    x = torch.as_tensor(contexts[:max_examples], device=device)
    y = torch.as_tensor(targets[:max_examples, 0], device=device)

    def accuracy() -> float:
        return float((model(x)[:, -1, :].argmax(-1) == y).float().mean())

    clean = accuracy()
    result = {"clean_first_value_accuracy": clean}
    for layer, block in enumerate(model.blocks):
        attn = block.attn
        for head in range(attn.n_heads):
            attn.head_mask[head] = 0.0
            result[f"L{layer + 1}H{head + 1}_ablation_drop"] = clean - accuracy()
            attn.head_mask[head] = 1.0
    drops = {k: v for k, v in result.items() if k.endswith("_ablation_drop")}
    result["max_ablation_drop"] = max(drops.values())
    result["max_ablation_head"] = max(drops, key=drops.get).replace("_ablation_drop", "")
    return result


@torch.no_grad()
def circuit_scores(model, contexts: np.ndarray, device: torch.device,
                   max_examples: int = 256) -> dict:
    """Score previous-token and induction-like attention patterns per head."""
    model.eval()
    tokens = torch.as_tensor(contexts[:max_examples], device=device)
    model(tokens)
    maps = model.attention_maps()
    query_pos = tokens.size(1) - 1
    query_symbols = tokens[:, query_pos]
    result = {}
    for layer, attention in enumerate(maps):
        previous = attention[:, :, 1:, :-1].diagonal(dim1=-2, dim2=-1).mean(dim=(0, 2))
        induction_by_example = []
        for i in range(tokens.size(0)):
            matching_symbols = (tokens[i, :query_pos] == query_symbols[i]).nonzero().flatten()
            value_positions = matching_symbols + 1
            value_positions = value_positions[value_positions < query_pos]
            induction_by_example.append(attention[i, :, query_pos, value_positions].sum(dim=-1))
        induction = torch.stack(induction_by_example).mean(dim=0)
        for head in range(attention.size(1)):
            result[f"L{layer + 1}H{head + 1}_previous_token"] = float(previous[head].cpu())
            result[f"L{layer + 1}H{head + 1}_induction"] = float(induction[head].cpu())
    previous_values = [v for k, v in result.items() if k.endswith("previous_token")]
    induction_values = [v for k, v in result.items() if k.endswith("induction")]
    result["best_previous_token_score"] = max(previous_values)
    result["best_induction_score"] = max(induction_values)
    return result


def save_metric_plots(metrics: pd.DataFrame, output_dir: Path) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    sns.set_theme(style="whitegrid")
    for metric in ["exact_accuracy", "first_value_accuracy", "loss", "perplexity",
                   "greedy_first_value_accuracy", "sampled_pair_consistency",
                   "sampled_symbol_entropy", "sampled_max_symbol_share",
                   "sampled_symbol_in_context", "trainlabel_first_value_accuracy",
                   "best_previous_token_score", "best_induction_score", "max_ablation_drop"]:
        if metric not in metrics or metrics[metric].isna().all():
            continue
        plt.figure(figsize=(7, 4))
        sns.lineplot(data=metrics, x="generation", y=metric, hue="task", marker="o")
        plt.title(metric.replace("_", " ").title() + " across generations")
        plt.tight_layout()
        plt.savefig(output_dir / f"{metric}.png", dpi=180)
        plt.close()


@torch.no_grad()
def save_attention_plot(model, context: np.ndarray, output: Path, device: torch.device) -> None:
    model.eval()
    tokens = torch.as_tensor(context[None, :], device=device)
    model(tokens)
    maps = model.attention_maps()
    rows, cols = len(maps), maps[0].shape[1]
    fig, axes = plt.subplots(rows, cols, figsize=(3 * cols, 3 * rows), squeeze=False)
    for layer, attention in enumerate(maps):
        for head in range(cols):
            sns.heatmap(attention[0, head].cpu(), ax=axes[layer, head], cmap="mako", cbar=False)
            axes[layer, head].set_title(f"Layer {layer + 1}, head {head + 1}")
            axes[layer, head].set_xlabel("Key position")
            axes[layer, head].set_ylabel("Query position")
    fig.tight_layout()
    fig.savefig(output, dpi=180)
    plt.close(fig)
