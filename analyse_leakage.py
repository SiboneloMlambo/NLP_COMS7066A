"""Post-hoc analysis of the extended task: where does generated-symbol
probability mass go across generations?

For each saved extended-task checkpoint, feeds the clean test contexts plus the
true first value, and reads the model's distribution over the next (symbol)
token. Reports the probability mass on symbols NOT shown in the context
("leaked mass"), and how concentrated that leaked mass is on one symbol.

    python analyse_leakage.py --config configs/submission.yaml
"""
from __future__ import annotations
import argparse, json
from pathlib import Path
import numpy as np, pandas as pd, torch, yaml

from run_experiment import build_model, make_splits


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/submission.yaml")
    cfg = yaml.safe_load(open(ap.parse_args().config, encoding="utf-8"))
    root, rows = Path(cfg["output_dir"]), []
    for seed in cfg["seeds"]:
        run = yaml.safe_load(yaml.safe_dump(cfg)); run["seed"] = seed
        s = seed + 10000
        vocab, _, _, (tx, ty) = make_splits("extended", run, s)
        n = vocab.num_symbols
        shown = np.zeros((len(tx), n), bool)
        for i, c in enumerate(tx):
            shown[i, c[1:-1:2]] = True
        inputs = torch.as_tensor(np.concatenate([tx, ty[:, :1]], 1))
        for g in range(cfg["collapse"]["generations"] + 1):
            m = build_model(run["model"], vocab.vocab_size, tx.shape[1] + 3)
            m.load_state_dict(torch.load(root / f"seed_{seed}/extended/model_generation_{g}.pt",
                                         map_location="cpu"))
            m.eval()
            with torch.no_grad():
                p = m(inputs)[:, -1, :].softmax(-1).numpy()
            sym = p[:, :n]
            leaked = (sym * ~shown).sum(1)
            leak_by_symbol = (sym * ~shown).sum(0)
            rows.append({"seed": seed, "generation": g,
                         "mass_on_symbols": float(sym.sum(1).mean()),
                         "leaked_mass": float(leaked.mean()),
                         "top_leaked_symbol": int(leak_by_symbol.argmax()),
                         "top_symbol_share_of_leak": float(leak_by_symbol.max() / leak_by_symbol.sum())})
    df = pd.DataFrame(rows)
    df.to_csv(root / "leakage_analysis.csv", index=False)
    print(df.round(4).to_string(index=False))


if __name__ == "__main__":
    main()
