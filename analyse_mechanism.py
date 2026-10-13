"""Direct logit attribution (DLA) at the extended task's symbol step.

At the position where the model chooses the next symbol, the residual stream is
a sum of components: token embedding, position embedding, each attention
head's output, and each MLP's output. Freezing the final LayerNorm's scale
(computed from the full residual), each component's contribution to every
logit is exact and additive:

    logit = sum_c  W_U ( gamma * (c - mean(c)) / sigma )  +  W_U beta

For every extended-task checkpoint we report, per component:
  * unseen_pref: mean logit contribution to symbols NOT shown in the context
    minus mean contribution to shown symbols (positive = pushes towards
    out-of-context symbols, i.e. causes leaked mass)
  * symbol_bias_sd: SD across symbol IDs of the component's context-averaged
    symbol logits (a context-independent preference for particular symbols)

    python analyse_mechanism.py --config configs/submission.yaml
"""
from __future__ import annotations
import argparse
from pathlib import Path
import numpy as np, pandas as pd, torch, yaml

from run_experiment import build_model, make_splits


@torch.no_grad()
def decompose(model, tokens: torch.Tensor) -> tuple[dict, torch.Tensor]:
    """Return {component: residual contribution at the last position} and the model's logits."""
    t = tokens.size(1)
    pos = torch.arange(t)
    comps = {"token_emb": model.token_embedding(tokens),
             "pos_emb": model.position_embedding(pos)[None].expand(tokens.size(0), -1, -1)}
    x = comps["token_emb"] + comps["pos_emb"]
    mask = torch.triu(torch.ones(t, t, dtype=torch.bool), 1)
    for li, block in enumerate(model.blocks):
        attn = block.attn
        h = block.norm1(x)
        q, k, v = attn.qkv(h).chunk(3, dim=-1)
        shape = (tokens.size(0), t, attn.n_heads, attn.head_dim)
        q, k, v = (z.view(shape).transpose(1, 2) for z in (q, k, v))
        a = (q @ k.transpose(-2, -1) / attn.head_dim ** 0.5).masked_fill(mask, float("-inf")).softmax(-1)
        z = a @ v                                                  # (b, H, t, hd)
        w = attn.out.weight                                        # (d, d)
        total = torch.zeros_like(x)
        for hi in range(attn.n_heads):
            sl = slice(hi * attn.head_dim, (hi + 1) * attn.head_dim)
            out = z[:, hi] @ w[:, sl].T
            comps[f"L{li + 1}H{hi + 1}"] = out
            total = total + out
        x = x + total
        m = block.mlp(block.norm2(x))
        comps[f"L{li + 1}MLP"] = m
        x = x + m
    last = {k: v[:, -1] for k, v in comps.items()}
    xf = x[:, -1]
    ln = model.norm
    sigma = torch.sqrt(xf.var(-1, unbiased=False, keepdim=True) + ln.eps)
    wu = model.lm_head.weight                                      # (V, d)
    logits = {k: ((c - c.mean(-1, keepdim=True)) / sigma * ln.weight) @ wu.T for k, c in last.items()}
    logits["ln_bias"] = (ln.bias @ wu.T).expand(tokens.size(0), -1)
    return logits, model(tokens)[:, -1]


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/submission.yaml")
    cfg = yaml.safe_load(open(ap.parse_args().config, encoding="utf-8"))
    root, rows = Path(cfg["output_dir"]), []
    for seed in cfg["seeds"]:
        run = yaml.safe_load(yaml.safe_dump(cfg)); run["seed"] = seed
        vocab, _, _, (tx, ty) = make_splits("extended", run, seed + 10000)
        n = vocab.num_symbols
        shown = np.zeros((len(tx), n), bool)
        for i, c in enumerate(tx):
            shown[i, c[1:-1:2]] = True
        keep = (~shown).any(1)                                    # needs >= 1 unseen symbol
        shown_t = torch.as_tensor(shown[keep])
        tokens = torch.as_tensor(np.concatenate([tx, ty[:, :1]], 1)[keep])
        for g in range(cfg["collapse"]["generations"] + 1):
            model = build_model(run["model"], vocab.vocab_size, tx.shape[1] + 3)
            model.load_state_dict(torch.load(root / f"seed_{seed}/extended/model_generation_{g}.pt",
                                             map_location="cpu"))
            model.eval()
            parts, logits = decompose(model, tokens)
            recon = sum(parts.values())
            err = float((recon - logits).abs().max())
            assert err < 1e-3, f"decomposition does not sum to logits (max error {err})"
            for name, part in parts.items():
                s = part[:, :n]
                unseen = (s * ~shown_t).sum(1) / (~shown_t).sum(1)
                seen = (s * shown_t).sum(1) / shown_t.sum(1)
                bias = s.mean(0)
                rows.append({"seed": seed, "generation": g, "component": name,
                             "unseen_pref": float((unseen - seen).mean()),
                             "symbol_bias_sd": float((bias - bias.mean()).std(unbiased=False))})
    df = pd.DataFrame(rows)
    df.to_csv(root / "mechanism_dla.csv", index=False)
    piv = df.groupby(["component", "generation"])[["unseen_pref", "symbol_bias_sd"]].mean()
    print(piv.unstack("generation").round(3).to_string())


if __name__ == "__main__":
    main()
