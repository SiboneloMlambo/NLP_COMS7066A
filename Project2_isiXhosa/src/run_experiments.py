"""Run every experiment reported in the extended abstract.

Research question: when a classifier separates Imvo Zabantsundu from
Isigidimi sama-Xosa (Nov 1884 - Dec 1888), how much of that separation is
political/thematic content and how much is newspaper identity, transcription
noise and style?  See README.txt for the experiment list.

Usage:  python src/run_experiments.py --data data --out outputs [--seeds 13 21 42]
"""
from __future__ import annotations

import argparse
import collections
import json
import os
import sys
import time

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(__file__))
from data import (build_corpus, issue_split, temporal_split, duplicate_mask,  # noqa: E402
                  level2, level4_tokens)
from features import (surface_features, SURFACE_NAMES, lexicon_counts,  # noqa: E402
                      inject_noise, ortho_features, ORTHO_NAMES)
from models import tune, metrics, cluster_bootstrap, word_lr  # noqa: E402
from sklearn.metrics import f1_score  # noqa: E402

LEVELS = ["L0", "L1", "L2", "L3", "L4"]
LEVEL_DESC = {
    "L0": "full page text (isiXhosa + English)",
    "L1": "isiXhosa paragraphs only",
    "L2": "L1 - self-references, digits, punctuation",
    "L3": "L2 + capitalised (proper-noun) tokens masked",
    "L4": "L3, lower-case, vocabulary shared by both papers",
}
TEXT_MODELS = ["word", "char", "nb", "fasttext"]
MAIN_START = "1884-11"           # Imvo's first issue; overlap window start


def log(*a):
    print(time.strftime("%H:%M:%S"), *a, flush=True)


# --------------------------------------------------------------------------
def shared_vocab(tokens_by_doc, y, min_count=5):
    c = [collections.Counter(), collections.Counter()]
    for toks, lab in zip(tokens_by_doc, y):
        c[lab].update(toks)
    vocab = sorted(w for w in c[0] if c[0][w] >= min_count and c[1][w] >= min_count)
    total = c[0] + c[1]
    return vocab, total


def filt(texts, vocab):
    v = set(vocab)
    return [" ".join(w for w in t.split() if w in v) for t in texts]


def prepare_split(seg, seg_full, split_map):
    """Attach split labels and drop val/test segments that near-duplicate
    training text (reprinted adverts / notices)."""
    seg = seg.assign(split=seg.issue.map(split_map))
    seg_full = seg_full.assign(split=seg_full.issue.map(split_map))
    info = {}
    for name, df in (("xh", seg), ("full", seg_full)):
        tr = df[df.split == "train"].raw.tolist()
        rest = df.split.isin(["val", "test"])
        dup = np.zeros(len(df), bool)
        dup[np.where(rest)[0]] = duplicate_mask(tr, df[rest].raw.tolist())
        info[name] = dict(dup_removed=int(dup.sum()),
                          dup_removed_test=int((dup & (df.split == "test")).sum()),
                          n_eval=int(rest.sum()))
        df["dup"] = dup
    return seg, seg_full, info


def texts_for(level, seg, seg_full, vocab=None):
    if level == "L0":
        return seg_full
    df = seg.copy()
    col = {"L1": "raw", "L2": "L2", "L3": "L3", "L4": "L4"}[level]
    df["text"] = df[col]
    if level == "L4":
        df["text"] = filt(df.text.tolist(), vocab)
    return df


def run_classification(seg, seg_full, split_map, models, levels, tag, out,
                       keep_models=False, drop_dups=True):
    seg, seg_full, dupinfo = prepare_split(seg, seg_full, split_map)
    if "text" not in seg_full:
        seg_full = seg_full.assign(text=seg_full.raw)
    tr = seg[seg.split == "train"]
    vocab, train_counts = shared_vocab([t.split() for t in tr.L4], tr.y.values)
    res, fitted = [], {}
    for level in levels:
        df = texts_for(level, seg, seg_full, vocab)
        keep = ~df.dup if drop_dups else np.ones(len(df), bool)
        d_tr = df[(df.split == "train")]
        d_va = df[(df.split == "val") & keep]
        d_te = df[(df.split == "test") & keep]
        for mname in models:
            t0 = time.time()
            m, hp, vs = tune(mname, d_tr.text.tolist(), d_tr.y.values,
                             d_va.text.tolist(), d_va.y.values)
            p = m.predict_proba(d_te.text.tolist())[:, 1]
            r = dict(tag=tag, level=level, model=mname, hp=str(hp), val_macro_f1=vs,
                     n_train=len(d_tr), n_test=len(d_te), **metrics(d_te.y.values, p))
            r.update(cluster_bootstrap(d_te.y.values, p, d_te.issue.values))
            res.append(r)
            log(tag, level, mname, f"F1={r['macro_f1']:.3f} BA={r['bal_acc']:.3f}",
                hp, f"{time.time() - t0:.0f}s")
            if keep_models and mname in ("word", "char"):
                fitted[(level, mname)] = (m, d_te.assign(p=p))
    return res, fitted, dict(seg=seg, vocab=vocab, train_counts=train_counts,
                             dupinfo=dupinfo)


# --------------------------------------------------------------------------
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default="data")
    ap.add_argument("--out", default="outputs")
    ap.add_argument("--seeds", type=int, nargs="+", default=[13, 21, 42])
    ap.add_argument("--fast", action="store_true", help="skip fastText (quick run)")
    ap.add_argument("--e1-only", action="store_true",
                    help="only the cleaning-ladder runs (used to parallelise seeds)")
    ap.add_argument("--smoke", action="store_true", help="tiny subsample, code check only")
    args = ap.parse_args()
    os.makedirs(args.out, exist_ok=True)
    models = [m for m in TEXT_MODELS if not (args.fast and m == "fasttext")]

    log("building corpus")
    C = build_corpus(args.data)
    seg_all, full_all = C.seg, C.seg_full
    if args.smoke:
        keep_iss = set(seg_all.issue.drop_duplicates().sample(frac=0.3, random_state=0))
        seg_all = seg_all[seg_all.issue.isin(keep_iss)].reset_index(drop=True)
        full_all = full_all[full_all.issue.isin(keep_iss)].reset_index(drop=True)
    date = lambda df: df.issue.str[5:12]                       # noqa: E731
    main_mask = date(seg_all) >= MAIN_START
    seg, seg_full = seg_all[main_mask].reset_index(drop=True), \
        full_all[date(full_all) >= MAIN_START].reset_index(drop=True)
    probe = seg_all[(~main_mask) & (seg_all.paper == "ISIG")].reset_index(drop=True)
    issues = seg[["issue", "paper", "year"]].drop_duplicates()

    stats = dict(
        n_segments=seg.groupby("paper").size().to_dict(),
        n_issues=seg.groupby("paper").issue.nunique().to_dict(),
        n_tokens_raw=seg.assign(n=seg.raw.str.split().str.len()).groupby("paper").n.sum().to_dict(),
        n_probe_segments=len(probe), n_probe_issues=probe.issue.nunique(),
        segments_by_year=seg.groupby(["paper", "year"]).size().unstack(0).fillna(0).astype(int).to_dict(),
    )
    log(stats)

    all_res = []
    # ---------------- E1: cleaning ladder x models, several split seeds ----
    keep = {}
    for s in args.seeds:
        smap = issue_split(issues, seed=s)
        r, fitted, ctx = run_classification(seg, seg_full, smap, models, LEVELS,
                                            f"random_s{s}", args.out,
                                            keep_models=(s == args.seeds[0]))
        all_res += r
        if s == args.seeds[0]:
            keep = dict(fitted=fitted, ctx=ctx, smap=smap)
            stats["dupinfo"] = ctx["dupinfo"]
            stats["shared_vocab_size"] = len(ctx["vocab"])
    if args.e1_only:
        pd.DataFrame(all_res).to_csv(f"{args.out}/results_e1_{'_'.join(map(str, args.seeds))}.csv",
                                     index=False)
        log("done (E1 only)")
        return
    # leakage check: same seed without the near-duplicate filter
    smap0 = keep["smap"]
    r, _, _ = run_classification(seg, seg_full, smap0, ["word", "char"], ["L0", "L1"],
                                 f"nodedup_s{args.seeds[0]}", args.out, drop_dups=False)
    all_res += r

    # ---------------- E2: temporal hold-out (train <=1887, test 1888) -------
    tmap = temporal_split(issues, seed=args.seeds[0])
    r, _, _ = run_classification(seg, seg_full, tmap, ["word", "char"], ["L1", "L4"],
                                 "temporal_1888", args.out)
    all_res += r

    # ---------------- E3: feature families on L4 / surface features -------
    ctx, s0 = keep["ctx"], args.seeds[0]
    sg = ctx["seg"]
    vocab, tc = ctx["vocab"], ctx["train_counts"]
    top = [w for w, _ in sorted(((w, tc[w]) for w in vocab), key=lambda x: -x[1])[:100]]
    content = [w for w in vocab if w not in set(top)]
    keepm = ~sg.dup
    tr, va, te = (sg[(sg.split == "train")], sg[(sg.split == "val") & keepm],
                  sg[(sg.split == "test") & keepm])
    fam = {}
    for fname, voc in (("style_top100", top), ("content_noTop100", content)):
        X = [filt(d.L4.tolist(), voc) for d in (tr, va, te)]
        # vocabulary-restricted unigram models: tune C on validation
        best = (-1, None, None)
        for Cc in [0.1, 0.3, 1, 3, 10]:
            mm = word_lr(C=Cc, vocabulary=voc).fit(X[0], tr.y.values)
            f = f1_score(va.y.values, mm.predict(X[1]), average="macro")
            if f > best[0]:
                best = (f, Cc, mm)
        p = best[2].predict_proba(X[2])[:, 1]
        r = dict(tag=f"family_s{s0}", level="L4", model=f"word[{fname}]", hp=f"C={best[1]}",
                 val_macro_f1=best[0], n_train=len(tr), n_test=len(te),
                 **metrics(te.y.values, p))
        r.update(cluster_bootstrap(te.y.values, p, te.issue.values))
        all_res.append(r)
        fam[fname] = (best[2], X, p)
        log(r["model"], f"F1={r['macro_f1']:.3f}")

    # non-content feature families: transcription quality / orthography
    raw_tr_counts = collections.Counter(w.lower() for t in tr.raw for w in t.split())
    SF, OF = {}, {}
    for k, d in (("tr", tr), ("va", va), ("te", te)):
        SF[k] = np.vstack([surface_features(t, raw_tr_counts) for t in d.raw])
        OF[k] = np.vstack([ortho_features(t) for t in d.raw])
    FAMS = {"surface": SF, "ortho": OF,
            "noncontent": {k: np.hstack([SF[k], OF[k]]) for k in SF}}
    surf_models = {}
    for fam_name, F in FAMS.items():
        for mname in ("surface_lr", "surface_gb"):
            m, hp, vs = tune(mname, F["tr"], tr.y.values, F["va"], va.y.values)
            p = m.predict_proba(F["te"])[:, 1]
            label = f"{fam_name}_{mname.split('_')[1]}"
            r = dict(tag=f"family_s{s0}", level="raw", model=label, hp=str(hp),
                     val_macro_f1=vs, n_train=len(tr), n_test=len(te),
                     **metrics(te.y.values, p))
            r.update(cluster_bootstrap(te.y.values, p, te.issue.values))
            all_res.append(r)
            surf_models[label] = (m, p)
            log(label, f"F1={r['macro_f1']:.3f}")
    sf_df = pd.DataFrame(np.vstack([FAMS["noncontent"][k] for k in ("tr", "va", "te")]),
                         columns=SURFACE_NAMES + ORTHO_NAMES)
    sf_df["paper"] = pd.concat([tr.paper, va.paper, te.paper]).values
    sf_df["split"] = ["train"] * len(tr) + ["val"] * len(va) + ["test"] * len(te)
    sf_df.to_csv(f"{args.out}/surface_features.csv", index=False)

    # ---------------- E4: non-content-matched evaluation -------------------
    # 1:1 nearest-neighbour matching (without replacement) of each Isigidimi
    # test segment to an Imvo test segment on standardised non-content
    # features, keeping pairs within a caliper (median pair distance).
    from sklearn.preprocessing import StandardScaler
    sc = StandardScaler().fit(FAMS["noncontent"]["tr"])
    Zte = sc.transform(FAMS["noncontent"]["te"])
    yte = te.y.values
    imvo_idx, isig_idx = np.where(yte == 1)[0], np.where(yte == 0)[0]
    used, pairs, dists = set(), [], []
    rng = np.random.default_rng(0)
    for i in rng.permutation(isig_idx):
        d = np.linalg.norm(Zte[imvo_idx] - Zte[i], axis=1)
        for j in np.argsort(d):
            if imvo_idx[j] not in used:
                used.add(imvo_idx[j]); pairs.append((i, imvo_idx[j])); dists.append(d[j])
                break
    dists = np.array(dists)
    cal = np.median(dists)
    sel = np.array([k for k, dd in enumerate(dists) if dd <= cal])
    msel = np.array([x for k in sel for x in pairs[k]])
    level_models = keep["fitted"]
    match_rows = []
    for (lv, mn), (m, dte) in level_models.items():
        pmap = dict(zip(dte.index, dte.p))
        if not all(te.index[k] in pmap for k in msel):
            continue
        p = np.array([pmap[te.index[k]] for k in msel])
        full_ = metrics(dte.y.values, dte.p.values)
        match_rows.append(dict(model=f"{lv}-{mn}", full_bal_acc=full_["bal_acc"],
                               **metrics(yte[msel], p)))
    for name, (m, p) in surf_models.items():
        match_rows.append(dict(model=name, full_bal_acc=metrics(yte, p)["bal_acc"],
                               **metrics(yte[msel], p[msel])))
    m_c, X_c, p_c = fam["content_noTop100"]
    match_rows.append(dict(model="L4-word[content]", full_bal_acc=metrics(yte, p_c)["bal_acc"],
                           **metrics(yte[msel], p_c[msel])))
    m_s, X_s, p_s = fam["style_top100"]
    match_rows.append(dict(model="L4-word[style]", full_bal_acc=metrics(yte, p_s)["bal_acc"],
                           **metrics(yte[msel], p_s[msel])))
    smd_before = np.abs(Zte[imvo_idx].mean(0) - Zte[isig_idx].mean(0))
    a, b = msel.reshape(-1, 2)[:, 0], msel.reshape(-1, 2)[:, 1]
    smd_after = np.abs(Zte[b].mean(0) - Zte[a].mean(0))
    stats["matching"] = dict(n_pairs=int(len(sel)), caliper=float(cal),
                             mean_abs_smd_before=float(smd_before.mean()),
                             mean_abs_smd_after=float(smd_after.mean()),
                             n_issues_matched=int(te.iloc[msel].issue.nunique()))
    pd.DataFrame(match_rows).to_csv(f"{args.out}/matched_eval.csv", index=False)
    log("matched", stats["matching"])

    # ---------------- E5: synthetic OCR-noise injection --------------------
    # Which paper carries more OCR garbage? (vowel-less + mixed letter/digit
    # tokens; the rare-token rate is not used because it is confounded with
    # each paper's share of the training vocabulary.)
    sfd = pd.DataFrame(SF["tr"], columns=SURFACE_NAMES).assign(paper=tr.paper.values)
    garbage = (sfd.frac_vowelless_tok + sfd.frac_mixed_alnum_tok).groupby(sfd.paper).mean()
    noisy_paper = garbage.idxmax()
    noisy_label = int(noisy_paper == "IMVO")
    clean_te = te[te.y == 1 - noisy_label]
    stats["garbage_rate_by_paper"] = garbage.to_dict()
    stats["noisier_paper"] = noisy_paper
    inj_rows = []
    for rate in [0.0, 0.02, 0.05, 0.10, 0.20, 0.40]:
        rng = np.random.default_rng(1)
        raw_noisy = [inject_noise(t, rate, rng) for t in clean_te.raw]
        L4n = filt([" ".join(level4_tokens(t)) for t in raw_noisy], vocab)
        for key in ("L1-word", "L1-char", "L4-word", "L4-char"):
            lv, mn = key.split("-")
            m = level_models[(lv, mn)][0]
            X = raw_noisy if lv == "L1" else L4n
            p = m.predict_proba(X)[:, 1]
            inj_rows.append(dict(rate=rate, model=key, frac_pred_noisy_paper=float(
                np.mean((p >= 0.5).astype(int) == noisy_label))))
        log("inject", rate, [round(r_["frac_pred_noisy_paper"], 3) for r_ in inj_rows[-4:]])
    pd.DataFrame(inj_rows).to_csv(f"{args.out}/noise_injection.csv", index=False)

    # ---------------- E6: interpretation: coefficients ----------------------
    clf = m_c.named_steps["logisticregression"]
    names = np.array(m_c.named_steps["tfidfvectorizer"].get_feature_names_out())
    order = np.argsort(clf.coef_[0])
    coef = pd.DataFrame(dict(token=np.r_[names[order[-30:]][::-1], names[order[:30]]],
                             coef=np.r_[clf.coef_[0][order[-30:]][::-1], clf.coef_[0][order[:30]]],
                             favours=["IMVO"] * 30 + ["ISIG"] * 30))
    coef.to_csv(f"{args.out}/top_content_features.csv", index=False)
    # char model at L1: top n-grams (shows identity / noise shortcuts)
    mc = level_models[("L1", "char")][0]
    cc = mc.named_steps["logisticregression"].coef_[0]
    cn = np.array(mc.named_steps["tfidfvectorizer"].get_feature_names_out())
    o = np.argsort(cc)
    pd.DataFrame(dict(ngram=np.r_[cn[o[-25:]][::-1], cn[o[:25]]],
                      coef=np.r_[cc[o[-25:]][::-1], cc[o[:25]]],
                      favours=["IMVO"] * 25 + ["ISIG"] * 25)).to_csv(
        f"{args.out}/top_L1_char_features.csv", index=False)
    mw = level_models[("L1", "word")][0]
    cw = mw.named_steps["logisticregression"].coef_[0]
    wn = np.array(mw.named_steps["tfidfvectorizer"].get_feature_names_out())
    o = np.argsort(cw)
    pd.DataFrame(dict(token=np.r_[wn[o[-25:]][::-1], wn[o[:25]]],
                      coef=np.r_[cw[o[-25:]][::-1], cw[o[:25]]],
                      favours=["IMVO"] * 25 + ["ISIG"] * 25)).to_csv(
        f"{args.out}/top_L1_word_features.csv", index=False)

    # ---------------- E7: editorial-era probe on Isigidimi ----------------
    # The content classifier never saw Isigidimi before Nov 1884.  If its
    # 'Imvo-ness' score for earlier Isigidimi issues peaks under Jabavu's
    # editorship (1881-Oct 1884) rather than rising monotonically into the
    # past, temporal drift alone does not explain it.
    def era(issue):
        d = issue[5:12]
        if d < "1881-01":
            return "ISIG 1876-80 (pre-Jabavu)"
        if d < MAIN_START:
            return "ISIG 1881-Oct 1884 (Jabavu)"
        return "ISIG Nov 1884-88 (post-Jabavu)"

    pr_L4 = filt(probe.L4.tolist(), content)
    p_probe = m_c.predict_proba(pr_L4)[:, 1]
    pdf = probe.assign(p=p_probe, era=probe.issue.map(era),
                       n_in_vocab=[len(t.split()) for t in pr_L4])
    tdf = te.assign(p=p_c, n_in_vocab=[len(t.split()) for t in X_c[2]])
    tdf["era"] = np.where(tdf.y == 1, "IMVO Nov 1884-88", tdf.issue.map(era))
    allp = pd.concat([pdf[["issue", "paper", "year", "era", "p", "n_in_vocab"]],
                      tdf[["issue", "paper", "year", "era", "p", "n_in_vocab"]]])
    allp.to_csv(f"{args.out}/probe_segments.csv", index=False)
    iss = allp.groupby(["issue", "paper", "year", "era"], as_index=False).agg(
        p=("p", "mean"), frac_imvo=("p", lambda x: float(np.mean(x >= 0.5))),
        n_in_vocab=("n_in_vocab", "mean"))
    iss.to_csv(f"{args.out}/probe_by_issue.csv", index=False)
    rng = np.random.default_rng(0)
    probe_summary = {}
    for e_, d in iss.groupby("era"):
        bs = [rng.choice(d.p.values, len(d)).mean() for _ in range(2000)]
        probe_summary[e_] = dict(mean_p_imvo=float(d.p.mean()),
                                 ci=[float(x) for x in np.percentile(bs, [2.5, 97.5])],
                                 frac_segments_imvo=float(np.mean(allp[allp.era == e_].p >= 0.5)),
                                 n_issues=int(len(d)),
                                 mean_in_vocab_tokens=float(d.n_in_vocab.mean()))
    # difference Jabavu era - post-Jabavu, and Jabavu - pre-Jabavu (issue bootstrap)
    def diff_ci(e1, e2):
        a1 = iss[iss.era == e1].p.values
        a2 = iss[iss.era == e2].p.values
        bs = [rng.choice(a1, len(a1)).mean() - rng.choice(a2, len(a2)).mean() for _ in range(2000)]
        return [float(a1.mean() - a2.mean()), *map(float, np.percentile(bs, [2.5, 97.5]))]
    J, PRE, POST = ("ISIG 1881-Oct 1884 (Jabavu)", "ISIG 1876-80 (pre-Jabavu)",
                    "ISIG Nov 1884-88 (post-Jabavu)")
    probe_summary["diff_jabavu_minus_post"] = diff_ci(J, POST)
    probe_summary["diff_jabavu_minus_pre"] = diff_ci(J, PRE)
    stats["probe"] = probe_summary
    stats["probe_by_year"] = {f"{a}-{b}": v for (a, b), v in
                              iss.groupby(["paper", "year"]).p.mean().round(4).items()}
    log("probe", probe_summary)

    # ---------------- E8: lexicon rates per issue ---------------------------
    rows = []
    for r_ in seg_all.itertuples():
        toks = level2(r_.raw).lower().split()
        rows.append(dict(issue=r_.issue, paper=r_.paper, year=r_.year,
                         month=r_.month, n=len(toks), **lexicon_counts(toks)))
    lx = pd.DataFrame(rows).groupby(["issue", "paper", "year", "month"], as_index=False).sum()
    lx["group"] = np.where(lx.paper == "IMVO", "IMVO Nov 1884-88", lx.issue.map(era))
    for k in ("political", "religious"):
        lx[f"{k}_per1k"] = 1000 * lx[k] / lx.n
    lx["pol_share"] = lx.political / (lx.political + lx.religious)
    lx.to_csv(f"{args.out}/lexicon_by_issue.csv", index=False)
    lex_summary = {}
    for g, d in lx.groupby("group"):
        e = {}
        for k in ("political_per1k", "religious_per1k", "pol_share"):
            bs = [rng.choice(d[k].values, len(d)).mean() for _ in range(2000)]
            e[k] = [float(d[k].mean()), *map(float, np.percentile(bs, [2.5, 97.5]))]
        e["n_issues"] = int(len(d))
        lex_summary[g] = e
    stats["lexicon"] = lex_summary
    yr = lx.groupby(["paper", "year"])[["political_per1k", "religious_per1k", "pol_share"]].mean()
    stats["lexicon_by_year"] = {f"{a}-{b}": v for (a, b), v in yr.round(4).to_dict("index").items()}
    # permutation test, issue-level political rate, IMVO vs ISIG in the same window
    mw_ = lx[lx.group.isin(["IMVO Nov 1884-88", POST])]
    vals, lab = mw_.political_per1k.values, (mw_.paper == "IMVO").values
    obs = vals[lab].mean() - vals[~lab].mean()
    perm = []
    for _ in range(5000):
        q = rng.permutation(lab)
        perm.append(vals[q].mean() - vals[~q].mean())
    stats["lexicon_perm_p_political"] = float((np.abs(perm) >= abs(obs)).mean())
    stats["lexicon_obs_diff_political"] = float(obs)
    log("lexicon", lex_summary)

    pd.DataFrame(all_res).to_csv(f"{args.out}/results.csv", index=False)
    with open(f"{args.out}/stats.json", "w") as fh:
        json.dump(stats, fh, indent=2, default=str)
    log("done")


if __name__ == "__main__":
    main()
