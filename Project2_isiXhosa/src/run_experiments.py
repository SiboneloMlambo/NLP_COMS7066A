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
import re
import os
import sys
import time

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(__file__))
from data import (build_corpus, issue_split, temporal_split, duplicate_mask,  # noqa: E402
                  level2, level4_tokens)
from features import (surface_features, SURFACE_NAMES, lexicon_counts,  # noqa: E402
                      inject_noise, ortho_features, ORTHO_NAMES, lexicon_hits,
                      REJECTED)
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


def _norm(w):
    return w.lower().replace("'", "").replace("bh", "b").replace("xh", "x")


def fragments_and_names(tr, vocab, prefix_thr=0.5, cap_thr=0.8):
    """Shared-vocabulary tokens that are (a) grammatical fragments - very short
    tokens or tokens that mostly stand directly before a masked name (Imvo's
    separated prefixes such as 'base', 'ase') - or (b) residual names - tokens
    capitalised in most of their raw occurrences (e.g. 'sprigg').
    Computed on training data only."""
    v = set(vocab)
    occ, before_name = collections.Counter(), collections.Counter()
    for t in tr.L3:
        toks = t.split()
        for i, w in enumerate(toks):
            if w == "NAME":
                continue
            k = _norm(w)
            if k in v:
                occ[k] += 1
                if i + 1 < len(toks) and toks[i + 1] == "NAME":
                    before_name[k] += 1
    cap, tot = collections.Counter(), collections.Counter()
    for t in tr.raw:
        for w in t.split():
            a = re.sub(r"[^A-Za-z']", "", w)
            if not a:
                continue
            k = _norm(a)
            if k in v:
                tot[k] += 1
                cap[k] += a[0].isupper()
    rows = []
    for w in vocab:
        pre = before_name[w] / max(1, occ[w])
        capr = cap[w] / max(1, tot[w])
        reason = ("short" if len(w) <= 3 else "prefix-before-name" if pre >= prefix_thr
                  else "capitalised" if capr >= cap_thr else "")
        if reason:
            rows.append(dict(token=w, reason=reason, frac_before_name=round(pre, 3),
                             frac_capitalised=round(capr, 3), train_count=occ[w]))
    return pd.DataFrame(rows)


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
    # Excluded tokens for the sensitivity analysis (fragments + residual names)
    excl = fragments_and_names(tr, vocab)
    excl.to_csv(f"{args.out}/excluded_fragments_names.csv", index=False)
    content_clean = [w for w in content if w not in set(excl.token)]
    stats["vocab_groups"] = dict(top100=len(top), remaining=len(content),
                                 remaining_clean=len(content_clean),
                                 excluded=excl.reason.value_counts().to_dict())
    fam = {}
    # labels: style_top100 = "top-100 frequent shared words";
    #         content_noTop100 = "remaining shared vocabulary";
    #         content_clean = remaining vocabulary minus fragments and names
    for fname, voc in (("style_top100", top), ("content_noTop100", content),
                       ("content_clean", content_clean)):
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
        log(r["model"], f"F1={r['macro_f1']:.3f} BA={r['bal_acc']:.3f}")

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
    pair_arr = msel.reshape(-1, 2)          # columns: Isigidimi idx, Imvo idx
    rng_m = np.random.default_rng(0)
    boot_pairs = [rng_m.integers(0, len(pair_arr), len(pair_arr)) for _ in range(1000)]

    def matched_row(name, p_full, full_ba):
        """BA on matched pairs + 95% CI from resampling pairs."""
        p_m = p_full[msel]
        bs = []
        for idx in boot_pairs:
            take = pair_arr[idx].ravel()
            bs.append(metrics(yte[take], p_full[take])["bal_acc"])
        lo, hi = np.percentile(bs, [2.5, 97.5])
        return dict(model=name, full_bal_acc=full_ba, **metrics(yte[msel], p_m),
                    bal_acc_ci_lo=lo, bal_acc_ci_hi=hi)

    match_rows = []
    pos = {ix: k for k, ix in enumerate(te.index)}
    for (lv, mn), (m, dte) in level_models.items():
        if lv == "L0" or not set(te.index) <= set(dte.index):
            continue
        p_full = np.zeros(len(te))
        for ix, pv in zip(dte.index, dte.p):
            if ix in pos:
                p_full[pos[ix]] = pv
        match_rows.append(matched_row(f"{lv}-{mn}", p_full,
                                      metrics(dte.y.values, dte.p.values)["bal_acc"]))
    for name, (m, p) in surf_models.items():
        match_rows.append(matched_row(name, p, metrics(yte, p)["bal_acc"]))
    m_c, X_c, p_c = fam["content_noTop100"]
    match_rows.append(matched_row("L4-word[content]", p_c, metrics(yte, p_c)["bal_acc"]))
    m_s, X_s, p_s = fam["style_top100"]
    match_rows.append(matched_row("L4-word[style]", p_s, metrics(yte, p_s)["bal_acc"]))
    p_cc = fam["content_clean"][2]
    match_rows.append(matched_row("L4-word[content_clean]", p_cc, metrics(yte, p_cc)["bal_acc"]))
    # balance diagnostics: standardised mean difference per feature (in
    # training-SD units), before matching (all test segments) and after
    a, b = pair_arr[:, 0], pair_arr[:, 1]
    smd_before = Zte[imvo_idx].mean(0) - Zte[isig_idx].mean(0)
    smd_after = Zte[b].mean(0) - Zte[a].mean(0)
    pd.DataFrame(dict(feature=SURFACE_NAMES + ORTHO_NAMES, smd_before=smd_before,
                      smd_after=smd_after)).to_csv(f"{args.out}/matching_balance.csv",
                                                    index=False)
    pd.DataFrame(dict(isig_issue=te.issue.values[a], imvo_issue=te.issue.values[b],
                      isig_year=te.year.values[a], imvo_year=te.year.values[b],
                      distance=dists[sel])).to_csv(f"{args.out}/matched_pairs.csv", index=False)
    stats["matching"] = dict(n_pairs=int(len(sel)), caliper=float(cal),
                             mean_abs_smd_before=float(np.abs(smd_before).mean()),
                             mean_abs_smd_after=float(np.abs(smd_after).mean()),
                             max_abs_smd_after=float(np.abs(smd_after).max()),
                             n_features_abs_smd_after_gt_0_25=int((np.abs(smd_after) > 0.25).sum()),
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
    inj_rows, base_pred = [], {}
    for rate in [0.0, 0.02, 0.05, 0.10, 0.20, 0.40]:
        rng = np.random.default_rng(1)
        raw_noisy = [inject_noise(t, rate, rng) for t in clean_te.raw]
        L4n = filt([" ".join(level4_tokens(t)) for t in raw_noisy], vocab)
        for key in ("L1-word", "L1-char", "L4-word", "L4-char"):
            lv, mn = key.split("-")
            m = level_models[(lv, mn)][0]
            X = raw_noisy if lv == "L1" else L4n
            pred = (m.predict_proba(X)[:, 1] >= 0.5).astype(int)
            if rate == 0.0:
                base_pred[key] = pred
            b0 = base_pred[key]
            inj_rows.append(dict(
                rate=rate, model=key,
                frac_pred_noisy_paper=float(np.mean(pred == noisy_label)),
                # per-segment changes relative to the uncorrupted text
                frac_flipped_to_noisy=float(np.mean((b0 != noisy_label) & (pred == noisy_label))),
                frac_flipped_any=float(np.mean(b0 != pred))))
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

    # Year-controlled comparison: only years in which BOTH papers appear in the
    # main window (Nov-Dec 1884, 1885, 1887, 1888; Isigidimi 1886 is missing).
    # Per-year issue-level differences, their equal-weight average, and a
    # permutation test that shuffles paper labels WITHIN each year.
    both = mw_.copy()
    years = sorted(set(both[both.paper == "IMVO"].year) & set(both[both.paper == "ISIG"].year))
    both = both[both.year.isin(years)]
    yrows = []
    for k in ("political_per1k", "religious_per1k", "pol_share"):
        diffs_y = []
        for yv in years:
            d = both[both.year == yv]
            a_, b_ = d[d.paper == "IMVO"][k].values, d[d.paper == "ISIG"][k].values
            bs = [rng.choice(a_, len(a_)).mean() - rng.choice(b_, len(b_)).mean()
                  for _ in range(2000)]
            lo, hi = np.percentile(bs, [2.5, 97.5])
            yrows.append(dict(measure=k, year=int(yv), imvo=a_.mean(), isig=b_.mean(),
                              diff=a_.mean() - b_.mean(), ci_lo=lo, ci_hi=hi,
                              n_imvo=len(a_), n_isig=len(b_)))
            diffs_y.append(a_.mean() - b_.mean())
        obs_y = float(np.mean(diffs_y))
        perm_y = []
        for _ in range(5000):
            dd = []
            for yv in years:
                d = both[both.year == yv]
                lab_y = rng.permutation((d.paper == "IMVO").values)
                v_ = d[k].values
                dd.append(v_[lab_y].mean() - v_[~lab_y].mean())
            perm_y.append(np.mean(dd))
        stats.setdefault("lexicon_year_controlled", {})[k] = dict(
            years=[int(x) for x in years], mean_within_year_diff=obs_y,
            perm_p_within_year=float(max(1, (np.abs(perm_y) >= abs(obs_y)).sum()) / 5000),
            n_years_imvo_higher=int(sum(x > 0 for x in diffs_y)))
    pd.DataFrame(yrows).to_csv(f"{args.out}/lexicon_by_year_controlled.csv", index=False)
    log("lexicon year-controlled", stats["lexicon_year_controlled"])

    # Lexicon audit: per pattern, matches per paper, most frequent matched
    # word types and two random concordance lines (main window only).
    hits = collections.defaultdict(list)
    main_seg = seg_all[seg_all.issue.str[5:12] >= MAIN_START]
    for r_ in main_seg.itertuples():
        toks = level2(r_.raw).lower().split()
        for i, w in enumerate(toks):
            for cat, pat, ww in lexicon_hits([w]):
                hits[(cat, pat)].append((r_.paper, w, " ".join(toks[max(0, i - 5):i + 6])))
    arows = []
    rng_a = np.random.default_rng(0)
    for (cat, pat), hs in sorted(hits.items()):
        types = collections.Counter(w for _, w, _ in hs).most_common(5)
        ex = [hs[j][2] for j in rng_a.choice(len(hs), size=min(2, len(hs)), replace=False)]
        arows.append(dict(category=cat, pattern=pat,
                          n_imvo=sum(p_ == "IMVO" for p_, _, _ in hs),
                          n_isig=sum(p_ == "ISIG" for p_, _, _ in hs),
                          top_types="; ".join(f"{w} ({c})" for w, c in types),
                          example_1=ex[0] if ex else "", example_2=ex[1] if len(ex) > 1 else ""))
    pd.DataFrame(arows).to_csv(f"{args.out}/lexicon_audit.csv", index=False)
    pd.DataFrame([dict(stem=k, reason=v) for k, v in REJECTED.items()]).to_csv(
        f"{args.out}/lexicon_rejected.csv", index=False)
    log("lexicon", lex_summary)

    pd.DataFrame(all_res).to_csv(f"{args.out}/results.csv", index=False)
    with open(f"{args.out}/stats.json", "w") as fh:
        json.dump(stats, fh, indent=2, default=str)
    log("done")


if __name__ == "__main__":
    # fastText (gensim) seeds its vectors with Python's string hash, which is
    # randomised per process unless PYTHONHASHSEED is fixed: re-launch with it.
    if os.environ.get("PYTHONHASHSEED") != "0":
        os.environ["PYTHONHASHSEED"] = "0"
        os.execv(sys.executable, [sys.executable] + sys.argv)
    main()
