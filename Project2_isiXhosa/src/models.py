"""Classifiers, validation-set hyper-parameter search and evaluation."""
from __future__ import annotations

import itertools
import warnings

import numpy as np
from gensim.models import FastText
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import balanced_accuracy_score, f1_score, roc_auc_score
from sklearn.naive_bayes import ComplementNB
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

warnings.filterwarnings("ignore", category=UserWarning)
TOKEN_PATTERN = r"(?u)\b[\w']+\b"


# --------------------------------------------------------------------------
# Model factories: each returns an object with fit(texts, y) / predict_proba
# --------------------------------------------------------------------------
def word_lr(C=1.0, ngram=(1, 1), min_df=2, vocabulary=None):
    vec = TfidfVectorizer(token_pattern=TOKEN_PATTERN, lowercase=True,
                          sublinear_tf=True, ngram_range=ngram,
                          min_df=1 if vocabulary is not None else min_df,
                          vocabulary=vocabulary)
    return make_pipeline(vec, LogisticRegression(C=C, class_weight="balanced",
                                                 max_iter=3000, solver="liblinear"))


def char_lr(C=1.0, ngram=(2, 5), min_df=3):
    vec = TfidfVectorizer(analyzer="char_wb", ngram_range=ngram, lowercase=True,
                          sublinear_tf=True, min_df=min_df, max_features=300_000)
    return make_pipeline(vec, LogisticRegression(C=C, class_weight="balanced",
                                                 max_iter=3000, solver="liblinear"))


def nb(alpha=0.5, min_df=2, vocabulary=None):
    vec = CountVectorizer(token_pattern=TOKEN_PATTERN, lowercase=True,
                          min_df=1 if vocabulary is not None else min_df,
                          vocabulary=vocabulary)
    return make_pipeline(vec, ComplementNB(alpha=alpha))


class FastTextLR:
    """Subword (fastText) embeddings trained on the TRAINING texts only,
    mean-pooled per segment, then a class-balanced logistic regression."""

    def __init__(self, dim=100, C=1.0, epochs=10, seed=0, vocabulary=None):
        self.dim, self.C, self.epochs, self.seed = dim, C, epochs, seed
        self.vocab = set(vocabulary) if vocabulary is not None else None

    def _tok(self, texts):
        out = [t.lower().split() for t in texts]
        if self.vocab is not None:
            out = [[w for w in s if w in self.vocab] for s in out]
        return out

    def _embed(self, toks):
        return np.vstack([self.ft.wv[s].mean(0) if s else np.zeros(self.dim)
                          for s in toks])

    _cache: dict = {}

    def fit(self, texts, y):
        toks = self._tok(texts)
        key = (self.dim, self.epochs, self.seed, len(toks),
               hash(tuple(" ".join(s) for s in toks[:200])),
               hash(tuple(" ".join(s) for s in toks[-200:])))
        if key not in FastTextLR._cache:
            FastTextLR._cache.clear()   # keep memory bounded
            FastTextLR._cache[key] = FastText(
                sentences=toks, vector_size=self.dim, window=5, min_count=3,
                min_n=3, max_n=5, epochs=self.epochs, workers=2, bucket=200_000,
                seed=self.seed, sg=1)
        self.ft = FastTextLR._cache[key]
        self.clf = make_pipeline(StandardScaler(),
                                 LogisticRegression(C=self.C, class_weight="balanced",
                                                    max_iter=3000))
        self.clf.fit(self._embed(toks), y)
        return self

    def predict_proba(self, texts):
        return self.clf.predict_proba(self._embed(self._tok(texts)))

    def predict(self, texts):
        return (self.predict_proba(texts)[:, 1] >= 0.5).astype(int)


def surface_lr(C=1.0):
    return make_pipeline(StandardScaler(), LogisticRegression(
        C=C, class_weight="balanced", max_iter=3000))


def surface_gb(lr=0.1, depth=3):
    return HistGradientBoostingClassifier(learning_rate=lr, max_depth=depth,
                                          max_iter=300, class_weight="balanced",
                                          random_state=0)


GRIDS = {
    "word": (word_lr, dict(C=[0.3, 1, 3, 10, 30], ngram=[(1, 1), (1, 2)], min_df=[2, 5])),
    "char": (char_lr, dict(C=[1, 3, 10, 30], ngram=[(2, 4), (2, 5)])),
    "nb": (nb, dict(alpha=[0.01, 0.03, 0.1, 0.3, 1.0], min_df=[2, 5])),
    "fasttext": (FastTextLR, dict(dim=[50, 100], C=[0.01, 0.1, 1.0])),
    "surface_lr": (surface_lr, dict(C=[0.001, 0.01, 0.1, 1, 10, 100])),
    "surface_gb": (surface_gb, dict(lr=[0.05, 0.1], depth=[2, 3, 5])),
}


def grid(params: dict):
    keys = list(params)
    for vals in itertools.product(*[params[k] for k in keys]):
        yield dict(zip(keys, vals))


# --------------------------------------------------------------------------
# Metrics
# --------------------------------------------------------------------------
def metrics(y, p):
    yhat = (p >= 0.5).astype(int)
    return dict(macro_f1=f1_score(y, yhat, average="macro"),
                bal_acc=balanced_accuracy_score(y, yhat),
                auroc=roc_auc_score(y, p) if len(set(y)) > 1 else np.nan)


def cluster_bootstrap(y, p, groups, B=1000, seed=0):
    """95% CI by resampling whole issues (segments of one issue are not
    independent)."""
    rng = np.random.default_rng(seed)
    y, p, groups = np.asarray(y), np.asarray(p), np.asarray(groups)
    uniq = np.unique(groups)
    idx = {g: np.where(groups == g)[0] for g in uniq}
    # resample within paper so both classes are always present
    by_class = [np.array([g for g in uniq if y[idx[g][0]] == c]) for c in (0, 1)]
    stats = []
    for _ in range(B):
        take = np.concatenate([idx[g] for c in by_class
                               for g in rng.choice(c, size=len(c), replace=True)])
        stats.append([metrics(y[take], p[take])[k] for k in ("macro_f1", "bal_acc")])
    lo, hi = np.percentile(np.array(stats), [2.5, 97.5], axis=0)
    return dict(macro_f1_ci=(lo[0], hi[0]), bal_acc_ci=(lo[1], hi[1]))


def tune(name, Xtr, ytr, Xva, yva, extra=None):
    """Select hyper-parameters by validation macro-F1; return refit-free best
    model (trained on train only, so test stays untouched) and val score."""
    factory, params = GRIDS[name]
    best = (-1, None, None)
    for hp in grid(params):
        kw = dict(hp, **(extra or {}))
        m = factory(**kw).fit(Xtr, ytr)
        s = f1_score(yva, (m.predict_proba(Xva)[:, 1] >= 0.5).astype(int), average="macro")
        if s > best[0]:
            best = (s, hp, m)
    return best[2], best[1], best[0]
