"""Surface (non-content) features, lexicons and synthetic OCR-noise injection."""
from __future__ import annotations

import re
import collections

import numpy as np

# --------------------------------------------------------------------------
# Surface / transcription-quality features computed on the raw window.
# None of these look at *which* words occur, only at how the text is formed.
# --------------------------------------------------------------------------
VOWELS = set("aeiouAEIOU")
EN_FUNC = {"the", "and", "of", "to", "in", "is", "that", "for", "with", "by",
           "was", "be", "on", "at", "as", "this", "which", "it", "from", "are"}
SURFACE_NAMES = [
    "mean_tok_len", "sd_tok_len", "frac_short_tok", "frac_long_tok",
    "frac_upper_chars", "frac_punct_chars", "frac_digit_chars",
    "frac_nonascii_chars", "frac_vowelless_tok", "frac_mixed_alnum_tok",
    "frac_rare_tok", "type_token_ratio", "frac_english_func",
    "frac_cap_initial_tok", "frac_split_fragment",
]


def surface_features(raw: str, vocab_counts: collections.Counter | None = None,
                     rare_thr: int = 2) -> np.ndarray:
    toks = raw.split()
    n = max(1, len(toks))
    chars = "".join(toks)
    nc = max(1, len(chars))
    alpha = [re.sub(r"[^A-Za-z]", "", t) for t in toks]
    lens = np.array([len(a) for a in alpha if a] or [0])
    low = [a.lower() for a in alpha if a]
    rare = 0.0
    if vocab_counts is not None and low:
        rare = sum(vocab_counts.get(w, 0) <= rare_thr for w in low) / len(low)
    return np.array([
        lens.mean(), lens.std(),
        np.mean(lens <= 2), np.mean(lens >= 15),
        sum(c.isupper() for c in chars) / nc,
        sum((not c.isalnum()) for c in chars) / nc,
        sum(c.isdigit() for c in chars) / nc,
        sum(ord(c) > 127 for c in chars) / nc,
        sum(1 for a in alpha if a and not (set(a) & VOWELS)) / n,
        sum(1 for t in toks if re.search(r"[A-Za-z]", t) and re.search(r"\d", t)) / n,
        rare,
        len(set(low)) / max(1, len(low)),
        sum(w in EN_FUNC for w in low) / n,
        sum(1 for t in toks if t[:1].isupper()) / n,
        sum(1 for a in alpha if 0 < len(a) <= 3 and a.islower()) / n,
    ], dtype=float)


# --------------------------------------------------------------------------
# Orthographic / editorial-convention features (also non-content):
# rates of <bh>, <xh> digraphs, apostrophe elision and hyphenated compounds.
# --------------------------------------------------------------------------
ORTHO_NAMES = ["frac_bh_tok", "frac_xh_tok", "frac_elision_tok", "frac_hyphen_tok"]


def ortho_features(raw: str) -> np.ndarray:
    toks = [t.lower() for t in raw.split()]
    n = max(1, len(toks))
    return np.array([
        sum("bh" in t for t in toks) / n,
        sum("xh" in t for t in toks) / n,
        sum(bool(re.search(r"[a-z]'($|[a-z])", t)) for t in toks) / n,
        sum(bool(re.search(r"[a-z]-[a-z]", t)) for t in toks) / n,
    ], dtype=float)


# --------------------------------------------------------------------------
# Thematic lexicons (old orthography, matched as stems inside lower-cased
# tokens).  Stems were checked against concordances of the corpus; stems
# with clear false positives (e.g. 'kolwa' -> 'kolwandle' "sea",
# 'yesu' -> 'yesuka' "start") were removed or anchored.
# --------------------------------------------------------------------------
LEXICON = {
    "political": [  # governance, law, franchise, colonial administration
        r"palamente",        # parliament
        r"voti", r"vota",    # vote, voter, to vote (incl. -votela)
        r"nyulo", r"nyulw", r"nyula",  # elect / election (umnyulo, ukunyula)
        r"rulumen",          # government (urulumente / urulumeni)
        r"mteto", r"miteto",  # law(s) (umteto, old orth. of umthetho)
        r"lungelo",          # right(s) (ilungelo, amalungelo)
        r"rafu",             # tax (irafu)
        r"mantyi",           # magistrate (imantyi)
        r"tyala",            # court case, charge (ityala)
        r"koloni",           # the Colony
        r"bhunga",           # council
    ],
    "religious": [
        r"tixo",             # God (uTixo, old orth. of uThixo)
        r"^yesu$", r"kristu",  # Jesus, Christ
        r"tandaz",           # pray / prayer
        r"zibalo",           # scriptures (izibalo)
        r"vangeli",          # gospel, evangelist
        r"ngcwele",          # holy
        r"sono$",            # sin (isono)
        r"mfundisi", r"bafundisi",  # minister(s) / missionary
        r"cawa",             # church / Sunday (icawa, ingcawa)
        r"nkonzo",           # (church) service
    ],
}
EXCLUDE = re.compile(r"q[au]nyul|kolwandle")   # 'nqanyulwa' = be cut off
LEX_RE = {k: [re.compile(p) for p in v] for k, v in LEXICON.items()}


def lexicon_counts(tokens: list[str]) -> dict[str, int]:
    out = {k: 0 for k in LEX_RE}
    for w in tokens:
        if EXCLUDE.search(w):
            continue
        for k, pats in LEX_RE.items():
            if any(p.search(w) for p in pats):
                out[k] += 1
    return out


# --------------------------------------------------------------------------
# Synthetic OCR noise: typical character-recognition confusions on Roman
# type (e.g. 'u'<->'n' as in IMVO 'ukntenga' for 'ukuthenga').
# --------------------------------------------------------------------------
CONFUSIONS = {"u": "n", "n": "u", "e": "c", "c": "e", "l": "i", "i": "l",
              "a": "o", "o": "a", "m": "rn", "h": "b", "b": "h", "t": "f"}


def inject_noise(text: str, rate: float, rng: np.random.Generator) -> str:
    """Corrupt a fraction `rate` of tokens with one character confusion."""
    toks = text.split()
    for i, t in enumerate(toks):
        if rng.random() < rate:
            pos = [j for j, c in enumerate(t) if c.lower() in CONFUSIONS]
            if pos:
                j = rng.choice(pos)
                rep = CONFUSIONS[t[j].lower()]
                toks[i] = t[:j] + rep + t[j + 1:]
    return " ".join(toks)
