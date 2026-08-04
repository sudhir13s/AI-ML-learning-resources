"""NLP Evaluation Metrics — the metric zoo, from scratch, plus correlation and significance.

Single seeded source of truth for the chapter "18 NLP Evaluation Metrics". Both the teaching
notebook (`18-NLP-Evaluation-Metrics.ipynb`) and the figure generator (`make_figures_18.py`)
import the functions defined here, so the prose, the notebook output, and every embedded
figure come from the SAME verified code and cannot silently drift apart.

This chapter is the NLP-evaluation CAPSTONE: rather than re-derive every task metric in full
(BLEU lives in ch.12 MT, ROUGE in ch.13 summarization, EM/F1 in ch.11 QA, perplexity in ch.04
n-grams, classification P/R/F1 in ch.10), it shows them SIDE BY SIDE on the same input so their
DISAGREEMENT is the lesson, and then derives the two meta-tools every metric is judged by:
metric-human correlation (Spearman) and bootstrap significance.

What it builds, from scratch (pure numpy / stdlib — no ML framework needed):

  * a "metric zoo": exact-match, token-level F1 (SQuAD style), BLEU (clipped n-gram precision +
    geometric mean + brevity penalty), ROUGE-L (LCS-based F), and chrF (character n-gram F-beta),
    all computed on ONE candidate/reference pair so their disagreement is visible. BLEU is
    verified against sacreBLEU and ROUGE-L against rouge-score (when those libs are importable).
  * a meaning-blindness demo: a faithful paraphrase scores LOW on n-gram overlap but HIGH on a
    small deterministic embedding-overlap metric — the cross-chapter punchline, in pure numpy
    (a fixed hashing embedding, no model download, so it runs anywhere and is reproducible).
  * a paired bootstrap CI on a metric DIFFERENCE: resample sentence pairs to get a confidence
    interval on (system A − system B), exposing a "win" whose interval straddles zero — i.e. not
    significant (Koehn 2004).
  * Spearman / Pearson / Kendall correlation of a metric against synthetic human scores — the
    meta-metric that decides whether any automatic metric can be trusted.

Everything is deterministic and seeded. The optional library cross-checks (sacrebleu,
rouge_score) are guarded by try/except so the core runs even without them. Verified on
Python 3.12 / numpy 2.4.6, CPU.

Run:
    python nlp_evaluation.py
"""

from __future__ import annotations

import hashlib
import math
import platform
from collections import Counter

import numpy as np

SEED = 0
RNG = np.random.default_rng(SEED)  # all resampling draws from this one seeded generator

# BLEU's standard configuration: equal weights over n-gram orders 1..4.
MAX_N = 4
BLEU_WEIGHTS = tuple(1.0 / MAX_N for _ in range(MAX_N))  # (0.25, 0.25, 0.25, 0.25)

# chrF standard configuration (matches sacreBLEU defaults): char n-grams 1..6, F-beta with beta=2
# (recall weighted twice as heavily as precision).
CHRF_MAX_N = 6
CHRF_BETA = 2.0

# A small fixed embedding dimension for the deterministic hashing embedding used by the
# meaning-blindness demo. 64 dims is plenty to separate "same meaning" from "unrelated" on the
# tiny vocabulary used here, while staying fast and fully reproducible (no model download).
EMBED_DIM = 64

# Bootstrap defaults: 10_000 resamples is the field-standard count (Koehn 2004 used ~1000; more
# is cheaper now and tightens the interval estimate). 95% CI -> the 2.5th / 97.5th percentiles.
N_BOOTSTRAP = 10_000
CI_LOW_PCT = 2.5
CI_HIGH_PCT = 97.5


# ===================================================================================================
# 0. Tokenization + n-gram helpers (shared by every overlap metric below).
# ===================================================================================================
def tokenize(text: str) -> list[str]:
    """Lowercase whitespace tokenization — deliberately simple so the math is hand-checkable.

    Real metric implementations (sacreBLEU, rouge-score) use richer, standardized tokenizers;
    we use whitespace + lowercase so every number on the page can be reproduced by hand. The
    library cross-checks below feed the libraries the SAME pre-tokenized text for a fair match.
    """
    return text.lower().split()


def ngram_counts(tokens: list[str], n: int) -> Counter:
    """Count every n-gram (as a tuple) in a token list. Empty if the list is shorter than n."""
    return Counter(tuple(tokens[i : i + n]) for i in range(len(tokens) - n + 1))


# ===================================================================================================
# 1. Exact match + token-level F1 (SQuAD style) — the QA pair of metrics.
# ===================================================================================================
def exact_match(pred: str, gold: str) -> int:
    """1 if the normalized prediction equals the normalized gold string, else 0.

    Strict and brittle: 'the Denver Broncos' vs 'Denver Broncos' scores 0 despite being right —
    which is exactly why SQuAD reports token-F1 alongside EM (see `token_f1`).
    """
    return int(tokenize(pred) == tokenize(gold))


def token_f1(pred: str, gold: str) -> float:
    """Bag-of-tokens F1 over the shared tokens — the forgiving QA metric that gives partial credit.

    Treats both strings as multisets of tokens, counts the overlap, and returns the harmonic mean
    of precision (shared / |pred|) and recall (shared / |gold|). Ignores word order (bag of
    tokens), which is fine for short extractive spans but never for full generation.
    """
    pred_toks, gold_toks = tokenize(pred), tokenize(gold)
    if not pred_toks or not gold_toks:
        return float(pred_toks == gold_toks)  # both empty -> 1.0; one empty -> 0.0
    shared = sum((Counter(pred_toks) & Counter(gold_toks)).values())  # multiset intersection size
    if shared == 0:
        return 0.0
    precision = shared / len(pred_toks)
    recall = shared / len(gold_toks)
    return harmonic_mean(precision, recall)


def harmonic_mean(precision: float, recall: float) -> float:
    """F1 = 2PR / (P + R) — the harmonic mean, which collapses toward the SMALLER of P, R.

    Using the harmonic (not arithmetic) mean is the whole reason F1 punishes lopsided scores: a
    system with precision 1.0 and recall 0.0 has arithmetic mean 0.5 but F1 = 0. You cannot win
    F1 by being great on one axis and terrible on the other.
    """
    if precision + recall == 0:
        return 0.0
    return 2 * precision * recall / (precision + recall)


# ===================================================================================================
# 2. BLEU, from scratch — clipped n-gram precision, geometric mean, brevity penalty.
#    (Full derivation lives in ch.12 Machine Translation; this is the capstone recap, runnable.)
# ===================================================================================================
def modified_precision(cand: list[str], ref: list[str], n: int) -> tuple[int, int]:
    """Clipped n-gram precision numerator and denominator for one candidate vs one reference.

    The 'modified' (clipped) precision is BLEU's key idea: a candidate n-gram is credited at most
    the number of times it appears in the reference. This kills the cheat of repeating one good
    word ('the the the the') to inflate precision. Returns (clipped_matches, candidate_total) so a
    corpus can sum numerators and denominators BEFORE dividing (correct micro-averaged corpus BLEU).
    """
    cand_ng = ngram_counts(cand, n)
    if not cand_ng:
        return 0, 0
    ref_ng = ngram_counts(ref, n)
    clipped = sum(min(count, ref_ng[gram]) for gram, count in cand_ng.items())
    total = sum(cand_ng.values())
    return clipped, total


def brevity_penalty(cand_len: int, ref_len: int) -> float:
    """BP = 1 if cand longer than ref, else exp(1 - ref_len/cand_len) — BLEU's stand-in for recall.

    Precision alone is gamed by being short ('the cat' scores 1.0 against a long reference). BLEU
    has no recall term, so BP shrinks the score exponentially when the candidate is shorter than
    the reference, forcing it to be roughly as long.
    """
    if cand_len == 0:
        return 0.0
    if cand_len > ref_len:
        return 1.0
    return math.exp(1 - ref_len / cand_len)


def sentence_bleu(cand_text: str, ref_text: str, max_n: int = MAX_N) -> dict:
    """Sentence-level BLEU from scratch: per-order clipped precision, geometric mean, times BP.

    Returns a dict with the per-order precisions, the geometric mean, the brevity penalty, and the
    final BLEU on a 0-100 scale (the conventional reporting scale). No smoothing — so a missing
    higher-order n-gram zeroes the score, exactly the corpus-vs-sentence brittleness the page warns
    about; this is a *teaching* sentence BLEU, fine for the worked example, not for production.
    """
    cand, ref = tokenize(cand_text), tokenize(ref_text)
    precisions = []
    for n in range(1, max_n + 1):
        clipped, total = modified_precision(cand, ref, n)
        precisions.append(clipped / total if total > 0 else 0.0)
    if min(precisions) > 0:
        geo_mean = math.exp(sum(w * math.log(p) for w, p in zip(BLEU_WEIGHTS, precisions)))
    else:
        geo_mean = 0.0  # any zero precision -> geometric mean is 0 (the brutal corpus-vs-sentence trap)
    bp = brevity_penalty(len(cand), len(ref))
    return {
        "precisions": precisions,
        "geo_mean": geo_mean,
        "bp": bp,
        "bleu": 100 * bp * geo_mean,
    }


# ===================================================================================================
# 3. ROUGE-L, from scratch — longest common subsequence -> P/R/F.
#    (Full derivation lives in ch.13 Summarization; capstone recap here.)
# ===================================================================================================
def lcs_length(a: list[str], b: list[str]) -> int:
    """Length of the longest common subsequence of two token lists (classic DP, O(|a|*|b|)).

    A subsequence keeps order but allows gaps: LCS('the cat sat on the mat', 'the cat on the mat')
    = 5. ROUGE-L applies this word-level LCS so in-order overlap is rewarded without demanding
    contiguous phrase matches — robust to small insertions a summary makes.
    """
    rows, cols = len(a), len(b)
    dp = [[0] * (cols + 1) for _ in range(rows + 1)]
    for i in range(1, rows + 1):
        for j in range(1, cols + 1):
            if a[i - 1] == b[j - 1]:
                dp[i][j] = dp[i - 1][j - 1] + 1
            else:
                dp[i][j] = max(dp[i - 1][j], dp[i][j - 1])
    return dp[rows][cols]


def rouge_l(cand_text: str, ref_text: str) -> dict:
    """ROUGE-L from scratch: LCS-based precision, recall, and balanced (beta=1) F-measure.

    Precision = LCS/|cand|, recall = LCS/|ref|. Summarization headlines RECALL (did the summary
    cover the reference?), but we return all three so the page can show P != R when lengths differ.
    """
    cand, ref = tokenize(cand_text), tokenize(ref_text)
    if not cand or not ref:
        return {"lcs": 0, "precision": 0.0, "recall": 0.0, "f": 0.0}
    lcs = lcs_length(cand, ref)
    precision = lcs / len(cand)
    recall = lcs / len(ref)
    return {
        "lcs": lcs,
        "precision": precision,
        "recall": recall,
        "f": harmonic_mean(precision, recall),  # balanced beta=1 F (most tools' default report)
    }


# ===================================================================================================
# 4. chrF, from scratch — character n-gram F-beta (sub-word overlap).
#    (Introduced in ch.12 MT; capstone recap here — the most language-agnostic surface metric.)
# ===================================================================================================
def char_ngrams(text: str, n: int) -> Counter:
    """Count character n-grams, ignoring whitespace — chrF's matching unit is below the word.

    Stripping spaces makes chrF tokenizer-free (its whole selling point) and lets it reward shared
    morphology: 'walked' and 'walking' share the 4-gram 'walk', so chrF gives partial credit where
    word-level BLEU sees two different tokens and gives none.
    """
    chars = list(text.lower().replace(" ", ""))
    return Counter(tuple(chars[i : i + n]) for i in range(len(chars) - n + 1))


def chrf(cand_text: str, ref_text: str, max_n: int = CHRF_MAX_N, beta: float = CHRF_BETA) -> float:
    """chrF from scratch: average character-n-gram F-beta over orders 1..max_n, scaled to 0-100.

    For each order it computes precision (shared / cand n-grams) and recall (shared / ref n-grams),
    then an F-beta that weights recall beta^2 times as much as precision (beta=2 default). The
    final score averages the per-order F-betas — a single number that rewards sub-word overlap.
    """
    f_scores = []
    for n in range(1, max_n + 1):
        cand_ng, ref_ng = char_ngrams(cand_text, n), char_ngrams(ref_text, n)
        if not cand_ng or not ref_ng:
            continue
        shared = sum((cand_ng & ref_ng).values())  # multiset intersection across char n-grams
        if shared == 0:
            f_scores.append(0.0)
            continue
        precision = shared / sum(cand_ng.values())
        recall = shared / sum(ref_ng.values())
        beta_sq = beta * beta
        denom = beta_sq * precision + recall
        f_scores.append((1 + beta_sq) * precision * recall / denom if denom > 0 else 0.0)
    return 100 * (sum(f_scores) / len(f_scores)) if f_scores else 0.0


def metric_zoo(cand_text: str, ref_text: str) -> dict:
    """Run EVERY surface metric on one candidate/reference pair — the disagreement IS the lesson.

    Returns each metric on a common 0-100 scale so a single bar chart can show how five metrics
    that all claim to 'measure overlap' assign very different scores to the same pair.
    """
    return {
        "Exact match": 100.0 * exact_match(cand_text, ref_text),
        "Token-F1": 100.0 * token_f1(cand_text, ref_text),
        "BLEU": sentence_bleu(cand_text, ref_text)["bleu"],
        "ROUGE-L": 100.0 * rouge_l(cand_text, ref_text)["f"],
        "chrF": chrf(cand_text, ref_text),
    }


# ===================================================================================================
# 5. Meaning-blindness — a deterministic embedding-overlap metric (pure numpy, no model download).
#    Shows the cross-chapter punchline: a paraphrase scores LOW on n-gram overlap, HIGH on meaning.
# ===================================================================================================
# A tiny hand-built synonym table so the deterministic embedding can place synonyms NEAR each
# other in vector space without a trained model. This is a TEACHING stand-in for a real contextual
# encoder (BERT): it captures just enough "same meaning -> similar vector" to make the point
# reproducibly. Real BERTScore numbers (which need a model) are quoted on the page from the
# pre-generated figure; this metric exists so the *mechanism* runs anywhere, instantly, seeded.
_SYNONYM_GROUPS = [
    {"fantastic", "wonderful", "incredible", "incredibly", "amazing", "great", "superb"},
    {"terrible", "awful", "horrible", "bad", "dreadful"},
    {"movie", "film"},
    {"absolutely", "utterly", "completely", "totally"},
    {"happy", "glad", "pleased", "delighted"},
]


def _word_vector(word: str) -> np.ndarray:
    """A fixed, deterministic unit vector for a word — same meaning -> same anchor direction.

    Words in the same synonym group share an anchor (so synonyms point the same way); every word
    also gets a small word-specific jitter (hashed, so it's deterministic) so distinct words are
    not identical. md5 (NOT Python's salted hash()) keeps the vectors STABLE across processes and
    machines — a hard reproducibility requirement for this chapter.
    """
    anchor_key = word
    for i, group in enumerate(_SYNONYM_GROUPS):
        if word in group:
            anchor_key = f"__group_{i}__"  # all synonyms collapse to one shared anchor direction
            break
    anchor = _hashed_vector(anchor_key)
    jitter = 0.15 * _hashed_vector(f"jitter::{word}")  # small per-word perturbation
    vec = anchor + jitter
    norm = np.linalg.norm(vec)
    return vec / norm if norm > 0 else vec


def _hashed_vector(key: str) -> np.ndarray:
    """A deterministic pseudo-random unit-ish vector seeded by a STABLE md5 of the key.

    Uses md5 of the key (not Python's per-process hash()) to seed a local RNG so the same key maps
    to the same vector in every run, process, and machine — the chapter's stable-hash requirement.
    """
    digest = hashlib.md5(key.encode("utf-8")).hexdigest()
    seed = int(digest[:8], 16)  # first 32 bits of the md5 -> a stable integer seed
    local_rng = np.random.default_rng(seed)
    return local_rng.standard_normal(EMBED_DIM)


def _embed_sentence(text: str) -> np.ndarray:
    """Mean-pool the word vectors of a sentence into one unit vector (a tiny sentence embedding)."""
    vectors = [_word_vector(tok) for tok in tokenize(text)]
    if not vectors:
        return np.zeros(EMBED_DIM)
    pooled = np.mean(vectors, axis=0)
    norm = np.linalg.norm(pooled)
    return pooled / norm if norm > 0 else pooled


def semantic_overlap(cand_text: str, ref_text: str) -> float:
    """Cosine similarity of the two sentence embeddings, scaled to 0-100 — a meaning-overlap proxy.

    This is the deterministic stand-in for an embedding metric (BERTScore/COMET): it scores a
    faithful PARAPHRASE high (synonyms point the same way) where n-gram overlap scores it low.
    Like real embedding metrics it is FOOLED by negation (antonyms still share sentence context),
    which the page calls out as the limit of the whole family.
    """
    cand_vec, ref_vec = _embed_sentence(cand_text), _embed_sentence(ref_text)
    cosine = float(np.dot(cand_vec, ref_vec))  # both are unit vectors -> dot product is cosine
    return 100.0 * max(0.0, cosine)  # clip tiny negatives from float noise to keep 0-100 scale


# ===================================================================================================
# 6. Paired bootstrap CI on a metric difference (Koehn 2004) — is the "win" significant?
# ===================================================================================================
def paired_bootstrap_diff(
    scores_a: np.ndarray,
    scores_b: np.ndarray,
    n_boot: int = N_BOOTSTRAP,
) -> dict:
    """Bootstrap a confidence interval on mean(A) - mean(B) over PAIRED per-sentence scores.

    Both systems are scored on the SAME sentences, so we resample sentence INDICES (with
    replacement) and recompute the mean difference each time, building its sampling distribution.
    Returns the observed difference, the 95% CI, and whether the CI excludes 0 (significant).

    The point of the demo: a positive observed difference whose CI straddles 0 is NOT a real win —
    reporting a single number ('A beats B by 1.2 BLEU') without this interval is how teams ship
    noise as progress (Koehn 2004).
    """
    assert scores_a.shape == scores_b.shape, "paired scores must align sentence-for-sentence"
    n = len(scores_a)
    observed = float(scores_a.mean() - scores_b.mean())
    idx = RNG.integers(0, n, size=(n_boot, n))  # n_boot resamples, each of n sentence indices
    boot_diffs = scores_a[idx].mean(axis=1) - scores_b[idx].mean(axis=1)  # vectorized over resamples
    ci_low, ci_high = np.percentile(boot_diffs, [CI_LOW_PCT, CI_HIGH_PCT])
    return {
        "observed_diff": observed,
        "ci_low": float(ci_low),
        "ci_high": float(ci_high),
        "significant": bool(ci_low > 0 or ci_high < 0),  # CI excludes 0 => significant
        "boot_diffs": boot_diffs,
    }


# ===================================================================================================
# 7. Metric-vs-human correlation — the meta-metric (Spearman / Pearson / Kendall).
# ===================================================================================================
def _rank(values: np.ndarray) -> np.ndarray:
    """Average ranks of an array (ties share the mean of their rank positions) — for Spearman."""
    order = values.argsort()
    ranks = np.empty(len(values), dtype=float)
    ranks[order] = np.arange(len(values), dtype=float)
    # average tied ranks so Spearman is correct under ties
    unique_vals, inverse = np.unique(values, return_inverse=True)
    for u in range(len(unique_vals)):
        mask = inverse == u
        if mask.sum() > 1:
            ranks[mask] = ranks[mask].mean()
    return ranks


def pearson(x: np.ndarray, y: np.ndarray) -> float:
    """Pearson correlation — linear association between two arrays, in [-1, 1]."""
    x_c, y_c = x - x.mean(), y - y.mean()
    denom = math.sqrt(float((x_c**2).sum()) * float((y_c**2).sum()))
    return float((x_c * y_c).sum() / denom) if denom > 0 else 0.0


def spearman(x: np.ndarray, y: np.ndarray) -> float:
    """Spearman correlation — Pearson on RANKS, so it measures MONOTONE (not just linear) agreement.

    This is the right metric for 'does the automatic metric ORDER outputs the way humans do?' —
    we never assume the metric's scale is linear in human preference, only that higher metric
    should mean higher human score.
    """
    return pearson(_rank(x), _rank(y))


def kendall_tau(x: np.ndarray, y: np.ndarray) -> float:
    """Kendall's tau — fraction of concordant minus discordant PAIRS, in [-1, 1].

    More robust than Spearman to a few wild points: it only asks, for every pair of items, whether
    the two rankings agree on their order. tau-a (no tie correction) is enough for the small,
    tie-free synthetic set used here.
    """
    n = len(x)
    concordant = discordant = 0
    for i in range(n):
        for j in range(i + 1, n):
            sign = (x[i] - x[j]) * (y[i] - y[j])
            if sign > 0:
                concordant += 1
            elif sign < 0:
                discordant += 1
    total = concordant + discordant
    return (concordant - discordant) / total if total > 0 else 0.0


# ===================================================================================================
# Demo scenarios (fixed data) — reused by the notebook and the figure generator so they agree.
# ===================================================================================================
# One reference, four candidates spanning the four cases that separate the metric families.
ZOO_REFERENCE = "the movie was absolutely fantastic"
ZOO_CANDIDATES = {
    "exact copy": "the movie was absolutely fantastic",
    "paraphrase": "the film was incredibly wonderful",  # same meaning, ~no shared content words
    "negation": "the movie was absolutely terrible",  # one word flipped, opposite meaning
    "unrelated": "i need to buy groceries today",  # nothing in common
}

# QA worked example (matches the page's Example A).
QA_PRED = "Denver Broncos"
QA_GOLD = "The Denver Broncos"

# BLEU worked example (matches the page's Example B).
BLEU_REFERENCE = "the cat sat on the warm mat"
BLEU_CANDIDATE = "the cat sat on the mat"


def make_correlation_data() -> tuple[np.ndarray, np.ndarray]:
    """A small synthetic set: human scores and a noisy automatic-metric score, monotonically tied.

    The automatic metric tracks the human ranking but with noise, so Spearman is high-but-not-1 —
    the realistic regime where a metric is useful for ranking systems yet unreliable per sentence.
    """
    human = np.array([1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0])
    noise = RNG.normal(0, 1.1, size=human.shape)  # seeded -> identical every run
    metric = human + noise  # monotone-with-noise: good ranking, imperfect per-item
    return human, metric


def make_two_systems() -> tuple[np.ndarray, np.ndarray]:
    """Per-sentence scores for two MT systems where A's small lead is NOT significant.

    A is centered just above B but with enough per-sentence variance that the paired bootstrap
    interval on (A − B) straddles zero — the 'win that isn't a win' the page makes concrete.
    """
    n = 60
    base = RNG.uniform(20, 80, size=n)  # shared per-sentence difficulty
    system_a = np.clip(base + RNG.normal(1.0, 12.0, size=n), 0, 100)  # tiny +1 mean lead, big noise
    system_b = np.clip(base + RNG.normal(0.0, 12.0, size=n), 0, 100)
    return system_a, system_b


# ===================================================================================================
# Optional library cross-checks — guarded so the core runs without sacrebleu / rouge_score.
# ===================================================================================================
def library_crosscheck() -> dict | None:
    """Verify our from-scratch BLEU and ROUGE-L against sacreBLEU and rouge-score, if importable.

    Returns the library numbers next to ours, or None if the libraries are missing. The libraries
    use their own tokenizers, so we feed them the SAME whitespace-tokenized text to compare the
    math, not the tokenization.
    """
    try:
        import sacrebleu
        from rouge_score import rouge_scorer
    except ImportError:
        return None
    lib_bleu = sacrebleu.sentence_bleu(
        BLEU_CANDIDATE, [BLEU_REFERENCE], smooth_method="none"
    ).score
    scorer = rouge_scorer.RougeScorer(["rougeL"], use_stemmer=False)
    rl = scorer.score(
        "the cat sat on the mat", "the cat was sitting on the mat"
    )["rougeL"]
    return {
        "lib_bleu": lib_bleu,
        "lib_rougeL_f": rl.fmeasure,
        "lib_rougeL_p": rl.precision,
        "lib_rougeL_r": rl.recall,
    }


def main() -> None:
    """Run every demo with assert-before-print, so a wrong number fails loudly, not silently."""
    print(f"python      : {platform.python_version()}")
    print(f"numpy       : {np.__version__}")
    print(f"device      : CPU (pure numpy / stdlib — no accelerator needed)")
    print(f"seed        : {SEED}\n")

    # --- 1. QA: EM vs token-F1 (Example A) ----------------------------------------------------
    em = exact_match(QA_PRED, QA_GOLD)
    f1 = token_f1(QA_PRED, QA_GOLD)
    assert em == 0, "EM should be 0 — the strings differ by 'The'"
    assert abs(f1 - 0.8) < 1e-9, f"token-F1 should be exactly 0.80, got {f1}"
    print("== QA: exact match vs token-F1 ==")
    print(f"  pred={QA_PRED!r}  gold={QA_GOLD!r}")
    print(f"  EM = {em}   token-F1 = {f1:.3f}   (strict EM=0 but mostly-right F1=0.80)\n")

    # --- 2. BLEU by hand (Example B) ----------------------------------------------------------
    b = sentence_bleu(BLEU_CANDIDATE, BLEU_REFERENCE)
    assert abs(b["bleu"] - 67.318) < 0.01, f"BLEU should be 67.318, got {b['bleu']}"
    print("== BLEU from scratch (Example B) ==")
    print(f"  precisions p1..p4 = {[round(p, 3) for p in b['precisions']]}")
    print(f"  geo_mean = {b['geo_mean']:.4f}   BP = {b['bp']:.4f}")
    print(f"  BLEU = {b['bleu']:.3f}\n")

    # --- 3. The metric zoo: five metrics disagree on the same pairs ---------------------------
    print("== The metric zoo: five surface metrics on ONE reference ==")
    print(f"  reference: {ZOO_REFERENCE!r}")
    header = f"  {'candidate':12} {'EM':>5} {'Tok-F1':>7} {'BLEU':>6} {'ROUGE-L':>8} {'chrF':>6} {'semantic':>9}"
    print(header)
    for name, cand in ZOO_CANDIDATES.items():
        z = metric_zoo(cand, ZOO_REFERENCE)
        sem = semantic_overlap(cand, ZOO_REFERENCE)
        print(
            f"  {name:12} {z['Exact match']:5.0f} {z['Token-F1']:7.1f} {z['BLEU']:6.1f} "
            f"{z['ROUGE-L']:8.1f} {z['chrF']:6.1f} {sem:9.1f}"
        )
    # The meaning-blindness assertion: the paraphrase scores LOW on BLEU but HIGH on semantic.
    para_bleu = metric_zoo(ZOO_CANDIDATES["paraphrase"], ZOO_REFERENCE)["BLEU"]
    para_sem = semantic_overlap(ZOO_CANDIDATES["paraphrase"], ZOO_REFERENCE)
    assert para_bleu < 20, f"paraphrase BLEU should be low (<20), got {para_bleu}"
    assert para_sem > 70, f"paraphrase semantic overlap should be high (>70), got {para_sem}"
    print(f"\n  meaning-blindness: paraphrase BLEU={para_bleu:.1f} (LOW) but semantic={para_sem:.1f} (HIGH)")
    # The negation warning: semantic is FOOLED — antonyms still score high.
    neg_sem = semantic_overlap(ZOO_CANDIDATES["negation"], ZOO_REFERENCE)
    assert neg_sem > 50, "negation should still score high on the embedding metric (the family's blind spot)"
    print(f"  negation warning : opposite-meaning sentence semantic={neg_sem:.1f} (still HIGH — fooled)\n")

    # --- 4. Paired bootstrap: is system A's lead significant? ----------------------------------
    sys_a, sys_b = make_two_systems()
    boot = paired_bootstrap_diff(sys_a, sys_b)
    assert boot["observed_diff"] > 0, "system A should have a small positive observed lead"
    assert not boot["significant"], "the lead should NOT be significant (CI straddles 0)"
    print("== Paired bootstrap on (A - B): a 'win' that isn't significant ==")
    print(f"  observed mean diff = {boot['observed_diff']:+.2f}")
    print(f"  95% CI = [{boot['ci_low']:+.2f}, {boot['ci_high']:+.2f}]   "
          f"significant: {boot['significant']} (CI straddles 0)\n")

    # --- 5. Metric-vs-human correlation -------------------------------------------------------
    human, metric = make_correlation_data()
    rho = spearman(metric, human)
    r = pearson(metric, human)
    tau = kendall_tau(metric, human)
    assert 0.7 < rho < 1.0, f"Spearman should be high-but-imperfect, got {rho}"
    print("== Metric-vs-human correlation (the meta-metric) ==")
    print(f"  Spearman rho = {rho:.3f}   Pearson r = {r:.3f}   Kendall tau = {tau:.3f}")
    print(f"  (high but < 1.0: good for RANKING systems, unreliable per single sentence)")

    # --- 6. Library cross-check ---------------------------------------------------------------
    cross = library_crosscheck()
    if cross is not None:
        assert abs(cross["lib_bleu"] - b["bleu"]) < 0.01, "our BLEU must match sacreBLEU"
        rl_ours = rouge_l("the cat was sitting on the mat", "the cat sat on the mat")
        assert abs(cross["lib_rougeL_f"] - rl_ours["f"]) < 1e-6, "our ROUGE-L must match rouge-score"
        print(f"\n== Library cross-check ==")
        print(f"  our BLEU {b['bleu']:.3f}  ==  sacreBLEU {cross['lib_bleu']:.3f}   (match)")
        print(f"  our ROUGE-L F {rl_ours['f']:.3f}  ==  rouge-score F {cross['lib_rougeL_f']:.3f}   (match)")
    else:
        print("\n(sacrebleu / rouge_score not installed — skipping library cross-check)")


if __name__ == "__main__":
    main()
