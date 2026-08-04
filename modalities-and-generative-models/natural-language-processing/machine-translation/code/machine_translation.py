"""Machine Translation — the evaluation core, from scratch, plus IBM Model 1 and beam search.

Single seeded source of truth for the chapter "12 Machine Translation". Both the teaching
notebook (`12-Machine-Translation.ipynb`) and the figure generator (`make_figures_12.py`)
import the functions defined here, so the prose, the notebook output, and every embedded
figure come from the SAME verified code and cannot silently drift apart.

What it builds, from scratch (pure numpy / stdlib — no ML framework needed for the core):

  * BLEU, derived term by term: modified n-gram precision p_n (with clipping), the brevity
    penalty BP, and BLEU = BP * exp(sum w_n log p_n). Verified to match sacreBLEU
    (tokenize='none') and nltk to ~13 significant digits on whitespace-tokenized input.
  * a BLEU "brittleness" demo: a meaning-preserving paraphrase scores ~0 BLEU, motivating
    character-level and learned metrics.
  * chrF from scratch (character n-gram F-score, beta=2), verified to match sacreBLEU's chrF.
  * IBM Model 1 word alignment learned by EM from a tiny parallel corpus — no dictionary,
    just iterated co-occurrence counts; returns the t(f|e) table and an alignment matrix.
  * beam-search length normalization: the worked example where raw log-prob prefers a short,
    truncated hypothesis but length normalization (alpha=0.6, the GNMT default) recovers the
    full faithful one.

An OPTIONAL block at the bottom (guarded by a try/except import) runs a REAL neural MT model
(Helsinki-NLP/opus-mt-fr-en) and scores it with sacreBLEU, so the page's measured NMT numbers
are reproducible. The core math above needs none of that — it runs anywhere numpy runs.

Everything is deterministic and seeded; there are no random components in the core math, and the
one seeded array (the EM init is uniform, not random) is set for clarity. Verified on
Python 3.12 / numpy 2.4.6, CPU.

Run:
    python machine_translation.py
"""

from __future__ import annotations

import math
import platform
from collections import Counter, defaultdict

import numpy as np

SEED = 0
np.random.seed(SEED)  # no randomness in the core math, but pin the global state for reproducibility

# BLEU's standard configuration: equal weights over n-gram orders 1..4.
MAX_N = 4
BLEU_WEIGHTS = tuple(1.0 / MAX_N for _ in range(MAX_N))  # (0.25, 0.25, 0.25, 0.25)

# chrF's standard configuration (matches sacreBLEU defaults): char n-grams 1..6, F-beta with beta=2
# (recall weighted twice as heavily as precision).
CHRF_MAX_N = 6
CHRF_BETA = 2.0


# ===================================================================================================
# 1. BLEU, from scratch — modified n-gram precision, brevity penalty, geometric mean.
# ===================================================================================================
def ngram_counts(tokens: list[str], n: int) -> Counter:
    """Count every n-gram (as a tuple) in a token list. Empty if the list is shorter than n."""
    return Counter(tuple(tokens[i : i + n]) for i in range(len(tokens) - n + 1))


def modified_precision(cand: list[str], refs: list[list[str]], n: int) -> tuple[int, int]:
    """Clipped n-gram precision numerator and denominator for one candidate vs its references.

    The 'modified' (clipped) precision is BLEU's key idea: a candidate n-gram can only be
    credited up to the MAXIMUM number of times it appears in any single reference. This stops
    the cheat of repeating one correct word ('the the the the') to inflate precision.

    Returns (clipped_matches, candidate_total) so a corpus can sum numerators and denominators
    across sentences BEFORE dividing — the correct (micro-averaged) corpus BLEU.
    """
    cand_ng = ngram_counts(cand, n)
    if not cand_ng:
        return 0, 0
    # max count of each n-gram across all references = the per-gram clip ceiling
    max_ref: Counter = Counter()
    for ref in refs:
        for gram, count in ngram_counts(ref, n).items():
            if count > max_ref[gram]:
                max_ref[gram] = count
    clipped = sum(min(count, max_ref[gram]) for gram, count in cand_ng.items())
    total = sum(cand_ng.values())
    return clipped, total


def closest_ref_len(cand_len: int, ref_lens: list[int]) -> int:
    """The reference length closest to the candidate's (ties -> the shorter), for the BP."""
    return min(ref_lens, key=lambda rl: (abs(rl - cand_len), rl))


def brevity_penalty(cand_len: int, ref_len: int) -> float:
    """BP = 1 if the candidate is at least as long as the reference, else exp(1 - r/c).

    BLEU is a precision metric, so a system could score high by emitting only the few words it
    is confident about. The brevity penalty is BLEU's stand-in for recall: it multiplicatively
    punishes outputs SHORTER than the reference, and never rewards longer ones.
    """
    if cand_len == 0:
        return 0.0
    if cand_len > ref_len:
        return 1.0
    return math.exp(1.0 - ref_len / cand_len)


def corpus_bleu(
    candidates: list[str],
    references: list[list[str]],
    *,
    max_n: int = MAX_N,
    weights: tuple[float, ...] = BLEU_WEIGHTS,
) -> dict[str, object]:
    """Corpus BLEU = BP * exp(sum_n w_n log p_n), micro-averaged over the corpus.

    candidates: one detokenized string per sentence (whitespace-tokenized internally).
    references: a list of reference-string lists, one list per candidate (>= 1 reference each).

    Matches sacreBLEU (tokenize='none') and nltk corpus_bleu on whitespace-split tokens to ~13
    significant digits — verified in the chapter's notebook. Returns a dict with the score and
    every intermediate term, so the page and figures can show the derivation, not just the number.
    """
    clipped = [0] * max_n
    total = [0] * max_n
    cand_len_sum = 0
    ref_len_sum = 0
    for cand, refs in zip(candidates, references):
        cand_tokens = cand.split()
        ref_token_lists = [r.split() for r in refs]
        cand_len_sum += len(cand_tokens)
        ref_len_sum += closest_ref_len(len(cand_tokens), [len(r) for r in ref_token_lists])
        for n in range(1, max_n + 1):
            cl, tot = modified_precision(cand_tokens, ref_token_lists, n)
            clipped[n - 1] += cl
            total[n - 1] += tot
    precisions = [(clipped[i] / total[i]) if total[i] > 0 else 0.0 for i in range(max_n)]
    # Geometric mean of the precisions, weighted. exp(sum w_n log p_n). If any p_n is 0, the log
    # is -inf and BLEU is 0 — the well-known cliff that makes raw BLEU unusable on short texts.
    if min(precisions) > 0:
        geo_mean = math.exp(sum(w * math.log(p) for w, p in zip(weights, precisions)))
    else:
        geo_mean = 0.0
    bp = brevity_penalty(cand_len_sum, ref_len_sum)
    score = bp * geo_mean * 100.0
    return {
        "score": score,
        "precisions": [p * 100.0 for p in precisions],  # as percentages, for display
        "bp": bp,
        "geo_mean": geo_mean,
        "cand_len": cand_len_sum,
        "ref_len": ref_len_sum,
        "clipped": clipped,
        "total": total,
    }


def sentence_bleu(candidate: str, references: list[str], **kwargs) -> dict[str, object]:
    """BLEU for a single sentence — a thin wrapper over corpus_bleu with one candidate."""
    return corpus_bleu([candidate], [references], **kwargs)


# ===================================================================================================
# 2. chrF, from scratch — character n-gram F-score (matches sacreBLEU's chrF).
# ===================================================================================================
def char_ngram_counts(text: str, n: int) -> Counter:
    """Character n-grams with whitespace removed — sacreBLEU's chrF convention."""
    stripped = text.replace(" ", "")
    return Counter(stripped[i : i + n] for i in range(len(stripped) - n + 1))


def chrf(candidate: str, reference: str, *, max_n: int = CHRF_MAX_N, beta: float = CHRF_BETA) -> float:
    """Character n-gram F-beta score, averaged over orders 1..max_n. Matches sacreBLEU chrF.

    chrF works on CHARACTERS, so a morphological near-miss ('sleeps' vs 'sleeping') still shares
    most of its character n-grams and scores partial credit — exactly the valid variation that
    word-level BLEU throws away. beta=2 weights recall twice as heavily as precision.
    """
    precisions, recalls = [], []
    for n in range(1, max_n + 1):
        cand_ng = char_ngram_counts(candidate, n)
        ref_ng = char_ngram_counts(reference, n)
        overlap = sum((cand_ng & ref_ng).values())  # min-count intersection
        cand_total = sum(cand_ng.values())
        ref_total = sum(ref_ng.values())
        if cand_total:
            precisions.append(overlap / cand_total)
        if ref_total:
            recalls.append(overlap / ref_total)
    p = float(np.mean(precisions)) if precisions else 0.0
    r = float(np.mean(recalls)) if recalls else 0.0
    if p + r == 0:
        return 0.0
    b2 = beta**2
    return 100.0 * (1 + b2) * p * r / (b2 * p + r)


# ===================================================================================================
# 3. IBM Model 1 — word alignment learned by EM (no dictionary, just co-occurrence counts).
# ===================================================================================================
DEFAULT_CORPUS = [
    ("la maison", "the house"),
    ("la fleur", "the flower"),
    ("la maison fleur", "the house flower"),
]


def ibm_model1(
    corpus: list[tuple[str, str]] = DEFAULT_CORPUS, *, n_iter: int = 20
) -> dict[str, dict[str, float]]:
    """Learn t(f|e) — the probability a French word f translates to an English word e — by EM.

    corpus: (french, english) sentence pairs. Model 1 assumes all alignments are a priori equally
    likely and each f_j is generated from the one e it aligns to. EM alternates:
      E-step: with the current t, compute SOFT alignment counts (each link gets fractional credit
              proportional to its normalized t).
      M-step: renormalize those expected counts into a new t.
    Model 1's objective is convex, so this converges to the global optimum from a uniform start.

    Returns t as {french_word: {english_word: prob}}.
    """
    fr_vocab = {w for fr, _ in corpus for w in fr.split()}
    en_vocab = {w for _, en in corpus for w in en.split()}
    # Uniform initialization: every English word equally likely for every French word.
    t = {f: {e: 1.0 / len(en_vocab) for e in en_vocab} for f in fr_vocab}
    for _ in range(n_iter):
        count: dict = defaultdict(lambda: defaultdict(float))
        total: dict = defaultdict(float)
        for fr_sent, en_sent in corpus:
            fs, es = fr_sent.split(), en_sent.split()
            for f in fs:  # E-step: distribute each f's unit of credit over the e's it co-occurs with
                denom = sum(t[f][e] for e in es)
                for e in es:
                    credit = t[f][e] / denom
                    count[f][e] += credit
                    total[e] += credit
        for f in fr_vocab:  # M-step: renormalize expected counts
            for e in en_vocab:
                if total[e] > 0:
                    t[f][e] = count[f][e] / total[e]
    return t


def alignment_matrix(
    fr_sentence: str, en_sentence: str, t: dict[str, dict[str, float]]
) -> np.ndarray:
    """Posterior alignment weights for one sentence pair, as a (len_en, len_fr) matrix.

    Cell [i, j] is the normalized t(f_j | e_i) — how strongly source word f_j aligns to target
    word e_i, given the learned table t. This is the soft alignment IBM Model 1 infers, and the
    direct ancestor of the attention weights neural MT learns by gradient descent instead of EM.
    """
    fs, es = fr_sentence.split(), en_sentence.split()
    matrix = np.zeros((len(es), len(fs)))
    for j, f in enumerate(fs):
        column = np.array([t.get(f, {}).get(e, 0.0) for e in es])
        s = column.sum()
        if s > 0:
            column = column / s  # normalize over target words so each source word's mass sums to 1
        matrix[:, j] = column
    return matrix


# ===================================================================================================
# 4. Beam search length normalization — why a longer faithful output can beat a short truncated one.
# ===================================================================================================
# Two decoded candidates for "Le chat noir dort sur le canape.":
#   A = "the black cat sleeps"            (4 tokens — faithful but TRUNCATED, drops "on the couch")
#   B = "the black cat sleeps on the couch" (7 tokens — full and faithful)
# B's three extra tokens are highly confident, so B's *average* per-token log-prob is BETTER, yet
# its *total* log-prob is worse purely because it is longer. Raw beam search (alpha=0) therefore
# prefers the truncated A; length normalization (divide by length^alpha) recovers the full B.
BEAM_A_LOGPROBS = (-0.30, -0.35, -0.30, -0.55)
BEAM_B_LOGPROBS = (-0.30, -0.35, -0.30, -0.55, -0.08, -0.10, -0.09)
BEAM_ALPHAS = (0.0, 0.6, 1.0)  # 0.6 is the GNMT default


def length_normalized_score(token_logprobs: tuple[float, ...], alpha: float) -> float:
    """Sum of per-token log-probs divided by length^alpha (GNMT-style length normalization).

    alpha=0 is raw log-prob (biased toward short sequences); alpha=1 is the per-token mean;
    alpha~0.6 is the empirical sweet spot used by Google's NMT.
    """
    return sum(token_logprobs) / (len(token_logprobs) ** alpha)


def beam_length_norm_demo() -> list[dict[str, object]]:
    """Run the length-normalization worked example across alphas; return per-alpha winners."""
    rows = []
    for alpha in BEAM_ALPHAS:
        sa = length_normalized_score(BEAM_A_LOGPROBS, alpha)
        sb = length_normalized_score(BEAM_B_LOGPROBS, alpha)
        rows.append(
            {
                "alpha": alpha,
                "score_a": sa,
                "score_b": sb,
                "winner": "A (short, truncated)" if sa > sb else "B (full, faithful)",
            }
        )
    return rows


# ===================================================================================================
# 5. Environment line (honest, reproducible).
# ===================================================================================================
def environment_line() -> str:
    """A reproducibility line: Python + numpy (+ torch if importable), and the CPU device."""
    parts = [f"Python {platform.python_version()}", f"numpy {np.__version__}"]
    try:
        import torch  # noqa: PLC0415 — optional; only for the NMT demo

        parts.append(f"torch {torch.__version__}")
    except ImportError:
        parts.append("torch not installed (core math needs only numpy)")
    parts.append("device: cpu (the from-scratch math is integer/float counting — CPU is exact)")
    return " | ".join(parts)


# ===================================================================================================
# 6. OPTIONAL — a real neural MT model translates and is scored (needs transformers + sacrebleu).
# ===================================================================================================
NMT_MODEL_NAME = "Helsinki-NLP/opus-mt-fr-en"
NMT_SRCS = (
    "Le chat noir dort sur le canapé.",
    "J'aime apprendre les langues étrangères.",
    "La traduction automatique a beaucoup progressé.",
)
NMT_REFS = (
    "The black cat is sleeping on the couch.",
    "I love learning foreign languages.",
    "Machine translation has progressed a lot.",
)


def run_real_nmt(beam_size: int = 5) -> dict[str, object] | None:
    """Translate NMT_SRCS with a real Transformer NMT model and score with sacreBLEU.

    Returns None (with a printed note) if transformers/sacrebleu are unavailable, so the core
    demo still runs in a minimal environment. When available, the numbers are exact and
    reproducible (greedy/beam decode is deterministic for this model).
    """
    try:
        import sacrebleu  # noqa: PLC0415
        from transformers import MarianMTModel, MarianTokenizer  # noqa: PLC0415
    except ImportError:
        print("(skipping real-NMT block: install `transformers sentencepiece sacremoses sacrebleu`)")
        return None
    tokenizer = MarianTokenizer.from_pretrained(NMT_MODEL_NAME)
    model = MarianMTModel.from_pretrained(NMT_MODEL_NAME)
    batch = tokenizer(list(NMT_SRCS), return_tensors="pt", padding=True)
    generated = model.generate(**batch, num_beams=beam_size, max_length=60)
    hyps = [tokenizer.decode(g, skip_special_tokens=True) for g in generated]
    per_sentence_chrf = [sacrebleu.sentence_chrf(h, [r]).score for h, r in zip(hyps, NMT_REFS)]
    corpus_bleu_score = sacrebleu.corpus_bleu(hyps, [list(NMT_REFS)]).score
    corpus_chrf_score = sacrebleu.corpus_chrf(hyps, [list(NMT_REFS)]).score
    return {
        "hyps": hyps,
        "chrf": per_sentence_chrf,
        "corpus_bleu": corpus_bleu_score,
        "corpus_chrf": corpus_chrf_score,
    }


# ===================================================================================================
# Driver — print every demo so `python machine_translation.py` is a full, readable transcript.
# ===================================================================================================
def main() -> None:
    print(environment_line())
    print()

    # --- BLEU derived on a worked example -----------------------------------------------------
    print("=" * 70)
    print("1. BLEU, derived (matches sacreBLEU tokenize='none' and nltk)")
    print("=" * 70)
    cand = "the the the black cat sat on the mat happily today"
    ref = "the black cat sat on the mat very happily today indeed"
    result = sentence_bleu(cand, [ref])
    print(f"candidate : {cand}")
    print(f"reference : {ref}")
    for n in range(MAX_N):
        print(
            f"  p_{n + 1} = {result['clipped'][n]:>2}/{result['total'][n]:>2} "
            f"= {result['precisions'][n]:5.1f}%   (clipped {n + 1}-gram precision)"
        )
    print(f"  brevity penalty BP = {result['bp']:.4f}  (cand_len={result['cand_len']}, ref_len={result['ref_len']})")
    print(f"  geometric mean of precisions = {result['geo_mean']:.4f}")
    print(f"  BLEU = BP * geo_mean * 100 = {result['score']:.6f}")

    # --- BLEU brittleness ---------------------------------------------------------------------
    print("\n" + "=" * 70)
    print("2. BLEU brittleness — a correct paraphrase scores ~0")
    print("=" * 70)
    ref_b = "the committee will convene on tuesday to discuss the budget"
    exact = "the committee will convene on tuesday to discuss the budget"
    paraphrase = "the panel meets tuesday to talk about the finances"
    for label, hyp in [("exact match", exact), ("valid paraphrase", paraphrase)]:
        b = sentence_bleu(hyp, [ref_b])
        f = chrf(hyp, ref_b)
        print(f"  {label:18s}: BLEU={b['score']:6.2f}   chrF={f:6.2f}   ->  {hyp}")
    print("  takeaway: the paraphrase is a PERFECT translation, yet BLEU collapses to 0 because")
    print("            it shares almost no word n-grams; chrF (characters) is far more forgiving.")

    # --- IBM Model 1 --------------------------------------------------------------------------
    print("\n" + "=" * 70)
    print("3. IBM Model 1 — word alignment by EM (no dictionary, just co-occurrence)")
    print("=" * 70)
    t = ibm_model1()
    for f in sorted(t):
        best_e = max(t[f], key=t[f].get)
        print(f"  {f:8s} -> {best_e:8s}  (p={t[f][best_e]:.2f})")
    matrix = alignment_matrix("la maison fleur", "the house flower", t)
    print("  alignment matrix (rows=English, cols=French) for 'la maison fleur' / 'the house flower':")
    print("           " + "  ".join(f"{w:>6s}" for w in "la maison fleur".split()))
    for i, e in enumerate("the house flower".split()):
        print(f"   {e:8s} " + "  ".join(f"{v:6.2f}" for v in matrix[i]))

    # --- Beam length normalization ------------------------------------------------------------
    print("\n" + "=" * 70)
    print("4. Beam search length normalization — short truncation vs full faithful output")
    print("=" * 70)
    print(f"  {'alpha':>6} | {'A (4 tok)':>12} | {'B (7 tok)':>12} | winner")
    print("  " + "-" * 56)
    for row in beam_length_norm_demo():
        print(
            f"  {row['alpha']:>6.1f} | {row['score_a']:>12.3f} | {row['score_b']:>12.3f} | {row['winner']}"
        )
    print("  raw log-prob (alpha=0) prefers the SHORT truncated output; length normalization")
    print("  (alpha=0.6, the GNMT default) recovers the full faithful translation.")

    # --- Real NMT (optional) ------------------------------------------------------------------
    print("\n" + "=" * 70)
    print("5. A real neural MT model, translated and scored (optional block)")
    print("=" * 70)
    nmt = run_real_nmt()
    if nmt is not None:
        for src, hyp, ref_s, cf in zip(NMT_SRCS, nmt["hyps"], NMT_REFS, nmt["chrf"]):
            print(f"  FR: {src}")
            print(f"   -> {hyp}   (ref: {ref_s})   chrF={cf:.1f}")
        print(f"  corpus BLEU = {nmt['corpus_bleu']:.1f}   corpus chrF = {nmt['corpus_chrf']:.1f}")


if __name__ == "__main__":
    main()
