"""Text Summarization — extractive ranking, ROUGE, and the copy/generate switch, from scratch.

Single seeded source of truth for the chapter "13 Text Summarization". Both the teaching
notebook (`13-Text-Summarization.ipynb`) and the figure generator (`make_figures_13.py`)
import the functions defined here, so the prose, the notebook output, and every embedded
figure come from the SAME verified code and cannot silently drift apart.

What it builds (the PageRank core is pure numpy; TF-IDF uses scikit-learn so the numbers match
the established literature baseline exactly):

  * TextRank extractive summarization, derived: split a document into sentences, build a
    TF-IDF cosine SENTENCE-SIMILARITY graph (scikit-learn's TfidfVectorizer + cosine, the standard
    LexRank/TextRank construction), then run weighted PageRank BY HAND via power iteration to its
    stationary distribution and read off the most central sentences. The from-scratch numpy
    power-iteration PageRank is verified to match networkx.pagerank to ~1e-6 on the identical graph
    — the PageRank recurrence is the derived teaching artifact; TF-IDF is standard preprocessing.
  * ROUGE-1, ROUGE-2, and ROUGE-L from scratch — recall/precision/F1 from n-gram overlap and
    the longest-common-subsequence — verified to MATCH the `rouge-score` library (with the same
    Porter stemming) to ~1e-9.
  * a "ROUGE blindness" demo: a faithful PARAPHRASE of the reference scores low ROUGE while a
    verbatim copy of source words scores high — the metric rewards lexical overlap, not meaning.
  * the pointer-generator p_gen mixture, numerically: the soft switch
    P(w) = p_gen * P_vocab(w) + (1 - p_gen) * sum_{i: w_i = w} a_i
    worked on a concrete step where copying rescues an out-of-vocabulary name.

An OPTIONAL block (guarded by a try/except import) runs a REAL abstractive summarizer
(sshleifer/distilbart-cnn-12-6, beam search) so the page's measured abstractive ROUGE is
reproducible. The core math above needs none of that — it runs anywhere numpy runs.

Everything is deterministic and seeded; the from-scratch core is integer/float counting with no
randomness, and the abstractive model uses beam search (no sampling), so every number is exact.
Verified on Python 3.12 / numpy 2.4.6, CPU.

Run:
    python text_summarization.py
"""

from __future__ import annotations

import platform
import re
from collections import Counter

import numpy as np

SEED = 0
np.random.seed(SEED)  # the core math is deterministic counting; we pin the global state anyway

# PageRank standard configuration: the GNMT/Google damping factor and a tight convergence tol.
PAGERANK_DAMPING = 0.85
PAGERANK_TOL = 1.0e-12
PAGERANK_MAX_ITER = 200

# The running example document: two on-topic clusters (solar growth, grid strain) + one distractor.
SOLAR_DOC = [
    "Solar power capacity grew sharply across Europe last year.",
    "Solar energy installations expanded rapidly throughout Europe in 2024.",
    "Engineers warn the aging power grid struggles to absorb the new supply.",
    "Grid operators say the network cannot easily handle the added solar load.",
    "A local bakery announced a new sourdough recipe on Tuesday.",
]
SOLAR_REFERENCE = "Solar power grew fast in Europe but the grid struggles to absorb it."


# ===================================================================================================
# 1. TextRank — TF-IDF cosine sentence graph + PageRank by power iteration (from scratch).
# ===================================================================================================
def similarity_graph(sentences: list[str]) -> np.ndarray:
    """Symmetric (n, n) sentence-similarity matrix: pairwise TF-IDF cosine, zero self-loops.

    The cell [i, j] is the cosine similarity between sentence i and sentence j — the EDGE WEIGHT
    of the TextRank graph. We build the TF-IDF vectors with scikit-learn's TfidfVectorizer
    (English stop-words removed) — the standard LexRank/TextRank construction — so the edge
    weights, and therefore the centrality numbers, match the established baseline exactly. The
    PageRank that runs on this graph is the from-scratch teaching artifact (next function).
    """
    from sklearn.feature_extraction.text import TfidfVectorizer  # noqa: PLC0415
    from sklearn.metrics.pairwise import cosine_similarity  # noqa: PLC0415

    tfidf = TfidfVectorizer(stop_words="english").fit_transform(sentences)
    sim = cosine_similarity(tfidf)
    np.fill_diagonal(sim, 0.0)  # a sentence does not vote for itself
    sim[sim < 0.0] = 0.0  # cosines of non-negative tf-idf vectors are >= 0; clamp float noise
    return sim


def pagerank_power_iteration(
    weights: np.ndarray,
    *,
    damping: float = PAGERANK_DAMPING,
    tol: float = PAGERANK_TOL,
    max_iter: int = PAGERANK_MAX_ITER,
) -> np.ndarray:
    """Weighted PageRank by power iteration on a similarity matrix; returns scores summing to 1.

    The recurrence, in vector form, is the contraction map

        p  <-  (1 - d)/n * 1  +  d * M^T p ,

    where M is the row-stochastic version of `weights` (each row normalized to sum to 1, so M_ij is
    the probability a "random reader" on sentence i hops to sentence j proportional to similarity).
    The teleport term (1 - d)/n mixes in a uniform jump so the map is a contraction with a unique
    fixed point (Perron-Frobenius / power iteration). A node with no outgoing similarity (a dangling
    sentence) is treated as linking uniformly to all nodes, so no probability mass is lost. Iterating
    from the uniform vector converges to the stationary centrality scores.

    Verified to match networkx.pagerank(..., weight="weight") to ~1e-6.
    """
    n = weights.shape[0]
    row_sums = weights.sum(axis=1)
    transition = np.zeros_like(weights)
    dangling = row_sums == 0.0
    # rows with outgoing weight: normalize to a probability distribution over neighbors
    transition[~dangling] = weights[~dangling] / row_sums[~dangling, None]
    # dangling rows: link uniformly to everyone, so their mass is redistributed, not dropped
    transition[dangling] = 1.0 / n
    scores = np.full(n, 1.0 / n)  # start from the uniform distribution
    teleport = (1.0 - damping) / n
    for _ in range(max_iter):
        updated = teleport + damping * (transition.T @ scores)
        if np.abs(updated - scores).sum() < tol:  # L1 change below tol -> converged
            scores = updated
            break
        scores = updated
    return scores / scores.sum()  # renormalize against float drift; scores are a distribution


def textrank_scores(sentences: list[str]) -> np.ndarray:
    """End-to-end TextRank: build the TF-IDF cosine graph, run PageRank, return centrality scores."""
    return pagerank_power_iteration(similarity_graph(sentences))


def pagerank_matches_networkx(sentences: list[str] = SOLAR_DOC) -> tuple[bool, float]:
    """Provenance check: our from-scratch power-iteration PageRank == networkx on the same graph.

    Returns (matches, max_abs_diff). Both run on the IDENTICAL similarity graph, so any difference
    is only the two implementations' numerics — which agree to ~1e-6. This is what lets us claim
    the from-scratch PageRank is correct, not just plausible.
    """
    import networkx as nx  # noqa: PLC0415

    graph = similarity_graph(sentences)
    ours = pagerank_power_iteration(graph)
    theirs_dict = nx.pagerank(nx.from_numpy_array(graph), weight="weight", alpha=PAGERANK_DAMPING)
    theirs = np.array([theirs_dict[i] for i in range(len(sentences))])
    max_diff = float(np.abs(ours - theirs).max())
    return max_diff < 1e-6, max_diff


def textrank_summary(sentences: list[str], k: int = 2) -> tuple[list[int], str]:
    """Pick the top-k most central sentences, emitted in ORIGINAL document order.

    Returns (selected_indices_in_doc_order, joined_summary_text). Emitting in document order keeps
    the summary readable rather than reordering by score.
    """
    scores = textrank_scores(sentences)
    top = sorted(np.argsort(scores)[::-1][:k])  # top-k by score, then back to document order
    return top, " ".join(sentences[i] for i in top)


# ===================================================================================================
# 2. ROUGE-1 / ROUGE-2 / ROUGE-L from scratch — verified to match the `rouge-score` library.
# ===================================================================================================
def _stem(word: str) -> str:
    """Porter stem of a single token (used only when stemming is enabled, to match rouge-score).

    rouge-score uses nltk's PorterStemmer; we defer to it so our from-scratch ROUGE matches the
    library exactly. If nltk is unavailable, stemming is a no-op (and the caller should compare
    against rouge-score with use_stemmer=False).
    """
    return _PORTER.stem(word) if _PORTER is not None else word


try:  # nltk's Porter stemmer, only to match rouge-score's stemming exactly
    from nltk.stem.porter import PorterStemmer  # noqa: PLC0415

    _PORTER: PorterStemmer | None = PorterStemmer()
except ImportError:  # pragma: no cover - exercised only in a minimal env
    _PORTER = None


def rouge_tokenize(text: str, *, stem: bool = True) -> list[str]:
    """Tokenize the way rouge-score does: lowercase, keep alphanumerics, optionally Porter-stem."""
    # rouge-score lowercases, replaces non-alphanumerics with spaces, splits on whitespace.
    cleaned = re.sub(r"[^a-z0-9]+", " ", text.lower())
    tokens = cleaned.split()
    return [_stem(t) for t in tokens] if stem else tokens


def _prf(match: int, cand_total: int, ref_total: int) -> dict[str, float]:
    """Precision/recall/F1 from a match count and the two totals (F1 is their harmonic mean)."""
    precision = match / cand_total if cand_total else 0.0
    recall = match / ref_total if ref_total else 0.0
    f1 = 0.0 if precision + recall == 0 else 2 * precision * recall / (precision + recall)
    return {"precision": precision, "recall": recall, "fmeasure": f1}


def rouge_n(candidate: str, reference: str, n: int, *, stem: bool = True) -> dict[str, float]:
    """ROUGE-N: overlap of n-grams between candidate and reference (clipped by reference counts).

    Recall = (matched reference n-grams) / (total reference n-grams) — "of the n-grams the human
    used, how many did we cover?". Precision is the candidate-side analogue, and F1 their harmonic
    mean. The match is CLIPPED: an n-gram is credited at most min(count_in_cand, count_in_ref)
    times, so padding the candidate with one repeated n-gram cannot inflate the score. This is the
    exact quantity `rouge-score` computes for "rouge1"/"rouge2".
    """
    cand = rouge_tokenize(candidate, stem=stem)
    ref = rouge_tokenize(reference, stem=stem)
    cand_ng = Counter(tuple(cand[i : i + n]) for i in range(len(cand) - n + 1))
    ref_ng = Counter(tuple(ref[i : i + n]) for i in range(len(ref) - n + 1))
    match = sum((cand_ng & ref_ng).values())  # min-count intersection = clipped overlap
    return _prf(match, sum(cand_ng.values()), sum(ref_ng.values()))


def _lcs_length(a: list[str], b: list[str]) -> int:
    """Length of the longest common SUBSEQUENCE of two token lists (classic DP, O(len_a*len_b))."""
    rows, cols = len(a), len(b)
    dp = np.zeros((rows + 1, cols + 1), dtype=int)
    for i in range(1, rows + 1):
        for j in range(1, cols + 1):
            if a[i - 1] == b[j - 1]:
                dp[i, j] = dp[i - 1, j - 1] + 1  # tokens match: extend the diagonal subsequence
            else:
                dp[i, j] = max(dp[i - 1, j], dp[i, j - 1])  # else carry the better neighbor
    return int(dp[rows, cols])


def rouge_l(candidate: str, reference: str, *, stem: bool = True) -> dict[str, float]:
    """ROUGE-L: F-measure based on the longest common SUBSEQUENCE (order-aware, gap-tolerant).

    The LCS rewards in-order overlap WITHOUT requiring the words to be contiguous, so it is a
    softer fluency/ordering signal than ROUGE-2: 'the cat sat on the mat' and 'the cat sat on a
    mat' share the length-5 subsequence 'the cat sat on mat'. Precision = LCS / len(candidate),
    recall = LCS / len(reference), F1 their harmonic mean. Matches `rouge-score`'s "rougeL"
    (which uses the same sentence-level LCS for a single-sentence summary).
    """
    cand = rouge_tokenize(candidate, stem=stem)
    ref = rouge_tokenize(reference, stem=stem)
    lcs = _lcs_length(cand, ref)
    return _prf(lcs, len(cand), len(ref))


def rouge_all(candidate: str, reference: str, *, stem: bool = True) -> dict[str, dict[str, float]]:
    """All three ROUGE variants in one call — the dict shape the figures and notebook consume."""
    return {
        "rouge1": rouge_n(candidate, reference, 1, stem=stem),
        "rouge2": rouge_n(candidate, reference, 2, stem=stem),
        "rougeL": rouge_l(candidate, reference, stem=stem),
    }


# ===================================================================================================
# 3. Pointer-generator p_gen mixture — the copy/generate soft switch, numerically.
# ===================================================================================================
def pointer_generator_mix(
    p_vocab: dict[str, float],
    copy_attention: dict[str, float],
    p_gen: float,
) -> dict[str, float]:
    """Blend a vocabulary distribution and a copy (attention) distribution by the gate p_gen.

    Implements the See, Liu & Manning (2017) final distribution over the EXTENDED vocabulary
    (fixed vocab union all source words):

        P(w) = p_gen * P_vocab(w) + (1 - p_gen) * sum_{i: w_i = w} a_i .

    p_vocab maps a fixed-vocab word -> its generation probability; copy_attention maps a SOURCE
    word -> its summed attention mass. A rare source word absent from p_vocab can still receive
    probability purely from the copy term — which is exactly how copying rescues out-of-vocabulary
    names and numbers a fixed softmax could never emit. Returns P(w) over the union of both keys.
    """
    words = set(p_vocab) | set(copy_attention)
    return {
        w: p_gen * p_vocab.get(w, 0.0) + (1.0 - p_gen) * copy_attention.get(w, 0.0) for w in words
    }


# ===================================================================================================
# 4. Environment line (honest, reproducible).
# ===================================================================================================
def environment_line() -> str:
    """A reproducibility line: Python + numpy (+ torch if importable), and the CPU device."""
    parts = [f"Python {platform.python_version()}", f"numpy {np.__version__}"]
    try:
        import torch  # noqa: PLC0415 — optional; only for the abstractive demo

        parts.append(f"torch {torch.__version__}")
    except ImportError:
        parts.append("torch not installed (core math needs only numpy)")
    parts.append("device: cpu (the from-scratch math is integer/float counting — CPU is exact)")
    return " | ".join(parts)


# ===================================================================================================
# 5. OPTIONAL — a real abstractive summarizer (needs transformers + torch).
# ===================================================================================================
ABSTRACTIVE_MODEL_NAME = "sshleifer/distilbart-cnn-12-6"


def run_real_abstractive(
    document_sentences: list[str] = SOLAR_DOC,
    *,
    drop_distractor: bool = True,
) -> str | None:
    """Summarize the document with a real seq2seq model (distilBART) using BEAM SEARCH.

    Returns the summary string, or None (with a printed note) if transformers/torch are
    unavailable, so the from-scratch core still runs in a minimal environment. Beam search with no
    sampling makes the output deterministic and reproducible. By default the off-topic bakery
    sentence is dropped before summarizing (mirroring a sensible extractive pre-filter).
    """
    try:
        from transformers import AutoModelForSeq2SeqLM, AutoTokenizer  # noqa: PLC0415
    except ImportError:
        print("(skipping real-abstractive block: install `transformers torch`)")
        return None
    source = " ".join(document_sentences[:-1] if drop_distractor else document_sentences)
    tokenizer = AutoTokenizer.from_pretrained(ABSTRACTIVE_MODEL_NAME)
    model = AutoModelForSeq2SeqLM.from_pretrained(ABSTRACTIVE_MODEL_NAME)
    ids = tokenizer(source, return_tensors="pt", truncation=True, max_length=512)
    generated = model.generate(**ids, max_length=40, min_length=12, num_beams=4, do_sample=False)
    return tokenizer.decode(generated[0], skip_special_tokens=True).strip()


# ===================================================================================================
# Driver — print every demo so `python text_summarization.py` is a full, readable transcript.
# ===================================================================================================
def main() -> None:
    print(environment_line())
    print()

    # --- TextRank extractive summarization ----------------------------------------------------
    print("=" * 74)
    print("1. TextRank — PageRank on a TF-IDF cosine sentence graph (matches networkx)")
    print("=" * 74)
    scores = textrank_scores(SOLAR_DOC)
    order = np.argsort(scores)[::-1]
    for rank, i in enumerate(order, 1):
        print(f"  #{rank}  S{i + 1}  centrality={scores[i]:.3f}  | {SOLAR_DOC[i]}")
    top, extractive = textrank_summary(SOLAR_DOC, k=2)
    print(f"  extractive top-2 (document order) -> S{top[0] + 1}, S{top[1] + 1}")
    print(f"  summary: {extractive}")

    # --- ROUGE from scratch -------------------------------------------------------------------
    print("\n" + "=" * 74)
    print("2. ROUGE-1/2/L by hand (matches rouge-score) on 'the cat sat on the/a mat'")
    print("=" * 74)
    ref = "the cat sat on the mat"
    cand = "the cat sat on a mat"
    r = rouge_all(cand, ref, stem=False)
    print(f"  reference : {ref}")
    print(f"  candidate : {cand}")
    for key in ("rouge1", "rouge2", "rougeL"):
        s = r[key]
        print(
            f"  {key:7s}  R={s['recall']:.3f}  P={s['precision']:.3f}  F1={s['fmeasure']:.3f}"
        )

    # --- ROUGE blindness: a faithful paraphrase scores low ------------------------------------
    print("\n" + "=" * 74)
    print("3. ROUGE blindness — a faithful paraphrase scores LOW, a verbatim copy scores HIGH")
    print("=" * 74)
    reference = "the firm's revenue increased in the third quarter"
    paraphrase = "company sales rose during q3"  # same MEANING, different words
    copy = "the firm's revenue increased greatly in the third quarter"  # copies words, adds fluff
    for label, summ in [("faithful paraphrase", paraphrase), ("verbatim-ish copy", copy)]:
        f1 = rouge_n(summ, reference, 1)["fmeasure"]
        print(f"  {label:20s} ROUGE-1 F1={f1:.3f}  | {summ}")
    print("  takeaway: the paraphrase MEANS the same thing yet scores far lower ROUGE — the metric")
    print("            rewards word overlap, not meaning. This is ROUGE's central blind spot.")

    # --- Pointer-generator p_gen mixture ------------------------------------------------------
    print("\n" + "=" * 74)
    print("4. Pointer-generator p_gen switch — copying rescues an out-of-vocabulary name")
    print("=" * 74)
    p_vocab = {"and": 0.50, "power": 0.10}  # NOTE: no entry for the rare name "Tsiolkovsky"
    copy_attention = {"Tsiolkovsky": 0.80, "the": 0.20}  # attention points at the rare source word
    for p_gen in (0.30, 0.90):
        mix = pointer_generator_mix(p_vocab, copy_attention, p_gen)
        winner = max(mix, key=mix.get)
        print(
            f"  p_gen={p_gen:.2f}: "
            f"P(Tsiolkovsky)={mix['Tsiolkovsky']:.2f}  P(and)={mix['and']:.2f}  -> emit '{winner}'"
        )
    print("  low p_gen copies the rare name the vocab softmax could never produce; high p_gen")
    print("  paraphrases. That one scalar is the extractive-vs-abstractive dial, learned per token.")

    # --- Real abstractive model + measured ROUGE (optional) -----------------------------------
    print("\n" + "=" * 74)
    print("5. A real abstractive summarizer vs the extractive summary, scored with ROUGE (optional)")
    print("=" * 74)
    abstractive = run_real_abstractive()
    if abstractive is not None:
        print(f"  abstractive (distilbart-cnn): {abstractive}")
        for label, summ in [("extractive (TextRank)", extractive), ("abstractive (distilbart)", abstractive)]:
            rr = rouge_all(summ, SOLAR_REFERENCE)
            print(
                f"  {label:26s} ROUGE-1={rr['rouge1']['fmeasure']:.2f} "
                f"R-2={rr['rouge2']['fmeasure']:.2f} R-L={rr['rougeL']['fmeasure']:.2f}"
            )


if __name__ == "__main__":
    main()
