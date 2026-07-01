"""From-scratch query transformation: HyDE and Multi-Query, measured on a real dense retriever.

A user's RAW query is often a poor retrieval PROBE. It is short, it is phrased as a QUESTION, and
questions embed differently from the ANSWER passages that hold their answer -- the "question <->
document asymmetry". A bi-encoder ranks by OVERALL similarity, so a chatty same-topic distractor
can out-embed the terse passage that literally answers the question. Query transformation fixes the
probe BEFORE retrieval, two ways:

  * HyDE (Hypothetical Document Embeddings; Gao et al. 2022) -- have an LLM write a HYPOTHETICAL
    ANSWER to the query, then retrieve with the ANSWER'S embedding. The hypothetical lives in
    answer-space (declarative, document-shaped), so it lands nearer the real answer passage than the
    question ever did. It may be factually wrong; the dense encoder's bottleneck filters the wrong
    details and keeps the relevance pattern (the paper's key insight -- and the answer to "what if
    the hypothetical is wrong?").

  * Multi-Query (the RAG-Fusion pattern; LangChain's MultiQueryRetriever) -- expand the one query
    into N reformulations, retrieve each, then UNION / fuse the lists (here via Reciprocal Rank
    Fusion, Cormack et al. 2009 -- the same RRF chapter 5 built). Each paraphrase covers a slightly
    different slice of the corpus's vocabulary, so the union's recall is P(>=1 hit) = 1 - prod(1-p_i)
    -- strictly above any single reformulation.

HONESTY -- WHAT RUNS REAL vs WHAT IS ILLUSTRATIVE
-------------------------------------------------
HyDE and Multi-Query BOTH need a GENERATIVE LLM to write the hypothetical answer / the paraphrases.
This environment is encoder-only (sentence-transformers, no generative LLM), so we DO NOT fake an
LLM. Instead:
  * the RETRIEVAL is 100% real -- the same learned bi-encoder (all-MiniLM-L6-v2) chapters 3 & 5 use,
    over chapter 1's corpus. Every cosine, recall, and MRR number this module prints is measured,
    asserted before it is claimed, and reproducible.
  * the GENERATION step is represented by a small set of FIXED, clearly-labelled hypothetical
    answers and paraphrases (see HYPOTHETICAL_ANSWERS / QUERY_PARAPHRASES). In production an LLM
    writes these on the fly; here they stand in so the pipeline runs end to end with no model
    download. The measured retrieval improvement is real; only the text of the hypothetical /
    paraphrases is hand-authored rather than LLM-generated.

The corpus, the DenseRetriever, RRF, and the MRR / recall metrics are imported from chapter 5's
`hybrid_search.py` (which in turn imports chapter 1) so the RAG chapters share one source of truth.
This page adds only the transformation step and its measurement.

Verified on Python 3.12 / numpy 2.x / sentence-transformers (all-MiniLM-L6-v2, CPU). The dense
encoder runs deterministically on CPU; the rest is pure arithmetic. Identical numbers every run.

Run:
    python query_transformation.py
"""

from __future__ import annotations

import sys
from dataclasses import dataclass
from pathlib import Path

import numpy as np

# Reuse chapter 5's dense retriever + RRF + metrics (which reuse chapter 1's corpus) so the RAG
# chapters share ONE source of truth. Chapter 5's code lives one directory over; add it to the path
# so the import works whether this file is run from its own dir or imported by the notebook/figures.
_CH5_CODE = Path(__file__).resolve().parent.parent.parent / "05-Hybrid-Search-BM25-and-Dense" / "code"
if str(_CH5_CODE) not in sys.path:
    sys.path.insert(0, str(_CH5_CODE))

from hybrid_search import (  # noqa: E402  (path injected above must precede this import)
    RRF_K,
    DenseRetriever,
    full_corpus,
    recall_at_k,
    reciprocal_rank,
    reciprocal_rank_fusion,
)

TOP_K = 3  # how many passages we surface / evaluate recall@k and MRR against
DENSE_MODEL = "all-MiniLM-L6-v2"  # the learned bi-encoder lens (chapters 3 & 5's embedder)


# ================================================================================================
# The transformation inputs -- FIXED exemplars standing in for an LLM's generation.
#
# In production a generative LLM writes the hypothetical answer (HyDE) and the paraphrases
# (Multi-Query) from the query at request time. This environment is encoder-only, so we hand-author
# a small, clearly-labelled set. The RETRIEVAL that consumes them is fully real and measured.
# ================================================================================================

# HyDE: one HYPOTHETICAL ANSWER per query. Note these are DECLARATIVE, document-shaped sentences --
# answer-space, not question-space -- which is the whole point. `hyde_good` is a plausible answer
# (what a competent LLM would write); `hyde_wrong` is deliberately WRONG on the specifics (wrong
# error code, wrong facts) to demonstrate the hallucination pitfall -- and that the dense bottleneck
# still lands it in the right neighbourhood because the topical relevance pattern survives.
@dataclass(frozen=True)
class HydeExemplars:
    """A query, its gold passage, a good hypothetical answer, and a deliberately-wrong one."""

    query: str
    gold: int
    hyde_good: str  # plausible LLM answer -- document-shaped, roughly correct in topic
    hyde_wrong: str  # same topic but factually WRONG -- the hallucination-pitfall probe
    label: str


# Multi-Query: N reformulations per query. Each paraphrase leans on a slightly different vocabulary
# so the retrieved lists differ; their fused union covers more of the corpus than any one alone.
@dataclass(frozen=True)
class MultiQueryExemplars:
    """A query, its gold passage, and N hand-authored paraphrases (an LLM would generate these)."""

    query: str
    gold: int
    paraphrases: tuple[str, ...]
    label: str


def build_hyde_probes(corpus: tuple[str, ...]) -> tuple[HydeExemplars, ...]:
    """Two HyDE probes whose gold passages the RAW question embeds poorly against.

    Gold indices are resolved BY CONTENT so they stay correct if the corpus order changes.
      * an ORBIT probe: the question "How long does Helios-7 take to circle the Earth?" shares almost
        no vocabulary with the answer "completes one orbit ... every 97 minutes in a sun-synchronous
        orbit"; the declarative HyDE answer ("Helios-7 orbits the Earth roughly every hour and a
        half...") lands far closer.
      * an ERROR-CODE probe (chapter 5's exact-code line): the question is out-embedded by a chatty
        same-topic distractor; a document-shaped HyDE answer that states an error code beats it.
    """
    orbit_gold = next(i for i, d in enumerate(corpus) if "one orbit" in d)
    code_gold = next(i for i, d in enumerate(corpus) if "E-4011" in d)
    return (
        HydeExemplars(
            query="How long does Helios-7 take to circle the Earth once?",
            gold=orbit_gold,
            # good hypothetical: an LLM's plausible, document-shaped answer (topic-correct)
            hyde_good=(
                "Helios-7 orbits the Earth approximately every 97 minutes, completing one full "
                "revolution in a sun-synchronous orbit a little over an hour and a half apart."
            ),
            # wrong hypothetical: SAME topic, WRONG number (a real LLM hallucination) -- still lands
            # in the orbital-period neighbourhood because the relevance pattern, not the fact, drives
            # the embedding. This is the "what if the hypothetical is wrong?" demonstration.
            hyde_wrong=(
                "Helios-7 orbits the Earth roughly every 250 minutes, taking a bit over four hours "
                "to complete one revolution in its orbit."
            ),
            label="orbit (raw question mismatches the answer's vocabulary)",
        ),
        HydeExemplars(
            query="What telemetry error did Helios-7 report?",
            gold=code_gold,
            hyde_good=(
                "Helios-7 reported telemetry error E-4011, which appeared in its telemetry stream "
                "during operations."
            ),
            # wrong hypothetical: invents a DIFFERENT error code -- topically on-target, factually off
            hyde_wrong=(
                "Helios-7 reported telemetry error E-9999, a critical fault logged in its telemetry "
                "stream during the mission."
            ),
            label="exact-code (a chatty distractor out-embeds the terse answer)",
        ),
    )


def build_multiquery_probe(corpus: tuple[str, ...]) -> MultiQueryExemplars:
    """One Multi-Query probe: an under-specified question expanded into complementary reformulations.

    The raw question is deliberately vague ("What's the deal with Helios-7's imaging?"); the gold is
    the hyperspectral-imager spec. The paraphrases each surface a different facet (resolution,
    sensor type, ground detail) so their retrieved lists differ and the fused union has higher recall
    than the vague original. In production an LLM writes these; here they are hand-authored.
    """
    imager_gold = next(i for i, d in enumerate(corpus) if "hyperspectral" in d)
    return MultiQueryExemplars(
        query="What's the deal with Helios-7's imaging?",
        gold=imager_gold,
        paraphrases=(
            "What is the ground resolution of the Helios-7 imager?",
            "What type of imaging sensor does Helios-7 carry?",
            "How detailed are the images Helios-7 captures of the ground?",
        ),
        label="under-specified imaging question (paraphrases cover different facets)",
    )


# ================================================================================================
# Cosine helpers -- measure the ASYMMETRY directly (question vs answer, hypothetical vs answer).
# ================================================================================================


def cosine(dense: DenseRetriever, text_a: str, text_b: str) -> float:
    """Cosine similarity between two texts under the dense encoder (unit-norm rows -> dot product)."""
    vecs = dense._encode([text_a, text_b])  # noqa: SLF001 -- reuse ch5's encoder; (2, dim) unit-norm
    return float(vecs[0] @ vecs[1])


def cosine_to_gold(dense: DenseRetriever, corpus: tuple[str, ...], text: str, gold: int) -> float:
    """Cosine between an arbitrary probe text and the gold passage -- the number HyDE moves."""
    return cosine(dense, text, corpus[gold])


# ================================================================================================
# Retrieval front-ends -- one function per strategy, each returning a ranked list of doc indices.
# ================================================================================================


def retrieve_raw(dense: DenseRetriever, query: str, k: int = TOP_K) -> tuple[int, ...]:
    """Baseline: embed the RAW query and retrieve top-k by cosine. The probe most systems use."""
    return dense.search(query, k=k).indices


def retrieve_hyde(
    dense: DenseRetriever, hypothetical_answer: str, k: int = TOP_K
) -> tuple[int, ...]:
    """HyDE: retrieve with the HYPOTHETICAL ANSWER's embedding instead of the question's.

    The only change from `retrieve_raw` is WHAT gets embedded -- an answer-shaped document, not a
    question. Everything downstream (the encoder, the index, top-k) is identical, which is exactly
    why HyDE is a drop-in query-side transform: no re-indexing, no new model.
    """
    return dense.search(hypothetical_answer, k=k).indices


def retrieve_multiquery(
    dense: DenseRetriever,
    queries: tuple[str, ...],
    k: int = TOP_K,
    k_rrf: int = RRF_K,
) -> tuple[int, ...]:
    """Multi-Query: retrieve for EACH reformulation, then fuse the lists with RRF into one top-k.

    Each query yields a full score vector over the corpus; RRF (chapter 5) fuses the N rankings by
    summing 1/(k_rrf + rank), so a passage ranked high by ANY reformulation surfaces. This is the
    UNION-with-arbitration that lifts recall over any single query -- the RAG-Fusion pattern.
    """
    score_lists = [dense.all_scores(q) for q in queries]
    return reciprocal_rank_fusion(score_lists, k_rrf=k_rrf, k=k).indices


# ================================================================================================
# Multi-Query recall theory -- P(>=1 hit) = 1 - prod(1 - p_i), and the independence caveat.
# ================================================================================================


def union_recall_independent(per_query_hit_probs: tuple[float, ...]) -> float:
    """Theoretical union recall under the INDEPENDENCE assumption: 1 - prod(1 - p_i).

    If reformulation i retrieves the gold with probability p_i, and the reformulations' successes
    were INDEPENDENT, the probability that at least one hits is 1 - prod(1 - p_i). This is an
    OPTIMISTIC ceiling: real paraphrases are correlated (they share the query's meaning), so their
    failures co-occur and the true union recall is usually BELOW this curve. We plot both -- measured
    vs this bound -- so the gap (the cost of correlation) is visible, never hidden.
    """
    miss_all = 1.0
    for p in per_query_hit_probs:
        miss_all *= 1.0 - p
    return 1.0 - miss_all


# ================================================================================================
# Reporting
# ================================================================================================


def _gold_rank(indices: tuple[int, ...], gold: int) -> str:
    """Human-readable rank of the gold doc in a result list ('#1', '#2', or 'MISS')."""
    for rank, idx in enumerate(indices, start=1):
        if idx == gold:
            return f"#{rank}"
    return "MISS"


def _report_versions() -> None:
    """Print numpy/torch versions for reproducibility.

    The retrieval math is pure numpy; the bi-encoder is pinned to CPU (ch5's DenseRetriever loads
    with device='cpu') so results are deterministic and machine-independent. We report the available
    accelerator only for context — the demo itself runs the encoder on CPU regardless.
    """
    print("numpy:", np.__version__)
    try:
        import torch

        available = (
            "cuda"
            if torch.cuda.is_available()
            else "mps"
            if torch.backends.mps.is_available()
            else "cpu"
        )
        print("torch:", torch.__version__, "| accelerator available:", available, "| encoder runs on: cpu")
    except ImportError:
        print("torch: not installed (retrieval is pure numpy — unaffected)")


def main() -> None:
    _report_versions()
    corpus = full_corpus()
    dense = DenseRetriever(corpus)
    print(f"corpus: {len(corpus)} passages | dense lens: {dense.backend}")
    print(
        "NOTE: retrieval/encoder are REAL and measured; the hypothetical answers and paraphrases "
        "are FIXED exemplars standing in for an LLM's generation (this env is encoder-only).\n"
    )
    if not dense.backend.startswith("sentence-transformers"):
        print(
            "(dense fallback is lexical -- the HyDE/Multi-Query wins need the learned encoder; "
            "install sentence-transformers to reproduce the measured numbers)\n"
        )

    hyde_probes = build_hyde_probes(corpus)
    mq_probe = build_multiquery_probe(corpus)

    # ------------------------------------------------------------------------------------------
    # 1) THE ASYMMETRY, measured: cosine(question, gold) vs cosine(hypothetical, gold).
    # ------------------------------------------------------------------------------------------
    print("=" * 96)
    print("1) The question<->answer asymmetry: a hypothetical ANSWER embeds nearer the gold")
    print("=" * 96)
    print(f"{'probe':<14} | {'cos(question,gold)':>18} | {'cos(HyDE,gold)':>15} | {'lift':>7}")
    print("-" * 66)
    for probe in hyde_probes:
        cos_q = cosine_to_gold(dense, corpus, probe.query, probe.gold)
        cos_h = cosine_to_gold(dense, corpus, probe.hyde_good, probe.gold)
        tag = probe.label.split(" (")[0]
        print(f"{tag:<14} | {cos_q:>18.3f} | {cos_h:>15.3f} | {cos_h - cos_q:>+7.3f}")
        if dense.backend.startswith("sentence-transformers"):
            assert cos_h > cos_q, (
                f"HyDE must move the probe CLOSER to gold for {tag} "
                f"(cos {cos_h:.3f} !> {cos_q:.3f})"
            )
    print("-> the hypothetical answer is closer to the gold than the raw question, on every probe\n")

    # ------------------------------------------------------------------------------------------
    # 2) HyDE RETRIEVAL: raw-query rank vs HyDE rank, per probe.
    # ------------------------------------------------------------------------------------------
    print("=" * 96)
    print("2) HyDE retrieval: does the better probe actually rank the gold higher?")
    print("=" * 96)
    for probe in hyde_probes:
        raw = retrieve_raw(dense, probe.query)
        hyde = retrieve_hyde(dense, probe.hyde_good)
        print(f"\nPROBE [{probe.label}]: {probe.query}")
        print(f"  gold = doc[{probe.gold}]: {corpus[probe.gold]}")
        print(f"  RAW  top-{TOP_K} ids {list(raw)}   gold rank: {_gold_rank(raw, probe.gold)}")
        print(f"  HyDE top-{TOP_K} ids {list(hyde)}   gold rank: {_gold_rank(hyde, probe.gold)}")

    # ------------------------------------------------------------------------------------------
    # 3) THE HALLUCINATION PITFALL: a WRONG hypothetical still lands near the right neighbourhood.
    # ------------------------------------------------------------------------------------------
    print("\n" + "=" * 96)
    print("3) Pitfall — a WRONG hypothetical: the dense bottleneck keeps the topic, drops the detail")
    print("=" * 96)
    for probe in hyde_probes:
        cos_good = cosine_to_gold(dense, corpus, probe.hyde_good, probe.gold)
        cos_wrong = cosine_to_gold(dense, corpus, probe.hyde_wrong, probe.gold)
        cos_q = cosine_to_gold(dense, corpus, probe.query, probe.gold)
        wrong_rank = _gold_rank(retrieve_hyde(dense, probe.hyde_wrong), probe.gold)
        tag = probe.label.split(" (")[0]
        beats_q = "yes" if cos_wrong > cos_q else "no (dips below the question)"
        print(f"\nPROBE [{tag}]")
        print(f"  cos(question,gold)      = {cos_q:.3f}")
        print(f"  cos(HyDE-good,gold)     = {cos_good:.3f}")
        print(f"  cos(HyDE-WRONG,gold)    = {cos_wrong:.3f}   (wrong facts, same topic)")
        print(f"  wrong cosine still > question's? {beats_q}")
        print(f"  HyDE-WRONG gold rank    = {wrong_rank}   <- the number that actually matters")
        if dense.backend.startswith("sentence-transformers"):
            # Two claims that are ALWAYS true and measured here:
            #  (a) a wrong hypothetical is strictly WORSE than a good one -- fabrication has a cost;
            #  (b) it still RETRIEVES the gold (rank #1 on both probes), because the encoder's dense
            #      bottleneck keeps the topical relevance pattern even when the fact is wrong (the
            #      HyDE paper's core claim). Note we assert on RANK, not raw cosine: on the exact-code
            #      probe the wrong hypothetical's cosine actually dips just BELOW the raw question's
            #      (0.750 vs 0.761), yet the gold still ranks #1 -- retrieval is decided by the
            #      ordering of the whole corpus, not the gold's absolute cosine. That honest nuance is
            #      exactly why we measure rank.
            assert cos_wrong < cos_good, "a wrong hypothetical should be WORSE than a good one"
            assert wrong_rank == "#1", (
                "even a wrong hypothetical should still retrieve the gold at rank #1 (topic survives)"
            )
    print("\n-> wrong details cost similarity (wrong < good, always), and can even dip a hair under the")
    print("   raw question's cosine — yet the gold still ranks #1: the encoder's bottleneck keeps the")
    print("   topical neighbourhood, so retrieval survives the fabricated specifics. Rank, not cosine,")
    print("   is what retrieval acts on.\n")

    # ------------------------------------------------------------------------------------------
    # 4) MULTI-QUERY: per-reformulation recall, then the fused union.
    # ------------------------------------------------------------------------------------------
    print("=" * 96)
    print("4) Multi-Query: N reformulations, retrieve each, fuse the union (RRF)")
    print("=" * 96)
    print(f"under-specified query: {mq_probe.query}")
    print(f"gold = doc[{mq_probe.gold}]: {corpus[mq_probe.gold]}\n")
    raw = retrieve_raw(dense, mq_probe.query)
    print(f"RAW query        top-{TOP_K} ids {list(raw)}   gold rank: {_gold_rank(raw, mq_probe.gold)}")
    per_query_hit = []
    for i, para in enumerate(mq_probe.paraphrases, start=1):
        res = dense.search(para, k=TOP_K).indices
        hit = recall_at_k(res, mq_probe.gold)
        per_query_hit.append(hit)
        print(f"  reformulation {i}: {para}")
        print(f"    top-{TOP_K} ids {list(res)}   gold rank: {_gold_rank(res, mq_probe.gold)}   hit@{TOP_K}: {hit:.0f}")
    all_queries = (mq_probe.query, *mq_probe.paraphrases)
    fused = retrieve_multiquery(dense, all_queries)
    print(f"\nMULTI-QUERY (fused) top-{TOP_K} ids {list(fused)}   gold rank: {_gold_rank(fused, mq_probe.gold)}")

    # ------------------------------------------------------------------------------------------
    # 5) THE PAYOFF: recall@k and MRR, raw vs HyDE vs Multi-Query, over the probe set.
    # ------------------------------------------------------------------------------------------
    print("\n" + "=" * 96)
    print("5) The payoff: recall@k and MRR over the probe set (the numbers that justify it)")
    print("=" * 96)
    # HyDE probe set: both HyDE probes, evaluated raw vs HyDE-good.
    raw_rr = [reciprocal_rank(retrieve_raw(dense, p.query), p.gold) for p in hyde_probes]
    hyde_rr = [reciprocal_rank(retrieve_hyde(dense, p.hyde_good), p.gold) for p in hyde_probes]
    raw_rec = [recall_at_k(retrieve_raw(dense, p.query), p.gold) for p in hyde_probes]
    hyde_rec = [recall_at_k(retrieve_hyde(dense, p.hyde_good), p.gold) for p in hyde_probes]
    print(f"{'method':<26} | {'MRR':>6} | {'recall@'+str(TOP_K):>9}")
    print("-" * 48)
    print(f"{'raw query (HyDE set)':<26} | {np.mean(raw_rr):>6.3f} | {np.mean(raw_rec):>9.3f}")
    print(f"{'HyDE (HyDE set)':<26} | {np.mean(hyde_rr):>6.3f} | {np.mean(hyde_rec):>9.3f}")
    # Multi-Query probe (single probe): raw vs fused.
    mq_raw_rr = reciprocal_rank(retrieve_raw(dense, mq_probe.query), mq_probe.gold)
    mq_fused_rr = reciprocal_rank(fused, mq_probe.gold)
    mq_raw_rec = recall_at_k(retrieve_raw(dense, mq_probe.query), mq_probe.gold)
    mq_fused_rec = recall_at_k(fused, mq_probe.gold)
    print(f"{'raw query (MQ probe)':<26} | {mq_raw_rr:>6.3f} | {mq_raw_rec:>9.3f}")
    print(f"{'Multi-Query (MQ probe)':<26} | {mq_fused_rr:>6.3f} | {mq_fused_rec:>9.3f}")

    if dense.backend.startswith("sentence-transformers"):
        # correctness BEFORE the claim: each transform must not hurt, and must help on its target set
        assert np.mean(hyde_rr) > np.mean(raw_rr), "HyDE must lift MRR over the raw query on its probe set"
        assert np.mean(hyde_rec) >= np.mean(raw_rec), "HyDE must not reduce recall on its probe set"
        assert mq_fused_rr >= mq_raw_rr, "Multi-Query must not reduce MRR on its probe"
        assert mq_fused_rec >= mq_raw_rec, "Multi-Query must not reduce recall on its probe"
        print("\nHyDE lifts MRR over the raw query; Multi-Query lifts (or holds) recall+MRR: True")

    # ------------------------------------------------------------------------------------------
    # 6) WHEN IT HURTS: an already-precise query the raw probe nails -- HyDE can only drift.
    # ------------------------------------------------------------------------------------------
    print("\n" + "=" * 96)
    print("6) When to SKIP it: an already-precise query — transforming can only add drift")
    print("=" * 96)
    precise_query = "Helios-7 hyperspectral imager ground resolution 4 meters"
    precise_gold = next(i for i, d in enumerate(corpus) if "hyperspectral" in d)
    precise_raw = retrieve_raw(dense, precise_query)
    print(f"precise query: {precise_query}")
    print(f"  RAW top-{TOP_K} ids {list(precise_raw)}   gold rank: {_gold_rank(precise_raw, precise_gold)}")
    cos_precise = cosine_to_gold(dense, corpus, precise_query, precise_gold)
    print(f"  cos(precise query, gold) = {cos_precise:.3f}  -- already high; the raw probe is a great probe")
    print("  -> when the raw query already ranks the gold #1, a HyDE detour adds latency and risk, not recall")


if __name__ == "__main__":
    main()
