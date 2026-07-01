"""From-scratch RAG evaluation: measure retrieval AND generation, the RAGAS way.

You built a RAG pipeline -- but is it GOOD? "Seems fine" is not a number, and a one-line change to
the chunker or the prompt can silently regress it. RAG has TWO failure surfaces, and you must
measure each independently:

  * RETRIEVAL -- did we fetch the right context? (context precision: are the relevant chunks ranked
    high; context recall: did we get all the relevant chunks at all).
  * GENERATION -- given that context, is the answer FAITHFUL to it (every claim supported, no
    hallucination) and RELEVANT (does it actually answer the question)?

The RAGAS triad names the generation side: FAITHFULNESS (answer grounded in context), ANSWER
RELEVANCE (answer addresses the question), CONTEXT RELEVANCE (retrieved context is on-topic). This
module computes all of them from primitives, then shows the headline contrast: a FAITHFUL answer
scores high faithfulness; a fluent-but-UNFAITHFUL answer (it adds a claim not in the context) scores
low -- measurably worse, even though both read fluently.

HONESTY -- WHAT RUNS REAL vs WHAT IS ILLUSTRATIVE
-------------------------------------------------
  * REAL and measured: ALL retrieval metrics (context precision@k, context recall, MRR, nDCG@k --
    computed on real ranked retrieval from ch5's all-MiniLM DenseRetriever, reusing ch6's ranking
    metrics); the SUPPORT / faithfulness signal (max cosine of each answer-claim vs each context
    sentence, thresholded -- ch8's real encoder-cosine groundedness proxy); and ANSWER / CONTEXT
    RELEVANCE (real embedding cosine). Every score printed here is measured, asserted before it is
    claimed, and reproducible.
  * ILLUSTRATIVE (labelled): in production an LLM does CLAIM DECOMPOSITION (split an answer into
    atomic claims) and acts as the JUDGE (decide support / relevance). This env is encoder-only, so
    claim-splitting is a transparent rule-based sentence splitter and the "judge" is the cosine
    threshold -- both clearly labelled. The mechanism (decompose -> check each claim -> aggregate)
    is demonstrated with real numbers; only the extractor/judge are stand-ins for an LLM.

CARRIED-FORWARD CAVEAT (ch8, still true): the cosine support proxy measures TOPICAL similarity, not
factual ENTAILMENT. A hallucinated *number* (a "2 meters" swapped for "4 meters") is topically
almost identical to the truth, so it slips past a raw-cosine gate -- shown explicitly in the pitfalls
demo. That is exactly why RAGAS uses an LLM judge (an NLI-style entailment decision), which the
cosine proxy only approximates.

The dense encoder + corpus are imported from ch5's hybrid_search (which imports ch1), and the
ranking metrics from ch6's reranking, so the RAG chapters share ONE source of truth.

Verified on Python 3.12.x / numpy 2.4.6 / torch 2.12.0 / sentence-transformers (all-MiniLM-L6-v2, CPU). Deterministic:
identical numbers every run given the same cached model.

Run:
    python rag_evaluation.py
"""

from __future__ import annotations

import re
import sys
from dataclasses import dataclass
from pathlib import Path

import numpy as np

# Reuse ch5's dense retriever + corpus and ch6's ranking metrics (each injects its own path
# transitively). ch5 and ch6 live two directories over; inject their code dirs so imports work
# whether this file is run from its own dir or imported by the notebook / figure scripts.
_CH5_CODE = Path(__file__).resolve().parent.parent.parent / "05-Hybrid-Search-BM25-and-Dense" / "code"
_CH6_CODE = Path(__file__).resolve().parent.parent.parent / "06-Re-ranking-Cross-Encoders" / "code"
for _p in (_CH5_CODE, _CH6_CODE):
    if str(_p) not in sys.path:
        sys.path.insert(0, str(_p))

from hybrid_search import (  # noqa: E402  (paths injected above must precede these imports)
    DenseRetriever,
    full_corpus,
)
from reranking import ndcg_at_k, reciprocal_rank  # noqa: E402  (reuse ch6's ranking metrics)

DENSE_MODEL = "all-MiniLM-L6-v2"  # the learned bi-encoder (ch3/5's embedder)

# ---- Support threshold: a claim is "supported" if its max cosine to any context sentence clears
# this bar. 0.5 is ch8's deliberate middle bar on unit-norm all-MiniLM cosines: paraphrase-level
# support (~0.6-0.9) clears it, unrelated text (~0.0-0.3) does not. It is a computable PROXY for a
# faithfulness JUDGE, not a trained entailment model -- see the module banner + the pitfalls demo.
SUPPORT_THRESHOLD = 0.5

_SENT_SPLIT_RE = re.compile(r"(?<=[.!?])\s+")  # split on sentence-ending punctuation + following space


# ================================================================================================
# Shared helpers: sentence splitting and the encoder-cosine support signal (ch8's real proxy)
# ================================================================================================


def split_into_claims(text: str) -> list[str]:
    """Split an answer into atomic CLAIMS -- here, one claim per sentence.

    ILLUSTRATIVE stand-in: in production an LLM extracts atomic factual claims (which can split a
    single sentence into several, or merge clauses). We use a transparent sentence splitter so the
    DECOMPOSE -> check -> aggregate mechanism is visible and reproducible with no LLM. The per-claim
    support check that follows is REAL (encoder cosine); only this extraction step is a stand-in.
    """
    flat = re.sub(r"\s+", " ", text).strip()
    return [s.strip() for s in _SENT_SPLIT_RE.split(flat) if s.strip()]


def claim_support(dense: DenseRetriever, claim: str, context: str) -> float:
    """Max cosine between one answer CLAIM and any sentence of the retrieved CONTEXT (ch8's proxy).

    REAL and measured: if the claim paraphrases something actually in the context, its best-matching
    context sentence scores high; if the claim is unsupported (a hallucination on a NEW topic), no
    context sentence matches and the score stays low. The trained faithfulness judge is illustrative
    here (see banner), but THIS grounding signal is a real encoder cosine.
    """
    context_sents = split_into_claims(context)
    if not context_sents:
        return 0.0
    claim_vec = dense._encode([claim])[0]  # noqa: SLF001 -- reuse ch5's encoder; unit-norm
    ctx_vecs = dense._encode(context_sents)  # (n_sents, dim) unit-norm
    return float(np.max(ctx_vecs @ claim_vec))  # best cosine to any context sentence


# ================================================================================================
# GENERATION metric 1 -- FAITHFULNESS = (# supported claims) / (# claims)
# ================================================================================================


@dataclass(frozen=True)
class FaithfulnessResult:
    """A faithfulness score plus the per-claim support breakdown that produced it."""

    score: float  # supported / total, in [0, 1]
    claims: tuple[str, ...]
    supports: tuple[float, ...]  # per-claim max-cosine support (parallel to claims)
    supported: tuple[bool, ...]  # per-claim supported? (support >= threshold)


def faithfulness(
    dense: DenseRetriever, answer: str, context: str, threshold: float = SUPPORT_THRESHOLD
) -> FaithfulnessResult:
    """Faithfulness of an ANSWER to its retrieved CONTEXT: the fraction of claims that are supported.

    Decompose the answer into claims, score each claim's support against the context (real cosine),
    count how many clear the threshold, and divide by the number of claims. Score 1.0 = every claim
    grounded; a hallucinated claim (unsupported) drags the fraction down. This is the RAGAS
    faithfulness definition (supported-claims / total-claims), computed with the encoder-cosine proxy
    standing in for the LLM judge.
    """
    claims = tuple(split_into_claims(answer))
    if not claims:
        return FaithfulnessResult(0.0, (), (), ())
    supports = tuple(claim_support(dense, c, context) for c in claims)
    supported = tuple(s >= threshold for s in supports)
    score = sum(supported) / len(claims)
    return FaithfulnessResult(score, claims, supports, supported)


# ================================================================================================
# GENERATION metric 2 -- ANSWER RELEVANCE (does the answer address the QUESTION?)
# ================================================================================================


def answer_relevance(dense: DenseRetriever, question: str, answer: str) -> float:
    """How well the ANSWER addresses the QUESTION -- cosine of their embeddings.

    RAGAS computes answer relevance by having an LLM generate N questions FROM the answer and
    averaging their cosine to the original question (an answer that addresses the question yields
    questions like it). ILLUSTRATIVE simplification (labelled): with no generator LLM we use the
    direct question<->answer cosine, which captures the same signal -- a faithful-but-off-topic
    answer (grounded, but about the wrong thing) scores LOW even though its faithfulness is high.
    That separation is the whole point: relevance is a SEPARATE axis from faithfulness.
    """
    q_vec = dense._encode([question])[0]  # noqa: SLF001 -- unit-norm
    a_vec = dense._encode([answer])[0]  # noqa: SLF001
    return float(q_vec @ a_vec)


# ================================================================================================
# GENERATION metric 3 -- CONTEXT RELEVANCE (is the retrieved context on-topic for the question?)
# ================================================================================================


def context_relevance(dense: DenseRetriever, question: str, context_chunks: tuple[str, ...]) -> float:
    """How on-topic the retrieved CONTEXT is for the QUESTION -- mean cosine of each chunk to the query.

    REAL and measured. Low context relevance means the retriever pulled off-topic chunks (a retrieval
    problem), which starves generation of signal and often forces a hallucination. Averaging over the
    retrieved chunks gives one number for "how good was the context we handed the generator?".
    """
    if not context_chunks:
        return 0.0
    q_vec = dense._encode([question])[0]  # noqa: SLF001
    chunk_vecs = dense._encode(list(context_chunks))  # (k, dim) unit-norm
    return float(np.mean(chunk_vecs @ q_vec))


# ================================================================================================
# RETRIEVAL metrics -- context precision@k and context recall (RAGAS retrieval side)
# ================================================================================================


def context_precision_at_k(ranked: tuple[int, ...], relevant: frozenset[int], k: int) -> float:
    """RAGAS context precision@k: mean of precision@i over the ranks (<=k) that hold a relevant chunk.

    For each rank i in 1..k, precision@i = (# relevant in the top i) / i. Context precision@k averages
    precision@i only at the ranks where a relevant chunk actually appears (so ranking relevant chunks
    HIGHER scores better -- a relevant chunk at rank 1 lifts every later precision@i). If no relevant
    chunk is in the top-k the score is 0. This rewards putting the useful context first, which matters
    because generators attend most to the top of the context window.
    """
    hits = 0
    precision_sum = 0.0
    for i, doc in enumerate(ranked[:k], start=1):
        if doc in relevant:
            hits += 1
            precision_sum += hits / i  # precision@i at this relevant-chunk rank
    n_relevant_in_top_k = sum(1 for d in ranked[:k] if d in relevant)
    if n_relevant_in_top_k == 0:
        return 0.0
    return precision_sum / n_relevant_in_top_k


def context_recall(ranked: tuple[int, ...], relevant: frozenset[int], k: int) -> float:
    """Context recall@k: fraction of the RELEVANT chunks that appear in the top-k.

    Recall answers "did we even retrieve the context we needed?" -- the ceiling on everything
    downstream, because a chunk that was never retrieved cannot ground the answer no matter how good
    generation is. 1.0 means every relevant chunk was retrieved into the top-k; 0.5 means half were
    missed (an upstream retrieval failure the generator cannot fix).
    """
    if not relevant:
        return 0.0
    retrieved_relevant = sum(1 for d in ranked[:k] if d in relevant)
    return retrieved_relevant / len(relevant)


# ================================================================================================
# The evaluation harness: one (question, answer, context, relevant-set) sample -> all metrics
# ================================================================================================


@dataclass(frozen=True)
class EvalSample:
    """One golden evaluation record: a question, its relevant chunk indices, and an answer to grade."""

    question: str
    answer: str
    relevant: frozenset[int]  # indices (into the corpus) of the chunks that SHOULD be retrieved
    label: str


@dataclass(frozen=True)
class EvalReport:
    """All metrics for one sample: retrieval (precision/recall/MRR/nDCG) + generation (the triad)."""

    retrieved: tuple[int, ...]
    context_precision: float
    context_recall: float
    mrr: float
    ndcg: float
    context_relevance: float
    faithfulness: FaithfulnessResult
    answer_relevance: float


def evaluate_sample(
    dense: DenseRetriever, corpus: tuple[str, ...], sample: EvalSample, k: int = 3
) -> EvalReport:
    """Retrieve for the question, then compute every retrieval + generation metric for this sample.

    Retrieval metrics run on the REAL ranked retrieval; generation metrics run on the answer vs the
    retrieved context. The single first relevant chunk is used as the MRR/nDCG "gold" (both are
    single-gold metrics here); precision/recall use the full relevant set.
    """
    ranked = dense.search(sample.question, k=len(corpus)).indices  # full ranking, for the metrics
    top_k = ranked[:k]
    context_chunks = tuple(corpus[i] for i in top_k)
    context_text = " ".join(context_chunks)
    first_gold = next(iter(sample.relevant))  # a single-gold anchor for MRR / nDCG
    return EvalReport(
        retrieved=top_k,
        context_precision=context_precision_at_k(ranked, sample.relevant, k),
        context_recall=context_recall(ranked, sample.relevant, k),
        mrr=reciprocal_rank(ranked, first_gold),
        ndcg=ndcg_at_k(ranked, first_gold, k),
        context_relevance=context_relevance(dense, sample.question, context_chunks),
        faithfulness=faithfulness(dense, sample.answer, context_text),
        answer_relevance=answer_relevance(dense, sample.question, sample.answer),
    )


# ================================================================================================
# The demo answers: a FAITHFUL one and a fluent-but-UNFAITHFUL one, over the same context.
# ================================================================================================

QUESTION = "What is the ground resolution of the Helios-7 imager, and when was it launched?"

# Both answers read fluently and share two TRUE claims; the unfaithful one appends a hallucinated
# claim (a fact absent from the corpus) -- the difference faithfulness is designed to catch.
FAITHFUL_ANSWER = (
    "The Helios-7 imager has a ground resolution of 4 meters. "
    "Helios-7 launched on March 3rd, 2024 from the Kourou spaceport."
)
UNFAITHFUL_ANSWER = (
    "The Helios-7 imager has a ground resolution of 4 meters. "
    "Helios-7 launched on March 3rd, 2024 from the Kourou spaceport. "
    "Helios-7 is powered entirely by solar panels."  # hallucinated -- NOT in the corpus
)

# A faithful-but-IRRELEVANT answer for the answer-relevance demo: every claim is grounded, but it
# answers the WRONG question (it talks about the project lead, not the imager resolution asked).
IRRELEVANT_ANSWER = "The project lead for Helios-7 is Dr. Amara Okoye, based in the Nairobi office."


def _report_versions() -> None:
    """Print numpy/torch versions + the detected accelerator (the encoder is CPU-pinned, ch5's loader)."""
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


def _print_faithfulness(result: FaithfulnessResult) -> None:
    """Pretty-print the per-claim support breakdown behind a faithfulness score."""
    for claim, support, ok in zip(result.claims, result.supports, result.supported):
        verdict = "SUPPORTED" if ok else "UNSUPPORTED"
        print(f"    [{verdict:<11} {support:.3f}] {claim}")


def main() -> None:
    _report_versions()
    corpus = full_corpus()
    dense = DenseRetriever(corpus)
    print(f"corpus: {len(corpus)} passages | dense lens: {dense.backend} | support threshold: {SUPPORT_THRESHOLD}")
    print(
        "NOTE: ALL retrieval metrics + support cosines + relevance sims are REAL and measured; "
        "the claim-splitter and the faithfulness/relevance JUDGE are illustrative stand-ins for an LLM.\n"
    )

    # ------------------------------------------------------------------------------------------
    # 1) THE HEADLINE: a FAITHFUL answer vs a fluent-but-UNFAITHFUL one, scored on faithfulness.
    # ------------------------------------------------------------------------------------------
    print("=" * 96)
    print("1) Faithfulness: a fluent answer that HALLUCINATES a claim scores measurably lower")
    print("=" * 96)
    # retrieve the context both answers are graded against (top-3 for the imager+launch question)
    ranked = dense.search(QUESTION, k=len(corpus)).indices
    context_chunks = tuple(corpus[i] for i in ranked[:3])
    context_text = " ".join(context_chunks)
    print(f"question: {QUESTION}")
    print("retrieved context (top-3):")
    for i in ranked[:3]:
        print(f"    doc[{i}]: {corpus[i]}")
    faithful = faithfulness(dense, FAITHFUL_ANSWER, context_text)
    unfaithful = faithfulness(dense, UNFAITHFUL_ANSWER, context_text)
    print(f"\n  FAITHFUL answer   -> faithfulness = {faithful.score:.3f}  ({sum(faithful.supported)}/{len(faithful.claims)} claims supported)")
    _print_faithfulness(faithful)
    print(f"\n  UNFAITHFUL answer -> faithfulness = {unfaithful.score:.3f}  ({sum(unfaithful.supported)}/{len(unfaithful.claims)} claims supported)")
    _print_faithfulness(unfaithful)
    # Correctness BEFORE the claim: the faithful answer is fully grounded; the hallucinated claim
    # drags the unfaithful answer's score below it.
    assert faithful.score == 1.0, "every claim of the faithful answer must be supported (faithfulness 1.0)"
    assert unfaithful.score < faithful.score, "the hallucinated claim must lower faithfulness"
    assert not unfaithful.supported[-1], "the 'solar panels' claim is NOT in the context -> unsupported"
    print(f"\n  -> both answers read fluently, but faithfulness catches the hallucination: {faithful.score:.2f} vs {unfaithful.score:.2f}.\n")

    # ------------------------------------------------------------------------------------------
    # 2) ANSWER RELEVANCE: a FAITHFUL answer can still be USELESS if it answers the wrong question.
    # ------------------------------------------------------------------------------------------
    print("=" * 96)
    print("2) Answer relevance: a grounded answer to the WRONG question scores low (a separate axis)")
    print("=" * 96)
    rel_relevant = answer_relevance(dense, QUESTION, FAITHFUL_ANSWER)
    rel_irrelevant = answer_relevance(dense, QUESTION, IRRELEVANT_ANSWER)
    # the irrelevant answer is fully grounded (every claim is a real corpus fact) -- prove it first
    irrelevant_faith = faithfulness(dense, IRRELEVANT_ANSWER, " ".join(corpus))
    print(f"  question: {QUESTION}")
    print(f"  on-topic answer   : {FAITHFUL_ANSWER[:60]}...")
    print(f"    answer relevance = {rel_relevant:.3f}")
    print(f"  faithful-but-off-topic answer: {IRRELEVANT_ANSWER}")
    print(f"    faithfulness     = {irrelevant_faith.score:.3f}  (grounded -- it IS a real corpus fact)")
    print(f"    answer relevance = {rel_irrelevant:.3f}  (but it does NOT answer the question)")
    assert irrelevant_faith.score == 1.0, "the off-topic answer is still fully grounded (faithfulness 1.0)"
    assert rel_relevant > rel_irrelevant, "the on-topic answer must be more relevant than the off-topic one"
    print(f"\n  -> faithful ({irrelevant_faith.score:.2f}) but irrelevant ({rel_irrelevant:.3f}): why relevance is a SEPARATE axis from faithfulness.\n")

    # ------------------------------------------------------------------------------------------
    # 3) RETRIEVAL METRICS: context precision@k / recall, good retrieval vs bad.
    # ------------------------------------------------------------------------------------------
    print("=" * 96)
    print("3) Retrieval metrics: context precision@k and recall on a good vs a bad ranking")
    print("=" * 96)
    # relevant set for a two-fact question = the imager chunk (1) and the launch chunk (0)
    relevant = frozenset({0, 1})
    k = 3
    good_ranking = dense.search(QUESTION, k=len(corpus)).indices  # the REAL ranking
    # a deliberately BAD ranking: bury both relevant chunks below the top-3 (distractors first)
    distractors = tuple(i for i in range(len(corpus)) if i not in relevant)
    bad_ranking = distractors[:3] + tuple(sorted(relevant)) + distractors[3:]
    print(f"  relevant chunks for the question: {sorted(relevant)} (imager + launch)")
    print(f"  GOOD ranking (real retrieval) top-{k}: {list(good_ranking[:k])}")
    print(f"    context precision@{k} = {context_precision_at_k(good_ranking, relevant, k):.3f}")
    print(f"    context recall@{k}    = {context_recall(good_ranking, relevant, k):.3f}")
    print(f"  BAD ranking (relevant buried) top-{k}: {list(bad_ranking[:k])}")
    print(f"    context precision@{k} = {context_precision_at_k(bad_ranking, relevant, k):.3f}")
    print(f"    context recall@{k}    = {context_recall(bad_ranking, relevant, k):.3f}")
    good_prec = context_precision_at_k(good_ranking, relevant, k)
    bad_prec = context_precision_at_k(bad_ranking, relevant, k)
    assert good_prec > bad_prec, "the good ranking must have higher context precision than the buried one"
    assert context_recall(bad_ranking, relevant, k) == 0.0, "the bad ranking retrieves NO relevant chunk in top-3"
    print(f"\n  -> precision/recall localize the failure to RETRIEVAL: good {good_prec:.2f} vs buried {bad_prec:.2f}.\n")

    # ------------------------------------------------------------------------------------------
    # 4) CONTEXT RELEVANCE: the third leg of the RAG triad (how on-topic is the retrieved context?).
    # ------------------------------------------------------------------------------------------
    print("=" * 96)
    print("4) Context relevance: how on-topic is the retrieved context for the question? (RAG triad leg 3)")
    print("=" * 96)
    good_context = tuple(corpus[i] for i in good_ranking[:k])
    off_topic_context = (corpus[4], corpus[5], corpus[6])  # photosynthesis / Eiffel Tower / chessboard
    cr_good = context_relevance(dense, QUESTION, good_context)
    cr_bad = context_relevance(dense, QUESTION, off_topic_context)
    print(f"  on-topic context (real top-{k}) : context relevance = {cr_good:.3f}")
    print(f"  off-topic context (unrelated)   : context relevance = {cr_bad:.3f}")
    assert cr_good > cr_bad, "on-topic context must score higher context relevance than unrelated text"
    print(f"\n  -> the RAG triad = context relevance ({cr_good:.2f}) + faithfulness + answer relevance, each a separate lens.\n")

    # ------------------------------------------------------------------------------------------
    # 5) THE PITFALL (ch8 caveat, still true): cosine support ~ topical, not ENTAILMENT.
    # ------------------------------------------------------------------------------------------
    print("=" * 96)
    print("5) Pitfall: the cosine support proxy measures TOPIC, not ENTAILMENT (a number-swap slips past)")
    print("=" * 96)
    true_claim = "The Helios-7 imager has a ground resolution of 4 meters."
    swap_claim = "The Helios-7 imager has a ground resolution of 2 meters."  # same shape, WRONG number
    imager_ctx = corpus[1]  # the true 4-meter passage
    s_true = claim_support(dense, true_claim, imager_ctx)
    s_swap = claim_support(dense, swap_claim, imager_ctx)
    print(f"  context: {imager_ctx}")
    print(f"  TRUE claim (4 meters)     -> support {s_true:.3f}  (>= {SUPPORT_THRESHOLD}: {s_true >= SUPPORT_THRESHOLD})")
    print(f"  SWAPPED claim (2 meters)  -> support {s_swap:.3f}  (>= {SUPPORT_THRESHOLD}: {s_swap >= SUPPORT_THRESHOLD})")
    assert s_swap >= SUPPORT_THRESHOLD, "the number-swap claim SLIPS PAST the cosine bar (a real limit)"
    assert s_true > s_swap, "the true claim still scores higher than the swapped one"
    print(f"\n  -> the false '2 meters' claim scores {s_swap:.3f} and CLEARS the {SUPPORT_THRESHOLD} bar: cosine ~ topical, not factual.")
    print("     This is why RAGAS uses an LLM judge (NLI-style entailment), which the cosine proxy only approximates.")

    # ------------------------------------------------------------------------------------------
    # 6) ONE GOLDEN RECORD -> ALL METRICS: the harness that a real eval set runs per sample.
    # ------------------------------------------------------------------------------------------
    print("\n" + "=" * 96)
    print("6) The harness: one golden record (question, answer, relevant set) -> a full EvalReport")
    print("=" * 96)
    sample = EvalSample(
        question=QUESTION,
        answer=FAITHFUL_ANSWER,
        relevant=frozenset({0, 1}),  # imager + launch chunks
        label="imager+launch, faithful answer",
    )
    report = evaluate_sample(dense, corpus, sample, k=3)
    print(f"  sample: {sample.label}")
    print(f"  retrieved top-3       : {list(report.retrieved)}")
    print(f"  context precision@3   : {report.context_precision:.3f}")
    print(f"  context recall@3      : {report.context_recall:.3f}")
    print(f"  MRR (first-gold)      : {report.mrr:.3f}")
    print(f"  nDCG@3 (first-gold)   : {report.ndcg:.3f}")
    print(f"  context relevance     : {report.context_relevance:.3f}")
    print(f"  faithfulness          : {report.faithfulness.score:.3f}")
    print(f"  answer relevance      : {report.answer_relevance:.3f}")
    # the report must agree with the individually-computed numbers from sections 1-4 above
    assert report.context_precision == good_prec, "report precision must match the section-3 number"
    assert report.context_recall == 1.0, "the faithful sample's relevant chunks are both in the top-3"
    assert report.faithfulness.score == faithful.score, "report faithfulness must match section 1"
    assert report.answer_relevance == rel_relevant, "report answer relevance must match section 2"
    assert report.context_relevance == cr_good, "report context relevance must match section 4"
    assert report.mrr == 0.5 and abs(report.ndcg - 0.631) < 1e-3, "single-gold MRR/nDCG for gold at rank 2"
    print("\n  -> one record -> every retrieval + generation metric, all agreeing with the per-metric calls above.")


if __name__ == "__main__":
    main()
