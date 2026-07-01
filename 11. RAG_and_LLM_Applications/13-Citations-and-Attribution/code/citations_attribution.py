"""From-scratch post-hoc citation & attribution for RAG: attach every claim to its source.

A RAG answer can *sound* authoritative and still be unverifiable: is a claim actually in the
retrieved passages, or did the model invent it? Without citations a fluent WRONG answer is
indistinguishable from a right one. This module makes each claim CHECKABLE. It decomposes an
answer into claims, attributes each claim to the retrieved passage that best supports it, assigns
an inline citation, and MEASURES the result -- citation precision and recall -- while flagging any
claim with NO supporting passage (uncitable -> the likely hallucination).

The headline demonstration: over one answer, the two grounded claims each get a valid citation to
the passage that entails them; the hallucinated claim ("powered by solar panels") matches NO
retrieved passage, clears no support bar, and is flagged UNCITABLE -- exactly the claim you cannot
trust. The cited claims check out; the fabricated one does not.

TWO ATTRIBUTION REGIMES
-----------------------
  * POST-HOC (what this module implements, from primitives): generate first, then map each claim
    back to the retrieved context after the fact. Works on ANY model's output -- no special
    decoding -- which is why it is the honest thing to build in an encoder-only environment.
  * GENERATION-TIME (labelled illustrative, shown as a library one-liner on the page): the model
    emits [1][2] AS it writes, so the citation is part of decoding (Anthropic Citations API,
    Vertex grounding, LlamaIndex CitationQueryEngine). We do NOT fake a generator here; we point
    at the real APIs and contrast the two regimes in prose.

HONESTY -- WHAT RUNS REAL vs WHAT IS ILLUSTRATIVE
-------------------------------------------------
  * REAL and measured: the ATTRIBUTION MATCHING (each claim's max cosine to each retrieved passage,
    via ch5's all-MiniLM DenseRetriever -- the same encoder ch8/ch11 use for their support proxy),
    the citation ASSIGNMENT (argmax passage above the support bar, else UNCITABLE), and the
    CITATION PRECISION / RECALL numbers. Every printed score is computed here, asserted before it
    is claimed, and reproducible.
  * ILLUSTRATIVE (labelled): in production an LLM does CLAIM DECOMPOSITION (split an answer into
    atomic claims) and GENERATION-TIME citation. This env is encoder-only, so claim-splitting is a
    transparent rule-based sentence splitter and there is no generator. The mechanism (decompose ->
    match to context -> assign / flag) is demonstrated with real numbers; only the extractor and
    the generator are stand-ins for an LLM.

CARRIED-FORWARD CAVEAT (ch8/ch11, still true): the cosine support proxy measures TOPICAL
similarity, not factual ENTAILMENT. A passage topically near a claim it does NOT actually support
scores high on cosine, so the matcher can assign a FALSE CITATION -- shown explicitly in the
`false_citation_demo`. That gap is precisely why ALCE (Gao et al. 2023) and AIS (Rashkin et al.
2021 / Bohnet et al. 2022) define citation quality with an NLI ENTAILMENT model, not cosine. The
cosine matcher approximates that entailment check; it does not replace it.

The dense encoder + corpus are imported from ch5's hybrid_search (which imports ch1); the sentence
splitter + support cosine mirror ch11's rag_evaluation, so the RAG chapters share ONE source of
truth.

Verified on Python 3.12.x / numpy 2.4.6 / torch 2.12.0 / sentence-transformers (all-MiniLM-L6-v2,
CPU). Deterministic: identical numbers every run given the same cached model.

Run:
    python citations_attribution.py
"""

from __future__ import annotations

import re
import sys
from dataclasses import dataclass
from pathlib import Path

import numpy as np

# Reuse ch5's dense retriever + corpus (ch5 injects ch1 transitively). ch5 lives two directories
# over; inject its code dir so the import works whether this file is run from its own dir or
# imported by the notebook / figure scripts.
_CH5_CODE = Path(__file__).resolve().parent.parent.parent / "05-Hybrid-Search-BM25-and-Dense" / "code"
if str(_CH5_CODE) not in sys.path:
    sys.path.insert(0, str(_CH5_CODE))

from hybrid_search import (  # noqa: E402  (path injected above must precede this import)
    DenseRetriever,
    full_corpus,
)

DENSE_MODEL = "all-MiniLM-L6-v2"  # the learned bi-encoder (ch3/5's embedder)

# ---- Support / citation threshold ---------------------------------------------------------------
# A claim is CITABLE to a passage only if their cosine clears this bar. 0.5 is ch8/ch11's deliberate
# middle bar on unit-norm all-MiniLM cosines: a claim genuinely paraphrasing a passage scores
# ~0.6-0.9 and clears it; an unsupported (hallucinated, off-topic) claim scores ~0.0-0.3 and does
# not. It is a computable PROXY for an NLI entailment judge -- see the module banner + the false-
# citation demo for why cosine != entailment.
SUPPORT_THRESHOLD = 0.5

_SENT_SPLIT_RE = re.compile(r"(?<=[.!?])\s+")  # split on sentence-ending punctuation + following space


# ================================================================================================
# Shared helpers: claim splitting and the encoder-cosine support signal (ch8/ch11's real proxy)
# ================================================================================================


def split_into_claims(text: str) -> list[str]:
    """Split an answer into atomic CLAIMS -- here, one claim per sentence.

    ILLUSTRATIVE stand-in: in production an LLM extracts atomic factual claims (which can split a
    single sentence into several, or merge clauses). We use a transparent sentence splitter so the
    DECOMPOSE -> attribute -> assign mechanism is visible and reproducible with no LLM. The per-claim
    attribution that follows is REAL (encoder cosine); only this extraction step is a stand-in.
    """
    flat = re.sub(r"\s+", " ", text).strip()
    return [s.strip() for s in _SENT_SPLIT_RE.split(flat) if s.strip()]


def claim_passage_scores(dense: DenseRetriever, claim: str, passages: tuple[str, ...]) -> np.ndarray:
    """Cosine of one answer CLAIM against every retrieved PASSAGE (parallel to `passages` order).

    REAL and measured. This is the attribution signal: a claim that paraphrases a passage scores
    high against THAT passage; an unsupported claim scores low against ALL of them. Encoder-cosine
    is a PROXY for an entailment judge (see banner + false_citation_demo) -- but the number itself
    is a real all-MiniLM cosine, unit-norm on both sides so the dot product IS the cosine.
    """
    if not passages:
        return np.zeros(0)
    claim_vec = dense._encode([claim])[0]  # noqa: SLF001 -- reuse ch5's encoder; unit-norm
    passage_vecs = dense._encode(list(passages))  # noqa: SLF001 -- reuse ch5's encoder; (n_passages, dim) unit-norm
    return passage_vecs @ claim_vec  # unit-norm rows => dot product == cosine, one per passage


# ================================================================================================
# Post-hoc attribution: decompose -> match each claim to its best passage -> assign / flag
# ================================================================================================


@dataclass(frozen=True)
class ClaimAttribution:
    """One claim's attribution: the claim text, its best-matching passage, the cosine, and a verdict.

    `citation` is the 1-based passage number the claim is cited to (e.g. 2 -> "[2]"), or None if no
    passage cleared the support bar (the claim is UNCITABLE -> the likely hallucination). `scores`
    is the claim's cosine to every retrieved passage, kept for the attribution heatmap.
    """

    claim: str
    citation: int | None  # 1-based passage index, or None if uncitable
    best_score: float  # cosine to the best-matching passage
    scores: tuple[float, ...]  # cosine to every passage (parallel to the passage list)
    citable: bool  # best_score >= threshold


def attribute_claims(
    dense: DenseRetriever,
    answer: str,
    passages: tuple[str, ...],
    threshold: float = SUPPORT_THRESHOLD,
) -> tuple[ClaimAttribution, ...]:
    """Post-hoc attribution: for each claim in the answer, cite the best passage above the bar.

    The whole mechanism, from primitives:
      1. DECOMPOSE the answer into claims (illustrative sentence split).
      2. For each claim, score it against every retrieved passage (REAL encoder cosine).
      3. ASSIGN the argmax passage as the citation IF its cosine clears `threshold`; otherwise mark
         the claim UNCITABLE (best_score < threshold -> citation is None -> the likely hallucination).
    Returns one ClaimAttribution per claim, in answer order. A grounded claim gets a real passage
    number; the hallucination gets None -- which is exactly the signal a reader needs to distrust it.
    """
    claims = split_into_claims(answer)
    out: list[ClaimAttribution] = []
    for claim in claims:
        scores = claim_passage_scores(dense, claim, passages)
        if scores.size == 0:  # no retrieved context at all -> nothing to cite
            out.append(ClaimAttribution(claim, None, 0.0, (), False))
            continue
        best_idx = int(np.argmax(scores))
        best_score = float(scores[best_idx])
        citable = best_score >= threshold
        out.append(
            ClaimAttribution(
                claim=claim,
                citation=(best_idx + 1) if citable else None,  # 1-based citation number, or None
                best_score=best_score,
                scores=tuple(float(s) for s in scores),
                citable=citable,
            )
        )
    return tuple(out)


def render_cited_answer(attributions: tuple[ClaimAttribution, ...]) -> str:
    """Reassemble the answer with an inline [n] after each cited claim, and [?] on uncitable ones.

    This is the reader-facing artifact: the same answer, now with every claim carrying a pointer to
    the passage it came from -- or a visible [?] where no passage supports it, so the unverifiable
    claim cannot hide behind fluent prose.
    """
    parts: list[str] = []
    for attr in attributions:
        marker = f"[{attr.citation}]" if attr.citation is not None else "[?]"
        parts.append(f"{attr.claim} {marker}")
    return " ".join(parts)


# ================================================================================================
# Metrics: citation precision & recall (ALCE-style, computed on the cosine proxy)
# ================================================================================================
#
# ALCE (Gao et al. 2023) defines citation quality with an NLI ENTAILMENT model; we compute the same
# quantities with the cosine support proxy (banner: cosine ~ topical, not entailment). To make the
# metric MEANINGFUL we need ground truth: for each claim, which passages ACTUALLY support it. We
# supply that as a labelled gold set on the demo (a stand-in for the human/NLI judgment), then score
# the matcher's assignments against it -- so the precision/recall numbers grade the ATTRIBUTOR, not
# the encoder's opinion of itself.


@dataclass(frozen=True)
class CitedClaimGold:
    """Ground truth for one claim: is it supportable at all, and by which passages (1-based).

    `supportable` is False for a hallucinated claim no passage supports. `supporting` is the set of
    passage numbers (1-based) that genuinely entail the claim -- the gold a correct citation must
    point into. For an unsupportable claim `supporting` is empty and a CORRECT system emits NO
    citation.
    """

    supportable: bool
    supporting: frozenset[int]  # 1-based passage numbers that truly support this claim


def citation_precision(
    attributions: tuple[ClaimAttribution, ...], golds: tuple[CitedClaimGold, ...]
) -> float:
    """Fraction of EMITTED citations that point to a genuinely-supporting passage.

    Precision answers "of the claims that cite a source, how many actually check out?" -- the
    metric that punishes a FALSE CITATION (citing a passage that does not support the claim) and
    OVER-citation. Denominator = claims that emitted a citation; numerator = those whose cited
    passage is in the gold supporting set. A perfect attributor cites only truly-supported claims,
    and always to a supporting passage.
    """
    emitted = 0
    correct = 0
    for attr, gold in zip(attributions, golds, strict=True):
        if attr.citation is None:  # no citation emitted -> not counted toward precision
            continue
        emitted += 1
        if attr.citation in gold.supporting:  # cited a genuinely-supporting passage
            correct += 1
    return correct / emitted if emitted else 0.0


def citation_recall(
    attributions: tuple[ClaimAttribution, ...], golds: tuple[CitedClaimGold, ...]
) -> float:
    """Fraction of SUPPORTABLE claims that received a correct citation.

    Recall answers "of the claims that COULD be cited, how many did we correctly cite?" -- the
    metric that punishes MISSING citations (a grounded claim left uncited). Denominator =
    supportable claims; numerator = supportable claims whose emitted citation lands in the gold
    supporting set. An unsupportable (hallucinated) claim is NOT in the denominator -- correctly
    leaving it uncited neither helps nor hurts recall (it helps precision by not being a false cite).
    """
    supportable = 0
    hit = 0
    for attr, gold in zip(attributions, golds, strict=True):
        if not gold.supportable:  # hallucinated claims are not part of recall's denominator
            continue
        supportable += 1
        if attr.citation is not None and attr.citation in gold.supporting:
            hit += 1
    return hit / supportable if supportable else 0.0


# ================================================================================================
# The demo: retrieve, generate (fixed exemplar answers), attribute, measure.
# ================================================================================================

QUESTION = "When did Helios-7 launch, what is its imager resolution, and what powers it?"

# A fixed exemplar answer standing in for a generator's output (this env has no generator). It reads
# fluently and mixes TWO grounded claims with ONE hallucination -- the case citations exist to catch.
ANSWER = (
    "Helios-7 launched on March 3rd, 2024 from the Kourou spaceport. "  # grounded (corpus doc 0)
    "Its hyperspectral imager has a ground resolution of 4 meters. "  # grounded (corpus doc 1)
    "Helios-7 is powered entirely by solar panels."  # HALLUCINATED -- no passage supports this
)

# For the false-citation demo: a claim topically NEAR a RETRIEVED passage it does NOT entail. The
# project-lead passage says the lead is "Dr. Amara Okoye, based in the Nairobi office"; this claim
# names a DIFFERENT lead in a DIFFERENT city -- a fact the corpus contradicts -- yet it shares the
# "project lead"/"office"/"Helios-7" vocabulary, so cosine rates it near that passage and would
# wrongly cite it. The passage is retrieved into the top-3, so the false cite lands on real context.
FALSE_CITE_CLAIM = "The Helios-7 project lead is Dr. Lars Vinter, based in the Oslo office."


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
        print("torch: not installed (attribution is pure numpy over cached embeddings — unaffected)")


def retrieve_passages(dense: DenseRetriever, corpus: tuple[str, ...], k: int = 3) -> tuple[tuple[str, ...], tuple[int, ...]]:
    """Retrieve the top-k passages the answer is attributed against (REAL ranked retrieval)."""
    ranked = dense.search(QUESTION, k=len(corpus)).indices
    top = ranked[:k]
    return tuple(corpus[i] for i in top), top


def build_golds(passages: tuple[str, ...]) -> tuple[CitedClaimGold, ...]:
    """Labelled ground truth for the demo's three claims (a stand-in for a human/NLI judge).

    Resolved by passage CONTENT so it stays correct if retrieval order changes: the launch claim is
    supported by the launch passage, the resolution claim by the imager passage, and the solar-panels
    claim by NOTHING (unsupportable -> a correct system leaves it uncited).
    """
    def passage_num(substr: str) -> int:  # 1-based number of the passage containing substr
        return next(i + 1 for i, p in enumerate(passages) if substr in p)

    launch_p = passage_num("Kourou")
    imager_p = passage_num("ground resolution")
    return (
        CitedClaimGold(supportable=True, supporting=frozenset({launch_p})),  # launch claim
        CitedClaimGold(supportable=True, supporting=frozenset({imager_p})),  # resolution claim
        CitedClaimGold(supportable=False, supporting=frozenset()),  # solar-panels hallucination
    )


def _print_attribution(attr: ClaimAttribution, passages: tuple[str, ...]) -> None:
    """Pretty-print one claim's attribution verdict + the passage it was (or wasn't) cited to."""
    if attr.citation is not None:
        cite = f"[{attr.citation}] (cos {attr.best_score:.3f})"
        target = passages[attr.citation - 1]
        print(f"    CITED   {cite}: {attr.claim}")
        print(f"            -> {target}")
    else:
        print(f"    UNCITABLE (best cos {attr.best_score:.3f} < {SUPPORT_THRESHOLD}): {attr.claim}")
        print("            -> no retrieved passage supports this claim (the likely hallucination)")


def false_citation_demo(dense: DenseRetriever, passages: tuple[str, ...]) -> tuple[int, float, bool]:
    """Show cosine assigning a FALSE citation: a claim topically near a passage it does not entail.

    Returns (cited_passage_1based, cosine, cleared_bar). The claim names a different lead in a
    different city than the project-lead passage states, but shares its vocabulary, so cosine rates
    it above the bar and the matcher CITES the project-lead passage -- a citation that is topically
    plausible but factually wrong (the passage in fact CONTRADICTS the claim). This is the concrete
    cosine != entailment failure, and the reason ALCE/AIS score citations with an NLI model.
    """
    scores = claim_passage_scores(dense, FALSE_CITE_CLAIM, passages)
    best_idx = int(np.argmax(scores))
    best_score = float(scores[best_idx])
    cleared = best_score >= SUPPORT_THRESHOLD
    return best_idx + 1, best_score, cleared


def main() -> None:
    _report_versions()
    corpus = full_corpus()
    dense = DenseRetriever(corpus)
    passages, top_idx = retrieve_passages(dense, corpus, k=3)
    print(f"corpus: {len(corpus)} passages | dense lens: {dense.backend} | support threshold: {SUPPORT_THRESHOLD}")
    print(
        "NOTE: the attribution matching + citation precision/recall are REAL and measured; "
        "the claim-splitter and the generator are illustrative stand-ins for an LLM.\n"
    )

    # ------------------------------------------------------------------------------------------
    # 1) THE HEADLINE: attribute each claim; the grounded ones get citations, the hallucination
    #    is flagged UNCITABLE.
    # ------------------------------------------------------------------------------------------
    print("=" * 96)
    print("1) Post-hoc attribution: cite the grounded claims, flag the hallucination as uncitable")
    print("=" * 96)
    print(f"question: {QUESTION}")
    print(f"retrieved context (top-{len(passages)}, ids {list(top_idx)}):")
    for n, passage in enumerate(passages, start=1):
        print(f"    [{n}] doc[{top_idx[n - 1]}]: {passage}")
    attributions = attribute_claims(dense, ANSWER, passages)
    print(f"\n  ANSWER (raw): {ANSWER}\n")
    for attr in attributions:
        _print_attribution(attr, passages)
    print(f"\n  ANSWER (with citations): {render_cited_answer(attributions)}")
    # Correctness BEFORE the claim: two claims cite a real passage; the solar-panels claim is uncitable.
    n_cited = sum(1 for a in attributions if a.citation is not None)
    assert n_cited == 2, "exactly the two grounded claims should be cited"
    assert attributions[-1].citation is None, "the solar-panels hallucination must be UNCITABLE"
    assert attributions[0].citation is not None and attributions[1].citation is not None, "both grounded claims cite a passage"
    print(f"\n  -> {n_cited}/3 claims cited to a real passage; the hallucination is flagged [?] and cannot hide.\n")

    # ------------------------------------------------------------------------------------------
    # 2) MEASURE it: citation precision & recall against the labelled gold.
    # ------------------------------------------------------------------------------------------
    print("=" * 96)
    print("2) Citation precision & recall: grade the attributor against a labelled gold")
    print("=" * 96)
    golds = build_golds(passages)
    precision = citation_precision(attributions, golds)
    recall = citation_recall(attributions, golds)
    print("  gold: launch & resolution are supportable (1 passage each); solar-panels is unsupportable")
    print(f"  citation precision = {precision:.3f}  (of emitted citations, how many point to a supporting passage)")
    print(f"  citation recall    = {recall:.3f}  (of supportable claims, how many got a correct citation)")
    # Correctness BEFORE the claim: on this clean case the attributor is perfect on both axes.
    assert precision == 1.0, "every emitted citation points to a genuinely-supporting passage"
    assert recall == 1.0, "both supportable claims received a correct citation"
    print("\n  -> perfect on this clean case: no false citation (precision 1.0), no missed citation (recall 1.0).\n")

    # ------------------------------------------------------------------------------------------
    # 3) THE PITFALL (ch8/ch11 caveat, still true): cosine != entailment -> a FALSE citation.
    # ------------------------------------------------------------------------------------------
    print("=" * 96)
    print("3) Pitfall: cosine ~ topic, not entailment -> a topically-near passage gets a FALSE citation")
    print("=" * 96)
    lead_passage_num = next(i + 1 for i, p in enumerate(passages) if "project lead" in p)
    cited_num, cos, cleared = false_citation_demo(dense, passages)
    print(f"  claim (CONTRADICTED by the passage it will be cited to): {FALSE_CITE_CLAIM}")
    print(f"  cosine cites passage [{cited_num}] at cos {cos:.3f}  (clears {SUPPORT_THRESHOLD}: {cleared})")
    print(f"    passage [{cited_num}]: {passages[cited_num - 1]}")
    # Correctness BEFORE the claim: the claim names a different lead in a different city, yet cosine
    # cites the project-lead passage above the bar -- a false citation the cosine proxy cannot catch.
    assert cleared, "the wrong-lead claim clears the cosine bar despite NOT being entailed -- the false-citation trap"
    assert cited_num == lead_passage_num, "cosine wrongly cites the topically-near PROJECT-LEAD passage"
    print("\n  -> a citation that is topically plausible but factually WRONG: the cited passage names a")
    print("     DIFFERENT lead in a DIFFERENT city. cosine cannot tell 'about the project lead' from")
    print("     'entails THIS lead' -- which is why ALCE/AIS score citations with an NLI entailment model.")

    # ------------------------------------------------------------------------------------------
    # 4) OVER-CITATION drops precision: cite EVERY claim (even the hallucination) and watch it fall.
    # ------------------------------------------------------------------------------------------
    print("\n" + "=" * 96)
    print("4) Over-citation: forcing a citation on every claim (threshold 0) tanks precision")
    print("=" * 96)
    greedy = attribute_claims(dense, ANSWER, passages, threshold=0.0)  # cite even the hallucination
    greedy_precision = citation_precision(greedy, golds)
    greedy_recall = citation_recall(greedy, golds)
    forced = greedy[-1]  # the solar-panels claim, now force-cited to its (wrong) best passage
    print(f"  with threshold 0, the hallucination is force-cited to passage [{forced.citation}] (cos {forced.best_score:.3f})")
    print(f"  citation precision: {precision:.3f} (thresholded)  ->  {greedy_precision:.3f} (over-cited)")
    print(f"  citation recall   : {recall:.3f} (thresholded)  ->  {greedy_recall:.3f} (unchanged)")
    # Correctness BEFORE the claim: over-citation adds a WRONG citation, so precision drops; recall,
    # which only counts supportable claims, is unchanged.
    assert greedy_precision < precision, "force-citing the hallucination adds a false citation -> precision drops"
    assert greedy_recall == recall, "over-citation does not change recall (same supportable claims, same hits)"
    assert forced.citation is not None, "with threshold 0 even the hallucination is (wrongly) cited"
    # Pin the tie: the hallucination's two best passages are a NEAR-tie at 2 decimals (both ~0.49), so
    # the forced citation column must be exactly np.argmax (which breaks the tie by the true 3rd-decimal
    # ordering, first-occurrence on an exact tie). This freezes the heatmap's red-box column == this
    # prose's passage, so a rounding-level drift can never make them disagree.
    forced_argmax = int(np.argmax(np.array(forced.scores))) + 1  # 1-based, exactly the citation logic
    assert forced.citation == forced_argmax, "the force-cited passage must equal np.argmax of the claim's passage scores"
    print(f"  (tie-break pinned: passage [{forced.citation}] == argmax; the two top passages are "
          f"{forced.scores[forced.citation - 1]:.3f} vs {sorted(forced.scores, reverse=True)[1]:.3f} — decided at the 3rd decimal)")
    print(f"\n  -> precision {precision:.2f} -> {greedy_precision:.2f}: over-citing the hallucination is a false citation.")
    print("     The support threshold is the precision/recall dial -- raise it to cite less but cleaner.\n")


if __name__ == "__main__":
    main()
