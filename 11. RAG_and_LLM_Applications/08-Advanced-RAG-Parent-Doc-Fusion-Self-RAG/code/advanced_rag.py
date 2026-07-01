"""From-scratch advanced RAG: parent-document retrieval, RAG-Fusion, and a Self-RAG support check.

Naive RAG has three structural weaknesses this chapter fixes:

  1. The small-vs-big chunk DILEMMA (from ch2). Small chunks retrieve PRECISELY (a focused embedding)
     but hand the LLM a context-starved FRAGMENT -- "It completes one orbit every 97 minutes" leaves
     the LLM asking *what* completes it. Big chunks carry context but embed fuzzily and retrieve worse.
     PARENT-DOCUMENT retrieval decouples the two units: index SMALL precise children, but at generation
     time feed the LLM the child's larger PARENT (its section) -- retrieve small, read large.

  2. A single query is a narrow probe (ch7). RAG-FUSION expands the query into N reformulations,
     retrieves each, and fuses the union with RRF -- the exact machinery ch7 built, named as a pattern.

  3. Naive RAG ALWAYS retrieves (even when the model knows the answer) and NEVER checks whether its
     output is actually grounded in what it retrieved. SELF-RAG (Asai et al. 2023) adds a retrieve-on-
     demand GATE and a self-critique of grounding via reflection tokens (Retrieve / ISREL / ISSUP /
     ISUSE). We implement the two decisions Self-RAG makes that ARE computable with an encoder -- the
     retrieve-on-demand gate and the SUPPORT check (is the answer's claim actually entailed by the
     retrieved context?) -- as real, measured signals.

HONESTY -- WHAT RUNS REAL vs WHAT IS ILLUSTRATIVE
-------------------------------------------------
  * REAL and measured: the dense encoder (all-MiniLM-L6-v2, ch3/5's embedder), ALL retrieval
    (child retrieval, parent dedup, RAG-Fusion), and the Self-RAG SUPPORT CHECK (a computable
    proxy: cosine similarity between an answer claim and the retrieved context, thresholded --
    "does the context actually support this claim?"). Every cosine, recall, and support score
    printed here is measured, asserted before it is claimed, and reproducible.
  * ILLUSTRATIVE (labelled): Self-RAG's true reflection tokens require a specially-TRAINED LLM; this
    env is encoder-only, so we do NOT fake trained ISREL/ISSUP/ISUSE tokens or an LLM's generated
    answer text. The generated answer strings and the human-readable critique verdicts are FIXED
    exemplars standing in for a generator LLM. The GATE decision and the SUPPORT score, however, are
    computed from real embeddings -- so the mechanism (retrieve-on-demand, ground-before-trust) is
    demonstrated with real numbers even though the generator text is illustrative.

The dense encoder + RRF + metrics are imported from ch5's hybrid_search (which imports ch1), and the
multi-query fusion from ch7's query_transformation, so the RAG chapters share ONE source of truth.

Verified on Python 3.12 / numpy 2.x / sentence-transformers (all-MiniLM-L6-v2, CPU). Deterministic:
identical numbers every run given the same cached model.

Run:
    python advanced_rag.py
"""

from __future__ import annotations

import re
import sys
from dataclasses import dataclass
from pathlib import Path

import numpy as np

# Reuse ch5's dense retriever + RRF + metrics and ch7's multi-query fusion (both add their code dir to
# the path transitively). ch5 and ch7 live two directories over; inject their paths so imports work
# whether this file is run from its own dir or imported by the notebook / figure scripts.
_CH5_CODE = Path(__file__).resolve().parent.parent.parent / "05-Hybrid-Search-BM25-and-Dense" / "code"
_CH7_CODE = Path(__file__).resolve().parent.parent.parent / "07-Query-Transformation-HyDE-Multi-Query" / "code"
for _p in (_CH5_CODE, _CH7_CODE):
    if str(_p) not in sys.path:
        sys.path.insert(0, str(_p))

from hybrid_search import (  # noqa: E402  (paths injected above must precede this import)
    RRF_K,
    DenseRetriever,
    recall_at_k,
)
from query_transformation import retrieve_multiquery  # noqa: E402  (RAG-Fusion = multi-query + RRF)

TOP_K = 3  # children to retrieve
DENSE_MODEL = "all-MiniLM-L6-v2"  # the learned bi-encoder (ch3/5's embedder)

# ---- Self-RAG support-check threshold ----------------------------------------------------------
# A claim is judged SUPPORTED if its max cosine to any retrieved-context sentence clears this bar.
# 0.5 is a deliberately middle bar on unit-norm all-MiniLM similarities: paraphrase-level support
# (~0.6-0.9) clears it, unrelated text (~0.0-0.3) does not. It is a computable PROXY for Self-RAG's
# ISSUP token, not the trained token itself -- see the module banner.
SUPPORT_THRESHOLD = 0.5

# ================================================================================================
# The corpus: ch2's multi-section Helios-7 spec. Sections are natural PARENTS; sentences are CHILDREN.
# This is the same document ch2 chunked -- reused so the chapters share one source of truth.
# ================================================================================================
DOCUMENT = """# Helios-7 Mission Overview

The Helios-7 satellite is an Earth-observation platform operated by the Nairobi office.
It was launched on March 3rd, 2024 from the Kourou spaceport aboard an Ariane 6 rocket.

# Instruments

Helios-7 carries a hyperspectral imager with a ground resolution of 4 meters.
The imager captures 200 spectral bands across the visible and near-infrared range.

# Orbit

Helios-7 completes one orbit of Earth every 97 minutes in a sun-synchronous orbit.
This orbit keeps the local solar time of each pass roughly constant for stable imaging."""

_SENT_SPLIT_RE = re.compile(r"(?<=[.!?])\s+")  # split on sentence-ending punctuation + space


@dataclass(frozen=True)
class ChildChunk:
    """One small retrieval unit (a sentence) and a pointer to the parent (section) it came from."""

    text: str  # the child sentence -- what we EMBED and retrieve on
    parent_id: int  # index of the section it belongs to -- what we FEED the LLM


@dataclass(frozen=True)
class Parent:
    """One larger generation unit (a document section: heading + its sentences)."""

    heading: str
    text: str  # the full section text -- heading + body sentences


# ================================================================================================
# Build the parent/child hierarchy from the document
# ================================================================================================


def build_hierarchy(document: str = DOCUMENT) -> tuple[tuple[Parent, ...], tuple[ChildChunk, ...]]:
    """Split the document into PARENT sections and their CHILD sentences (each child knows its parent).

    A section starts at a '# heading' line and runs to the next heading. The section (heading + body)
    is the PARENT -- the context we feed the LLM. Each body sentence is a CHILD -- a small, focused
    unit we embed and retrieve on. Every child stores its parent_id so a retrieved child can be
    resolved back to the section that gives it meaning. This is the retrieve-small / read-large split.
    """
    parents: list[Parent] = []
    children: list[ChildChunk] = []
    # split into sections on heading lines (keep the heading with its body)
    blocks = re.split(r"\n(?=# )", document.strip())
    for block in blocks:
        lines = block.strip().split("\n", 1)
        heading = lines[0].lstrip("# ").strip()
        body = lines[1].strip() if len(lines) > 1 else ""
        parent_id = len(parents)
        parents.append(Parent(heading=heading, text=block.strip()))
        # flatten the body to sentences -> children, each pointing back at this parent
        flat = re.sub(r"\s+", " ", body).strip()
        for sent in _SENT_SPLIT_RE.split(flat):
            sent = sent.strip()
            if sent:
                children.append(ChildChunk(text=sent, parent_id=parent_id))
    return tuple(parents), tuple(children)


# ================================================================================================
# Parent-document retriever: retrieve small children, dedupe to their parents
# ================================================================================================


@dataclass(frozen=True)
class ParentDocResult:
    """One parent-document retrieval: the child hits (with scores) and the deduped parents returned."""

    child_indices: tuple[int, ...]
    child_scores: tuple[float, ...]
    parent_ids: tuple[int, ...]  # deduped, in order of first appearance among the child hits


class ParentDocumentRetriever:
    """Retrieve on SMALL child chunks; return their (deduped) larger PARENTS for generation.

    The child index is a real dense index over the sentence-level children (all-MiniLM). At query
    time we retrieve the top-k children, then map each to its parent and DEDUPE (several children can
    share one parent -- see the dedup pitfall), preserving first-hit order. The caller feeds the
    parents (full sections) to the LLM: sharp retrieval on children, full context from parents.
    """

    def __init__(
        self, parents: tuple[Parent, ...], children: tuple[ChildChunk, ...], model_name: str = DENSE_MODEL
    ) -> None:
        self.parents = parents
        self.children = children
        # a dense index over the CHILD texts only -- children are what we retrieve on
        self._dense = DenseRetriever(tuple(c.text for c in children), model_name=model_name)
        self.backend = self._dense.backend

    def retrieve(self, query: str, k: int = TOP_K) -> ParentDocResult:
        """Top-k children by cosine, then their deduped parents (first-hit order preserved)."""
        res = self._dense.search(query, k=k)
        parent_ids: list[int] = []
        for child_idx in res.indices:
            pid = self.children[child_idx].parent_id
            if pid not in parent_ids:  # dedupe: several retrieved children may share one parent
                parent_ids.append(pid)
        return ParentDocResult(res.indices, res.scores, tuple(parent_ids))

    def child_context(self, result: ParentDocResult) -> str:
        """The naive baseline: just the retrieved child sentences concatenated (context-starved)."""
        return " ".join(self.children[i].text for i in result.child_indices)

    def top_child_context(self, result: ParentDocResult) -> str:
        """The sharpest naive baseline: only the #1 child (what top-1 child-only RAG feeds the LLM).

        This is the most context-starved case -- a single retrieved sentence -- which is exactly where
        the fragment problem bites: the #1 child can open with a pronoun whose referent lives in a
        neighbouring sentence it does not include.
        """
        return self.children[result.child_indices[0]].text

    def parent_context(self, result: ParentDocResult) -> str:
        """The parent-document context: the full deduped parent sections (retrieve small, read large)."""
        return "\n\n".join(self.parents[pid].text for pid in result.parent_ids)


# ================================================================================================
# Self-RAG: the two decisions that ARE computable with an encoder
#   (a) retrieve-on-demand GATE, (b) SUPPORT check (a proxy for the ISSUP reflection token)
# ================================================================================================


def sentence_split(text: str) -> list[str]:
    """Flatten text to sentences -- the granularity Self-RAG checks support at."""
    flat = re.sub(r"\s+", " ", text).strip()
    return [s.strip() for s in _SENT_SPLIT_RE.split(flat) if s.strip()]


def support_score(dense: DenseRetriever, claim: str, context: str) -> float:
    """Max cosine between an answer CLAIM and any sentence of the retrieved CONTEXT.

    A computable PROXY for Self-RAG's ISSUP ("is the output supported by the evidence?") reflection
    token: if the claim is a paraphrase of something actually in the context, its best-matching
    context sentence scores high; if the claim is unsupported (hallucinated), no context sentence
    matches and the score stays low. Real embeddings, real number -- the trained ISSUP token is
    illustrative here (see banner), but THIS grounding signal is measured.
    """
    context_sents = sentence_split(context)
    if not context_sents:
        return 0.0
    claim_vec = dense._encode([claim])[0]  # noqa: SLF001 -- reuse ch5's encoder; unit-norm
    ctx_vecs = dense._encode(context_sents)  # (n_sents, dim) unit-norm
    return float(np.max(ctx_vecs @ claim_vec))  # best cosine to any context sentence


def is_supported(dense: DenseRetriever, claim: str, context: str, threshold: float = SUPPORT_THRESHOLD) -> bool:
    """ISSUP proxy: True if the claim's max cosine to the context clears the support threshold."""
    return support_score(dense, claim, context) >= threshold


# The retrieve-on-demand gate: a tiny illustrative rule standing in for Self-RAG's `Retrieve` token.
# Self-RAG TRAINS an LM to emit Retrieve=yes/no; here we use a transparent heuristic (does the query
# ask about a specific external fact vs a self-contained/greeting/arithmetic query?) so the GATE
# BEHAVIOUR is demonstrable without a trained model. The decision is illustrative; the point it makes
# -- "don't retrieve when retrieval can't help" -- is the real Self-RAG lesson.
_NO_RETRIEVE_MARKERS = (
    "hello",
    "hi ",
    "thank",
    "what is 2+2",
    "translate",
    "write a poem",
)


def needs_retrieval(query: str) -> bool:
    """Retrieve-on-demand GATE (illustrative proxy for Self-RAG's `Retrieve` token).

    Returns False for queries that don't need external factual grounding (greetings, arithmetic,
    pure-style requests) and True otherwise. Self-RAG learns this decision; we hard-code a transparent
    version so the *gate behaviour* -- skip retrieval when it can't help -- is visible with no LLM.
    """
    q = query.lower().strip()
    return not any(marker in q for marker in _NO_RETRIEVE_MARKERS)


# ================================================================================================
# Reporting
# ================================================================================================


def _report_versions() -> None:
    """Print numpy/torch versions for reproducibility. The encoder is CPU-pinned (ch5's DenseRetriever)."""
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
    parents, children = build_hierarchy()
    retriever = ParentDocumentRetriever(parents, children)
    print(f"document: {len(parents)} parent sections, {len(children)} child sentences")
    print(f"dense lens: {retriever.backend}")
    print(
        "NOTE: encoder + ALL retrieval + the support-check are REAL and measured; the generated "
        "answer text and human critique verdicts are FIXED exemplars standing in for a generator LLM.\n"
    )

    # ------------------------------------------------------------------------------------------
    # 1) THE DILEMMA: a child-only retrieval hands the LLM a context-starved fragment.
    # ------------------------------------------------------------------------------------------
    print("=" * 96)
    print("1) Parent-document retrieval: retrieve the small child, FEED the larger parent")
    print("=" * 96)
    # a query whose top answer sentence opens with a pronoun ("This orbit ...") -- meaningless alone:
    # the referent ("sun-synchronous orbit") lives in the PREVIOUS sentence / the parent section.
    query = "What keeps the local solar time of each pass roughly constant?"
    result = retriever.retrieve(query, k=TOP_K)
    top_child_ctx = retriever.top_child_context(result)  # the #1 child alone (top-1 child-only RAG)
    parent_ctx = retriever.parent_context(result)
    top_child = children[result.child_indices[0]]
    print(f"query: {query}")
    print(f"  top child hit: doc-child[{result.child_indices[0]}] (cos={result.child_scores[0]:.3f})")
    print(f"    child text (what top-1 child-only RAG feeds the LLM): {top_child.text!r}")
    print("    -> opens with 'This orbit' — the referent (which orbit?) is NOT in the child; a fragment.")
    print(f"  deduped parents returned: {list(result.parent_ids)}")
    print("  PARENT context (what parent-doc RAG feeds the LLM):")
    for line in parent_ctx.splitlines():
        print(f"    | {line}")
    # measure: the #1 child alone LACKS the referent ('sun-synchronous orbit'); its parent SUPPLIES it
    referent = "sun-synchronous orbit"
    child_has = referent in top_child_ctx
    parent_has = referent in parent_ctx
    print(f"\n  referent {referent!r} in the #1 child alone: {child_has}")
    print(f"  referent {referent!r} in its parent section: {parent_has}")
    assert not child_has, "the #1 child alone should LACK the referent (the fragment problem)"
    assert parent_has, "the parent section must SUPPLY the referent (retrieve small, read large)"
    print("  -> the #1 child is a context-starved fragment; its parent supplies the referent. Retrieve small, read large.\n")

    # ------------------------------------------------------------------------------------------
    # 2) PRECISION x CONTEXT, measured: child retrieval is sharp; parents add the context.
    # ------------------------------------------------------------------------------------------
    print("=" * 96)
    print("2) Precision (retrieval) x Context (generation), measured")
    print("=" * 96)
    # child retrieval precision: does the top child belong to the RIGHT section (the Orbit section)?
    orbit_parent = next(i for i, p in enumerate(parents) if p.heading == "Orbit")
    top_child_parent = children[result.child_indices[0]].parent_id
    print(f"  top child's parent section: {parents[top_child_parent].heading!r} (target: 'Orbit')")
    assert top_child_parent == orbit_parent, "the sharp child retrieval must hit the Orbit section"
    # context delivered: parent is larger than the child (more tokens of context)
    child_chars = len(top_child.text)
    parent_chars = len(parents[top_child_parent].text)
    print(f"  child delivered to LLM: {child_chars} chars | parent delivered: {parent_chars} chars")
    print(f"  context multiplier (parent/child): {parent_chars / child_chars:.1f}x more context, SAME sharp retrieval")
    assert parent_chars > child_chars, "the parent must carry more context than the child"

    # ------------------------------------------------------------------------------------------
    # 3) RAG-FUSION: multi-query + RRF (ch7's machinery), over the child index.
    # ------------------------------------------------------------------------------------------
    print("\n" + "=" * 96)
    print("3) RAG-Fusion: expand the query, retrieve each, fuse the children with RRF (ch7 reused)")
    print("=" * 96)
    # a vague query whose gold child a single probe MISSES at top-2 but the fused reformulations catch.
    # gold = the "200 spectral bands" sentence (the specific imaging-capability fact).
    fusion_k = 2  # a tight top-2 so recall is not trivially saturated on this 6-child index
    vague = "What are the imaging capabilities of Helios-7?"
    reformulations = (
        "How many spectral bands does Helios-7 capture?",
        "What wavelength range does the imager cover?",
        "What kind of imager is on Helios-7?",
    )
    bands_child = next(i for i, c in enumerate(children) if "spectral bands" in c.text)
    raw = retriever._dense.search(vague, k=fusion_k).indices  # noqa: SLF001 -- child index is the retrieval unit
    fused = retrieve_multiquery(retriever._dense, (vague, *reformulations), k=fusion_k, k_rrf=RRF_K)  # noqa: SLF001
    print(f"  vague query: {vague}")
    print(f"    RAW child hits (top-{fusion_k}) {list(raw)}   gold child[{bands_child}] rank: {_rank(raw, bands_child)}")
    print(f"    RAG-FUSION child hits (top-{fusion_k}) {list(fused)}   gold child[{bands_child}] rank: {_rank(fused, bands_child)}")
    raw_recall = recall_at_k(raw, bands_child)
    fused_recall = recall_at_k(fused, bands_child)
    print(f"    recall@{fusion_k}: raw {raw_recall:.0f} -> RAG-Fusion {fused_recall:.0f}")
    assert raw_recall == 0.0, "the vague single query should MISS the gold child at top-2"
    assert fused_recall == 1.0, "RAG-Fusion must recover the gold child the single query missed"
    print("    -> the single vague probe misses the specific fact; fusing reformulations recovers it (0 -> 1)")

    # ------------------------------------------------------------------------------------------
    # 4) SELF-RAG (a): retrieve-on-demand GATE -- don't retrieve when it can't help.
    # ------------------------------------------------------------------------------------------
    print("\n" + "=" * 96)
    print("4) Self-RAG gate: retrieve ON DEMAND (skip retrieval when the query doesn't need it)")
    print("=" * 96)
    gate_queries = [
        ("When was Helios-7 launched?", True),
        ("Hello, how are you today?", False),
        ("What is 2+2?", False),
        ("What is the ground resolution of the Helios-7 imager?", True),
    ]
    print(f"  {'query':<52} | {'gate: retrieve?':>15} | expected")
    print("  " + "-" * 84)
    for q, expected in gate_queries:
        decision = needs_retrieval(q)
        print(f"  {q:<52} | {str(decision):>15} | {expected}")
        assert decision == expected, f"gate decision wrong for {q!r}"
    print("  -> the gate (illustrative proxy for Self-RAG's Retrieve token) skips retrieval when it can't help\n")

    # ------------------------------------------------------------------------------------------
    # 5) SELF-RAG (b): the SUPPORT check -- ground before you trust (real, measured proxy for ISSUP).
    # ------------------------------------------------------------------------------------------
    print("=" * 96)
    print("5) Self-RAG support check: is the answer's claim actually grounded in the retrieved context?")
    print("=" * 96)
    # retrieve context for a launch-date query, then score three claims of decreasing groundedness
    launch_q = "When was Helios-7 launched?"
    launch_res = retriever.retrieve(launch_q, k=TOP_K)
    launch_ctx = retriever.parent_context(launch_res)
    dense = retriever._dense  # noqa: SLF001 -- reuse ch5's encoder for the support cosines
    grounded_claim = "Helios-7 launched on March 3rd, 2024 from the Kourou spaceport."  # true, in context
    offtopic_claim = "Photosynthesis converts carbon dioxide and water into glucose."   # unrelated -> rejected
    swap_claim = "Helios-7 launched in July 2021 from Cape Canaveral."                   # same shape, WRONG facts
    s_good = support_score(dense, grounded_claim, launch_ctx)
    s_off = support_score(dense, offtopic_claim, launch_ctx)
    s_swap = support_score(dense, swap_claim, launch_ctx)
    print("  retrieved context (parents):")
    for line in launch_ctx.splitlines():
        print(f"    | {line}")
    print(f"\n  {'claim type':<34} | {'support':>7} | {'>= '+str(SUPPORT_THRESHOLD)+'?':>8}")
    print("  " + "-" * 58)
    print(f"  {'grounded (true)':<34} | {s_good:>7.3f} | {is_supported(dense, grounded_claim, launch_ctx)!s:>8}")
    print(f"  {'off-topic hallucination':<34} | {s_off:>7.3f} | {is_supported(dense, offtopic_claim, launch_ctx)!s:>8}")
    print(f"  {'same-structure false (date swap)':<34} | {s_swap:>7.3f} | {is_supported(dense, swap_claim, launch_ctx)!s:>8}")
    # what the proxy RELIABLY does: accept the grounded claim, reject the off-topic hallucination
    assert s_good > s_off, "the grounded claim must score higher than the off-topic hallucination"
    assert is_supported(dense, grounded_claim, launch_ctx), "grounded claim must clear the support bar"
    assert not is_supported(dense, offtopic_claim, launch_ctx), "off-topic hallucination must FAIL the bar"
    print("\n  -> the cosine support-proxy ACCEPTS the grounded claim and REJECTS the off-topic hallucination.")
    # the HONEST limitation: cosine measures topical similarity, not factual entailment
    assert s_swap > SUPPORT_THRESHOLD, "the same-structure false claim slips past the cosine proxy (a real limit)"
    assert s_good > s_swap, "the true claim still scores higher than the same-structure false one"
    print("     LIMITATION (measured): the same-structure date-swap claim scores {:.3f} and SLIPS PAST the".format(s_swap))
    print("     0.5 cosine bar — cosine measures TOPICAL similarity, not factual entailment. It still ranks")
    print("     below the true claim ({:.3f}), but a raw-cosine gate can't catch a plausible fact-swap.".format(s_good))
    print("     This is exactly why Self-RAG TRAINS an ISSUP token (an entailment/NLI judgement), rather")
    print("     than thresholding cosine — the trained critic is what the encoder proxy only approximates.")


def _rank(indices: tuple[int, ...], gold: int) -> str:
    """Human-readable rank of a gold index in a result list ('#1', '#2', or 'MISS')."""
    for rank, idx in enumerate(indices, start=1):
        if idx == gold:
            return f"#{rank}"
    return "MISS"


if __name__ == "__main__":
    main()
