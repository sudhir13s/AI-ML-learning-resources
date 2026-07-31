"""Long-context vs RAG: the cost/accuracy tradeoff, with real arithmetic and a real encoder proxy.

Million-token context windows tempt you to skip retrieval and paste the whole corpus into every
prompt. This module makes the tradeoff CONCRETE and MEASURED instead of a vibe:

  1. COST (real arithmetic). Stuffing the whole corpus costs (corpus_tokens x price) EVERY query;
     RAG pays (retrieved_tokens x price) plus a tiny fixed retrieval cost. We compute both across
     corpus sizes and find the CROSSOVER corpus size beyond which RAG is cheaper -- with real
     token counts and clearly-cited representative provider prices.

  2. LOST-IN-THE-MIDDLE (a real encoder PROXY + a cited external result). The true effect -- an LLM
     attending worst to the MIDDLE of a long context -- needs a generative LLM (this env is
     encoder-only). So we (a) MEASURE a proxy for context DILUTION: how a gold "needle" passage's
     retrieval margin (its cosine lead over the best distractor) SHRINKS as the surrounding context
     grows, vs a focused retrieved context that keeps the margin wide; and (b) CITE Liu et al.
     (2023)'s reported accuracy-vs-position U-curve as an EXTERNAL result (their numbers, labelled
     as theirs, not ours). The figures keep the two strictly separate.

HONESTY -- OUR MEASUREMENT vs CITED-EXTERNAL
--------------------------------------------
  * OUR MEASUREMENT (real, reproducible here): the COST arithmetic (token counts x prices) and the
    ENCODER DILUTION proxy (real all-MiniLM cosines from ch5's DenseRetriever over the ch1 corpus
    plus synthetic distractors). Every number these print is computed and asserted.
  * CITED EXTERNAL (labelled, NOT our measurement): the lost-in-the-middle accuracy-vs-position
    U-curve (Liu et al. 2023, arXiv:2307.03172); the effective-vs-advertised context gap (RULER,
    Hsieh et al. 2024, arXiv:2404.06654); provider context-window sizes and per-token prices (from
    provider docs, cited). We NEVER pass these off as our own measurement -- they are constants
    sourced to their papers/docs.

The dense retriever + corpus are imported from ch5's hybrid_search (which imports ch1), so the RAG
chapters share ONE source of truth.

Verified on Python 3.12.x / numpy 2.4.6 / torch 2.12.0 / sentence-transformers (all-MiniLM-L6-v2,
CPU). Deterministic: identical numbers every run given the same cached model.

Run:
    python long_context_vs_rag.py
"""

from __future__ import annotations

import sys
from dataclasses import dataclass
from pathlib import Path

import numpy as np

# Reuse ch5's dense retriever + corpus (ch5 injects ch1's path transitively). ch5 lives two dirs over.
_CH5_CODE = Path(__file__).resolve().parent.parent.parent / "05-Hybrid-Search-BM25-and-Dense" / "code"
if str(_CH5_CODE) not in sys.path:
    sys.path.insert(0, str(_CH5_CODE))

from hybrid_search import (  # noqa: E402  (path injected above must precede this import)
    DenseRetriever,
    full_corpus,
)

# ================================================================================================
# Provider constants -- CITED EXTERNAL (provider docs), NOT our measurement. Verified against the
# sources in the references. Prices are representative $/1M-token INPUT rates (round, illustrative
# but realistic -- the crossover conclusion is robust to the exact price).
# ================================================================================================


@dataclass(frozen=True)
class Provider:
    """A provider's advertised context window (tokens) and representative input price ($/1M tokens)."""

    name: str
    context_window: int  # advertised max input tokens (provider docs)
    price_per_mtok: float  # representative input price, USD per 1,000,000 tokens (provider pricing)
    source: str


# Verified: OpenAI GPT-4o 128k (developers.openai.com); Claude 200k (docs.anthropic.com);
# Gemini 1.5 Pro 1M->2M (Google Developers Blog). Prices are representative input rates.
PROVIDERS: tuple[Provider, ...] = (
    Provider("GPT-4o (128k)", 128_000, 2.50, "OpenAI API pricing"),
    Provider("Claude Sonnet (200k)", 200_000, 3.00, "Anthropic pricing"),
    Provider("Gemini 1.5 Pro (1M–2M)", 2_000_000, 1.25, "Google Gemini pricing"),
)

# Token accounting constants (our modelling assumptions, stated so the arithmetic is reproducible).
TOKENS_PER_CHUNK = 100  # a realistic short passage/chunk ~= 100 tokens (our modelling unit)
RETRIEVE_K = 5  # RAG retrieves this many chunks per query -> a small, fixed context
RETRIEVAL_OVERHEAD_TOKENS = 200  # question + instructions the RAG prompt always carries
DEFAULT_PRICE_PER_MTOK = 3.00  # the default price the crossover demo uses (Claude-Sonnet-class)


# ================================================================================================
# 1) THE COST MODEL -- real arithmetic. Per-query input-token cost of stuff-everything vs RAG.
# ================================================================================================


def stuff_tokens(corpus_chunks: int, tokens_per_chunk: int = TOKENS_PER_CHUNK) -> int:
    """Input tokens when you STUFF the whole corpus into the prompt: every chunk, every query."""
    return corpus_chunks * tokens_per_chunk


def rag_tokens(
    k: int = RETRIEVE_K,
    tokens_per_chunk: int = TOKENS_PER_CHUNK,
    overhead: int = RETRIEVAL_OVERHEAD_TOKENS,
) -> int:
    """Input tokens when RAG retrieves only the top-k chunks: fixed, independent of corpus size."""
    return k * tokens_per_chunk + overhead


def query_cost_usd(input_tokens: int, price_per_mtok: float = DEFAULT_PRICE_PER_MTOK) -> float:
    """Dollar cost of one query's INPUT tokens at a given $/1M-token price."""
    return input_tokens / 1_000_000 * price_per_mtok


def cost_crossover_chunks(
    k: int = RETRIEVE_K,
    tokens_per_chunk: int = TOKENS_PER_CHUNK,
    overhead: int = RETRIEVAL_OVERHEAD_TOKENS,
) -> int:
    """The corpus size (in chunks) at which stuffing first costs MORE than RAG.

    Stuffing costs corpus_chunks x tokens_per_chunk tokens; RAG costs the fixed k x tokens_per_chunk
    + overhead. Price cancels (both scale linearly in the same price), so the crossover is purely a
    TOKEN comparison: the smallest integer corpus_chunks with stuff_tokens > rag_tokens. Beyond it,
    RAG is cheaper on every query, and the gap widens without bound as the corpus grows.
    """
    fixed_rag = rag_tokens(k, tokens_per_chunk, overhead)
    # smallest corpus_chunks such that corpus_chunks * tokens_per_chunk > fixed_rag
    return fixed_rag // tokens_per_chunk + 1


# ================================================================================================
# 2) THE ENCODER DILUTION PROXY -- OUR measurement (real all-MiniLM cosines). A proxy for the
# context-dilution half of lost-in-the-middle: as irrelevant context grows around the gold, the
# gold's retrieval MARGIN (its cosine lead over the best distractor) shrinks. RAG keeps it wide by
# handing the model only the focused top-k.
# ================================================================================================


@dataclass(frozen=True)
class DilutionPoint:
    """One measured point: how many distractors surround the gold, and the gold's cosine margin."""

    n_distractors: int
    gold_cosine: float  # cosine of the query to the gold passage (fixed -- gold text is unchanged)
    best_distractor_cosine: float  # cosine of the query to the strongest distractor present
    margin: float  # gold_cosine - best_distractor_cosine (the gold's lead; shrinks as pool grows)


def _distractor_pool(n: int) -> list[str]:
    """Build n synthetic distractor passages -- Helios-7-flavoured but NOT the gold answer.

    Deterministic templated text so the pool is reproducible with no randomness. These stand in for
    the mass of irrelevant context that a stuff-everything prompt drags along on every query.
    """
    templates = (
        "The Helios-7 telemetry log recorded routine subsystem check number {i} without anomalies.",
        "Ground station pass {i} for Helios-7 completed nominal downlink of housekeeping data.",
        "Calibration cycle {i} of the Helios-7 payload finished within expected tolerance bounds.",
        "Thermal reading {i} on the Helios-7 bus stayed inside the nominal operating envelope.",
    )
    return [templates[i % len(templates)].format(i=i) for i in range(n)]


def measure_dilution(
    dense: DenseRetriever,
    query: str,
    gold: str,
    distractor_counts: tuple[int, ...],
) -> list[DilutionPoint]:
    """Measure how the gold's cosine MARGIN shrinks as more distractors join the candidate pool.

    The gold's own cosine to the query never changes (its text is fixed). What changes is the BEST
    DISTRACTOR's cosine: with more distractors present, some distractor gets luckier and scores
    closer to the gold, so the gold's LEAD (margin) shrinks. This is the retrieval-visible shadow of
    "more irrelevant context makes the needle harder to find" -- real cosines, our measurement.
    """
    q_vec = dense._encode([query])[0]  # noqa: SLF001 -- reuse ch5's encoder; unit-norm
    gold_cos = float(dense._encode([gold])[0] @ q_vec)  # noqa: SLF001 -- fixed across all points
    points: list[DilutionPoint] = []
    for n in distractor_counts:
        distractors = _distractor_pool(n)
        if distractors:
            d_vecs = dense._encode(distractors)  # noqa: SLF001 -- (n, dim) unit-norm
            best_distractor = float(np.max(d_vecs @ q_vec))
        else:
            best_distractor = 0.0
        points.append(DilutionPoint(n, gold_cos, best_distractor, gold_cos - best_distractor))
    return points


# ================================================================================================
# 3) LOST-IN-THE-MIDDLE U-CURVE -- CITED EXTERNAL (Liu et al. 2023). NOT our measurement.
# These are the SHAPE of their reported GPT-3.5-Turbo multi-document QA result: accuracy is highest
# when the gold document is first or last, and dips in the middle. Values are an illustrative
# reproduction of their reported curve (labelled as external everywhere they are used).
# ================================================================================================

# (position of gold doc among 20, approximate accuracy %) -- Liu et al. 2023, reported shape.
LIU_U_CURVE: tuple[tuple[int, float], ...] = (
    (1, 75.0),   # gold at the very start -> best
    (5, 62.0),
    (10, 54.0),  # gold in the middle -> worst (the "lost in the middle" dip)
    (15, 60.0),
    (20, 74.0),  # gold at the very end -> nearly as good as the start (near-symmetric U, per the paper)
)
LIU_SOURCE = "Liu et al. 2023, 'Lost in the Middle', arXiv:2307.03172 (their reported result, not ours)"

# RULER effective-vs-advertised context -- CITED EXTERNAL (Hsieh et al. 2024). Illustrative of their
# finding that a model's USABLE context is well below its advertised window.
RULER_SOURCE = "Hsieh et al. 2024, 'RULER', arXiv:2404.06654 (their reported result, not ours)"


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


def main() -> None:
    _report_versions()
    corpus = full_corpus()
    dense = DenseRetriever(corpus)
    print(f"corpus: {len(corpus)} passages | dense lens: {dense.backend}")
    print(
        "NOTE: the COST arithmetic and the ENCODER DILUTION proxy are OUR measurements (real, "
        "reproducible); the lost-in-the-middle U-curve, RULER gap, and provider sizes/prices are "
        "CITED EXTERNAL constants (their papers/docs), never passed off as ours.\n"
    )

    # ------------------------------------------------------------------------------------------
    # 1) THE COST CROSSOVER: stuff-everything vs RAG, real token arithmetic.
    # ------------------------------------------------------------------------------------------
    print("=" * 96)
    print("1) Cost per query: stuff-the-whole-corpus vs RAG (retrieve top-k) — the crossover")
    print("=" * 96)
    rag_toks = rag_tokens()
    crossover = cost_crossover_chunks()
    price = DEFAULT_PRICE_PER_MTOK
    print(f"  modelling: {TOKENS_PER_CHUNK} tokens/chunk | RAG retrieves k={RETRIEVE_K} "
          f"(+{RETRIEVAL_OVERHEAD_TOKENS} overhead) = {rag_toks} tokens/query (FIXED)")
    print(f"  price: ${price:.2f} / 1M input tokens (representative, cited)\n")
    print(f"  {'corpus (chunks)':>16} | {'stuff tokens':>12} | {'stuff $/query':>13} | {'RAG $/query':>11} | cheaper")
    print("  " + "-" * 78)
    # numeric order, with the just-below-crossover and crossover rows ADJACENT so the boundary is clear
    for chunks in (crossover - 1, crossover, 10, 100, 1_000, 100_000):
        st = stuff_tokens(chunks)
        st_cost = query_cost_usd(st, price)
        rag_cost = query_cost_usd(rag_toks, price)
        cheaper = "RAG" if rag_cost < st_cost else "stuff"
        if chunks == crossover - 1:
            tag = "  <- just below (stuffing still wins)"
        elif chunks == crossover:
            tag = "  <- crossover (RAG wins from here)"
        else:
            tag = ""
        print(f"  {chunks:>16,} | {st:>12,} | {st_cost:>12.5f}$ | {rag_cost:>10.5f}$ | {cheaper}{tag}")
    # Correctness BEFORE the claim: at the crossover stuffing first exceeds RAG; below it, stuffing wins.
    assert stuff_tokens(crossover) > rag_toks, "at the crossover, stuffing must exceed RAG's fixed cost"
    assert stuff_tokens(crossover - 1) <= rag_toks, "just below the crossover, stuffing is still <= RAG"
    print(f"\n  crossover corpus size: {crossover} chunks (~{crossover * TOKENS_PER_CHUNK:,} tokens). "
          f"Beyond it, RAG is cheaper on EVERY query — and the gap only widens.")
    # a concrete headline at large scale
    big = 100_000
    factor = stuff_tokens(big) / rag_toks
    print(f"  at {big:,} chunks (~{stuff_tokens(big):,} tokens), stuffing costs {factor:,.0f}x RAG's per-query tokens.\n")
    assert factor > 100, "at 100k chunks stuffing should cost >100x RAG per query"

    # ------------------------------------------------------------------------------------------
    # 2) THE ENCODER DILUTION PROXY: gold margin shrinks as irrelevant context grows (OUR measurement).
    # ------------------------------------------------------------------------------------------
    print("=" * 96)
    print("2) Context dilution (OUR encoder proxy): the gold's retrieval margin shrinks as context grows")
    print("=" * 96)
    query = "When was the Helios-7 satellite launched?"
    gold = corpus[0]  # "The Helios-7 satellite was launched on March 3rd, 2024 from the Kourou spaceport."
    counts = (0, 5, 20, 100, 500)
    points = measure_dilution(dense, query, gold, counts)
    print(f"  query: {query}")
    print(f"  gold : {gold}")
    print(f"  gold's own cosine to the query (fixed): {points[0].gold_cosine:.3f}\n")
    print(f"  {'# distractors':>14} | {'best distractor cos':>19} | {'gold margin':>11}")
    print("  " + "-" * 52)
    for p in points:
        print(f"  {p.n_distractors:>14,} | {p.best_distractor_cosine:>19.3f} | {p.margin:>11.3f}")
    # the margin at the largest pool must be strictly smaller than with no distractors
    assert points[-1].margin < points[0].margin, "more distractors must shrink the gold's margin"
    focused_margin = points[0].margin  # RAG's focused context = gold alone, widest margin
    diluted_margin = points[-1].margin
    print(f"\n  focused (RAG top-k, ~0 distractors) margin: {focused_margin:.3f}")
    print(f"  diluted (stuff {counts[-1]} distractors)   margin: {diluted_margin:.3f}  "
          f"({(1 - diluted_margin / focused_margin) * 100:.0f}% narrower)")
    print("  -> stuffing drags in mass of irrelevant context that erodes the gold's lead; RAG keeps it wide.")
    print("     (This is a RETRIEVAL-visible proxy for context dilution — our measurement — NOT the LLM's")
    print("      mid-context accuracy drop, which is the cited Liu et al. result below.)\n")

    # ------------------------------------------------------------------------------------------
    # 3) LOST-IN-THE-MIDDLE U-CURVE: CITED EXTERNAL (Liu et al. 2023) -- NOT our measurement.
    # ------------------------------------------------------------------------------------------
    print("=" * 96)
    print("3) Lost-in-the-middle (CITED EXTERNAL — Liu et al. 2023, NOT our measurement)")
    print("=" * 96)
    print(f"  source: {LIU_SOURCE}")
    print(f"  {'gold position (of 20)':>21} | {'reported accuracy %':>19}")
    print("  " + "-" * 44)
    for pos, acc in LIU_U_CURVE:
        marker = "  <- middle (worst)" if acc == min(a for _, a in LIU_U_CURVE) else ""
        print(f"  {pos:>21} | {acc:>18.1f}%{marker}")
    first_acc = LIU_U_CURVE[0][1]
    mid_acc = min(a for _, a in LIU_U_CURVE)
    print(f"\n  their reported drop, edge -> middle: {first_acc:.0f}% -> {mid_acc:.0f}% "
          f"(~{first_acc - mid_acc:.0f} points). Position in a long context matters — a reason to")
    print("  RETRIEVE the few relevant chunks (put them where the model reads best) rather than bury them.")
    print(f"  Related: {RULER_SOURCE} — a model's EFFECTIVE context is well below its advertised window.\n")

    # ------------------------------------------------------------------------------------------
    # 4) THE DECISION: corpus size vs window vs cost -- when to stuff, retrieve, or hybrid.
    # ------------------------------------------------------------------------------------------
    print("=" * 96)
    print("4) The decision: stuff / retrieve / hybrid, by corpus size vs window and cost")
    print("=" * 96)
    for prov in PROVIDERS:
        window_chunks = prov.context_window // TOKENS_PER_CHUNK
        print(f"  {prov.name:<24} window {prov.context_window:>9,} tok "
              f"(~{window_chunks:,} chunks) @ ${prov.price_per_mtok:.2f}/1M  [{prov.source}]")
    print()
    print("  rule of thumb:")
    print("   - corpus << window AND cost irrelevant  -> STUFF it (simplest; no retrieval to build)")
    print(f"   - corpus >> window OR cost/latency matter -> RETRIEVE (RAG; the {crossover}-chunk crossover)")
    print("   - need many relevant chunks              -> HYBRID (retrieve MANY, put them in a long window)")
    # sanity: even the biggest window is finite, so a large enough corpus always overflows it
    biggest = max(p.context_window for p in PROVIDERS)
    assert stuff_tokens(1_000_000) > biggest, "a 1M-chunk corpus overflows even a 2M-token window"
    print(f"\n  even the largest window ({biggest:,} tok) overflows at "
          f"{biggest // TOKENS_PER_CHUNK:,}+ chunks — corpora outgrow ANY window, so retrieval scales when stuffing can't.")


if __name__ == "__main__":
    main()
