"""From-scratch semantic cache + cost model for LLM apps: serve repeated/similar work for free.

Every query to a RAG app costs money and time: embed + retrieve + an LLM call ($/token + latency).
At scale, a query stream is full of REPEATS and PARAPHRASES -- "how tall is the imager?" and "imager
height?" want the same answer -- and a naive app re-pays the FULL cost for each. A SEMANTIC CACHE
serves the cached answer whenever a new query is close enough (by embedding cosine) to one it has
already answered, so repeated/similar work becomes nearly free. The saving is exactly
hit_rate x per-query cost.

This module builds a semantic cache from primitives, then MEASURES it on a query stream with repeats
and paraphrases:
  * embed the query (REAL ch5 all-MiniLM encoder), find the nearest cached query by cosine;
  * if the best match clears a threshold tau -> HIT (return the cached answer, pay only the tiny
    lookup-embedding cost); else MISS -> compute the answer, pay the full cost, and STORE it;
  * report HIT RATE, cost saved, latency saved;
  * SWEEP tau to expose the FALSE-HIT vs MISS tradeoff -- a too-low tau serves a genuinely different
    query the WRONG cached answer (cosine != exact intent), a too-high tau misses real paraphrases.

The cost model (per-query token cost) is reused from ch12; the encoder from ch5.

HONESTY -- WHAT RUNS REAL vs WHAT IS ILLUSTRATIVE
-------------------------------------------------
  * REAL and measured: the SEMANTIC CACHE (embed -> nearest cached query by cosine -> HIT/MISS at a
    threshold), the HIT RATE over a real query stream, and the COST/LATENCY arithmetic (ch12's token
    cost model + a modelled per-call latency). Every printed number is computed here, asserted before
    it is claimed, and reproducible.
  * ILLUSTRATIVE (labelled): the ANSWER text a MISS "computes" is a fixed template (no LLM in this
    env); the per-call LATENCY is a modelled constant (a real LLM call is ~hundreds of ms, a cache
    hit ~a few ms). The CACHE MECHANISM and the hit-rate/cost math are real; only the answer text and
    the exact latency constants are stand-ins.

CARRIED-FORWARD CAVEAT (ch11/ch13/ch14, still true): the cache matches by COSINE, which measures
TOPICAL similarity, not exact intent -- so a too-low threshold serves a FALSE HIT (a near-but-different
query gets the wrong cached answer). This is the same cosine!=exact-meaning gap the earlier chapters
carry; here it is the false-hit risk, shown explicitly in the threshold sweep.

The encoder + corpus come from ch5 (via ch13); the cost model from ch12, so the RAG chapters share ONE
source of truth.

Verified on Python 3.12.x / numpy 2.4.6 / torch 2.12.0 / sentence-transformers (all-MiniLM-L6-v2,
CPU). Deterministic: identical numbers every run given the same cached model.

Run:
    python caching_cost.py
"""

from __future__ import annotations

import sys
from dataclasses import dataclass
from pathlib import Path

import numpy as np

# Reuse ch13's encoder (which re-exports ch5's DenseRetriever) and ch12's cost model. Inject their
# code dirs so imports work whether this file is run from its own dir or imported by the notebook /
# figure scripts.
_APP = Path(__file__).resolve().parent.parent.parent
for _rel in (
    ("13-Citations-and-Attribution", "code"),
    ("12-Long-Context-vs-RAG", "code"),
):
    _p = _APP.joinpath(*_rel)
    if str(_p) not in sys.path:
        sys.path.insert(0, str(_p))

from citations_attribution import DenseRetriever, full_corpus  # noqa: E402  (paths injected above)
from long_context_vs_rag import query_cost_usd, rag_tokens  # noqa: E402  (ch12 cost model)

DENSE_MODEL = "all-MiniLM-L6-v2"

# ---- Cache threshold ----------------------------------------------------------------------------
# A new query is a HIT on a cached query only if their cosine clears this bar. 0.8 is a deliberately
# HIGH bar (higher than the 0.5 grounding bar of ch14): a cache serves a *stored answer verbatim*, so
# a wrong hit is worse than a wrong retrieval -- we want near-duplicate paraphrases, not merely
# same-topic queries. The sweep below shows why: drop tau too low and a different-intent query gets a
# FALSE HIT. It is a computable proxy for "same question", carried caveat: cosine != exact intent.
CACHE_THRESHOLD = 0.8

# ---- Cost / latency constants (ch12 cost model + modelled latency) ------------------------------
# Per-query cost of a full (MISS) call: RAG input tokens (ch12's rag_tokens) at the default price,
# plus a small output-token cost. A cache HIT pays only the tiny lookup-embedding cost.
FULL_CALL_TOKENS = rag_tokens()  # ch12: k*tokens_per_chunk + overhead = the RAG prompt size
OUTPUT_TOKENS = 60  # a short answer; billed at the same modelled input price for simplicity
LOOKUP_TOKENS = 12  # embedding the query for the cache lookup -- the only cost a HIT pays
# Modelled latency (ILLUSTRATIVE constants; a real LLM call is ~hundreds of ms, a cache hit ~a few ms):
FULL_CALL_MS = 800.0  # a full retrieve+generate round trip
HIT_MS = 5.0  # a cache hit: one embedding + a nearest-neighbour lookup


@dataclass(frozen=True)
class CacheEntry:
    """One stored (query, answer) pair plus the query's unit-norm embedding (for fast cosine lookup)."""

    query: str
    answer: str
    embedding: np.ndarray  # unit-norm, so cosine to a new query is a plain dot product


@dataclass(frozen=True)
class LookupResult:
    """The outcome of one cache lookup: hit?, the served answer, the best cosine, and its entry index."""

    hit: bool
    answer: str
    best_cosine: float
    best_index: int  # index into the cache's entries, or -1 if the cache was empty


# ================================================================================================
# The semantic cache: embed the query, find the nearest stored query, HIT if cosine >= threshold.
# ================================================================================================


class SemanticCache:
    """Serve a cached answer when a new query is semantically close to one already answered.

    The whole mechanism, from primitives:
      * `lookup(query)` embeds the query (REAL ch5 encoder), computes its cosine to every cached
        query's embedding, and returns a HIT with the stored answer iff the best cosine clears the
        threshold -- else a MISS.
      * `store(query, answer)` embeds and appends a new entry (what a MISS does after computing).
    Cosine is a plain dot product because all embeddings are unit-norm (ch5's geometry). Ties in the
    nearest-neighbour search are broken by np.argmax (first occurrence) -- deterministic.
    """

    def __init__(self, dense: DenseRetriever, threshold: float = CACHE_THRESHOLD) -> None:
        self._dense = dense
        self.threshold = threshold
        self.entries: list[CacheEntry] = []

    def _embed(self, text: str) -> np.ndarray:
        """Unit-norm embedding of one text via ch5's encoder (the same lens the whole corpus uses)."""
        return self._dense._encode([text])[0]  # noqa: SLF001 -- reuse ch5's encoder; unit-norm

    def lookup(self, query: str) -> LookupResult:
        """Return the nearest cached answer as a HIT iff its cosine clears the threshold, else a MISS."""
        if not self.entries:  # cold start: an empty cache always misses
            return LookupResult(hit=False, answer="", best_cosine=0.0, best_index=-1)
        q_vec = self._embed(query)
        cosines = np.array([q_vec @ e.embedding for e in self.entries])  # unit-norm => dot == cosine
        best = int(np.argmax(cosines))  # first-occurrence tie-break -> deterministic
        best_cos = float(cosines[best])
        if best_cos >= self.threshold:
            return LookupResult(hit=True, answer=self.entries[best].answer, best_cosine=best_cos, best_index=best)
        return LookupResult(hit=False, answer="", best_cosine=best_cos, best_index=best)

    def store(self, query: str, answer: str) -> None:
        """Embed and append a new (query, answer) entry -- what a MISS does after computing the answer."""
        self.entries.append(CacheEntry(query=query, answer=answer, embedding=self._embed(query)))


# ================================================================================================
# The cost model: per-query cost of a full call (MISS) vs a cache hit; expected cost under a hit rate.
# ================================================================================================


def full_call_cost_usd() -> float:
    """Dollar cost of a MISS: the full RAG prompt (input) + output tokens at ch12's default price."""
    return query_cost_usd(FULL_CALL_TOKENS + OUTPUT_TOKENS)


def hit_cost_usd() -> float:
    """Dollar cost of a HIT: only the query-embedding lookup tokens (no LLM generation)."""
    return query_cost_usd(LOOKUP_TOKENS)


def expected_cost_per_query(hit_rate: float) -> float:
    """Expected per-query cost under a given hit rate: (1-h)*full + h*hit.

    The core caching identity: a fraction h of queries are served from cache at the tiny hit cost, the
    rest pay the full call. Savings vs no cache = h * (full - hit) per query -- linear in the hit rate.
    """
    return (1.0 - hit_rate) * full_call_cost_usd() + hit_rate * hit_cost_usd()


def savings_fraction(hit_rate: float) -> float:
    """Fraction of the no-cache cost saved at a given hit rate: 1 - expected/full."""
    full = full_call_cost_usd()
    return 1.0 - expected_cost_per_query(hit_rate) / full


# ================================================================================================
# The query stream + the run: process a stream, count hits, measure cost/latency saved.
# ================================================================================================


@dataclass(frozen=True)
class StreamResult:
    """The outcome of running a query stream through the cache: per-query hit flags + the aggregates."""

    hits: tuple[bool, ...]  # per-query: was it a cache hit?
    hit_rate: float
    cost_no_cache_usd: float  # every query pays the full call
    cost_cached_usd: float  # hits pay only the lookup
    latency_no_cache_ms: float
    latency_cached_ms: float

    @property
    def cost_saved_usd(self) -> float:
        return self.cost_no_cache_usd - self.cost_cached_usd

    @property
    def latency_saved_ms(self) -> float:
        return self.latency_no_cache_ms - self.latency_cached_ms


def run_stream(dense: DenseRetriever, stream: tuple[str, ...], answer_of: dict[str, str], threshold: float = CACHE_THRESHOLD) -> StreamResult:
    """Process a query stream through a fresh cache; return per-query hits + cost/latency aggregates.

    For each query: look it up; on a HIT serve the cached answer (pay the lookup cost only); on a MISS
    compute the answer (illustrative: from `answer_of`), pay the full call, and store it. The first
    time a distinct question is seen it MUST miss (cold entry); repeats and near-paraphrases of it hit.
    """
    cache = SemanticCache(dense, threshold=threshold)
    hits: list[bool] = []
    cost_cached = 0.0
    latency_cached = 0.0
    for query in stream:
        result = cache.lookup(query)
        if result.hit:
            hits.append(True)
            cost_cached += hit_cost_usd()
            latency_cached += HIT_MS
        else:
            hits.append(False)
            answer = answer_of[query]  # illustrative: the "computed" answer for a fresh query
            cache.store(query, answer)
            cost_cached += full_call_cost_usd()
            latency_cached += FULL_CALL_MS
    n = len(stream)
    hit_rate = sum(hits) / n if n else 0.0
    return StreamResult(
        hits=tuple(hits),
        hit_rate=hit_rate,
        cost_no_cache_usd=n * full_call_cost_usd(),
        cost_cached_usd=cost_cached,
        latency_no_cache_ms=n * FULL_CALL_MS,
        latency_cached_ms=latency_cached,
    )


# ================================================================================================
# The false-hit / miss tradeoff: sweep the threshold over a labelled probe set.
# ================================================================================================


@dataclass(frozen=True)
class CacheProbe:
    """One labelled probe: a query, the cached query it is compared against, and whether it SHOULD hit."""

    query: str
    cached_query: str
    should_hit: bool  # gold: True = a genuine paraphrase (should hit); False = different intent (must miss)
    label: str


def false_hit_and_miss_rates(dense: DenseRetriever, probes: tuple[CacheProbe, ...], threshold: float) -> tuple[float, float]:
    """Return (false_hit_rate, miss_rate) for the cache at `threshold` over the labelled probes.

    * FALSE HIT: of the probes that should MISS (different intent), the fraction whose cosine wrongly
      clears the threshold -- a genuinely different query served the WRONG cached answer.
    * MISS (of should-hit): of the probes that should HIT (real paraphrase), the fraction whose cosine
      wrongly falls below the threshold -- a real paraphrase not caught, so it re-pays the full cost.
    Raising the threshold trades one for the other -- the same false-hit/false-refuse shape as ch14.
    """
    def cos(a: str, b: str) -> float:
        va = dense._encode([a])[0]  # noqa: SLF001 -- unit-norm
        vb = dense._encode([b])[0]  # noqa: SLF001
        return float(va @ vb)

    should_hit = [p for p in probes if p.should_hit]
    should_miss = [p for p in probes if not p.should_hit]
    false_hit = np.mean([cos(p.query, p.cached_query) >= threshold for p in should_miss]) if should_miss else 0.0
    miss = np.mean([cos(p.query, p.cached_query) < threshold for p in should_hit]) if should_hit else 0.0
    return float(false_hit), float(miss)


# ================================================================================================
# The demo fixtures: a query stream (repeats + paraphrases) and the false-hit probe set.
# ================================================================================================

# The distinct questions the corpus can answer, each with its illustrative "computed" answer.
BASE_ANSWERS: dict[str, str] = {
    "What is the ground resolution of the Helios-7 imager?": "The Helios-7 imager has a ground resolution of 4 meters.",
    "When was Helios-7 launched?": "Helios-7 launched on March 3rd, 2024 from the Kourou spaceport.",
    "Who is the Helios-7 project lead?": "The Helios-7 project lead is Dr. Amara Okoye, based in the Nairobi office.",
}

# A realistic query stream: the three base questions, plus PARAPHRASES (should hit) and REPEATS (hit).
# The order matters -- the first occurrence of each distinct question misses (cold), the rest hit.
QUERY_STREAM: tuple[str, ...] = (
    "What is the ground resolution of the Helios-7 imager?",  # MISS (cold) -> stores
    "When was Helios-7 launched?",  # MISS (cold) -> stores
    "What is the ground resolution of the Helios-7 imager?",  # exact REPEAT -> HIT
    "How fine is the Helios-7 imager's ground resolution?",  # PARAPHRASE of imager -> HIT
    "Who is the Helios-7 project lead?",  # MISS (cold) -> stores
    "What date did Helios-7 lift off?",  # PARAPHRASE of launch -> HIT
    "Who leads the Helios-7 project?",  # PARAPHRASE of lead -> HIT
    "When was Helios-7 launched?",  # exact REPEAT -> HIT
)

# Every stream query resolves to a base answer (paraphrases share their base's answer).
STREAM_ANSWERS: dict[str, str] = {
    "What is the ground resolution of the Helios-7 imager?": BASE_ANSWERS["What is the ground resolution of the Helios-7 imager?"],
    "When was Helios-7 launched?": BASE_ANSWERS["When was Helios-7 launched?"],
    "Who is the Helios-7 project lead?": BASE_ANSWERS["Who is the Helios-7 project lead?"],
    "How fine is the Helios-7 imager's ground resolution?": BASE_ANSWERS["What is the ground resolution of the Helios-7 imager?"],
    "What date did Helios-7 lift off?": BASE_ANSWERS["When was Helios-7 launched?"],
    "Who leads the Helios-7 project?": BASE_ANSWERS["Who is the Helios-7 project lead?"],
}


def build_probes() -> tuple[CacheProbe, ...]:
    """Probes for the false-hit/miss sweep: genuine paraphrases (should hit) + near-but-different (must miss).

    The should-MISS probes are the dangerous ones: a query TOPICALLY near a cached query but with a
    DIFFERENT intent (a different Helios-7 attribute). At a low threshold cosine serves them the wrong
    cached answer -- the false hit. At a high threshold they correctly miss.
    """
    return (
        # genuine paraphrases -> SHOULD HIT
        CacheProbe("How fine is the Helios-7 imager's ground resolution?", "What is the ground resolution of the Helios-7 imager?", True, "paraphrase: imager resolution"),
        CacheProbe("What date did Helios-7 lift off?", "When was Helios-7 launched?", True, "paraphrase: launch date"),
        CacheProbe("Who leads the Helios-7 project?", "Who is the Helios-7 project lead?", True, "paraphrase: project lead"),
        # near-but-DIFFERENT intent -> MUST MISS (a false hit here serves the wrong answer)
        CacheProbe("What is the orbital period of Helios-7?", "What is the ground resolution of the Helios-7 imager?", False, "different: orbit vs imager"),
        CacheProbe("Where is the Helios-7 project lead based?", "Who is the Helios-7 project lead?", False, "different: WHERE vs WHO (lead)"),
        CacheProbe("From which spaceport did Helios-7 launch?", "When was Helios-7 launched?", False, "different: WHERE vs WHEN (launch)"),
    )


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
        print("torch: not installed (the cost model is pure arithmetic; the cache needs the encoder)")


def main() -> None:
    _report_versions()
    corpus = full_corpus()
    dense = DenseRetriever(corpus)
    print(f"corpus: {len(corpus)} passages | dense lens: {dense.backend} | cache threshold: {CACHE_THRESHOLD}")
    print(f"cost model (ch12): full call = {FULL_CALL_TOKENS + OUTPUT_TOKENS} tok = ${full_call_cost_usd():.6f}; "
          f"cache hit = {LOOKUP_TOKENS} tok = ${hit_cost_usd():.6f}")
    print(
        "NOTE: the semantic cache + hit rate + cost/latency arithmetic are REAL and measured; only the "
        "MISS answer text and the exact latency constants are illustrative stand-ins.\n"
    )

    # ------------------------------------------------------------------------------------------
    # 1) THE STREAM: run a query stream (repeats + paraphrases) through the cache; count hits.
    # ------------------------------------------------------------------------------------------
    print("=" * 96)
    print("1) A query stream through the semantic cache: repeats and paraphrases become HITS")
    print("=" * 96)
    result = run_stream(dense, QUERY_STREAM, STREAM_ANSWERS)
    cache = SemanticCache(dense)  # a parallel cache to report per-query cosines for the trace
    for query, hit in zip(QUERY_STREAM, result.hits):
        lookup = cache.lookup(query)
        verdict = f"HIT  (cos {lookup.best_cosine:.3f} >= {CACHE_THRESHOLD})" if hit else (
            f"MISS (cos {lookup.best_cosine:.3f} < {CACHE_THRESHOLD}) -> compute + store" if cache.entries
            else "MISS (cold cache) -> compute + store")
        print(f"  {verdict:<44} {query}")
        if not hit:
            cache.store(query, STREAM_ANSWERS[query])
    print(f"\n  hit rate: {sum(result.hits)}/{len(result.hits)} = {result.hit_rate:.3f}")
    # Correctness BEFORE the claim: 3 distinct questions -> 3 cold misses; the other 5 hit.
    assert sum(result.hits) == 5, "5 of the 8 stream queries are repeats/paraphrases -> hits"
    assert result.hit_rate == 5 / 8, "hit rate is 5/8 on this stream"
    assert result.hits == (False, False, True, True, False, True, True, True), "the exact hit pattern (cold misses first)"
    print("  -> the 3 distinct questions miss once (cold); every repeat and paraphrase after is a HIT.\n")

    # ------------------------------------------------------------------------------------------
    # 2) THE SAVINGS: cost and latency saved on the stream, and the expected-cost identity.
    # ------------------------------------------------------------------------------------------
    print("=" * 96)
    print("2) The payoff: cost and latency saved = hit_rate x per-query cost")
    print("=" * 96)
    print(f"  cost   : no-cache ${result.cost_no_cache_usd:.6f}  ->  cached ${result.cost_cached_usd:.6f}  "
          f"(saved ${result.cost_saved_usd:.6f}, {result.cost_saved_usd / result.cost_no_cache_usd:.1%})")
    print(f"  latency: no-cache {result.latency_no_cache_ms:.0f}ms  ->  cached {result.latency_cached_ms:.0f}ms  "
          f"(saved {result.latency_saved_ms:.0f}ms, {result.latency_saved_ms / result.latency_no_cache_ms:.1%})")
    # the expected-cost identity should match the measured cached cost (within float noise)
    predicted = len(QUERY_STREAM) * expected_cost_per_query(result.hit_rate)
    print(f"  identity check: {len(QUERY_STREAM)} x expected_cost_per_query({result.hit_rate:.3f}) = ${predicted:.6f} "
          f"== measured cached ${result.cost_cached_usd:.6f}")
    assert abs(predicted - result.cost_cached_usd) < 1e-9, "the expected-cost identity must match the measured cost"
    assert result.cost_saved_usd > 0 and result.latency_saved_ms > 0, "caching saves both cost and latency here"
    # savings fraction is (essentially) the hit rate, because a hit is ~free vs a full call
    assert abs(savings_fraction(result.hit_rate) - result.hit_rate) < 0.02, "savings fraction ~ hit rate (a hit is nearly free)"
    print(f"  -> savings ~ hit rate: at {result.hit_rate:.1%} hits you cut ~{savings_fraction(result.hit_rate):.1%} of the bill.\n")

    # ------------------------------------------------------------------------------------------
    # 3) THE FALSE-HIT / MISS TRADEOFF: sweep the threshold over the labelled probes.
    # ------------------------------------------------------------------------------------------
    print("=" * 96)
    print("3) The false-hit / miss tradeoff: too-low tau serves the WRONG answer, too-high misses paraphrases")
    print("=" * 96)
    probes = build_probes()
    thresholds = (0.50, 0.60, 0.70, 0.80, 0.90)
    print(f"  {'threshold':>9} | {'false-hit':>9} | {'miss (of should-hit)':>21}")
    print("  " + "-" * 46)
    rates: list[tuple[float, float, float]] = []
    for tau in thresholds:
        fh, miss = false_hit_and_miss_rates(dense, probes, tau)
        rates.append((tau, fh, miss))
        print(f"  {tau:>9.2f} | {fh:>9.3f} | {miss:>21.3f}")
    fhs = [fh for _, fh, _ in rates]
    misses = [m for _, _, m in rates]
    # Correctness BEFORE the claim: false-hit falls as tau rises; miss rises as tau rises (monotone).
    assert fhs == sorted(fhs, reverse=True), "false-hit must be non-increasing as tau rises"
    assert misses == sorted(misses), "miss must be non-decreasing as tau rises"
    assert fhs[0] > fhs[-1], "a low threshold produces MORE false hits than a high one (the tradeoff is real)"
    print("\n  -> raising tau cuts false hits but misses more paraphrases: the semantic-cache tradeoff.")
    print("     A false hit serves the WRONG cached answer -- cosine matches TOPIC, not exact intent.\n")

    # ------------------------------------------------------------------------------------------
    # 4) THE FALSE HIT, CONCRETELY: at a low tau a different-intent query gets the wrong cached answer.
    # ------------------------------------------------------------------------------------------
    print("=" * 96)
    print("4) A false hit, concretely: a 'wavelength' query wrongly served the 'resolution' answer")
    print("=" * 96)
    cached_q = "What is the ground resolution of the Helios-7 imager?"
    danger = "What is the wavelength range of the Helios-7 imager?"  # DIFFERENT imager attribute
    low_tau = 0.50
    cache = SemanticCache(dense, threshold=low_tau)
    cache.store(cached_q, BASE_ANSWERS[cached_q])
    lookup = cache.lookup(danger)
    print(f"  cache holds: '{cached_q}' -> '{BASE_ANSWERS[cached_q]}'")
    print(f"  new query  : '{danger}'  (a DIFFERENT question -- asks WAVELENGTH, not RESOLUTION)")
    print(f"  at tau={low_tau}: cosine {lookup.best_cosine:.3f} -> {'FALSE HIT' if lookup.hit else 'correct miss'}")
    if lookup.hit:
        print(f"    served (WRONG): '{lookup.answer}'  <- this answers RESOLUTION, not WAVELENGTH")
    # Correctness BEFORE the claim: at low tau this different-intent query wrongly hits.
    assert lookup.hit, "at tau=0.50 the wavelength query wrongly hits the resolution answer (a false hit)"
    # and at the default (high) tau it correctly misses
    strict = SemanticCache(dense, threshold=CACHE_THRESHOLD)
    strict.store(cached_q, BASE_ANSWERS[cached_q])
    strict_lookup = strict.lookup(danger)
    assert not strict_lookup.hit, f"at tau={CACHE_THRESHOLD} the different-intent query correctly misses"
    print(f"  at tau={CACHE_THRESHOLD} (default): same query, cosine {strict_lookup.best_cosine:.3f} < {CACHE_THRESHOLD} -> correct MISS, no wrong answer served.")
    print("  -> the fix for false hits is a HIGH threshold (near-duplicates only) + an exact-match fast path.\n")


if __name__ == "__main__":
    main()
