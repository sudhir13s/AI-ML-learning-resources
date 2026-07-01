"""From-scratch orchestrator for an LLM app: compose typed steps into a chain, route, and run a graph.

A real LLM app is not one prompt -- it is a PIPELINE: retrieve -> rerank -> guardrail -> generate ->
format, often with BRANCHING (route to the right path by query type), state passing, retries, and
error handling. Wire that as one giant glue function and it breaks: no retry, no branching, untestable,
unobservable -- when it fails you can't see WHICH step failed. An orchestrator makes the pipeline a
first-class object: typed STEPS that transform a shared STATE, a CHAIN that composes them, a ROUTER
that picks the path, and a GRAPH that adds cycles/retries -- with a TRACE you can inspect.

This module builds that orchestrator from primitives, then WIRES a real mini-RAG app out of the
earlier chapters' actual steps:
  * RETRIEVE  -- ch5's all-MiniLM DenseRetriever (real dense retrieval);
  * RERANK    -- ch6's CrossEncoderReranker (real joint-encoding re-rank, with its transparent fallback);
  * GUARDRAIL -- ch14's grounding-based abstention gate (emit iff grounded, else "I don't know");
  * ROUTE     -- a cosine router (ch10's idea: score the query against each path's description).

The headline demonstration: the app runs END TO END and produces a step trace; the router sends a
fact-lookup query down the RAG path and a chit-chat query down a direct-answer path; and a RETRY
recovers a simulated transient failure (a step that fails once, then succeeds) -- all measured.

HONESTY -- WHAT RUNS REAL vs WHAT IS ILLUSTRATIVE
-------------------------------------------------
  * REAL and measured: the ORCHESTRATION PRIMITIVES (Step protocol, Chain compose, Router argmax,
    StatefulGraph run-loop with a step budget, retry/backoff), and every WIRED step -- retrieval
    (ch5), re-rank (ch6), the grounding gate (ch14), and the cosine route (ch10). Every step trace,
    router score, and retry count printed here is computed and asserted before it is claimed.
  * ILLUSTRATIVE (labelled): the GENERATE step's answer TEXT is a fixed template over the top passage
    (no LLM in this env). The pipeline that produces it -- retrieve, rerank, ground-check, format --
    is real; only the final natural-language generation is a stand-in for an LLM call.

CARRIED-FORWARD CAVEAT (ch11/ch13/ch14, still true): the grounding gate uses encoder cosine, which
scores TOPIC not ENTAILMENT -- so the guardrail step is a real signal, not a perfect one. The
orchestration is honest about wiring real steps; the steps keep their own documented limits.

The retriever/corpus come from ch5 (via ch13), the reranker from ch6, and the grounding gate from
ch14, so the RAG chapters share ONE source of truth.

Verified on Python 3.12.x / numpy 2.4.6 / torch 2.12.0 / sentence-transformers (all-MiniLM-L6-v2,
CPU). Deterministic: identical numbers every run given the same cached model.

Run:
    python orchestration.py
"""

from __future__ import annotations

import sys
import time
from collections.abc import Callable
from dataclasses import dataclass, field, replace
from pathlib import Path
from typing import Protocol

import numpy as np

# Reuse the earlier chapters' real steps. ch13 re-exports ch5's DenseRetriever + corpus; ch6 has the
# reranker; ch14 has the grounding gate. Inject their code dirs so imports work whether this file is
# run from its own dir or imported by the notebook / figure scripts.
_APP = Path(__file__).resolve().parent.parent.parent
for _rel in (
    ("13-Citations-and-Attribution", "code"),
    ("06-Re-ranking-Cross-Encoders", "code"),
    ("14-Guardrails-and-Hallucination-Mitigation", "code"),
):
    _p = _APP.joinpath(*_rel)
    if str(_p) not in sys.path:
        sys.path.insert(0, str(_p))

from citations_attribution import (  # noqa: E402  (paths injected above must precede these imports)
    DenseRetriever,
    full_corpus,
)
from guardrails import (  # noqa: E402
    ABSTAIN_MESSAGE,
    GROUNDING_THRESHOLD,
    answer_grounding,
)
from reranking import CrossEncoderReranker  # noqa: E402

DENSE_MODEL = "all-MiniLM-L6-v2"
RETRIEVE_K = 4  # first-stage candidate pool the reranker re-orders
TOP_K = 2  # passages kept after re-rank and handed to generation
MAX_GRAPH_STEPS = 8  # the run-loop budget: a cyclic graph must terminate within this many hops


# ================================================================================================
# The shared STATE: an immutable record threaded through every step (each step returns a NEW state).
# ================================================================================================


@dataclass(frozen=True)
class AppState:
    """The typed state threaded through the pipeline. Immutable: each step returns a modified COPY.

    Immutability is the fix for the state-mutation bug (a step quietly clobbering a field another step
    relies on): because `replace()` returns a new object, a step can only ADD to the state, never
    silently overwrite the original -- and the trace records every version.
    """

    query: str
    route: str = ""  # which path the router chose ("rag" or "direct")
    retrieved: tuple[int, ...] = ()  # candidate doc indices from first-stage retrieval
    reranked: tuple[int, ...] = ()  # doc indices after re-rank (best-first)
    context: tuple[str, ...] = ()  # the passages handed to generation
    answer: str = ""  # the generated (or abstained) answer
    grounding: float = 0.0  # the answer's grounding score (from the guardrail step)
    abstained: bool = False  # did the guardrail refuse?
    trace: tuple[str, ...] = ()  # human-readable log of each step that ran (for observability)

    def log(self, message: str) -> AppState:
        """Return a copy with `message` appended to the trace -- the observability primitive."""
        return replace(self, trace=(*self.trace, message))


# ================================================================================================
# The Step protocol + Chain (sequential composition) -- the WHAT (steps) separated from the HOW (run).
# ================================================================================================


class Step(Protocol):
    """A step is any callable state -> state with a name. This is the single orchestration contract.

    Everything downstream (Chain, Router, StatefulGraph) composes objects that satisfy THIS protocol,
    so a step can be a plain function, a wired retriever, or a whole sub-chain -- they're interchangeable.
    """

    name: str

    def __call__(self, state: AppState) -> AppState: ...


@dataclass(frozen=True)
class FnStep:
    """Wrap a plain `state -> state` function as a named Step (the concrete Step implementation)."""

    name: str
    fn: Callable[[AppState], AppState]

    def __call__(self, state: AppState) -> AppState:
        return self.fn(state)


@dataclass(frozen=True)
class Chain:
    """A sequential composition of steps: run each in order, threading the state through.

    This is function composition over a typed state -- `Chain([a, b, c])(s) == c(b(a(s)))` -- PLUS the
    two things a bare `c(b(a(s)))` doesn't give you: a name for every step (observability) and a single
    place to add cross-cutting concerns (the run loop below can wrap each call in retry/timing).
    """

    steps: tuple[Step, ...]

    def __call__(self, state: AppState) -> AppState:
        for step in self.steps:
            state = step(state)
        return state


# ================================================================================================
# The Router -- pick the next path by scoring the query against each path's description (ch10's idea).
# ================================================================================================


@dataclass(frozen=True)
class Route:
    """One routable path: a name, a natural-language description (embedded for scoring), and its chain."""

    name: str
    description: str
    chain: Chain


class Router:
    """Route a query to the best path by cosine similarity to each path's description (ch10's router).

    Embeds every path's DESCRIPTION once (ch5's real encoder), then at route time embeds the query and
    picks the path whose description it is most similar to -- an argmax over a real learned-embedding
    score, exactly ch10's tool-routing decision applied to whole sub-chains. A fact-lookup query lands
    on the RAG path; a greeting lands on the direct path.
    """

    def __init__(self, routes: tuple[Route, ...], dense: DenseRetriever) -> None:
        self.routes = routes
        self._dense = dense
        self._desc_index = dense._encode([r.description for r in routes])  # noqa: SLF001 -- reuse ch5's encoder; unit-norm

    def route_scores(self, query: str) -> np.ndarray:
        """Cosine of the query against every route description (parallel to self.routes)."""
        q_vec = self._dense._encode([query])[0]  # noqa: SLF001 -- unit-norm query embedding
        return self._desc_index @ q_vec  # unit-norm rows => dot product == cosine

    def route(self, query: str) -> tuple[Route, float]:
        """Return (chosen route, its cosine score) -- the argmax path for this query."""
        scores = self.route_scores(query)
        best = int(np.argmax(scores))  # np.argmax breaks ties by first occurrence -- deterministic
        return self.routes[best], float(scores[best])


# ================================================================================================
# The StatefulGraph -- nodes + edges + a run loop with a step budget (cycles allowed, like ch10's loop).
# ================================================================================================

# An edge function inspects the state and returns the NAME of the next node, or "" to stop. This is
# what lets a graph BRANCH (return different names) and LOOP (return a name already visited).
EdgeFn = Callable[[AppState], str]


@dataclass(frozen=True)
class StatefulGraph:
    """Nodes (named steps) + edges (state -> next-node-name) + a bounded run loop. Cycles allowed.

    Generalizes the Chain: instead of a fixed order, each node's outgoing EDGE decides the next node
    from the current state, so the graph can branch and loop (a retry edge points back at the node
    that failed). The run loop enforces a STEP BUDGET so a cyclic graph must terminate -- the same
    stop-condition ch10's agent loop uses. Returns the final state, whose trace records the path taken.
    """

    nodes: dict[str, Step]
    edges: dict[str, EdgeFn]  # node name -> function picking the next node ("" = stop)
    entry: str
    max_steps: int = MAX_GRAPH_STEPS

    def run(self, state: AppState) -> AppState:
        current = self.entry
        for _ in range(self.max_steps):
            if not current:  # an edge returned "" -> the graph is done
                return state
            state = self.nodes[current](state)
            current = self.edges.get(current, lambda _s: "")(state)
        return state.log(f"[graph] stopped at step budget ({self.max_steps})")


# ================================================================================================
# Retry / fallback -- wrap a step so a transient failure is retried with backoff before giving up.
# ================================================================================================


class StepError(RuntimeError):
    """Raised by a step to signal a (possibly transient) failure the orchestrator may retry."""


def with_retry(step: Step, *, max_attempts: int = 3, base_delay: float = 0.0) -> FnStep:
    """Wrap a step so a StepError is retried up to `max_attempts` times with exponential backoff.

    This is the concrete "the orchestrator handles retries" claim: a step that fails transiently
    (a flaky network call, a rate limit) is retried instead of crashing the whole pipeline. Each
    attempt is logged to the trace, so a reader can SEE the retry happen -- observability, not magic.
    `base_delay=0` keeps the demo instant; a real deployment uses a nonzero backoff.
    """

    def run(state: AppState) -> AppState:
        last_error: Exception | None = None
        for attempt in range(1, max_attempts + 1):
            try:
                out = step(state)
                if attempt > 1:
                    out = out.log(f"[retry] '{step.name}' succeeded on attempt {attempt}")
                return out
            except StepError as err:  # only retry the errors a step explicitly marks as retryable
                last_error = err
                state = state.log(f"[retry] '{step.name}' failed attempt {attempt}: {err}")
                if attempt < max_attempts and base_delay > 0:
                    time.sleep(base_delay * 2 ** (attempt - 1))  # exponential backoff
        raise StepError(f"'{step.name}' exhausted {max_attempts} attempts") from last_error

    return FnStep(f"{step.name}+retry", run)


# ================================================================================================
# The WIRED steps -- real retrieve / rerank / guardrail / generate, each as a Step over AppState.
# ================================================================================================


def make_retrieve_step(dense: DenseRetriever, k: int = RETRIEVE_K) -> FnStep:
    """RETRIEVE (ch5): dense top-k candidate pool. Real all-MiniLM cosine retrieval."""

    def run(state: AppState) -> AppState:
        ranked = dense.search(state.query, k=k)
        state = replace(state, retrieved=ranked.indices)
        return state.log(f"[retrieve] top-{k} candidates: {list(ranked.indices)}")

    return FnStep("retrieve", run)


def make_rerank_step(reranker: CrossEncoderReranker, corpus: tuple[str, ...], top_k: int = TOP_K) -> FnStep:
    """RERANK (ch6): re-order the candidate pool by joint encoding, keep the top-k. Real reranker."""

    def run(state: AppState) -> AppState:
        reranked = reranker.rerank(state.query, state.retrieved, corpus)
        kept = reranked.indices[:top_k]
        state = replace(state, reranked=kept, context=tuple(corpus[i] for i in kept))
        return state.log(f"[rerank] kept top-{top_k}: {list(kept)}")

    return FnStep("rerank", run)


def make_guardrail_step(dense: DenseRetriever, threshold: float = GROUNDING_THRESHOLD) -> FnStep:
    """GUARDRAIL (ch14): does the retrieved context actually answer the QUERY? Abstain if not.

    The real "should we even answer?" gate: score the QUERY against the retrieved context (ch14's
    context-relevance / grounding signal). If the best passage clears `threshold`, retrieval found
    something on-topic and we may generate an answer; if not, no passage answers the query, so we
    ABSTAIN -- "I don't know" -- rather than let generation fabricate over off-topic context. We
    ground the QUERY (not a passage-echo) so the signal isn't circular: a fact query whose answer is
    in the corpus scores high; an unanswerable query whose corpus has nothing on-topic scores low.
    """

    def run(state: AppState) -> AppState:
        # context relevance = max cosine of the QUERY to any retrieved passage (ch14's grounding proxy,
        # reused via answer_grounding with the query as the text being grounded).
        grounding = answer_grounding(dense, state.query, state.context) if state.context else 0.0
        abstained = grounding < threshold
        state = replace(state, grounding=grounding, abstained=abstained)
        verdict = "ABSTAIN (no on-topic context)" if abstained else "context is on-topic -> allow"
        return state.log(f"[guardrail] context relevance {grounding:.3f} -> {verdict}")

    return FnStep("guardrail", run)


def make_generate_step() -> FnStep:
    """GENERATE (illustrative): draft the answer from the top passage, OR emit the abstain message.

    Runs AFTER the guardrail so it can honour its verdict: if the guardrail abstained (no on-topic
    context), emit "I don't know"; otherwise surface the top reranked passage as the answer. The
    pipeline feeding it is real; only this natural-language drafting is an LLM stand-in.
    """

    def run(state: AppState) -> AppState:
        if state.abstained:
            answer = ABSTAIN_MESSAGE
            note = "abstained per guardrail"
        else:
            answer = state.context[0] if state.context else ABSTAIN_MESSAGE  # illustrative: top passage
            note = f"from doc pool {list(state.reranked)}"
        state = replace(state, answer=answer)
        return state.log(f"[generate] {note}")

    return FnStep("generate", run)


def make_direct_step() -> FnStep:
    """The DIRECT path's single step: answer a chit-chat query with no retrieval (illustrative text)."""

    def run(state: AppState) -> AppState:
        state = replace(state, answer="Hello! How can I help you with the Helios-7 knowledge base?")
        return state.log("[direct] answered without retrieval (chit-chat)")

    return FnStep("direct", run)


# ================================================================================================
# Assemble the app: a RAG chain (retrieve->rerank->guardrail->generate) and a DIRECT chain, behind a router.
# ================================================================================================


@dataclass(frozen=True)
class OrchestratedApp:
    """The wired app: the router plus the two route chains it dispatches to."""

    router: Router
    dense: DenseRetriever

    def run(self, query: str) -> AppState:
        """Route the query, run the chosen chain, and return the final state (with its full trace)."""
        chosen, score = self.router.route(query)
        state = AppState(query=query, route=chosen.name)
        state = state.log(f"[route] '{query[:32]}...' -> '{chosen.name}' (cosine {score:.3f})")
        return chosen.chain(state)


def build_app(corpus: tuple[str, ...]) -> OrchestratedApp:
    """Wire the real steps into two routed chains behind a cosine router. This is the whole app."""
    dense = DenseRetriever(corpus)
    reranker = CrossEncoderReranker()
    rag_chain = Chain((
        make_retrieve_step(dense),
        make_rerank_step(reranker, corpus),
        make_guardrail_step(dense),  # decide "should we answer?" BEFORE generation
        make_generate_step(),  # honour the guardrail: answer, or emit the abstain message
    ))
    direct_chain = Chain((make_direct_step(),))
    routes = (
        Route("rag", "Answer a factual question about the Helios-7 satellite from the knowledge base.", rag_chain),
        Route("direct", "Respond to a greeting or casual chit-chat with no factual lookup needed.", direct_chain),
    )
    return OrchestratedApp(Router(routes, dense), dense)


# ================================================================================================
# The demo fixtures + a flaky step for the retry demo.
# ================================================================================================

FACT_QUERY = "What is the ground resolution of the Helios-7 imager?"
CHITCHAT_QUERY = "Hey there, how are you doing today?"
# An ungrounded query: retrievable-looking but the corpus can't answer it -> the guardrail abstains.
UNANSWERABLE_QUERY = "What was the total budget of the Helios-7 mission in dollars?"


@dataclass
class FlakyStep:
    """A step that raises StepError on its first `fail_times` calls, then succeeds. For the retry demo.

    It is stateful ON PURPOSE (a mutable call counter) to simulate a transient fault; the orchestrator's
    with_retry wrapper is what turns that fault into a recovered success instead of a crash.
    """

    name: str = "flaky-retrieve"
    fail_times: int = 1
    _calls: int = field(default=0)

    def __call__(self, state: AppState) -> AppState:
        self._calls += 1
        if self._calls <= self.fail_times:
            raise StepError(f"transient fault (call {self._calls})")
        return state.log(f"[flaky] succeeded on call {self._calls}")


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
        print("torch: not installed (retrieval needs the encoder; orchestration primitives are pure python)")


def _print_trace(state: AppState) -> None:
    """Pretty-print a state's step trace -- the observability payoff of naming every step."""
    for line in state.trace:
        print(f"    {line}")


def main() -> None:
    _report_versions()
    corpus = full_corpus()
    app = build_app(corpus)
    print(f"corpus: {len(corpus)} passages | dense lens: {app.dense.backend} | reranker: {CrossEncoderReranker().backend}")
    print(f"grounding threshold: {GROUNDING_THRESHOLD} | retrieve K: {RETRIEVE_K} | keep top-K: {TOP_K}")
    print(
        "NOTE: the orchestration primitives + wired retrieve/rerank/guardrail steps are REAL and "
        "measured; only the GENERATE step's answer text is an illustrative LLM stand-in.\n"
    )

    # ------------------------------------------------------------------------------------------
    # 1) THE WIRED APP, end to end: route -> retrieve -> rerank -> guardrail -> generate, with a trace.
    # ------------------------------------------------------------------------------------------
    print("=" * 96)
    print("1) The wired mini-RAG app: one query flows through the whole pipeline, with a step trace")
    print("=" * 96)
    result = app.run(FACT_QUERY)
    print(f"  query: {FACT_QUERY}")
    print(f"  route: {result.route} | grounding: {result.grounding:.3f} | abstained: {result.abstained}")
    print("  trace:")
    _print_trace(result)
    print(f"  ANSWER: {result.answer}")
    # Correctness BEFORE the claim: the fact query takes the RAG path, runs all four steps, and answers.
    assert result.route == "rag", "the fact query must route to the RAG chain"
    assert len(result.trace) == 5, "route + 4 RAG steps = 5 trace lines"
    # Pin the retrieve/rerank indices BY VALUE so a library drift can't silently change the numbers
    # the page bolds (retrieve top-4, rerank top-2).
    assert list(result.retrieved) == [1, 0, 2, 10], "retrieve returns the imager+launch+lead+distractor pool"
    assert list(result.reranked) == [1, 10], "rerank keeps the imager passage and the top telemetry line"
    assert not result.abstained and result.answer, "a grounded fact query is answered, not abstained"
    assert "resolution" in result.answer, "the answer surfaces the imager-resolution passage"
    print("  -> one object wires four real steps; the trace shows exactly what ran.\n")

    # ------------------------------------------------------------------------------------------
    # 2) ROUTING: the router sends contrasting queries down different chains (real cosine argmax).
    # ------------------------------------------------------------------------------------------
    print("=" * 96)
    print("2) Routing: a fact query and a greeting take DIFFERENT paths (cosine of query vs route desc)")
    print("=" * 96)
    for query, expected in ((FACT_QUERY, "rag"), (CHITCHAT_QUERY, "direct")):
        scores = app.router.route_scores(query)
        chosen, score = app.router.route(query)
        pairs = ", ".join(f"{r.name}={s:.3f}" for r, s in zip(app.router.routes, scores))
        print(f"  '{query[:40]}...' -> {chosen.name} (cosine {score:.3f})   [{pairs}]")
        assert chosen.name == expected, f"router mis-routed {query!r} to {chosen.name} (expected {expected})"
    print("  -> fact-lookups take the RAG path; chit-chat takes the direct path, by embedding score.\n")

    # ------------------------------------------------------------------------------------------
    # 3) THE GUARDRAIL BRANCH: an unanswerable query flows the SAME pipeline but abstains at the gate.
    # ------------------------------------------------------------------------------------------
    print("=" * 96)
    print("3) The guardrail step in the chain: an unanswerable query runs the pipeline, then ABSTAINS")
    print("=" * 96)
    unanswerable = app.run(UNANSWERABLE_QUERY)
    print(f"  query: {UNANSWERABLE_QUERY}")
    print(f"  route: {unanswerable.route} | grounding: {unanswerable.grounding:.3f} | abstained: {unanswerable.abstained}")
    print(f"  ANSWER: {unanswerable.answer}")
    assert unanswerable.route == "rag", "the (factual-looking) query still routes to RAG"
    assert unanswerable.abstained, "no passage answers it -> the guardrail step abstains"
    assert unanswerable.answer == ABSTAIN_MESSAGE, "the app emits 'I don't know', not a fabrication"
    print("  -> orchestration doesn't remove the guardrail's judgement -- it WIRES it in as a step.\n")

    # ------------------------------------------------------------------------------------------
    # 4) RETRY: a step fails once (transient), the orchestrator retries it, and the pipeline recovers.
    # ------------------------------------------------------------------------------------------
    print("=" * 96)
    print("4) Retry: a flaky step fails once, then the orchestrator's retry wrapper recovers it")
    print("=" * 96)
    flaky = FlakyStep(fail_times=1)
    guarded = with_retry(flaky, max_attempts=3)
    naive_state = AppState(query="probe")
    # First show the naive (unwrapped) step CRASHES on the transient fault:
    naive_flaky = FlakyStep(fail_times=1)
    crashed = False
    try:
        naive_flaky(naive_state)
    except StepError as err:
        crashed = True
        print(f"  naive step (no retry): CRASHED -> {err}")
    assert crashed, "the naive step crashes on the transient fault"
    # Now the retry-wrapped step recovers:
    recovered = guarded(AppState(query="probe"))
    print("  retry-wrapped step trace:")
    _print_trace(recovered)
    assert any("failed attempt 1" in line for line in recovered.trace), "attempt 1 fails (logged)"
    assert any("succeeded on attempt 2" in line for line in recovered.trace), "attempt 2 succeeds (logged)"
    assert flaky._calls == 2, "the step was called exactly twice (1 fail + 1 success)"  # noqa: SLF001 -- inspect the demo counter
    print("  -> the same transient fault crashes naive glue but is recovered by the orchestrator.\n")

    # ------------------------------------------------------------------------------------------
    # 5) A STATEFUL GRAPH: the RAG steps as a graph whose edges route retrieve->rerank->guardrail->generate.
    # ------------------------------------------------------------------------------------------
    print("=" * 96)
    print("5) The same steps as a StatefulGraph: edges pick the next node; a step budget bounds cycles")
    print("=" * 96)
    dense = app.dense
    reranker = CrossEncoderReranker()
    nodes: dict[str, Step] = {
        "retrieve": make_retrieve_step(dense),
        "rerank": make_rerank_step(reranker, corpus),
        "guardrail": make_guardrail_step(dense),
        "generate": make_generate_step(),
    }
    # linear edges here, but each is a state->name function, so any could branch on the state instead
    edges: dict[str, EdgeFn] = {
        "retrieve": lambda _s: "rerank",
        "rerank": lambda _s: "guardrail",
        "guardrail": lambda _s: "generate",
        "generate": lambda _s: "",  # "" stops the run loop
    }
    graph = StatefulGraph(nodes, edges, entry="retrieve")
    graph_result = graph.run(AppState(query=FACT_QUERY))
    print("  graph trace:")
    _print_trace(graph_result)
    print(f"  ANSWER: {graph_result.answer}")
    # the graph produces the SAME answer as the chain -- graph is a superset (adds branch/cycle ability)
    chain_answer = app.run(FACT_QUERY).answer
    assert graph_result.answer == chain_answer, "the graph and the chain produce the same answer here"
    assert len(graph_result.trace) == 4, "the graph ran exactly the 4 nodes (no route line)"
    print("  -> a chain is a graph with fixed edges; the graph adds branching + cycles + a step budget.")


if __name__ == "__main__":
    main()
