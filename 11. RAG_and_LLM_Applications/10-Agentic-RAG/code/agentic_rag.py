"""From-scratch Agentic RAG: a real ReAct control loop, real tools, real routing.

Static RAG is a FIXED pipeline -- embed the query once, retrieve top-k once, stuff the passages
into the prompt, generate once. That single shot cannot: do MULTI-STEP retrieval, pick WHICH
source, run a calculation on what it retrieved, or notice a miss and try again. A COMPOUND query
-- "how many complete orbits does Helios-7 make in a day, and what is its imager's ground
resolution?" -- needs (1) a retrieval, (2) an arithmetic step on the retrieved number, and (3)
a second retrieval. Static single-shot RAG structurally cannot deliver the orbit COUNT: it has
no place to divide 1440 by the retrieved period. This module shows that failure, then shows an
AGENT solve the same query in a handful of Thought -> Action -> Observation steps.

The agentic idea: put an LLM in the driver's seat of a LOOP. It REASONS about what it still
needs (a Thought), picks a TOOL and its input (an Action), reads the tool's result (an
Observation), and repeats -- retrieve again, compute, route to a different source, or finish --
until it can answer. That is the ReAct loop (Yao et al. 2022): interleaved reasoning and acting.

HONESTY -- WHAT RUNS REAL vs WHAT IS ILLUSTRATIVE
-------------------------------------------------
  * REAL and measured: the CONTROL LOOP (a genuine Thought/Action/Observation state machine with
    a step budget and a stop condition), the TOOL REGISTRY, and every TOOL the agent invokes --
    a real dense retriever (ch5's all-MiniLM DenseRetriever), a real calculator (a safe AST
    evaluator, never eval), and a real router (cosine of the query against each tool's
    description, an actual scoring decision). Every step count, tool call, cosine, and assembled
    fact printed here is computed and asserted before it is claimed.
  * ILLUSTRATIVE (labelled): in production an LLM policy decides the next Action from the running
    trace (that needs a generative model; this env is encoder-only). So the ACTION-SELECTION
    POLICY here is a transparent DETERMINISTIC rule-based stand-in -- it makes the SAME sequence
    of decisions an LLM would, but by hand-written rules, so the loop is fully reproducible with
    no LLM. The reasoning TEXT in each Thought is a fixed exemplar. The tools those actions
    invoke, and the routing/arithmetic/retrieval they perform, are entirely real.

The dense retriever + corpus are imported from ch5's hybrid_search (which imports ch1), so the
RAG chapters share ONE source of truth.

Verified on Python 3.12 / numpy 2.x / sentence-transformers (all-MiniLM-L6-v2, CPU). Deterministic:
identical numbers every run given the same cached model.

Run:
    python agentic_rag.py
"""

from __future__ import annotations

import ast
import operator
import re
import sys
from collections.abc import Callable
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

# Reuse ch5's dense retriever + corpus (ch5 injects ch1's path transitively). ch5 lives two
# directories over; inject its code dir so imports work whether this file is run from its own dir
# or imported by the notebook / figure scripts.
_CH5_CODE = Path(__file__).resolve().parent.parent.parent / "05-Hybrid-Search-BM25-and-Dense" / "code"
if str(_CH5_CODE) not in sys.path:
    sys.path.insert(0, str(_CH5_CODE))

from hybrid_search import (  # noqa: E402  (path injected above must precede this import)
    DenseRetriever,
    full_corpus,
)

DENSE_MODEL = "all-MiniLM-L6-v2"  # the learned bi-encoder (ch3/5's embedder)
RETRIEVE_K = 1  # a tool call returns its single best passage -- the agent asks a FOCUSED question
MAX_STEPS = 6  # the step budget: the loop's hard stop, the fix for the infinite-loop pitfall
MINUTES_PER_DAY = 1440  # 24 * 60 -- the constant the orbit-count arithmetic divides


# ================================================================================================
# Tools -- each is REAL and measured. A tool is a name + a description (used for routing) + a fn.
# ================================================================================================


@dataclass(frozen=True)
class Observation:
    """What a tool returns to the agent: a short text result plus the raw value for assertions."""

    text: str  # the human-readable observation the agent "reads" in its trace
    value: object = None  # the raw payload (a passage index, a number, a route) for tests/plots


@dataclass(frozen=True)
class Tool:
    """One tool in the registry: a name, a natural-language description, and the callable itself.

    The DESCRIPTION is not decoration -- the router embeds it and scores the query against it, so
    a good description is what lets the agent (or router) pick the right tool. This mirrors how
    real tool/function-calling stacks expose tools to an LLM via their descriptions/schemas.
    """

    name: str
    description: str
    run: Callable[[str], Observation]


class RetrieverTool:
    """A dense-retrieval tool: ask a focused question, get back the single best passage.

    Wraps ch5's real all-MiniLM DenseRetriever over the shared Helios-7 corpus. The agent calls it
    with a SUB-question (not the whole compound query) -- which is exactly why the agent can succeed
    where one-shot retrieval fails: each call is a sharp, single-fact probe.
    """

    def __init__(self, corpus: tuple[str, ...], model_name: str = DENSE_MODEL) -> None:
        self.corpus = corpus
        self._dense = DenseRetriever(corpus, model_name=model_name)
        self.backend = self._dense.backend

    def __call__(self, query: str) -> Observation:
        res = self._dense.search(query, k=RETRIEVE_K)
        idx = res.indices[0]
        return Observation(text=self.corpus[idx], value=(idx, float(res.scores[0])))


# ---- Calculator tool: a SAFE arithmetic evaluator (never eval) ---------------------------------
# Only these AST node types / operators are allowed -- everything else raises. This is the standard
# safe-eval pattern: parse to an AST, walk it, and permit only numeric literals and basic operators,
# so a tool input can never execute arbitrary code (the security reason to never use eval()).
_ALLOWED_BINOPS: dict[type, Callable[[float, float], float]] = {
    ast.Add: operator.add,
    ast.Sub: operator.sub,
    ast.Mult: operator.mul,
    ast.Div: operator.truediv,
    ast.FloorDiv: operator.floordiv,
    ast.Mod: operator.mod,
    ast.Pow: operator.pow,
}
_ALLOWED_UNARYOPS: dict[type, Callable[[float], float]] = {
    ast.UAdd: operator.pos,
    ast.USub: operator.neg,
}
# Exponent magnitude cap. `**` is safe against code execution, but a nested tower like `9**9**9`
# is a pure-arithmetic resource blowup (huge result -> CPU/memory), so we bound the exponent. This
# is a denial-of-service guard, not a code-exec one -- the AST whitelist already blocks execution.
MAX_EXPONENT = 100.0


def safe_eval(expression: str) -> float:
    """Evaluate a pure-arithmetic expression safely via AST walking (no eval, no names, no calls).

    Parses the string to an AST and recursively evaluates ONLY numeric literals and the whitelisted
    operators above; any other node (a name, an attribute, a function call, an import) raises
    ValueError. This is why the calculator tool can take model/agent-provided input without the
    code-execution risk that bare eval() would introduce. As a denial-of-service guard, an exponent
    whose magnitude exceeds MAX_EXPONENT is refused (so `9**9**9` cannot blow up CPU/memory).
    """

    def _eval(node: ast.AST) -> float:
        if isinstance(node, ast.Expression):
            return _eval(node.body)
        if isinstance(node, ast.Constant) and isinstance(node.value, (int, float)):
            return float(node.value)
        if isinstance(node, ast.BinOp) and type(node.op) in _ALLOWED_BINOPS:
            left, right = _eval(node.left), _eval(node.right)
            if isinstance(node.op, ast.Pow) and abs(right) > MAX_EXPONENT:
                raise ValueError(f"exponent {right:g} exceeds the MAX_EXPONENT={MAX_EXPONENT:g} guard")
            return _ALLOWED_BINOPS[type(node.op)](left, right)
        if isinstance(node, ast.UnaryOp) and type(node.op) in _ALLOWED_UNARYOPS:
            return _ALLOWED_UNARYOPS[type(node.op)](_eval(node.operand))
        raise ValueError(f"disallowed expression element: {ast.dump(node)}")

    return _eval(ast.parse(expression, mode="eval"))


def calculator_tool(expression: str) -> Observation:
    """Compute an arithmetic expression and return the numeric result -- the step static RAG lacks."""
    result = safe_eval(expression)
    return Observation(text=f"{expression} = {result:g}", value=result)


# ================================================================================================
# Router -- pick WHICH tool a query needs, by scoring the query against each tool's description.
# This is a REAL decision: cosine similarity (via the shared encoder) between the query and every
# tool description; the argmax is the route. Routing = a scoring/classification step.
# ================================================================================================


class Router:
    """Route a query to the best tool by cosine similarity to each tool's description.

    Embeds every tool's DESCRIPTION once (with ch5's real encoder), then at route time embeds the
    query and returns the tool whose description it is most similar to. This is the concrete form
    of "the agent picks the right source": a learned-embedding scoring decision, argmax over tools.
    A math-flavoured query lands on the calculator; a fact-lookup query lands on the retriever.
    """

    def __init__(self, tools: tuple[Tool, ...], dense: DenseRetriever) -> None:
        self.tools = tools
        self._dense = dense
        # embed each tool's description once -> a (n_tools, dim) unit-norm matrix
        self._desc_index = dense._encode([t.description for t in tools])  # noqa: SLF001 -- reuse ch5's encoder

    def route_scores(self, query: str) -> np.ndarray:
        """Cosine of the query against every tool description (parallel to self.tools)."""
        q_vec = self._dense._encode([query])[0]  # noqa: SLF001 -- unit-norm query embedding
        return self._desc_index @ q_vec  # unit-norm rows => dot product == cosine

    def route(self, query: str) -> tuple[Tool, float]:
        """Return (chosen tool, its cosine score) -- the argmax route for this query."""
        scores = self.route_scores(query)
        best = int(np.argmax(scores))
        return self.tools[best], float(scores[best])


# ================================================================================================
# The ReAct agent -- a REAL control loop. Thought -> Action -> Observation, until finish or budget.
#
# The ACTION-SELECTION POLICY (which tool + input to use next) is the one illustrative piece: a
# deterministic rule-based stand-in for an LLM policy, clearly labelled. It makes the same decisions
# an LLM would on this query, so the loop is reproducible with no generative model. The loop, the
# tools it calls, and the budget/stop logic are all real.
# ================================================================================================


@dataclass(frozen=True)
class Step:
    """One iteration of the ReAct loop, recorded for the trace and for assertions."""

    thought: str  # the reasoning text (illustrative exemplar -- an LLM would generate this)
    action: str  # the tool name chosen (e.g. "retrieve", "calculator", "finish")
    action_input: str  # the tool's input string
    observation: str  # the tool's returned text (real -- the tool actually ran)


@dataclass
class AgentResult:
    """The outcome of a run: the full step trace, the final answer, and the tools used."""

    steps: list[Step] = field(default_factory=list)
    answer: str = ""
    hit_budget: bool = False  # True if the loop stopped because it hit MAX_STEPS (no finish)

    @property
    def n_steps(self) -> int:
        return len(self.steps)

    @property
    def tools_used(self) -> list[str]:
        return [s.action for s in self.steps]


# A Policy maps the running trace (the steps so far) + the query to the next (thought, tool, input).
# In production an LLM IS this function. Here we pass in a deterministic rule-based policy so the
# whole loop runs with no generative model -- see compound_orbit_policy below.
Policy = Callable[[str, list[Step]], tuple[str, str, str]]


class ReActAgent:
    """A minimal, real ReAct control loop over a tool registry, driven by a pluggable policy.

    The loop is the lesson: at each step it asks the policy for (thought, tool_name, tool_input),
    RUNS that tool for real, appends the Observation, and repeats -- until the policy emits the
    special `finish` action or the step budget MAX_STEPS is hit. The budget is not optional
    decoration: without it a policy that never says `finish` loops forever (see the pitfalls demo).
    """

    FINISH = "finish"  # the sentinel tool name that ends the loop with an answer

    def __init__(self, tools: tuple[Tool, ...], max_steps: int = MAX_STEPS) -> None:
        self.tools = {t.name: t for t in tools}
        self.max_steps = max_steps

    def run(self, query: str, policy: Policy) -> AgentResult:
        """Drive the Thought -> Action -> Observation loop until finish or the step budget."""
        result = AgentResult()
        for _ in range(self.max_steps):
            thought, tool_name, tool_input = policy(query, result.steps)
            if tool_name == self.FINISH:
                result.answer = tool_input  # finish's "input" is the assembled final answer
                result.steps.append(Step(thought, tool_name, tool_input, observation="(done)"))
                return result
            if tool_name not in self.tools:
                raise ValueError(f"policy chose unknown tool {tool_name!r}")
            obs = self.tools[tool_name].run(tool_input)
            result.steps.append(Step(thought, tool_name, tool_input, obs.text))
        # fell out of the loop without a finish -> budget exhausted (the infinite-loop guard fired)
        result.hit_budget = True
        result.answer = "(step budget exhausted -- no final answer)"
        return result


# ================================================================================================
# Static single-shot RAG baseline -- the FIXED pipeline the agent is contrasted against.
# ================================================================================================


@dataclass(frozen=True)
class StaticResult:
    """One static single-shot RAG answer: the passages it retrieved and what it could assemble."""

    retrieved: tuple[str, ...]
    orbit_count: int | None  # the arithmetic answer -- None because static RAG cannot compute it
    resolution: str | None


def static_rag(query: str, dense: DenseRetriever, corpus: tuple[str, ...], k: int = 3) -> StaticResult:
    """Static RAG: embed the query ONCE, retrieve top-k ONCE, and try to answer from those passages.

    This is the fixed retrieve-then-generate pipeline. It can surface passages that CONTAIN the
    orbit period and the resolution, but it has no step to DIVIDE the period into a day -- so the
    "how many complete orbits per day" part of a compound query is structurally unanswerable here.
    We model that honestly: static RAG returns the passages and the resolution it can read off, but
    orbit_count is None because there is no arithmetic step in the pipeline.
    """
    res = dense.search(query, k=k)
    passages = tuple(corpus[i] for i in res.indices)
    joined = " ".join(passages)
    res_match = re.search(r"(\d+)\s*meters?", joined)
    resolution = f"{res_match.group(1)} meters" if res_match else None
    # orbit_count stays None: no pipeline step turns "97 minutes" into "how many per day"
    return StaticResult(retrieved=passages, orbit_count=None, resolution=resolution)


# ================================================================================================
# The illustrative policy for the compound orbit+resolution query.
#   Rule-based stand-in for an LLM policy: it inspects how many steps have run and returns the next
#   (thought, tool, input). An LLM would decide these from the trace text; the DECISIONS are the
#   same, the mechanism (hard-coded rules) is what makes it reproducible. Clearly labelled.
# ================================================================================================

COMPOUND_QUERY = (
    "How many complete orbits does Helios-7 make in one day, "
    "and what is the ground resolution of its imager?"
)


def compound_orbit_policy(query: str, steps: list[Step]) -> tuple[str, str, str]:
    """Deterministic ReAct policy for COMPOUND_QUERY (illustrative stand-in for an LLM policy).

    Decomposes the compound question the way an LLM would: retrieve the orbit period, compute
    orbits/day from it, retrieve the imager resolution, then finish by assembling both facts. Each
    branch keys off how many steps have run so far -- a transparent state machine. The tools it
    calls (retrieve, calculator) run for REAL; only this decision logic is hand-written.
    """
    n = len(steps)
    if n == 0:
        return (
            "The question has two parts. First I need the orbit period, then I can compute "
            "orbits per day.",
            "retrieve",
            "Helios-7 orbit period in minutes",
        )
    if n == 1:
        # the previous observation is the orbit passage; extract the period and divide the day by it
        period = _extract_int(steps[-1].observation, r"(\d+)\s*minutes?")
        return (
            f"The passage says the period is {period} minutes. A day has {MINUTES_PER_DAY} "
            f"minutes, so I divide to get orbits per day.",
            "calculator",
            f"{MINUTES_PER_DAY} / {period}",
        )
    if n == 2:
        return (
            "That gives the orbits per day; the whole-number part is the count of COMPLETE "
            "orbits. Now I still need the imager's ground resolution.",
            "retrieve",
            "Helios-7 imager ground resolution",
        )
    # n == 3: assemble both retrieved/computed facts into the final answer
    orbits_per_day = float(steps[1].observation.split("=")[-1])
    complete_orbits = int(orbits_per_day)  # floor: only COMPLETE orbits count
    resolution = _extract_int(steps[-1].observation, r"(\d+)\s*meters?")
    answer = (
        f"Helios-7 completes {complete_orbits} full orbits per day "
        f"({orbits_per_day:.2f} raw), and its imager has a ground resolution of {resolution} meters."
    )
    return ("I now have both facts: the orbit count and the resolution. I can answer.", "finish", answer)


def _extract_int(text: str, pattern: str) -> int:
    """Pull the first integer matching `pattern` out of an observation string (helper for the policy)."""
    match = re.search(pattern, text)
    if not match:
        raise ValueError(f"expected {pattern!r} in observation {text!r}")
    return int(match.group(1))


# A deliberately BROKEN policy for the pitfalls demo: it NEVER emits `finish`, so without a step
# budget it would loop forever. It just keeps re-retrieving. The budget is what saves it.
def never_finishing_policy(query: str, steps: list[Step]) -> tuple[str, str, str]:
    """A broken policy that never finishes -- used to demonstrate the infinite-loop pitfall + the cap."""
    return ("I'm not sure yet, let me search again...", "retrieve", "Helios-7")


# ================================================================================================
# Build the standard tool registry + agent used by the demos, page, notebook, and figures.
# ================================================================================================


def build_tools(corpus: tuple[str, ...]) -> tuple[Tool, ...]:
    """The three-tool registry: a real dense retriever, a real calculator, and (for routing) both."""
    retriever = RetrieverTool(corpus)
    return (
        Tool(
            name="retrieve",
            description=(
                "Look up a fact about the Helios-7 satellite from the knowledge base: launch date, "
                "orbit, instruments, imager resolution, mission team, or telemetry."
            ),
            run=retriever,
        ),
        Tool(
            name="calculator",
            description=(
                "Compute an arithmetic expression: add, subtract, multiply, divide numbers, "
                "for example to work out a rate, a count, or a total from figures."
            ),
            run=calculator_tool,
        ),
    )


def build_agent(corpus: tuple[str, ...]) -> tuple[ReActAgent, tuple[Tool, ...], RetrieverTool]:
    """Assemble the agent, its tools, and the retriever tool (whose encoder the router reuses)."""
    tools = build_tools(corpus)
    agent = ReActAgent(tools)
    retriever_tool = next(t.run for t in tools if t.name == "retrieve")
    return agent, tools, retriever_tool


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


def _print_trace(result: AgentResult) -> None:
    """Pretty-print a ReAct trace: Thought / Action / Observation per step."""
    for i, step in enumerate(result.steps, start=1):
        print(f"  step {i}")
        print(f"    Thought:     {step.thought}")
        print(f"    Action:      {step.action}({step.action_input!r})")
        print(f"    Observation: {step.observation}")


def main() -> None:
    _report_versions()
    corpus = full_corpus()
    agent, tools, retriever_tool = build_agent(corpus)
    dense = retriever_tool._dense  # noqa: SLF001 -- the shared encoder, reused by the router
    router = Router(tools, dense)
    print(f"corpus: {len(corpus)} passages | dense lens: {retriever_tool.backend}")
    print(f"tools: {[t.name for t in tools]} | step budget: {agent.max_steps}")
    print(
        "NOTE: the control loop, the tools (retriever/calculator), and routing are REAL and "
        "measured; the action-selection POLICY and Thought text are a deterministic illustrative "
        "stand-in for an LLM policy.\n"
    )

    # ------------------------------------------------------------------------------------------
    # 1) THE FAILURE: static single-shot RAG cannot answer the compound query's arithmetic part.
    # ------------------------------------------------------------------------------------------
    print("=" * 96)
    print("1) Static single-shot RAG on the COMPOUND query — where the fixed pipeline breaks")
    print("=" * 96)
    print(f"query: {COMPOUND_QUERY}")
    static = static_rag(COMPOUND_QUERY, dense, corpus, k=3)
    print(f"  static retrieved (top-3): {[p[:52] for p in static.retrieved]}")
    print(f"  resolution it can read off passages : {static.resolution}")
    print(f"  orbits-per-day it can compute        : {static.orbit_count}  (no arithmetic step in the pipeline)")
    assert static.orbit_count is None, "static single-shot RAG has no step to compute the orbit count"
    print("  -> static RAG surfaces facts but CANNOT divide 1440 by the period; the count part is unanswerable.\n")

    # ------------------------------------------------------------------------------------------
    # 2) THE SOLVE: the ReAct agent decomposes the query and solves it in N steps (full trace).
    # ------------------------------------------------------------------------------------------
    print("=" * 96)
    print("2) The ReAct agent SOLVES the same query — Thought → Action → Observation, iterated")
    print("=" * 96)
    result = agent.run(COMPOUND_QUERY, compound_orbit_policy)
    _print_trace(result)
    print(f"\n  FINAL ANSWER: {result.answer}")
    print(f"  steps taken: {result.n_steps} | tools used: {result.tools_used} | hit budget: {result.hit_budget}")
    # Correctness BEFORE the claim: the agent used both retrieval AND the calculator, finished
    # cleanly (not by budget), and assembled the RIGHT facts (14 complete orbits, 4 meters).
    assert not result.hit_budget, "the agent should finish via `finish`, not by exhausting the budget"
    assert result.tools_used == ["retrieve", "calculator", "retrieve", "finish"], (
        f"unexpected tool sequence: {result.tools_used}"
    )
    assert "14 full orbits" in result.answer, "the agent must compute 14 complete orbits (1440 // 97)"
    assert "4 meters" in result.answer, "the agent must retrieve the 4-meter imager resolution"
    print("  -> the agent solved a query static RAG could not: multi-step retrieve + compute + retrieve.\n")

    # ------------------------------------------------------------------------------------------
    # 3) ROUTING: pick the right tool by scoring the query against each tool's description (real cosine).
    # ------------------------------------------------------------------------------------------
    print("=" * 96)
    print("3) Routing: which tool does a query need? (cosine of query vs each tool description)")
    print("=" * 96)
    route_probes = [
        ("Who is the project lead for Helios-7?", "retrieve"),
        ("What is 1440 divided by 97?", "calculator"),
        ("When was Helios-7 launched?", "retrieve"),
        ("Compute 200 times 4.", "calculator"),
    ]
    print(f"  {'query':<44} | {'-> route':>11} | scores (retrieve / calculator) | expected")
    print("  " + "-" * 92)
    for q, expected in route_probes:
        scores = router.route_scores(q)
        chosen, _ = router.route(q)
        score_str = " / ".join(f"{s:+.3f}" for s in scores)
        print(f"  {q:<44} | {chosen.name:>11} | {score_str:^30} | {expected}")
        assert chosen.name == expected, f"router mis-routed {q!r} to {chosen.name} (expected {expected})"
    print("  -> the router sends fact-lookups to the retriever and math to the calculator, by embedding score.\n")

    # ------------------------------------------------------------------------------------------
    # 4) PITFALL: no step budget -> infinite loop. The cap is the fix.
    # ------------------------------------------------------------------------------------------
    print("=" * 96)
    print("4) Pitfall: a policy that never finishes would loop forever — the step budget stops it")
    print("=" * 96)
    broken = agent.run(COMPOUND_QUERY, never_finishing_policy)
    print(f"  broken policy (never emits `finish`) ran {broken.n_steps} steps, hit budget: {broken.hit_budget}")
    print(f"  final answer: {broken.answer}")
    assert broken.hit_budget, "the never-finishing policy must be stopped by the budget"
    assert broken.n_steps == agent.max_steps, "it should run exactly MAX_STEPS steps, then stop"
    print(f"  -> without MAX_STEPS this loops forever; the cap ({agent.max_steps}) turns a hang into a bounded stop.\n")

    # ------------------------------------------------------------------------------------------
    # 5) COST: the agent's power costs steps. Count the per-query work vs static's single shot.
    # ------------------------------------------------------------------------------------------
    print("=" * 96)
    print("5) The cost of agency: steps (≈ LLM calls) per query, agent vs static")
    print("=" * 96)
    agent_steps = result.n_steps
    static_steps = 1  # one retrieval + one generation = a single shot
    print(f"  static single-shot RAG : {static_steps} pass  (1 retrieval + 1 generation)")
    print(f"  ReAct agent            : {agent_steps} steps ({result.tools_used.count('retrieve')} retrievals, "
          f"{result.tools_used.count('calculator')} calc, 1 finish)")
    print(f"  ratio                  : {agent_steps}x the per-query work of static RAG")
    assert agent_steps > static_steps, "the agent trades more per-query work for the ability to solve harder queries"
    print("  -> agency is not free: each step is (in production) an LLM call. Use an agent when the query NEEDS")
    print("     multi-step reasoning; reach for static RAG when a single shot suffices (the over-agentic pitfall).")


if __name__ == "__main__":
    main()
