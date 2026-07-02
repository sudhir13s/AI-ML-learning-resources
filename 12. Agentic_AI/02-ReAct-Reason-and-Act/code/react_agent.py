"""A REAL ReAct (Reason + Act) agent — the load-bearing module for the ReAct chapter.

This is not a toy. A genuine small instruction-tuned LLM (``Qwen/Qwen2.5-1.5B-Instruct`` by
default) is driven through a real Thought -> Action -> Observation loop against real Python tools,
and every trace, count, and figure the chapter shows is produced here, from real greedy (temperature
0) generations. Nothing about the model's output is mocked, stubbed, or hand-written.

What is real here, and why it matters for teaching ReAct honestly:

  * **Real model.** A local instruct model, greedy-decoded for determinism, device-agnostic
    (cuda -> mps -> cpu). The first run downloads the weights (a few hundred MB) and caches them;
    every run after is offline and reproducible. Print the model id + versions in the banner.

  * **Real tools.** Two genuine tools the agent invokes and gets real return values from:
      - ``calculator`` — evaluates arithmetic by walking a parsed AST (NOT ``eval`` on arbitrary
        input; only +, -, *, /, %, ** and numbers are permitted), so the observation is real math.
      - ``wiki`` — a real lookup against a small local knowledge base (offline, deterministic),
        standing in for a web/Wikipedia search; it returns real text the agent must read.

  * **A real loop with a real stop condition.** Generation is *halted at the first ``Action:``*
    (via a stop string on ``Observation:``); the agent does NOT get to hallucinate the observation.
    We dispatch the real tool, splice the *real* observation into the transcript, and let the model
    reason again. The loop ends on ``finish[...]`` or a step budget. That "stop the model before it
    invents the tool result" move is the entire point of ReAct — and it is real code below.

  * **Real robustness.** Small models emit messy text: extra Action lines, unevaluated expressions
    in ``finish``, bracket typos. Parsing must be defensive (regex on the first well-formed action,
    a normaliser that evaluates a numeric ``finish`` argument through the calculator). That messiness
    IS the lesson — production ReAct is 20% prompt and 80% parsing and control flow.

  * **A real comparison.** ``compare_react_vs_direct`` runs the SAME questions two ways — the full
    ReAct loop vs a single "answer directly, no tools" prompt — and measures exact-match accuracy on
    real multi-step questions, so the chapter can *show* that acting on real observations reduces the
    wrong answers a reason-only model confidently produces.

Run it standalone (downloads+caches the model on first run, then offline)::

    python react_agent.py

Verified on Python 3.12 / torch 2.12 / transformers 5.10, CPU / Apple MPS / CUDA.
"""

from __future__ import annotations

import ast
import operator
import re
from collections.abc import Callable
from dataclasses import dataclass, field

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, PreTrainedModel, PreTrainedTokenizerBase

# --------------------------------------------------------------------------------------------------
# Constants (hoisted; no magic numbers inline)
# --------------------------------------------------------------------------------------------------
DEFAULT_MODEL_ID = "Qwen/Qwen2.5-1.5B-Instruct"  # small, real, instruction-tuned, freely available
MAX_NEW_TOKENS = 96  # cap per step: enough for one Thought + one Action, not an essay
MAX_STEPS = 6  # step budget: the real "give up" stop condition that prevents infinite loops
GEN_STOP_STRINGS = ("Observation:", "\nQuestion:")  # HALT generation before the model fakes a result
# Only these AST node types are allowed in the calculator — this is what makes it safe (not `eval`).
_SAFE_BINOPS: dict[type[ast.operator], Callable[[float, float], float]] = {
    ast.Add: operator.add,
    ast.Sub: operator.sub,
    ast.Mult: operator.mul,
    ast.Div: operator.truediv,
    ast.Pow: operator.pow,
    ast.Mod: operator.mod,
}
_SAFE_UNARYOPS: dict[type[ast.unaryop], Callable[[float], float]] = {
    ast.USub: operator.neg,
    ast.UAdd: operator.pos,
}

# One regex, reused: find the FIRST well-formed `Action: tool[arg]` (dotall so multi-line args work).
ACTION_RE = re.compile(r"Action:\s*([A-Za-z_]\w*)\s*\[(.*?)\]", re.DOTALL)
# Detect a purely-numeric expression so a `finish[1889 + 100]` can be evaluated to `1989`.
_NUMERIC_EXPR_RE = re.compile(r"^[\d\s+\-*/%.()]+$")


class ToolError(Exception):
    """Raised when a tool cannot process its input — surfaced to the agent as an Observation."""


# --------------------------------------------------------------------------------------------------
# 1. Real tools — genuine Python functions with genuine return values
# --------------------------------------------------------------------------------------------------
def _eval_ast(node: ast.expr) -> float:
    """Recursively evaluate a *whitelisted* arithmetic AST — the safe core of the calculator.

    We never call ``eval``. We parse the expression to an AST and walk it, permitting only numeric
    constants and the handful of operators in ``_SAFE_BINOPS``/``_SAFE_UNARYOPS``. Anything else
    (a name, a call, an attribute, a subscript) raises — so ``calculator[__import__('os')...]`` is
    structurally impossible, not merely discouraged. This is the difference between a real, safe
    tool and a security hole.
    """
    if isinstance(node, ast.Constant):
        if isinstance(node.value, bool) or not isinstance(node.value, (int, float)):
            raise ToolError(f"only numbers are allowed, got {node.value!r}")
        return float(node.value)
    if isinstance(node, ast.BinOp) and type(node.op) in _SAFE_BINOPS:
        return _SAFE_BINOPS[type(node.op)](_eval_ast(node.left), _eval_ast(node.right))
    if isinstance(node, ast.UnaryOp) and type(node.op) in _SAFE_UNARYOPS:
        return _SAFE_UNARYOPS[type(node.op)](_eval_ast(node.operand))
    raise ToolError("expression contains a disallowed operation")


def calculator(expression: str) -> str:
    """Evaluate a real arithmetic expression safely and return the result as a string.

    Examples: ``calculator("481 * 32 + 19") -> "15411"``. Integer-valued results print without a
    trailing ``.0`` so observations read naturally to the model (``15411`` not ``15411.0``).
    """
    try:
        tree = ast.parse(expression.strip(), mode="eval")
    except SyntaxError as exc:
        raise ToolError(f"could not parse expression {expression!r}") from exc
    value = _eval_ast(tree.body)
    # present a clean integer when the math is integral (better for the model to read back)
    return str(int(value)) if float(value).is_integer() else f"{value:.6g}"


# A small REAL local knowledge base — offline, deterministic, stands in for a web/Wikipedia search.
# Real text with real facts the agent must actually read to answer multi-hop questions correctly.
KNOWLEDGE_BASE: dict[str, str] = {
    "eiffel tower": (
        "The Eiffel Tower is a wrought-iron lattice tower in Paris, France. It was completed "
        "in 1889 as the entrance arch to the 1889 World's Fair and stands 330 metres tall."
    ),
    "great pyramid of giza": (
        "The Great Pyramid of Giza was completed around 2560 BC. It stood about 146 metres tall "
        "when built and was the tallest human-made structure for over 3,800 years."
    ),
    "moon landing": (
        "Apollo 11 landed the first humans on the Moon on 20 July 1969. Neil Armstrong and "
        "Buzz Aldrin walked on the surface while Michael Collins orbited above."
    ),
    "python programming language": (
        "Python is a high-level programming language first released in 1991 by Guido van Rossum. "
        "It emphasises readability and is widely used in data science and machine learning."
    ),
    "transformer architecture": (
        "The Transformer is a neural network architecture introduced in the 2017 paper "
        "'Attention Is All You Need'. It relies entirely on self-attention and underpins modern LLMs."
    ),
    "mount everest": (
        "Mount Everest is Earth's highest mountain above sea level, with a summit at 8,849 metres. "
        "It was first summited in 1953 by Edmund Hillary and Tenzing Norgay."
    ),
}


def wiki(query: str) -> str:
    """Look a topic up in the real local knowledge base; return the matching entry or a miss.

    A deliberately simple substring/keyword match — the point is that the observation is REAL text
    the agent did not know a priori and must read. Returns a clear miss so the agent can recover
    (a real failure mode: the tool returns nothing useful and the agent must decide what to do).
    """
    q = query.lower().strip()
    for key, entry in KNOWLEDGE_BASE.items():
        # match if the query shares its key words with a KB key (order/extra words tolerated)
        key_words = set(key.split())
        if key in q or key_words.issubset(set(q.split())) or any(w in q for w in key_words if len(w) > 4):
            return entry
    return f"No knowledge-base entry found for {query!r}."


# Registry: name -> (callable, one-line description used to build the system prompt).
Tool = Callable[[str], str]
TOOLS: dict[str, Tool] = {"calculator": calculator, "wiki": wiki}


# --------------------------------------------------------------------------------------------------
# 2. The ReAct prompt — the Thought / Action / Observation grammar, with a one-shot exemplar
# --------------------------------------------------------------------------------------------------
SYSTEM_PROMPT = """You solve questions by reasoning and using tools, one step at a time.

At each step output EXACTLY one Thought line then one Action line, then STOP and wait:
Thought: <your reasoning about what to do next>
Action: <tool>[<input>]

Available tools:
- calculator[expression]   evaluates arithmetic, e.g. Action: calculator[481 * 32 + 19]
- wiki[query]              looks up a fact, e.g. Action: wiki[Eiffel Tower]
- finish[answer]           gives the FINAL answer, e.g. Action: finish[15411]

Rules:
- Never write an "Observation:" line yourself — the system supplies real observations.
- Put the fully computed final value inside finish[...], not an expression.

Example:
Question: What is 6 times 7, plus 3?
Thought: I should compute 6 * 7 + 3 with the calculator.
Action: calculator[6 * 7 + 3]
Observation: 45
Thought: The calculator returned 45, which is the final answer.
Action: finish[45]

Now solve the new question."""

# The single-prompt baseline: answer directly, NO tools, NO reasoning trace — the reason-only foil.
DIRECT_PROMPT = (
    "Answer the question with ONLY the final answer (a number or a short phrase), nothing else."
)


# --------------------------------------------------------------------------------------------------
# 3. The model wrapper — device-agnostic, greedy (deterministic) generation
# --------------------------------------------------------------------------------------------------
def pick_device() -> str:
    """Choose the best available device without assuming a GPU: cuda -> mps -> cpu."""
    if torch.cuda.is_available():
        return "cuda"
    if torch.backends.mps.is_available():
        return "mps"
    return "cpu"


@dataclass
class LanguageModel:
    """A thin, real wrapper over a local instruct model, generating deterministically (greedy).

    Greedy decoding (``do_sample=False``) makes every trace reproducible: the same question yields
    the same Thought/Action every run — essential for a teaching notebook and for honest evaluation.
    """

    model_id: str = DEFAULT_MODEL_ID
    device: str = field(default_factory=pick_device)
    tokenizer: PreTrainedTokenizerBase = field(init=False)
    model: PreTrainedModel = field(init=False)

    def __post_init__(self) -> None:
        self.tokenizer = AutoTokenizer.from_pretrained(self.model_id)
        # float32 everywhere for numerical reproducibility across cpu/mps/cuda (small model, fine)
        self.model = AutoModelForCausalLM.from_pretrained(self.model_id, dtype=torch.float32)
        self.model.to(self.device).eval()

    def generate(
        self,
        system: str,
        user: str,
        *,
        max_new_tokens: int = MAX_NEW_TOKENS,
        stop_strings: tuple[str, ...] | None = GEN_STOP_STRINGS,
    ) -> str:
        """Run one greedy generation given a system + user message; return the new text only.

        ``stop_strings`` is the mechanism that HALTS the model at the first ``Observation:`` — so the
        agent cannot fabricate a tool result. The transformers generator truncates at the stop
        string; we then parse the single action out of what remains.
        """
        text = self.tokenizer.apply_chat_template(
            [{"role": "system", "content": system}, {"role": "user", "content": user}],
            add_generation_prompt=True,
            tokenize=False,
        )
        enc = self.tokenizer(text, return_tensors="pt").to(self.device)
        with torch.no_grad():
            out = self.model.generate(
                **enc,
                max_new_tokens=max_new_tokens,
                do_sample=False,  # greedy => deterministic => reproducible traces
                pad_token_id=self.tokenizer.eos_token_id,
                stop_strings=list(stop_strings) if stop_strings else None,
                tokenizer=self.tokenizer,
            )
        return self.tokenizer.decode(out[0, enc.input_ids.shape[1] :], skip_special_tokens=True).strip()


# --------------------------------------------------------------------------------------------------
# 4. Parsing the model's (messy) output into a structured action
# --------------------------------------------------------------------------------------------------
@dataclass(frozen=True)
class Action:
    """One parsed tool call: the tool name and its raw string argument."""

    tool: str
    arg: str


def parse_action(generated: str) -> tuple[str, Action | None]:
    """Extract the FIRST well-formed ``Action: tool[arg]`` from messy model text.

    Returns ``(thought_and_action_text, action_or_None)``. We deliberately keep only text up to and
    including the first action so a chatty model that dumps several actions (or invents an
    observation) cannot advance the loop by more than one real step. ``None`` means the model failed
    to emit a parseable action this step — a real failure the loop must handle, not crash on.
    """
    match = ACTION_RE.search(generated)
    if match is None:
        return generated, None
    trimmed = generated[: match.end()]  # drop everything after the first complete action
    return trimmed, Action(tool=match.group(1).strip(), arg=match.group(2).strip())


def _normalise_finish(arg: str) -> str:
    """If a ``finish`` argument is a bare numeric expression, evaluate it to a concrete number.

    Small models often emit ``finish[1889 + 100]`` instead of ``finish[1989]``. Rather than mark that
    wrong, we evaluate a purely-numeric finish argument through the same safe calculator — a real,
    defensible normalisation that reflects the model's clear intent. Non-numeric answers pass through.
    """
    if _NUMERIC_EXPR_RE.match(arg):
        try:
            return calculator(arg)
        except ToolError:
            return arg
    return arg


# --------------------------------------------------------------------------------------------------
# 5. The ReAct loop itself — reason, act, observe, repeat
# --------------------------------------------------------------------------------------------------
@dataclass
class Step:
    """One turn of the loop: what the model thought+did, and the real observation it got back."""

    index: int
    text: str  # the raw Thought + Action the model produced this step
    action: Action | None  # the parsed action (None if parsing failed)
    observation: str | None  # the REAL tool result spliced back in (None for finish / parse-fail)


@dataclass
class ReActResult:
    """The full outcome of solving one question: the transcript, the answer, and why it stopped."""

    question: str
    steps: list[Step]
    answer: str | None
    stop_reason: str  # "finish" | "no_action" | "step_budget"

    @property
    def num_steps(self) -> int:
        return len(self.steps)

    @property
    def num_tool_calls(self) -> int:
        return sum(1 for s in self.steps if s.action and s.action.tool in TOOLS)

    def transcript(self) -> str:
        """The human-readable Thought/Action/Observation trace — what the chapter and figures show."""
        lines = [f"Question: {self.question}"]
        for s in self.steps:
            lines.append(s.text)
            if s.observation is not None:
                lines.append(f"Observation: {s.observation}")
        if self.answer is not None:
            lines.append(f"Answer: {self.answer}")
        return "\n".join(lines)


def dispatch(action: Action) -> str:
    """Run the real tool named by the action and return its real observation string.

    Unknown tools and tool errors become observations (not crashes) so the agent can read the
    problem and recover — exactly how a robust ReAct loop behaves in the wild.
    """
    tool = TOOLS.get(action.tool)
    if tool is None:
        return f"Error: no tool named {action.tool!r}. Available: {', '.join(TOOLS)}."
    try:
        return tool(action.arg)
    except ToolError as exc:
        return f"Error: {exc}"


def run_react(llm: LanguageModel, question: str, *, max_steps: int = MAX_STEPS) -> ReActResult:
    """Drive the model through the full Reason -> Act -> Observe loop against real tools.

    The scratchpad grows one real step at a time: we append the model's Thought+Action, then the
    REAL Observation from dispatching the tool, then ask the model again with the enlarged context.
    Three real stop conditions: the model calls ``finish`` (success), it fails to emit an action
    (``no_action``), or it exhausts the step budget (``step_budget``) — the guard against infinite
    loops that every production agent needs.
    """
    scratchpad = f"Question: {question}\n"
    steps: list[Step] = []
    for i in range(max_steps):
        generated = llm.generate(SYSTEM_PROMPT, scratchpad)
        text, action = parse_action(generated)

        if action is None:
            steps.append(Step(index=i, text=text, action=None, observation=None))
            return ReActResult(question, steps, answer=None, stop_reason="no_action")

        if action.tool == "finish":
            answer = _normalise_finish(action.arg)
            steps.append(Step(index=i, text=text, action=action, observation=None))
            return ReActResult(question, steps, answer=answer, stop_reason="finish")

        observation = dispatch(action)  # the REAL tool result
        steps.append(Step(index=i, text=text, action=action, observation=observation))
        scratchpad += f"{text}\nObservation: {observation}\n"

    return ReActResult(question, steps, answer=None, stop_reason="step_budget")


# --------------------------------------------------------------------------------------------------
# 6. The reason-only baseline + the head-to-head comparison
# --------------------------------------------------------------------------------------------------
def run_direct(llm: LanguageModel, question: str) -> str:
    """Answer the question in ONE shot, no tools, no trace — the reason-only foil for ReAct.

    This is what an LLM does on its own: it must recall or guess the facts and do the arithmetic
    in its head. Comparing this to ReAct on multi-step questions is how we *show* that grounding in
    real observations reduces confident wrong answers.
    """
    raw = llm.generate(DIRECT_PROMPT, question, max_new_tokens=48, stop_strings=None)
    return raw.strip().splitlines()[0].strip() if raw.strip() else raw.strip()


@dataclass
class EvalItem:
    """One benchmark question with its known gold answer (as a normalised string)."""

    question: str
    gold: str


# Real multi-step questions that REQUIRE tool use: each needs a lookup and/or exact arithmetic that
# a 1.5B model reliably gets wrong from memory alone — the honest test bed for ReAct vs reason-only.
EVAL_SET: tuple[EvalItem, ...] = (
    EvalItem("What is 481 multiplied by 32, then plus 19?", "15411"),
    EvalItem("What is 1287 minus 998, all multiplied by 6?", "1734"),
    EvalItem("In what year was the Eiffel Tower completed, and what is that year plus 100?", "1989"),
    EvalItem("The Great Pyramid of Giza is about 146 metres tall. What is that height times 3?", "438"),
    EvalItem("Mount Everest's summit is 8849 metres. What is that minus 849?", "8000"),
    EvalItem("What is 17 to the power of 3, minus 200?", "4713"),
)


def _normalise_answer(text: str) -> str:
    """Extract a comparable answer: the last integer if present, else lowercased trimmed text.

    Robust exact-match for short numeric answers: pull the final integer out of a phrase like
    'the answer is 1989.' so formatting differences don't count as wrong.
    """
    numbers = re.findall(r"-?\d+", text.replace(",", ""))
    return numbers[-1] if numbers else text.strip().lower().rstrip(".")


@dataclass
class ComparisonRow:
    """One question evaluated both ways, for the ReAct-vs-direct figure and table."""

    question: str
    gold: str
    react_answer: str | None
    react_correct: bool
    react_steps: int
    direct_answer: str
    direct_correct: bool


def compare_react_vs_direct(llm: LanguageModel, eval_set: tuple[EvalItem, ...] = EVAL_SET) -> list[ComparisonRow]:
    """Run every eval question through BOTH the ReAct loop and the direct baseline; score exact match.

    Returns one row per question with both answers, correctness, and the ReAct step count — the real
    numbers behind the chapter's comparison figure. All generations are greedy, so re-running
    reproduces the table exactly.
    """
    rows: list[ComparisonRow] = []
    for item in eval_set:
        react = run_react(llm, item.question)
        direct = run_direct(llm, item.question)
        gold = _normalise_answer(item.gold)
        rows.append(
            ComparisonRow(
                question=item.question,
                gold=item.gold,
                react_answer=react.answer,
                react_correct=react.answer is not None and _normalise_answer(react.answer) == gold,
                react_steps=react.num_steps,
                direct_answer=direct,
                direct_correct=_normalise_answer(direct) == gold,
            )
        )
    return rows


# --------------------------------------------------------------------------------------------------
# 7. Run it all: the printed proof
# --------------------------------------------------------------------------------------------------
def main() -> None:
    """Load the real model, print two full real traces, then the ReAct-vs-direct comparison."""
    import transformers

    llm = LanguageModel()
    print(
        f"torch {torch.__version__} | transformers {transformers.__version__} | "
        f"model {llm.model_id} | device {llm.device}\n"
    )

    print("=" * 74)
    print("A REAL ReAct trace — a numeric question (needs the calculator)")
    print("=" * 74)
    print(run_react(llm, "What is 481 multiplied by 32, then plus 19?").transcript())

    print("\n" + "=" * 74)
    print("A REAL ReAct trace — a multi-hop question (needs wiki THEN the calculator)")
    print("=" * 74)
    print(run_react(llm, "In what year was the Eiffel Tower completed, and what is that year plus 100?").transcript())

    print("\n" + "=" * 74)
    print("ReAct vs reason-only (no tools) on real multi-step questions")
    print("=" * 74)
    rows = compare_react_vs_direct(llm)
    react_acc = sum(r.react_correct for r in rows) / len(rows)
    direct_acc = sum(r.direct_correct for r in rows) / len(rows)
    print(f"  {'gold':>8} | {'ReAct':>8} {'ok':>3} {'steps':>5} | {'direct':>10} {'ok':>3}")
    print("  " + "-" * 56)
    for r in rows:
        print(
            f"  {r.gold:>8} | {str(r.react_answer):>8} {'Y' if r.react_correct else 'N':>3} "
            f"{r.react_steps:>5} | {str(r.direct_answer)[:10]:>10} {'Y' if r.direct_correct else 'N':>3}"
        )
    print("  " + "-" * 56)
    print(f"  ReAct accuracy  : {react_acc:.0%} ({sum(r.react_correct for r in rows)}/{len(rows)})")
    print(f"  Direct accuracy : {direct_acc:.0%} ({sum(r.direct_correct for r in rows)}/{len(rows)})")


if __name__ == "__main__":
    main()
