"""A REAL function-calling agent — the load-bearing module for the Tool Use & Function Calling chapter.

This is not a toy. A genuine small instruction-tuned LLM with *native* tool-calling support
(``Qwen/Qwen2.5-1.5B-Instruct`` by default) is driven through the real, structured function-calling
protocol: JSON-schema tool declarations go into the chat template, the model emits **structured**
``<tool_call>{"name": ..., "arguments": {...}}</tool_call>`` blocks, we parse that JSON, validate the
arguments against the schema, run the real Python tool, append a **tool-role result message**, and let
the model finish. Every trace, count, and figure the chapter shows is produced here, from real greedy
(temperature 0) generations. Nothing about the model's output is mocked, stubbed, or hand-written.

The whole point of this chapter — and what makes it *complementary* to the ReAct chapter, not
redundant — is the contrast between two ways to let a model call a tool:

  * **ReAct (the sibling chapter)** parses the model's *free text* (``Action: calculator[481*32]``)
    with a regex. Reliable-ish, but brittle: the model can drift off the grammar, mangle brackets,
    or emit prose, and every such drift is a parse failure you must defend against.

  * **Function calling (this chapter)** declares each tool with a **JSON schema** and relies on the
    model having been *trained/templated* to emit a **structured** tool call. The runtime parses
    well-formed JSON — a data format with an unambiguous grammar — instead of guessing at prose.
    Argument *shape* is reliable; the runtime's job shifts from "parse fuzzy text" to "validate typed
    JSON and dispatch." The chapter measures this reliability gap on real output.

What is real here, and why it matters for teaching function calling honestly:

  * **Real model, native tool calling.** A local instruct model whose chat template has a
    ``tools=`` slot (``tokenizer.apply_chat_template(messages, tools=TOOL_SCHEMAS, ...)``). We verified
    it reliably emits valid ``<tool_call>`` JSON, selects the right tool, and extracts arguments —
    including *parallel* calls (two tool_calls in one turn) and *sequential* multi-tool tasks. Greedy
    decoding (``do_sample=False``) makes every trace reproduce exactly; device-agnostic (cuda -> mps ->
    cpu); float32 for cross-device reproducibility; first run downloads+caches the weights, then offline.

  * **Real JSON-schema tools.** Three genuine tools, each with a proper JSON schema (name, description,
    typed parameters, ``required``):
      - ``calculator`` — evaluates arithmetic by walking a parsed AST (NOT ``eval``; only +, -, *, /,
        %, ** and numbers are permitted). It also *normalises* the model's common ``^`` (caret) into
        Python's ``**`` — a real argument-cleaning step, because the model's JSON is valid *shape* but
        not always valid *semantics*.
      - ``convert_units`` — a real length/mass/temperature converter over a genuine factor table,
        with unit-alias normalisation (``"Celsius" -> "C"``) and a real error path for unknown units.
      - ``get_exchange_rate`` — a real (offline, deterministic) rate table, standing in for a live FX
        API, so a currency-then-math task needs two different tools in sequence.

  * **A real protocol loop with real message roles.** ``build_messages -> apply_chat_template(tools=)
    -> generate -> parse structured tool_calls -> validate args against the schema -> dispatch the real
    tool -> append a tool-role result message -> generate again``, until the model answers with no
    tool call (or a step budget is hit). Both **parallel** (independent calls in one turn) and
    **sequential** (each call chosen from the previous result) multi-tool traces are produced for real.

  * **Real validation.** Structured calling guarantees the *shape* of the arguments, never their
    *correctness*. The model can emit ``{"expression": "1287 - 998 * 6"}`` when the user meant
    ``(1287 - 998) * 6`` — valid JSON, wrong maths. And it can omit a required field or send the wrong
    type. We validate every call against its JSON schema (required keys present, types coercible) and
    turn a violation into a real tool-error result the model can read and recover from — exactly what a
    robust agent does.

  * **A real comparison.** ``compare_structured_vs_text`` runs the SAME questions two ways — the
    structured function-calling path (JSON, schema-validated) vs a ReAct-style *text* path that
    prompts for ``TOOL: name(args)`` and regex-parses it — and measures how often each yields a
    *parseable, dispatchable* call on real model output. That reliability gap is the empirical core of
    "why structured beats text-parsing," and it ties directly back to the ReAct chapter.

Run it standalone (downloads+caches the model on first run, then offline)::

    python function_calling_agent.py

Verified on Python 3.12 / torch 2.12 / transformers 5.10, CPU / Apple MPS / CUDA.
"""

from __future__ import annotations

import ast
import json
import operator
import re
from collections.abc import Callable
from dataclasses import dataclass, field
from typing import Any

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, PreTrainedModel, PreTrainedTokenizerBase

# --------------------------------------------------------------------------------------------------
# Constants (hoisted; no magic numbers inline)
# --------------------------------------------------------------------------------------------------
DEFAULT_MODEL_ID = "Qwen/Qwen2.5-1.5B-Instruct"  # small, real, NATIVE tool-calling, freely available
MAX_NEW_TOKENS = 200  # cap per turn: enough for one or several tool calls, or a short final answer
MAX_TOOL_TURNS = 5  # step budget: the real "give up" stop condition that prevents infinite loops

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

# The model's structured tool call arrives wrapped in <tool_call>...</tool_call> (the Qwen/Hermes
# template convention). One regex, reused, pulls out the JSON payload(s) — DOTALL for multi-line JSON.
TOOL_CALL_RE = re.compile(r"<tool_call>\s*(\{.*?\})\s*</tool_call>", re.DOTALL)

# The TEXT-protocol foil (the ReAct-style parsing path), for the reliability comparison: `TOOL: name(args)`.
TEXT_CALL_RE = re.compile(r"TOOL:\s*([A-Za-z_]\w*)\s*\((.*?)\)", re.DOTALL)


class ToolError(Exception):
    """Raised when a tool cannot process its input — surfaced to the agent as a tool-result message."""


# --------------------------------------------------------------------------------------------------
# 1. Real tools — genuine Python functions with genuine return values
# --------------------------------------------------------------------------------------------------
def _eval_ast(node: ast.expr) -> float:
    """Recursively evaluate a *whitelisted* arithmetic AST — the safe core of the calculator.

    We never call ``eval``. We parse the expression to an AST and walk it, permitting only numeric
    constants and the handful of operators in ``_SAFE_BINOPS``/``_SAFE_UNARYOPS``. Anything else
    (a name, a call, an attribute, a subscript) raises — so an argument like ``__import__('os')...``
    is structurally impossible, not merely discouraged. The model's ``arguments`` are untrusted
    input, exactly like user input; a real tool treats them that way.
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


def calculator(*, expression: str) -> str:
    """Evaluate a real arithmetic expression safely and return the result as a string.

    Keyword-only ``expression`` mirrors the JSON schema's named parameter — the dispatcher unpacks the
    model's ``arguments`` dict straight into this signature. We normalise the model's frequent ``^``
    (a real observed slip: it emits ``17^3`` where Python wants ``17**3``) before parsing. Integer
    results print without a trailing ``.0`` so the tool-result message reads naturally (``4713``).
    """
    normalised = expression.strip().replace("^", "**")  # real cleanup: caret power -> Python power
    try:
        tree = ast.parse(normalised, mode="eval")
    except SyntaxError as exc:
        raise ToolError(f"could not parse expression {expression!r}") from exc
    value = _eval_ast(tree.body)
    return str(int(value)) if float(value).is_integer() else f"{value:.6g}"


# Real conversion factors to a canonical base unit per dimension (length->metre, mass->kilogram).
# Temperature is affine (offset + scale), so it is handled separately below — a real edge the schema
# alone can't express, and a good lesson in "the tool, not the schema, owns semantics".
_CANONICAL_UNIT: dict[str, str] = {
    "km": "km",
    "kilometre": "km",
    "kilometer": "km",
    "kilometres": "km",
    "kilometers": "km",
    "m": "m",
    "metre": "m",
    "meter": "m",
    "metres": "m",
    "meters": "m",
    "mi": "mi",
    "mile": "mi",
    "miles": "mi",
    "ft": "ft",
    "foot": "ft",
    "feet": "ft",
    "kg": "kg",
    "kilogram": "kg",
    "kilograms": "kg",
    "g": "g",
    "gram": "g",
    "grams": "g",
    "lb": "lb",
    "pound": "lb",
    "pounds": "lb",
    "c": "C",
    "celsius": "C",
    "centigrade": "C",
    "f": "F",
    "fahrenheit": "F",
}
_TO_BASE: dict[str, tuple[str, float]] = {  # unit -> (dimension, factor-to-base)
    "km": ("length", 1000.0),
    "m": ("length", 1.0),
    "mi": ("length", 1609.344),
    "ft": ("length", 0.3048),
    "kg": ("mass", 1.0),
    "g": ("mass", 0.001),
    "lb": ("mass", 0.45359237),
}


def convert_units(*, value: float, from_unit: str, to_unit: str) -> str:
    """Convert ``value`` from ``from_unit`` to ``to_unit`` (length, mass, or temperature).

    A real converter over a genuine factor table, with alias normalisation ('kilometres' -> 'km',
    'Celsius' -> 'C') because the model sends whatever word the user used. Temperature is affine and
    handled explicitly. Unknown or mismatched-dimension units raise a real ``ToolError`` that becomes a
    tool-result the model reads — the schema guarantees three string/number args arrive, not that they
    name real, compatible units.
    """
    src = _CANONICAL_UNIT.get(from_unit.strip().lower())
    dst = _CANONICAL_UNIT.get(to_unit.strip().lower())
    if src is None or dst is None:
        raise ToolError(f"unknown unit(s): {from_unit!r} -> {to_unit!r}")
    if {src, dst} <= {"C", "F"}:  # temperature: affine, not a simple factor
        if src == dst:
            result = float(value)
        elif src == "C":
            result = float(value) * 9.0 / 5.0 + 32.0
        else:
            result = (float(value) - 32.0) * 5.0 / 9.0
        return f"{result:.4g} {dst}"
    if src not in _TO_BASE or dst not in _TO_BASE:
        raise ToolError(f"cannot convert between {from_unit!r} and {to_unit!r}")
    src_dim, src_factor = _TO_BASE[src]
    dst_dim, dst_factor = _TO_BASE[dst]
    if src_dim != dst_dim:
        raise ToolError(f"incompatible dimensions: {src_dim} ({from_unit}) vs {dst_dim} ({to_unit})")
    result = float(value) * src_factor / dst_factor
    return f"{result:.6g} {dst}"


# A small REAL offline rate table — deterministic so the notebook reproduces exactly; stands in for a
# live FX API. A currency-then-math task must call THIS, read the rate, then call the calculator.
_EXCHANGE_RATES: dict[tuple[str, str], float] = {
    ("USD", "EUR"): 0.92,
    ("EUR", "USD"): 1.09,
    ("USD", "GBP"): 0.79,
    ("GBP", "USD"): 1.27,
    ("USD", "JPY"): 157.0,
    ("JPY", "USD"): 0.0064,
}


def get_exchange_rate(*, from_currency: str, to_currency: str) -> str:
    """Return the (offline, deterministic) exchange rate multiplier from one currency to another.

    Real lookup with a real miss path: unknown pairs raise a ``ToolError``. The returned rate is a
    multiplier the agent must then *use* (multiply the source amount by it) — which is why a full
    currency question needs this tool AND the calculator, chosen in sequence.
    """
    key = (from_currency.strip().upper(), to_currency.strip().upper())
    if key not in _EXCHANGE_RATES:
        raise ToolError(f"no rate for {from_currency!r} -> {to_currency!r}")
    return str(_EXCHANGE_RATES[key])


# --------------------------------------------------------------------------------------------------
# 2. The tool REGISTRY: JSON schema + the real callable, side by side
# --------------------------------------------------------------------------------------------------
@dataclass(frozen=True)
class ToolSpec:
    """One tool: its JSON schema (what the model sees) and its real callable (what the runtime runs)."""

    schema: dict[str, Any]  # the OpenAI/JSON-schema-style function declaration handed to the template
    fn: Callable[..., str]  # the real Python function, called with **arguments

    @property
    def name(self) -> str:
        return self.schema["function"]["name"]

    @property
    def parameters(self) -> dict[str, Any]:
        return self.schema["function"]["parameters"]


def _fn_schema(name: str, description: str, properties: dict[str, Any], required: list[str]) -> dict[str, Any]:
    """Build one OpenAI/JSON-schema-style function declaration (the format apply_chat_template expects)."""
    return {
        "type": "function",
        "function": {
            "name": name,
            "description": description,
            "parameters": {"type": "object", "properties": properties, "required": required},
        },
    }


# The registry: name -> ToolSpec. The schemas here are the SINGLE source of truth — they are handed to
# the model (via the chat template) AND validated against at dispatch time, so declaration and
# enforcement can never drift.
TOOL_REGISTRY: dict[str, ToolSpec] = {
    "calculator": ToolSpec(
        schema=_fn_schema(
            "calculator",
            "Evaluate an arithmetic expression and return the numeric result. "
            "Use for any exact calculation (multiplication, powers, parenthesised expressions).",
            {"expression": {"type": "string", "description": "An arithmetic expression, e.g. '481 * 32 + 19'."}},
            ["expression"],
        ),
        fn=calculator,
    ),
    "convert_units": ToolSpec(
        schema=_fn_schema(
            "convert_units",
            "Convert a value from one unit to another. Supports length (km, m, mi, ft), "
            "mass (kg, g, lb), and temperature (C, F).",
            {
                "value": {"type": "number", "description": "The numeric value to convert."},
                "from_unit": {"type": "string", "description": "Source unit, e.g. 'km'."},
                "to_unit": {"type": "string", "description": "Target unit, e.g. 'mi'."},
            },
            ["value", "from_unit", "to_unit"],
        ),
        fn=convert_units,
    ),
    "get_exchange_rate": ToolSpec(
        schema=_fn_schema(
            "get_exchange_rate",
            "Get the exchange-rate multiplier from one currency to another. "
            "Multiply the source amount by this rate to get the target amount.",
            {
                "from_currency": {"type": "string", "description": "ISO code, e.g. 'USD'."},
                "to_currency": {"type": "string", "description": "ISO code, e.g. 'EUR'."},
            },
            ["from_currency", "to_currency"],
        ),
        fn=get_exchange_rate,
    ),
}


def tool_schemas() -> list[dict[str, Any]]:
    """The list of JSON-schema declarations to hand to ``apply_chat_template(tools=...)``."""
    return [spec.schema for spec in TOOL_REGISTRY.values()]


# --------------------------------------------------------------------------------------------------
# 3. Structured parsing + schema validation (the heart of "structured beats text")
# --------------------------------------------------------------------------------------------------
@dataclass(frozen=True)
class ToolCall:
    """One structured tool call the model emitted: a tool name and a JSON arguments dict."""

    name: str
    arguments: dict[str, Any]


def parse_tool_calls(generated: str) -> list[ToolCall]:
    """Extract every well-formed structured ``<tool_call>{...}</tool_call>`` from the model's output.

    Unlike ReAct's regex-on-prose, we parse a *data format*: pull each JSON payload and ``json.loads``
    it. A block whose JSON is malformed, or that lacks a ``name``, is skipped (it will surface as "no
    dispatchable call this turn"). Returning a *list* is what makes **parallel** tool calls first-class:
    the model can emit two independent calls in one turn and we get both.
    """
    calls: list[ToolCall] = []
    for payload in TOOL_CALL_RE.findall(generated):
        try:
            obj = json.loads(payload)
        except json.JSONDecodeError:
            continue  # malformed JSON -> not a dispatchable call (rare, but real: handle, don't crash)
        name = obj.get("name")
        args = obj.get("arguments", {})
        if isinstance(name, str) and isinstance(args, dict):
            calls.append(ToolCall(name=name, arguments=args))
    return calls


# JSON-schema "type" -> the Python types we accept for it (numbers may arrive as int/float/str-of-num).
_JSON_TYPE_OK: dict[str, tuple[type, ...]] = {
    "string": (str,),
    "number": (int, float),
    "integer": (int,),
    "boolean": (bool,),
}


def validate_arguments(call: ToolCall, spec: ToolSpec) -> dict[str, Any]:
    """Validate a structured call's arguments against its JSON schema; return coerced kwargs or raise.

    Structured calling guarantees the *shape* is JSON — it does NOT guarantee required fields are
    present or types are right. This is the real second line of defence: every required property must
    be present, and each provided value must match (or coerce to) its declared type. A ``number`` sent
    as the string ``"42"`` is coerced; a missing required field or an uncoercible value raises a
    ``ToolError`` that becomes a tool-result the model can read and correct. This is the function-
    calling analogue of ReAct's defensive parsing — but on typed JSON, not fuzzy prose.
    """
    props: dict[str, Any] = spec.parameters.get("properties", {})
    required: list[str] = spec.parameters.get("required", [])
    missing = [k for k in required if k not in call.arguments]
    if missing:
        raise ToolError(f"missing required argument(s) {missing} for {spec.name}")
    coerced: dict[str, Any] = {}
    for key, raw in call.arguments.items():
        if key not in props:
            continue  # ignore extraneous keys rather than fail — models occasionally add stray fields
        want = props[key].get("type", "string")
        ok_types = _JSON_TYPE_OK.get(want, (str,))
        if want in ("number", "integer") and isinstance(raw, str):
            try:
                coerced[key] = float(raw) if want == "number" else int(raw)
                continue
            except ValueError as exc:
                raise ToolError(f"argument {key!r} must be a {want}, got {raw!r}") from exc
        if isinstance(raw, ok_types) and not (want != "boolean" and isinstance(raw, bool)):
            coerced[key] = raw
        else:
            raise ToolError(f"argument {key!r} must be a {want}, got {type(raw).__name__} {raw!r}")
    return coerced


def dispatch(call: ToolCall) -> str:
    """Validate + run the real tool named by a structured call; return its real result string.

    Unknown tools, schema violations, and tool errors all become *result strings* (not crashes) so the
    model can read the problem and recover on the next turn — exactly how a robust function-calling
    loop behaves in the wild.
    """
    spec = TOOL_REGISTRY.get(call.name)
    if spec is None:
        return f"Error: no tool named {call.name!r}. Available: {', '.join(TOOL_REGISTRY)}."
    try:
        kwargs = validate_arguments(call, spec)
        return spec.fn(**kwargs)
    except ToolError as exc:
        return f"Error: {exc}"


# --------------------------------------------------------------------------------------------------
# 4. The model wrapper — device-agnostic, greedy (deterministic), native tool-calling template
# --------------------------------------------------------------------------------------------------
def pick_device() -> str:
    """Choose the best available device without assuming a GPU: cuda -> mps -> cpu."""
    if torch.cuda.is_available():
        return "cuda"
    if torch.backends.mps.is_available():
        return "mps"
    return "cpu"


Message = dict[str, Any]  # a chat message: {"role": ..., "content": ..., optionally "tool_calls"/"name"}


@dataclass
class ToolCallingModel:
    """A thin, real wrapper over a local instruct model with NATIVE tool-calling, greedy-decoded.

    The key call is ``apply_chat_template(messages, tools=..., add_generation_prompt=True)``: the
    template renders the JSON tool schemas into the exact system section the model was trained to read,
    so the model knows which tools exist and emits ``<tool_call>`` JSON to use them. Greedy decoding
    (``do_sample=False``) makes every trace reproducible.
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
        messages: list[Message],
        *,
        tools: list[dict[str, Any]] | None = None,
        max_new_tokens: int = MAX_NEW_TOKENS,
    ) -> str:
        """Render the messages (+ optional tool schemas) with the chat template and greedily generate.

        When ``tools`` is provided, ``apply_chat_template`` injects the tool declarations — this is the
        native function-calling path. When it is ``None``, we get an ordinary chat turn (used by the
        text-protocol foil in the comparison). Returns only the newly generated text.
        """
        text = self.tokenizer.apply_chat_template(messages, tools=tools, add_generation_prompt=True, tokenize=False)
        enc = self.tokenizer(text, return_tensors="pt").to(self.device)
        with torch.no_grad():
            out = self.model.generate(
                **enc,
                max_new_tokens=max_new_tokens,
                do_sample=False,  # greedy => deterministic => reproducible traces
                pad_token_id=self.tokenizer.eos_token_id,
            )
        return self.tokenizer.decode(out[0, enc.input_ids.shape[1] :], skip_special_tokens=True).strip()


# --------------------------------------------------------------------------------------------------
# 5. The function-calling LOOP — schema -> call -> validate -> execute -> tool-result -> answer
# --------------------------------------------------------------------------------------------------
@dataclass
class Turn:
    """One turn of the loop: the raw generation, the structured calls parsed, and their real results."""

    index: int
    raw: str  # the raw text the model produced this turn
    calls: list[ToolCall]  # the structured tool calls parsed (possibly several -> parallel)
    results: list[str]  # the real tool-result strings, aligned with ``calls``


@dataclass
class AgentResult:
    """The full outcome of answering one query: the message trace, the answer, and why it stopped."""

    query: str
    turns: list[Turn]
    answer: str | None
    stop_reason: str  # "answered" | "step_budget"
    messages: list[Message]  # the full chat transcript, including tool-role result messages

    @property
    def num_turns(self) -> int:
        return len(self.turns)

    @property
    def num_tool_calls(self) -> int:
        return sum(len(t.calls) for t in self.turns)

    def transcript(self) -> str:
        """A human-readable trace of the protocol: user -> tool_call(s) -> tool result(s) -> answer."""
        lines = [f"User: {self.query}"]
        for t in self.turns:
            for call in t.calls:
                lines.append(f"Assistant -> tool_call: {call.name}({json.dumps(call.arguments)})")
            for call, result in zip(t.calls, t.results):
                lines.append(f"Tool [{call.name}] -> {result}")
        if self.answer is not None:
            lines.append(f"Assistant: {self.answer}")
        return "\n".join(lines)


def run_agent(model: ToolCallingModel, query: str, *, max_turns: int = MAX_TOOL_TURNS) -> AgentResult:
    """Drive the model through the full structured function-calling protocol against real tools.

    The message list grows by the book: the user query, then for each turn an *assistant* message
    recording the structured ``tool_calls``, then one *tool-role* result message per call, then the
    model is asked again. The loop ends when the model replies with **no** tool call (that reply is the
    answer) or the turn budget is hit (the guard against infinite loops). Parallel calls in a single
    turn are handled naturally — each gets its own tool-result message.
    """
    messages: list[Message] = [{"role": "user", "content": query}]
    turns: list[Turn] = []
    for i in range(max_turns):
        raw = model.generate(messages, tools=tool_schemas())
        calls = parse_tool_calls(raw)
        if not calls:  # no structured call -> the model is answering in plain text
            messages.append({"role": "assistant", "content": raw})
            return AgentResult(query, turns, answer=raw, stop_reason="answered", messages=messages)

        results = [dispatch(c) for c in calls]  # the REAL tool results (validated + executed)
        turns.append(Turn(index=i, raw=raw, calls=calls, results=results))
        # record the assistant's tool_calls, then one tool-role result message per call (the protocol)
        messages.append(
            {
                "role": "assistant",
                "content": "",
                "tool_calls": [
                    {"type": "function", "function": {"name": c.name, "arguments": c.arguments}} for c in calls
                ],
            }
        )
        for call, result in zip(calls, results):
            messages.append({"role": "tool", "name": call.name, "content": result})

    return AgentResult(query, turns, answer=None, stop_reason="step_budget", messages=messages)


# --------------------------------------------------------------------------------------------------
# 6. The TEXT-protocol foil + the structured-vs-text reliability comparison
# --------------------------------------------------------------------------------------------------
# A ReAct-style *text* instruction: emit `TOOL: name(arg=value, ...)` as prose, which we regex-parse.
# This is the honest foil — the same job (call a tool) done by parsing free text instead of JSON.
_TEXT_PROTOCOL_SYSTEM = (
    "You can call tools by writing a line of the form:\n"
    "TOOL: <name>(<arg>=<value>, ...)\n"
    "Available tools:\n"
    '- calculator(expression="...")  evaluates arithmetic\n'
    '- convert_units(value=..., from_unit="...", to_unit="...")  converts units\n'
    '- get_exchange_rate(from_currency="...", to_currency="...")  gets an FX rate\n'
    "Write exactly one TOOL: line and nothing else."
)


def _parse_text_call(generated: str) -> ToolCall | None:
    """Parse a ReAct-style ``TOOL: name(args)`` line into a ToolCall — the brittle text path.

    Real prose parsing: we must split ``arg=value`` pairs ourselves, strip quotes, and guess types.
    Any drift (missing parens, prose instead of the line, an un-splittable arg list) yields ``None`` —
    a parse failure. This is deliberately the same *kind* of code the ReAct chapter leans on, so the
    comparison is fair: it can and does fail on messy output that structured JSON handles cleanly.
    """
    match = TEXT_CALL_RE.search(generated)
    if match is None:
        return None
    name, arg_blob = match.group(1).strip(), match.group(2).strip()
    args: dict[str, Any] = {}
    if arg_blob:
        for pair in re.split(r",(?![^()]*\))", arg_blob):  # split on top-level commas only
            if "=" not in pair:
                return None  # a positional/malformed arg -> the text parser gives up (a real failure)
            k, v = pair.split("=", 1)
            v = v.strip().strip("\"'")
            try:
                args[k.strip()] = float(v) if re.fullmatch(r"-?\d+(\.\d+)?", v) else v
            except ValueError:
                args[k.strip()] = v
    return ToolCall(name=name, arguments=args)


@dataclass
class ReliabilityRow:
    """One query, tried both ways: did structured (JSON) and did text (prose) yield a dispatchable call?"""

    query: str
    structured_ok: bool  # model emitted valid JSON that parsed AND validated against the schema
    text_ok: bool  # the TOOL: line parsed AND validated against the schema
    structured_call: str | None
    text_call: str | None


# Real queries that each need exactly one tool call — the fair test bed for "did we get a usable call?"
RELIABILITY_QUERIES: tuple[str, ...] = (
    "What is 481 multiplied by 32, then plus 19?",
    "What is 17 to the power of 3, minus 200?",
    "Convert 42 kilometres to miles.",
    "How many pounds is 5 kilograms?",
    "Convert 100 degrees Celsius to Fahrenheit.",
    "What is the exchange rate from US dollars to euros?",
    "What is 1000 grams in pounds?",
    "What is 250 times 4, then divided by 5?",
)


def _validates(call: ToolCall | None) -> bool:
    """True iff a parsed call names a real tool and its args pass schema validation (i.e. is dispatchable)."""
    if call is None:
        return False
    spec = TOOL_REGISTRY.get(call.name)
    if spec is None:
        return False
    try:
        validate_arguments(call, spec)
        return True
    except ToolError:
        return False


def compare_structured_vs_text(
    model: ToolCallingModel, queries: tuple[str, ...] = RELIABILITY_QUERIES
) -> list[ReliabilityRow]:
    """For each query, ask the model BOTH ways and measure which yields a parseable, valid tool call.

    Structured path: native tool schemas -> parse ``<tool_call>`` JSON -> schema-validate. Text path:
    the ``TOOL: name(args)`` prose instruction -> regex-parse -> schema-validate. We score *reliability
    of getting a dispatchable call*, not answer correctness — that is the axis on which structured
    calling beats text-parsing, and it is exactly the pain the ReAct chapter's parser fights. All
    generations are greedy, so this reproduces exactly.
    """
    rows: list[ReliabilityRow] = []
    for q in queries:
        structured_raw = model.generate([{"role": "user", "content": q}], tools=tool_schemas())
        s_calls = parse_tool_calls(structured_raw)
        s_call = s_calls[0] if s_calls else None

        text_raw = model.generate(
            [{"role": "system", "content": _TEXT_PROTOCOL_SYSTEM}, {"role": "user", "content": q}]
        )
        t_call = _parse_text_call(text_raw)

        rows.append(
            ReliabilityRow(
                query=q,
                structured_ok=_validates(s_call),
                text_ok=_validates(t_call),
                structured_call=f"{s_call.name}({json.dumps(s_call.arguments)})" if s_call else None,
                text_call=f"{t_call.name}({json.dumps(t_call.arguments)})" if t_call else None,
            )
        )
    return rows


# --------------------------------------------------------------------------------------------------
# 7. Run it all: the printed proof
# --------------------------------------------------------------------------------------------------
def main() -> None:
    """Load the real model, print a single-tool trace, a sequential and a parallel multi-tool trace,
    then the structured-vs-text reliability comparison."""
    import transformers

    model = ToolCallingModel()
    print(
        f"torch {torch.__version__} | transformers {transformers.__version__} | "
        f"model {model.model_id} | device {model.device}\n"
    )

    print("=" * 78)
    print("A REAL single tool call — schema -> structured call -> execute -> tool-result -> answer")
    print("=" * 78)
    print(run_agent(model, "What is 481 multiplied by 32, then plus 19?").transcript())

    print("\n" + "=" * 78)
    print("A REAL sequential multi-tool trace — look up the rate, THEN ground the answer on it")
    print("=" * 78)
    print(
        run_agent(
            model,
            "First look up the USD to JPY exchange rate with the get_exchange_rate tool, "
            "then multiply 40 dollars by that rate to get the yen amount.",
        ).transcript()
    )

    print("\n" + "=" * 78)
    print("A REAL parallel multi-tool trace — two independent unit conversions in ONE turn")
    print("=" * 78)
    print(
        run_agent(model, "Convert 42 kilometres to miles, and separately convert 5 kilograms to pounds.").transcript()
    )

    print("\n" + "=" * 78)
    print("Structured function-calling vs text-parsing: which yields a dispatchable call?")
    print("=" * 78)
    rows = compare_structured_vs_text(model)
    s_ok = sum(r.structured_ok for r in rows)
    t_ok = sum(r.text_ok for r in rows)
    print(f"  {'structured':>10} {'text':>6} | query")
    print("  " + "-" * 62)
    for r in rows:
        print(f"  {'OK' if r.structured_ok else 'FAIL':>10} {'OK' if r.text_ok else 'FAIL':>6} | {r.query[:44]}")
    print("  " + "-" * 62)
    print(f"  structured (JSON, schema-validated): {s_ok}/{len(rows)} dispatchable")
    print(f"  text (TOOL: prose, regex-parsed)   : {t_ok}/{len(rows)} dispatchable")


if __name__ == "__main__":
    main()
