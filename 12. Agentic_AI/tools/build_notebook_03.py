"""Generate the step-by-step function-calling teaching notebook (03-Tool-Use-and-Function-Calling.ipynb).

The notebook mirrors ``function_calling_agent.py`` one operation at a time, so a reader can open it, run
every cell live against the REAL model, and *teach* function calling from it. Each numbered step has a
short markdown lead-in (the intuition) followed by ONE focused code cell with real output. This
generator writes the .ipynb; a separate nbconvert pass executes it headless so the real outputs are
embedded.

    python build_notebook_03.py         # writes the .ipynb (unexecuted) into the chapter's code/
    python -m nbconvert --to notebook --execute --inplace \
        "../03-Tool-Use-and-Function-Calling/code/03-Tool-Use-and-Function-Calling.ipynb"

This generator lives in the domain-level ``12. Agentic_AI/tools/`` folder; the notebook it writes (and
the module it mirrors) stay in the chapter's ``code/`` folder. Kept as a generator (not a hand-edited
.ipynb) so the notebook and the module stay in lockstep: the same agent, typed once in the module,
demonstrated step-by-step here. Each code cell imports ONLY the names it uses, so the notebook is clean
under ``ruff check`` (no unused-import warnings).
"""

from __future__ import annotations

import json
from pathlib import Path

NB_PATH = (
    Path(__file__).resolve().parent.parent
    / "03-Tool-Use-and-Function-Calling"
    / "code"
    / "03-Tool-Use-and-Function-Calling.ipynb"
)

_CELL_ID = 0


def _next_id() -> str:
    """Stable, sequential cell id (silences nbformat's MissingIDFieldWarning)."""
    global _CELL_ID
    _CELL_ID += 1
    return f"cell-{_CELL_ID:02d}"


def md(source: str) -> dict:
    """A markdown cell."""
    return {
        "cell_type": "markdown",
        "id": _next_id(),
        "metadata": {},
        "source": source.splitlines(keepends=True),
    }


def code(source: str) -> dict:
    """A code cell (outputs filled in by the nbconvert execute pass)."""
    return {
        "cell_type": "code",
        "id": _next_id(),
        "execution_count": None,
        "metadata": {},
        "outputs": [],
        "source": source.splitlines(keepends=True),
    }


CELLS: list[dict] = []


def add_md(source: str) -> None:
    CELLS.append(md(source))


def add_code(source: str) -> None:
    CELLS.append(code(source))


# ============================ Title ============================================================
add_md(
    "# Tool Use & Function Calling — a step-by-step, runnable teaching notebook\n"
    "\n"
    "This notebook builds a **real function-calling agent** from the ground up, one operation at a time, "
    "driving a genuine small instruction-tuned LLM with *native* tool calling "
    "(`Qwen/Qwen2.5-1.5B-Instruct`) through the real **structured** protocol: JSON-schema tool "
    "declarations → the model emits a structured `<tool_call>{...}</tool_call>` → we parse the JSON, "
    "validate the arguments against the schema, run the real tool, feed the result back as a "
    "**tool-role message**, and let the model answer. It is the executable companion to the chapter and "
    "to `function_calling_agent.py` — every function used here lives in that module, imported so the "
    "notebook and the module can never drift apart.\n"
    "\n"
    "Nothing about the model's output is mocked. The traces below are what the model actually generates "
    "(greedy / temperature 0, so they are reproducible). By the end you will have **seen**, with a real "
    "model and real tools:\n"
    "\n"
    "1. why free-text tool invocation (the ReAct way) is **brittle** — and a real measurement of it;\n"
    "2. three **real JSON-schema tools** — a safe calculator, a unit converter, an FX-rate lookup;\n"
    "3. how the schemas enter the prompt via `apply_chat_template(tools=...)`;\n"
    "4. a **real structured `<tool_call>`** the model emits, parsed as JSON (not regex-on-prose);\n"
    "5. **schema validation** — because valid JSON is not the same as valid arguments;\n"
    "6. the **full protocol loop** — call → tool-role result → answer — on single, sequential, and "
    "**parallel** multi-tool tasks;\n"
    "7. a **head-to-head**: structured function-calling vs a ReAct-style text protocol, on reliability.\n"
    "\n"
    "The first run downloads the model (a few hundred MB) and caches it; every run after is offline and "
    "reproducible. It runs on CPU, Apple MPS, or CUDA — whatever you have."
)

# ---- Step 0: setup + banner ----
add_md(
    "## Step 0 — Setup and version banner\n"
    "\n"
    "We import the real pieces from the chapter module (so this notebook uses the *exact same code* the "
    "chapter and figures use) and print the library + model versions and the device the results were "
    "produced on. `pick_device()` chooses `cuda → mps → cpu` without assuming a GPU."
)
add_code(
    "import torch\n"
    "import transformers\n"
    "\n"
    "from function_calling_agent import pick_device\n"
    "\n"
    "print(f'torch {torch.__version__} | transformers {transformers.__version__} '\n"
    "      f'| device {pick_device()}')"
)

# ============================ 1. The problem =====================================================
add_md(
    "## Step 1 — The problem: free-text tool calls are brittle\n"
    "\n"
    "The sibling **ReAct** chapter has the model write its tool call as *free text* — "
    "`Action: calculator[481 * 32 + 19]` — which the runtime then **regex-parses**. That works until the "
    "model drifts: a missing bracket, prose instead of the line, the wrong shape. Feel that brittleness "
    "directly. We give the model a ReAct-style text instruction and a query, and try to parse a call out "
    "of whatever prose it returns."
)
add_code(
    "from function_calling_agent import ToolCallingModel, _TEXT_PROTOCOL_SYSTEM, _parse_text_call\n"
    "\n"
    "model = ToolCallingModel()   # loads the real model (downloads+caches on first run)\n"
    "print(f'loaded {model.model_id} on {model.device}\\n')\n"
    "\n"
    "q = 'What is 481 multiplied by 32, then plus 19?'\n"
    "text_raw = model.generate([\n"
    "    {'role': 'system', 'content': _TEXT_PROTOCOL_SYSTEM},\n"
    "    {'role': 'user', 'content': q},\n"
    "])\n"
    "print('--- the model prose ---')\n"
    "print(text_raw[:300])\n"
    "print('\\n--- regex-parsed call ---')\n"
    "print(_parse_text_call(text_raw))   # often None: the prose did not match the TOOL: grammar"
)
add_md(
    "The model answers helpfully but *not* in the rigid `TOOL: name(args)` shape the parser needs — so "
    "the parse returns `None`. This is the felt inadequacy function calling removes: instead of hoping "
    "the model formats prose correctly, we **declare a schema** and rely on the model having been trained "
    "to emit a **structured** call the runtime can parse as data."
)

# ============================ 2. Real tools =====================================================
add_md(
    "## Step 2 — Real tool #1: a safe calculator (AST-walked, not `eval`)\n"
    "\n"
    "The first tool is a real calculator. Crucially it is **not** `eval(expr)` — the model's arguments "
    "are untrusted input, exactly like user input. Instead we parse the expression to an abstract syntax "
    "tree and walk it, permitting *only* numbers and a whitelist of arithmetic operators. It also cleans "
    "the model's frequent `^` (caret) into Python's `**`. Safe tool design is part of the lesson."
)
add_code(
    "from function_calling_agent import calculator\n"
    "\n"
    "print(calculator(expression='481 * 32 + 19'))   # the real answer\n"
    "print(calculator(expression='17^3 - 200'))      # caret power is normalised to ** -> 4713\n"
    "print(calculator(expression='(1287 - 998) * 6'))# parentheses respected\n"
    "\n"
    "# and it refuses anything that is not pure arithmetic (this is what makes it SAFE):\n"
    "try:\n"
    "    calculator(expression='__import__(\"os\").system(\"echo hi\")')\n"
    "except Exception as e:\n"
    "    print('rejected unsafe input:', type(e).__name__, '-', e)"
)

add_md(
    "## Step 3 — Real tools #2 and #3: a unit converter and an FX-rate lookup\n"
    "\n"
    "Two more real tools give the agent a reason to *choose* between tools and to *chain* them. "
    "`convert_units` converts length/mass/temperature over a genuine factor table (with unit-alias "
    "normalisation and a real error path); `get_exchange_rate` returns a real (offline) rate the agent "
    "must then *use*. Both are ordinary Python functions — the model never runs them, our runtime does."
)
add_code(
    "from function_calling_agent import convert_units, get_exchange_rate\n"
    "\n"
    "print(convert_units(value=42, from_unit='km', to_unit='mi'))       # 26.0976 mi\n"
    "print(convert_units(value=100, from_unit='Celsius', to_unit='F'))  # alias 'Celsius' -> C; 212 F\n"
    "print(get_exchange_rate(from_currency='USD', to_currency='JPY'))   # a real rate: 157.0\n"
    "\n"
    "# real error paths (the schema guarantees args ARRIVE, not that they name real, compatible units):\n"
    "try:\n"
    "    convert_units(value=5, from_unit='kg', to_unit='mi')  # mass -> length is incompatible\n"
    "except Exception as e:\n"
    "    print('rejected:', e)"
)

# ============================ 4. the schema ======================================================
add_md(
    "## Step 4 — The JSON schema: what the model actually sees\n"
    "\n"
    "This is the pivot from ReAct. Each tool is declared with a **JSON schema** — a name, a description, "
    "and *typed* parameters with a `required` list. The registry pairs each schema with its real "
    "callable, so the declaration handed to the model and the function the runtime runs can never drift. "
    "Here is the real schema for the calculator."
)
add_code(
    "import json\n"
    "\n"
    "from function_calling_agent import TOOL_REGISTRY, tool_schemas\n"
    "\n"
    "print('registered tools:', list(TOOL_REGISTRY))\n"
    "print()\n"
    "print(json.dumps(TOOL_REGISTRY['calculator'].schema, indent=2))"
)

add_md(
    "## Step 5 — The schemas enter the prompt via `apply_chat_template(tools=...)`\n"
    "\n"
    "How does the model *know* these tools exist? The chat template has a `tools=` slot. When we render "
    "the messages with `apply_chat_template(messages, tools=tool_schemas(), ...)`, the tokenizer injects "
    "the JSON tool declarations into the exact system section the model was **trained** to read. Let's "
    "look at the rendered prompt so the mechanism is not magic."
)
add_code(
    "rendered = model.tokenizer.apply_chat_template(\n"
    "    [{'role': 'user', 'content': q}],\n"
    "    tools=tool_schemas(),\n"
    "    add_generation_prompt=True,\n"
    "    tokenize=False,\n"
    ")\n"
    "# show the part that declares the tools (the template wraps them in a <tools>...</tools> block)\n"
    "start = rendered.find('You are')\n"
    "print(rendered[start:start + 900])"
)

# ============================ 6. one structured call =============================================
add_md(
    "## Step 6 — A real structured `<tool_call>`: parse JSON, not prose\n"
    "\n"
    "Now the payoff. We generate *with* the tool schemas and the model emits a **structured** tool call "
    "wrapped in `<tool_call>...</tool_call>` — valid JSON with a `name` and an `arguments` object. "
    "`parse_tool_calls` pulls out the JSON and `json.loads` it. Compare this to Step 1: we are parsing a "
    "**data format** with an unambiguous grammar, not guessing at free text."
)
add_code(
    "from function_calling_agent import parse_tool_calls\n"
    "\n"
    "raw = model.generate([{'role': 'user', 'content': q}], tools=tool_schemas())\n"
    "print('--- raw model output ---')\n"
    "print(raw)\n"
    "print('\\n--- parsed structured calls ---')\n"
    "calls = parse_tool_calls(raw)\n"
    "for c in calls:\n"
    "    print(c)"
)

# ============================ 7. validation =====================================================
add_md(
    "## Step 7 — Validation: valid JSON is not the same as valid arguments\n"
    "\n"
    "Structured calling guarantees the *shape* of the arguments — never their *correctness*. The model "
    "can omit a required field, send the wrong type, or (famously) drop parentheses so "
    "`(1287 - 998) * 6` becomes `1287 - 998 * 6`. `validate_arguments` checks every call against its "
    "schema (required keys present, types coercible) and raises a real error on a violation — which the "
    "loop turns into a tool-result the model can read and fix."
)
add_code(
    "from function_calling_agent import ToolCall, validate_arguments\n"
    "\n"
    "good = ToolCall(name='calculator', arguments={'expression': '481 * 32 + 19'})\n"
    "print('valid   ->', validate_arguments(good, TOOL_REGISTRY['calculator']))\n"
    "\n"
    "missing = ToolCall(name='convert_units', arguments={'value': 42, 'from_unit': 'km'})  # no to_unit\n"
    "try:\n"
    "    validate_arguments(missing, TOOL_REGISTRY['convert_units'])\n"
    "except Exception as e:\n"
    "    print('invalid ->', e)"
)

# ============================ 8. dispatch =======================================================
add_md(
    "## Step 8 — Dispatch: validate, then run the real tool\n"
    "\n"
    "`dispatch` ties Steps 2–7 together: look the tool up in the registry, validate the arguments against "
    "its schema, call the real Python function, and return its real result string. Unknown tools and "
    "validation/tool errors become *result strings* (not crashes), so the agent can read the problem and "
    "recover — how a robust function-calling loop behaves in the wild."
)
add_code(
    "from function_calling_agent import dispatch\n"
    "\n"
    "print('good call   ->', dispatch(calls[0]))\n"
    "print('bad tool    ->', dispatch(ToolCall(name='translate', arguments={'text': 'hi'})))\n"
    "print('bad args    ->', dispatch(ToolCall(name='convert_units', arguments={'value': 5,\n"
    "                                          'from_unit': 'kg', 'to_unit': 'mi'})))"
)

# ============================ 9. the full loop, single ==========================================
add_md(
    "## Step 9 — The full protocol loop on a single-tool task\n"
    "\n"
    "Now assemble everything into the loop. `run_agent` builds the message list by the book: the user "
    "query, then for each turn an *assistant* message recording the structured `tool_calls`, then one "
    "**tool-role** result message per call, then the model is asked again — until it answers with no tool "
    "call (that reply is the answer) or the turn budget is hit. Here is the complete real trace for the "
    "query the text parser choked on in Step 1."
)
add_code(
    "from function_calling_agent import run_agent\n"
    "\n"
    "result = run_agent(model, q)\n"
    "print(result.transcript())\n"
    "print()\n"
    "print(f'stop_reason={result.stop_reason} | turns={result.num_turns} | '\n"
    "      f'tool_calls={result.num_tool_calls} | answer={result.answer!r}')"
)
add_md(
    "The model emitted a structured call, our runtime executed the real calculator, we handed the result "
    "back as a tool-role message, and the model produced a grounded final answer. That round-trip — "
    "**schema → call → execute → tool-result → answer** — is the entire function-calling protocol."
)

# ============================ 10. sequential multi-tool =========================================
add_md(
    "## Step 10 — A sequential multi-tool trace: look up, THEN compute\n"
    "\n"
    "The power shows on tasks that need one tool's result to drive the next. The agent looks up the "
    "exchange rate, **reads the real rate**, then grounds its answer on it. It has to decide *which* tool "
    "to call first based on the task, and *what to do* based on what the first tool returned."
)
add_code(
    "seq = run_agent(model, 'First look up the USD to JPY exchange rate with the get_exchange_rate tool, '\n"
    "                       'then multiply 40 dollars by that rate to get the yen amount.')\n"
    "print(seq.transcript())"
)
add_md(
    "The rate **157.0** came from the tool, not from the model's memory — and the final answer is built "
    "on that real observation. (Notice the model does the final multiply in prose here; a common, "
    "realistic behaviour. The lesson stands: the *fact* was grounded in a real tool result.)"
)

# ============================ 11. parallel calls ================================================
add_md(
    "## Step 11 — Parallel tool calls: two independent calls in ONE turn\n"
    "\n"
    "When two sub-tasks are *independent*, a capable model emits **both** tool calls in a single turn — "
    "and because `parse_tool_calls` returns a *list*, our loop dispatches both and returns a tool-result "
    "for each. This is a real efficiency win (no need to round-trip twice) and it is why structured "
    "calling models the tool_calls as a list, not a single call."
)
add_code(
    "par = run_agent(model, 'Convert 42 kilometres to miles, and separately convert 5 kilograms to pounds.')\n"
    "print(par.transcript())\n"
    "print()\n"
    "print(f'tool calls this run: {par.num_tool_calls} across {par.num_turns} turn(s)')"
)
add_md(
    "Two `convert_units` calls, one turn, two tool-result messages, one grounded answer. The model "
    "recognised the two conversions are independent and issued them together."
)

# ============================ 12. the comparison ================================================
add_md(
    "## Step 12 — Structured vs text: the reliability head-to-head\n"
    "\n"
    "Now the honest measurement behind this whole chapter. `compare_structured_vs_text` asks the model "
    "the **same** real queries two ways — the structured function-calling path (JSON, schema-validated) "
    "vs the ReAct-style text path from Step 1 (`TOOL: name(args)`, regex-parsed) — and scores how often "
    "each yields a **parseable, dispatchable** call. Every generation is greedy, so this reproduces "
    "exactly."
)
add_code(
    "from function_calling_agent import compare_structured_vs_text\n"
    "\n"
    "rows = compare_structured_vs_text(model)\n"
    "print(f\"{'structured':>10} {'text':>6} | query\")\n"
    "print('-' * 64)\n"
    "for r in rows:\n"
    "    print(f\"{'OK' if r.structured_ok else 'FAIL':>10} {'OK' if r.text_ok else 'FAIL':>6} \"\n"
    "          f'| {r.query[:44]}')"
)
add_code(
    "s_ok = sum(r.structured_ok for r in rows)\n"
    "t_ok = sum(r.text_ok for r in rows)\n"
    "print(f'structured (JSON, schema-validated): {s_ok}/{len(rows)} dispatchable '\n"
    "      f'({s_ok / len(rows):.0%})')\n"
    "print(f'text (TOOL: prose, regex-parsed)   : {t_ok}/{len(rows)} dispatchable '\n"
    "      f'({t_ok / len(rows):.0%})')"
)
add_md(
    "On this real set, structured calling yields a dispatchable call almost every time, while the "
    "text-parsing path fails on most queries — the model simply does not reliably format prose into the "
    "rigid `TOOL:` grammar the regex needs. **That gap is the value of structured function calling**, "
    "measured on real output. It is the same brittleness the ReAct chapter's parser fights, quantified."
)

# ============================ 13. tie to figures ================================================
add_md(
    "## Step 13 — The figures on the chapter page come from exactly this run\n"
    "\n"
    "Every figure in the chapter is generated from the same real module you just ran — no hand-typed "
    "numbers. The single-call and parallel trace figures are real solved traces (Steps 9 and 11); the "
    "reliability bars are the real comparison (Step 12). You can regenerate them yourself:\n"
    "\n"
    "```bash\n"
    "python \"../../tools/make_figures_03.py\"   # writes agentic03_*.png into ../../images/\n"
    "```\n"
    "\n"
    "That closes the loop between the page, the notebook, and the module: one real agent, demonstrated "
    "three ways, always in agreement."
)
add_code(
    "# Confirm the numbers behind the chapter's reliability figure, from THIS run:\n"
    "print('structured dispatchable :', sum(r.structured_ok for r in rows), '/', len(rows))\n"
    "print('text dispatchable       :', sum(r.text_ok for r in rows), '/', len(rows))\n"
    "print('parallel-trace tool calls:', par.num_tool_calls, 'in', par.num_turns, 'turn(s)')"
)

# ============================ Recap =============================================================
add_md(
    "## Recap — what you built\n"
    "\n"
    "You built a **real function-calling agent** end to end: three real JSON-schema tools, the "
    "schema-into-prompt mechanism, structured `<tool_call>` parsing, schema validation, the "
    "call → tool-result → answer protocol loop with real message roles, single / sequential / **parallel** "
    "multi-tool traces, and an honest structured-vs-text reliability comparison — all driving a genuine "
    "LLM with reproducible greedy decoding.\n"
    "\n"
    "The one idea to keep: **declaring a typed schema and parsing a structured call is more reliable than "
    "parsing free text.** ReAct proved a model can *act*; function calling makes that acting robust by "
    "moving the tool call from fuzzy prose into a data format the runtime can trust — and by validating "
    "the arguments before it ever runs a tool.\n"
    "\n"
    "Next: [Model Context Protocol (MCP)](../../08-Model-Context-Protocol-MCP/08-Model-Context-Protocol-MCP.md) "
    "(the open standard that lets *any* client discover and call *any* server's tools over the same "
    "schema-and-structured-call idea), and back to "
    "[ReAct](../../02-ReAct-Reason-and-Act/02-ReAct-Reason-and-Act.md) (the reasoning loop these tool "
    "calls slot into)."
)


def main() -> None:
    notebook = {
        "cells": CELLS,
        "metadata": {
            "kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
            "language_info": {"name": "python", "version": "3.12"},
        },
        "nbformat": 4,
        "nbformat_minor": 5,
    }
    NB_PATH.parent.mkdir(parents=True, exist_ok=True)
    NB_PATH.write_text(json.dumps(notebook, indent=1) + "\n", encoding="utf-8")
    print(f"wrote {NB_PATH} ({len(CELLS)} cells)")


if __name__ == "__main__":
    main()
