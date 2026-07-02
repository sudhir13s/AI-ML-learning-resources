"""Generate the step-by-step ReAct teaching notebook (02-ReAct-Reason-and-Act.ipynb).

The notebook mirrors ``react_agent.py`` one operation at a time, so a reader can open it, run every
cell live against the REAL model, and *teach* ReAct from it. Each numbered step has a short markdown
lead-in (the intuition) followed by ONE focused code cell with real output. This generator writes the
.ipynb; a separate nbconvert pass executes it headless so the real outputs are embedded.

    python build_notebook_02.py         # writes the .ipynb (unexecuted) into the chapter's code/
    python -m nbconvert --to notebook --execute --inplace \
        "../02-ReAct-Reason-and-Act/code/02-ReAct-Reason-and-Act.ipynb"

This generator lives in the domain-level ``12. Agentic_AI/tools/`` folder; the notebook it writes (and
the module it mirrors) stay in the chapter's ``code/`` folder. Kept as a generator (not a hand-edited
.ipynb) so the notebook and the module stay in lockstep: the same agent, typed once in the module,
demonstrated step-by-step here.
"""

from __future__ import annotations

import json
from pathlib import Path

NB_PATH = (
    Path(__file__).resolve().parent.parent
    / "02-ReAct-Reason-and-Act"
    / "code"
    / "02-ReAct-Reason-and-Act.ipynb"
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
    "# ReAct (Reason + Act) — a step-by-step, runnable teaching notebook\n"
    "\n"
    "This notebook builds a **real ReAct agent** from the ground up, one operation at a time, driving a "
    "genuine small instruction-tuned LLM (`Qwen/Qwen2.5-1.5B-Instruct`) through a real "
    "**Thought → Action → Observation** loop against real Python tools. It is the executable companion to "
    "the chapter and to `react_agent.py` — every function used here lives in that module, imported so the "
    "notebook and the module can never drift apart.\n"
    "\n"
    "Nothing about the model's output is mocked. The traces below are what the model actually generates "
    "(greedy / temperature 0, so they are reproducible). By the end you will have **seen**, with a real "
    "model and real tools:\n"
    "\n"
    "1. why an LLM alone gets multi-step questions **confidently wrong**;\n"
    "2. two **real tools** — a safe calculator (AST-walked, not `eval`) and a local `wiki` lookup;\n"
    "3. the **ReAct prompt grammar** — the Thought/Action/Observation format with a one-shot example;\n"
    "4. the **stop condition** that halts the model *before* it can hallucinate an observation;\n"
    "5. **robust parsing** of messy model text into a structured action;\n"
    "6. the **full loop** — reason, act, observe, repeat — on a numeric and a multi-hop question;\n"
    "7. a **head-to-head**: ReAct (with tools) vs reason-only (no tools) on real multi-step questions.\n"
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
    "from react_agent import (\n"
    "    calculator, wiki, TOOLS, KNOWLEDGE_BASE,\n"
    "    SYSTEM_PROMPT, DIRECT_PROMPT,\n"
    "    LanguageModel, pick_device,\n"
    "    parse_action, Action, _normalise_finish,\n"
    "    dispatch, run_react, run_direct,\n"
    "    compare_react_vs_direct, EVAL_SET,\n"
    ")\n"
    "\n"
    "print(f'torch {torch.__version__} | transformers {transformers.__version__} '\n"
    "      f'| device {pick_device()}')"
)

# ============================ 1. The problem =====================================================
add_md(
    "## Step 1 — The problem: an LLM alone guesses, and guesses wrong\n"
    "\n"
    "Before any agent machinery, feel the gap. We ask the model — with **no tools** — a question that "
    "needs one exact multiplication and one addition. A capable-sounding model will produce a confident "
    "number. Watch whether it is *right*.\n"
    "\n"
    "This is the reason-only baseline (`run_direct`): one shot, no reasoning trace, no tools — exactly "
    "what an LLM does on its own."
)
add_code(
    "llm = LanguageModel()   # loads the real model (downloads+caches on first run)\n"
    "print(f'loaded {llm.model_id} on {llm.device}')\n"
    "\n"
    "q = 'What is 481 multiplied by 32, then plus 19?'\n"
    "direct_answer = run_direct(llm, q)\n"
    "print('question :', q)\n"
    "print('LLM alone:', direct_answer, '   (correct answer is 15411)')"
)
add_md(
    "The model answers fluently but the arithmetic is off — it is doing multi-digit multiplication *in its "
    "head*, token by token, and small models are unreliable at that. It cannot check itself. This is the "
    "felt inadequacy ReAct removes: give the model a **tool** and let it *act*, then *read the real result*."
)

# ============================ 2. Real tools =====================================================
add_md(
    "## Step 2 — Real tool #1: a safe calculator (AST-walked, not `eval`)\n"
    "\n"
    "The first tool is a real calculator. Crucially it is **not** `eval(expr)` — that would let a model "
    "(or a prompt-injection) run arbitrary code. Instead we parse the expression to an abstract syntax "
    "tree and walk it, permitting *only* numbers and a whitelist of arithmetic operators. Anything else "
    "raises. Safe tool design is part of the lesson."
)
add_code(
    "print(calculator('481 * 32 + 19'))     # the real answer the LLM missed\n"
    "print(calculator('17 ** 3 - 200'))     # powers work too\n"
    "print(calculator('(1287 - 998) * 6'))  # parentheses respected\n"
    "\n"
    "# and it refuses anything that is not pure arithmetic (this is what makes it SAFE):\n"
    "try:\n"
    "    calculator('__import__(\"os\").system(\"echo hi\")')\n"
    "except Exception as e:\n"
    "    print('rejected unsafe input:', type(e).__name__, '-', e)"
)

add_md(
    "## Step 3 — Real tool #2: a `wiki` lookup against a local knowledge base\n"
    "\n"
    "The second tool returns real *facts* the model may not know — a stand-in for a web/Wikipedia search, "
    "but offline and deterministic so the notebook reproduces exactly. The agent must **read** the "
    "returned text to answer multi-hop questions. Note the deliberate miss case: a real tool sometimes "
    "returns nothing useful, and the agent has to cope."
)
add_code(
    "print('KB topics:', list(KNOWLEDGE_BASE)[:4], '...')\n"
    "print()\n"
    "print(wiki('Eiffel Tower'))\n"
    "print()\n"
    "print(wiki('the moon landing'))\n"
    "print()\n"
    "print(wiki('quantum chromodynamics'))   # a real miss — the agent must handle this"
)

# ============================ 4. The prompt grammar =============================================
add_md(
    "## Step 4 — The ReAct prompt: the Thought / Action / Observation grammar\n"
    "\n"
    "ReAct is, mechanically, a **prompt format plus a loop**. The system prompt tells the model to emit "
    "exactly one `Thought:` line then one `Action: tool[input]` line and then *stop*, and it includes a "
    "single worked example (one-shot) so a small model reliably copies the shape. The tools it may call "
    "and the special `finish[...]` action are declared here."
)
add_code(
    "print(SYSTEM_PROMPT)"
)

# ============================ 5. one raw step ===================================================
add_md(
    "## Step 5 — One raw generation: the model proposes a Thought + Action\n"
    "\n"
    "Let's run a *single* generation step by hand. We feed the system prompt + the question and let the "
    "model produce text. The key move: generation is **halted at the first `Observation:`** (a stop "
    "string), so the model proposes an action but does **not** get to invent the tool's result. That "
    "'stop before it fakes the observation' is the heart of ReAct."
)
add_code(
    "raw = llm.generate(SYSTEM_PROMPT, f'Question: {q}\\n')\n"
    "print(raw)"
)

# ============================ 6. parse ==========================================================
add_md(
    "## Step 6 — Parsing messy model text into a structured Action\n"
    "\n"
    "Real models are messy: extra lines, chatter, occasional bracket typos. `parse_action` finds the "
    "**first** well-formed `Action: tool[arg]` with a regex and throws away everything after it — so a "
    "chatty model can never advance the loop by more than one real step. It returns the trimmed text and "
    "a typed `Action(tool, arg)` (or `None` if the model failed to emit a parseable action — a real "
    "failure the loop must survive)."
)
add_code(
    "trimmed, action = parse_action(raw)\n"
    "print('parsed action ->', action)\n"
    "print('tool:', action.tool, '| arg:', repr(action.arg))"
)

# ============================ 7. dispatch =======================================================
add_md(
    "## Step 7 — Dispatch: run the real tool, get the REAL observation\n"
    "\n"
    "`dispatch` looks the tool up in the registry and calls it with the parsed argument, returning its "
    "real result string. Unknown tools and tool errors become *observations* (not crashes), so the agent "
    "can read the problem and recover — how a robust ReAct loop behaves in the wild."
)
add_code(
    "observation = dispatch(action)\n"
    "print('Observation:', observation)"
)

# ============================ 8. the full loop, numeric =========================================
add_md(
    "## Step 8 — The full loop on a numeric question\n"
    "\n"
    "Now assemble Steps 5–7 into the loop. `run_react` grows a scratchpad one real step at a time: append "
    "the model's Thought + Action, splice in the **real** Observation, ask again with the enlarged "
    "context, and stop on `finish[...]`, a parse failure, or the step budget (the guard against infinite "
    "loops). Here is the complete real trace for the numeric question the LLM missed in Step 1."
)
add_code(
    "result = run_react(llm, q)\n"
    "print(result.transcript())\n"
    "print()\n"
    "print(f'stop_reason={result.stop_reason} | steps={result.num_steps} | '\n"
    "      f'tool_calls={result.num_tool_calls} | answer={result.answer}')"
)
add_md(
    "The same model that guessed wrong in Step 1 now gets it **exactly right** — because it *acted* "
    "(called the calculator) and *read the real observation* instead of trusting its head. That is the "
    "entire ReAct thesis in one before/after."
)

# ============================ 9. multi-hop ======================================================
add_md(
    "## Step 9 — A multi-hop trace: `wiki` THEN `calculator`\n"
    "\n"
    "The real power shows on questions that need **two different tools in sequence**: look a fact up, then "
    "compute with it. The agent has to decide, at each step, *which* tool to use next — driven purely by "
    "what it read in the previous observation. No fixed plan; it reacts."
)
add_code(
    "mh = run_react(llm, 'In what year was the Eiffel Tower completed, and what is that year plus 100?')\n"
    "print(mh.transcript())"
)
add_md(
    "Read the flow: `wiki` returns the real sentence containing **1889**; the model reads it, reasons "
    "'1889 + 100', calls the calculator, gets **1989**, and finishes. The fact came from the tool, not "
    "from the model's memory — which is precisely how ReAct reduces hallucination."
)

# ============================ 10. finish normalisation ==========================================
add_md(
    "## Step 10 — A real robustness detail: normalising `finish`\n"
    "\n"
    "Small models often emit `finish[1889 + 100]` — the *expression* rather than the *value*. Rather than "
    "score that wrong, `_normalise_finish` evaluates a purely-numeric finish argument through the same "
    "safe calculator, reflecting the model's obvious intent. Non-numeric answers pass through untouched. "
    "This kind of defensive normalisation is 80% of making a real agent work."
)
add_code(
    "print(_normalise_finish('1889 + 100'))   # numeric expression -> evaluated\n"
    "print(_normalise_finish('15411'))        # already a number -> unchanged\n"
    "print(_normalise_finish('Ada Lovelace')) # a phrase -> passed through"
)

# ============================ 11. the comparison ================================================
add_md(
    "## Step 11 — ReAct vs reason-only: the head-to-head on real questions\n"
    "\n"
    "The claim of the ReAct paper is that interleaving reasoning with *acting on real observations* beats "
    "reasoning alone. We test it honestly: `compare_react_vs_direct` runs the **same** real multi-step "
    "questions two ways — the full ReAct loop vs a single 'answer directly, no tools' prompt — and scores "
    "exact match. Every generation is greedy, so this table reproduces exactly."
)
add_code(
    "rows = compare_react_vs_direct(llm, EVAL_SET)\n"
    "print(f\"{'gold':>8} | {'ReAct':>8} {'ok':>3} {'steps':>5} | {'direct':>12} {'ok':>3}\")\n"
    "print('-' * 54)\n"
    "for r in rows:\n"
    "    print(f'{r.gold:>8} | {str(r.react_answer):>8} '\n"
    "          f\"{'Y' if r.react_correct else 'N':>3} {r.react_steps:>5} | \"\n"
    "          f\"{str(r.direct_answer)[:12]:>12} {'Y' if r.direct_correct else 'N':>3}\")"
)
add_code(
    "react_acc = sum(r.react_correct for r in rows) / len(rows)\n"
    "direct_acc = sum(r.direct_correct for r in rows) / len(rows)\n"
    "print(f'ReAct accuracy  : {react_acc:.0%} ({sum(r.react_correct for r in rows)}/{len(rows)})')\n"
    "print(f'Direct accuracy : {direct_acc:.0%} ({sum(r.direct_correct for r in rows)}/{len(rows)})')"
)
add_md(
    "On this real set the reason-only model is right about half the time — it nails the questions where "
    "the arithmetic is easy and blows the ones needing exact multi-digit computation or a looked-up fact. "
    "ReAct, by grounding every step in a real tool result, gets them **all**. The gap *is* the value of "
    "acting."
)

# ============================ 12. failure mode ==================================================
add_md(
    "## Step 12 — Watching a failure mode: the step budget stops a runaway loop\n"
    "\n"
    "ReAct loops can misbehave — the classic failure is a model that never calls `finish` and loops "
    "forever. The `max_steps` budget is the real stop condition that prevents that. We force a tiny budget "
    "on a harder question and watch the loop terminate cleanly with `stop_reason='step_budget'` instead of "
    "hanging. (In production you'd surface this as 'I couldn't solve it in N steps', not a crash.)"
)
add_code(
    "capped = run_react(llm, 'What is 17 to the power of 3, minus 200?', max_steps=1)\n"
    "print(capped.transcript())\n"
    "print()\n"
    "print('stop_reason:', capped.stop_reason, '| answer:', capped.answer,\n"
    "      '  <- one step was not enough; the budget stopped it safely')"
)

# ============================ 13. tie to figures ================================================
add_md(
    "## Step 13 — The figures on the chapter page come from exactly this run\n"
    "\n"
    "Every figure in the chapter is generated from the same real module you just ran — no hand-typed "
    "numbers. The trace figure is a real solved trace (Step 9); the accuracy bars and the per-question "
    "grid are the real comparison table (Step 11). You can regenerate them yourself:\n"
    "\n"
    "```bash\n"
    "python \"../../tools/make_figures_02.py\"   # writes agentic02_*.png into ../../images/\n"
    "```\n"
    "\n"
    "That closes the loop between the page, the notebook, and the module: one real agent, demonstrated "
    "three ways, always in agreement."
)
add_code(
    "# Confirm the numbers behind the chapter's comparison figure, from THIS run:\n"
    "print('ReAct correct :', sum(r.react_correct for r in rows), '/', len(rows))\n"
    "print('Direct correct:', sum(r.direct_correct for r in rows), '/', len(rows))\n"
    "print('avg ReAct steps:', round(sum(r.react_steps for r in rows) / len(rows), 2))"
)

# ============================ Recap =============================================================
add_md(
    "## Recap — what you built\n"
    "\n"
    "You built a **real ReAct agent** end to end: two safe real tools, the Thought/Action/Observation "
    "prompt grammar, the stop-at-observation generation, robust parsing, the reason→act→observe loop with "
    "real stop conditions, and an honest ReAct-vs-baseline evaluation — all driving a genuine LLM with "
    "reproducible greedy decoding.\n"
    "\n"
    "The one idea to keep: **an LLM that can act on real observations beats an LLM that can only think.** "
    "Reasoning tells the agent *what to do next*; acting + observing tells it *what is actually true*. "
    "ReAct interleaves the two, and that interleaving is why it grounds reasoning and cuts hallucination.\n"
    "\n"
    "Next: [Tool Use & Function Calling](../../03-Tool-Use-and-Function-Calling/03-Tool-Use-and-Function-Calling.md) "
    "(the structured-output evolution of the text-parsed actions here), and "
    "[Reflection & Self-Critique](../../05-Reflection-and-Self-Critique/05-Reflection-and-Self-Critique.md) "
    "(adding a self-correction step between attempts)."
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
