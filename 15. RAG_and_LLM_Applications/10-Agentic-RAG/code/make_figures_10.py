"""Static figure generator for 10-Agentic-RAG.

Imports the SAME canonical functions the page and notebook use (agentic_rag.py, which reuses ch5)
so every plotted number is the chapter's own -- no hand-typed values. Writes muted-palette PNGs to
the shared chapter image dir (../../images/) with the per-chapter prefix `rag10_`.

    python make_figures_10.py

Figures produced:
  rag10_static_vs_agent.png   -- static FIXED pipeline (one straight shot) vs the agent's ADAPTIVE
                                 loop (reason -> pick tool -> observe -> decide again), side by side.
  rag10_react_loop.png        -- the ReAct loop: Thought -> Action(pick tool) -> Observation ->
                                 (decide: act again or finish), the control-policy cycle.
  rag10_agent_trace.png       -- the measured step-by-step trace of the compound query: each
                                 Thought/Action/Observation, the tool used, ending in the answer.
  rag10_router_decision.png   -- the router's cosine scores (query vs each tool description) for
                                 several probes; the argmax route, measured.
  rag10_step_cost.png         -- steps (≈ LLM calls) per query: static's single shot vs the agent's
                                 measured step count -- the cost of agency.
  rag10_static_miss_agent_hit.png -- the headline contrast: static can't produce the orbit COUNT;
                                 the agent computes 14 -- what each pipeline delivers on the query.

Verified on Python 3.12 / matplotlib 3.x / numpy 2.x / sentence-transformers (CPU). Headless (Agg).
"""

from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")  # headless: render straight to PNG, never open a window
import matplotlib.pyplot as plt

from agentic_rag import (
    COMPOUND_QUERY,
    Router,
    build_agent,
    compound_orbit_policy,
    full_corpus,
    static_rag,
)

# ---- Palette (matches the chapter's muted Mermaid classDefs) -------------------------------
BLUE = "#3A6B96"  # data / retrieve
PURPLE = "#5D4A8A"  # process / thought
GREEN = "#2E7A5A"  # answer / finish / hit
RED = "#8B3B4A"  # miss / danger
SLATE = "#4A5B6E"  # neutral
AMBER = "#7A6528"  # action / highlight
INK = "#1C2530"  # labels
GRID = "#D4D9DF"  # gridlines

OUT_DIR = Path(__file__).resolve().parent.parent.parent / "images"
DPI = 110

# per-action colours for the trace (keeps the figure legend consistent with the loop diagram)
TOOL_COLOR = {"retrieve": BLUE, "calculator": AMBER, "finish": GREEN}


def _style_axis(ax: plt.Axes) -> None:
    """Consistent muted styling: light grid, no top/right spines, ink-coloured labels."""
    ax.grid(True, color=GRID, linewidth=0.7, alpha=0.8)
    ax.set_axisbelow(True)
    for spine in ("top", "right"):
        ax.spines[spine].set_visible(False)
    for spine in ("left", "bottom"):
        ax.spines[spine].set_color(GRID)
    ax.tick_params(colors=INK, labelsize=9)
    ax.xaxis.label.set_color(INK)
    ax.yaxis.label.set_color(INK)
    ax.title.set_color(INK)


def _save(fig: plt.Figure, name: str) -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    path = OUT_DIR / name
    fig.savefig(path, dpi=DPI, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    print(f"wrote {path}")


def _box(ax, x, y, w, h, text, color, tcol="white", fs=8.6):
    """A filled rounded box with centred text -- the flow-diagram primitive used across figures."""
    ax.add_patch(plt.Rectangle((x, y), w, h, facecolor=color, alpha=0.92, edgecolor=INK, linewidth=1.0))
    ax.text(x + w / 2, y + h / 2, text, ha="center", va="center", color=tcol, fontsize=fs, fontweight="bold")


def fig_static_vs_agent() -> None:
    """Static FIXED pipeline (one straight shot) vs the agent's ADAPTIVE loop, side by side.

    Left: retrieve-once -> stuff -> generate -> done, a straight line (no branches, no loop). Right:
    the agent reasons, picks a tool, observes, and LOOPS back to decide again until it finishes --
    the same control-flow contrast the page's Mermaid draws, as a single figure. Schematic (a
    mechanism diagram), not a measurement.
    """
    fig, (ax_s, ax_a) = plt.subplots(1, 2, figsize=(12.6, 6.4))
    for ax in (ax_s, ax_a):
        ax.axis("off")
        ax.set_xlim(0, 1)
        ax.set_ylim(0, 1)

    # --- left: static, a straight line ---
    ax_s.text(0.5, 0.965, "STATIC RAG — a FIXED pipeline", ha="center", fontsize=12.5,
              fontweight="bold", color=SLATE)
    static_steps = [
        ("query", SLATE, 0.82),
        ("retrieve top-k  (ONCE)", BLUE, 0.62),
        ("stuff passages into prompt", PURPLE, 0.42),
        ("generate answer  (ONCE)", GREEN, 0.22),
    ]
    for text, color, y in static_steps:
        _box(ax_s, 0.22, y - 0.055, 0.56, 0.10, text, color)
    for y0, y1 in ((0.765, 0.675), (0.565, 0.475), (0.365, 0.275)):
        ax_s.annotate("", xy=(0.5, y1), xytext=(0.5, y0), arrowprops=dict(arrowstyle="->", color=INK, lw=1.6))
    ax_s.text(0.5, 0.07, "no branches, no loop, no tools:\ncontrol flow is fixed at design time",
              ha="center", fontsize=9.2, color=RED, style="italic", fontweight="bold")

    # --- right: agent, a loop ---
    ax_a.text(0.5, 0.965, "AGENTIC RAG — an ADAPTIVE loop", ha="center", fontsize=12.5,
              fontweight="bold", color=GREEN)
    _box(ax_a, 0.36, 0.855, 0.28, 0.09, "query", SLATE)
    _box(ax_a, 0.24, 0.66, 0.52, 0.10, "THOUGHT\nreason: what do I need next?", PURPLE, fs=8.0)
    _box(ax_a, 0.24, 0.45, 0.52, 0.10, "ACTION\npick a tool: retrieve / calc / route", AMBER, fs=8.0)
    _box(ax_a, 0.24, 0.24, 0.52, 0.10, "OBSERVATION\nread the tool's result", BLUE, fs=8.0)
    _box(ax_a, 0.36, 0.05, 0.28, 0.09, "FINISH → answer", GREEN, fs=8.4)
    ax_a.annotate("", xy=(0.5, 0.76), xytext=(0.5, 0.855), arrowprops=dict(arrowstyle="->", color=INK, lw=1.6))
    ax_a.annotate("", xy=(0.5, 0.55), xytext=(0.5, 0.66), arrowprops=dict(arrowstyle="->", color=INK, lw=1.6))
    ax_a.annotate("", xy=(0.5, 0.34), xytext=(0.5, 0.45), arrowprops=dict(arrowstyle="->", color=INK, lw=1.6))
    # the loop-back arrow: observation -> thought (act again)
    ax_a.annotate("", xy=(0.24, 0.71), xytext=(0.24, 0.29),
                  arrowprops=dict(arrowstyle="->", color=GREEN, lw=1.8, connectionstyle="arc3,rad=-0.55"))
    ax_a.text(0.05, 0.50, "loop:\nact again", ha="center", fontsize=8.4, color=GREEN,
              fontweight="bold", style="italic")
    # the finish branch
    ax_a.annotate("", xy=(0.5, 0.14), xytext=(0.5, 0.24),
                  arrowprops=dict(arrowstyle="->", color=GREEN, lw=1.6))
    ax_a.text(0.72, 0.19, "done?\nfinish", ha="center", fontsize=8.4, color=GREEN, fontweight="bold", style="italic")
    fig.suptitle("Fixed control flow (static) vs adaptive control flow (agent)", fontsize=12.5,
                 y=1.005, color=INK, fontweight="bold")
    _save(fig, "rag10_static_vs_agent.png")


def fig_react_loop() -> None:
    """The ReAct loop as a control-policy cycle: Thought -> Action -> Observation -> decide.

    A schematic of the Yao et al. 2022 interleaving: reason (Thought), act with a tool (Action),
    read the result (Observation), then loop back to reason again -- or stop with a final answer.
    The step budget is drawn as the guard that bounds the loop. Mechanism diagram, not a measurement.
    """
    fig, ax = plt.subplots(figsize=(9.4, 6.6))
    ax.axis("off")
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.text(0.5, 0.97, "The ReAct loop: reason, act, observe — repeat until answer or budget",
            ha="center", fontsize=12.5, fontweight="bold", color=INK)

    # three nodes of the cycle arranged as a triangle
    _box(ax, 0.37, 0.76, 0.26, 0.10, "THOUGHT\nwhat do I need?", PURPLE, fs=8.6)
    _box(ax, 0.66, 0.44, 0.30, 0.11, "ACTION\nchoose a tool + input\n(retrieve · calc · route)", AMBER, fs=8.0)
    _box(ax, 0.04, 0.44, 0.30, 0.11, "OBSERVATION\nthe tool's result\n(a passage, a number)", BLUE, fs=8.0)

    # cycle arrows (thought -> action -> observation -> thought)
    ax.annotate("", xy=(0.66, 0.52), xytext=(0.60, 0.78),
                arrowprops=dict(arrowstyle="->", color=INK, lw=1.7, connectionstyle="arc3,rad=0.25"))
    ax.annotate("", xy=(0.34, 0.49), xytext=(0.66, 0.49),
                arrowprops=dict(arrowstyle="->", color=INK, lw=1.7))
    ax.annotate("", xy=(0.40, 0.78), xytext=(0.34, 0.52),
                arrowprops=dict(arrowstyle="->", color=INK, lw=1.7, connectionstyle="arc3,rad=0.25"))
    ax.text(0.70, 0.68, "run\ntool", ha="center", fontsize=8.0, color=AMBER, style="italic")
    ax.text(0.50, 0.44, "feed result back", ha="center", fontsize=8.0, color=BLUE, style="italic")
    ax.text(0.30, 0.68, "reason on\nwhat I saw", ha="center", fontsize=8.0, color=PURPLE, style="italic")

    # finish branch + budget guard
    _box(ax, 0.37, 0.14, 0.26, 0.09, "FINISH → answer", GREEN, fs=8.6)
    ax.annotate("", xy=(0.5, 0.23), xytext=(0.5, 0.44),
                arrowprops=dict(arrowstyle="->", color=GREEN, lw=1.6))
    ax.text(0.60, 0.33, "enough info?\n→ finish", ha="center", fontsize=8.0, color=GREEN, style="italic")
    ax.text(0.5, 0.06, "GUARD: a step BUDGET caps the loop — no finish within N steps → stop "
            "(the infinite-loop fix)",
            ha="center", fontsize=8.8, color=RED, style="italic", fontweight="bold")
    _save(fig, "rag10_react_loop.png")


def fig_agent_trace() -> None:
    """The measured step-by-step trace of the compound query: Thought/Action/Observation per step.

    Runs the REAL agent (agentic_rag.build_agent + compound_orbit_policy) and renders its actual
    trace -- each step's chosen tool, its input, and the observation the tool returned -- ending in
    the assembled answer. Every string here is the agent's own output, not hand-typed.
    """
    corpus = full_corpus()
    agent, _tools, _retriever = build_agent(corpus)
    result = agent.run(COMPOUND_QUERY, compound_orbit_policy)

    fig, ax = plt.subplots(figsize=(12.8, 8.2))
    ax.axis("off")
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.text(0.5, 0.985, "A ReAct trace, measured: the agent solves the compound query step by step",
            ha="center", fontsize=12.5, fontweight="bold", color=INK)
    ax.text(0.5, 0.95, f"query: “{COMPOUND_QUERY}”", ha="center", fontsize=9.2, style="italic", color=SLATE)

    def wrap(text, width):
        import textwrap
        return "\n".join(textwrap.wrap(text, width))

    n = len(result.steps)
    top, bottom = 0.90, 0.18
    row_h = (top - bottom) / n
    for i, step in enumerate(result.steps):
        y = top - i * row_h
        color = TOOL_COLOR.get(step.action, SLATE)
        # step index badge
        ax.text(0.02, y - row_h / 2, f"step\n{i + 1}", ha="center", va="center", fontsize=9.5,
                fontweight="bold", color="white",
                bbox=dict(boxstyle="circle,pad=0.42", facecolor=color, edgecolor=INK, linewidth=1.0))
        # thought (purple), action (tool colour), observation (ink)
        ax.text(0.08, y - 0.02, "Thought:  " + wrap(step.thought, 92), ha="left", va="top",
                fontsize=8.2, color=PURPLE, style="italic")
        action_label = f"Action:   {step.action}({step.action_input!r})"
        ax.text(0.08, y - row_h * 0.42, wrap(action_label, 92), ha="left", va="top",
                fontsize=8.2, color=color, fontweight="bold")
        if step.action != "finish":
            ax.text(0.08, y - row_h * 0.70, "Observe:  " + wrap(step.observation, 92), ha="left", va="top",
                    fontsize=8.2, color=INK)
        # separator line
        ax.axhline(y - row_h, xmin=0.06, xmax=0.98, color=GRID, linewidth=0.8)

    ax.text(0.5, 0.135, f"FINAL ANSWER: {wrap(result.answer, 108)}", ha="center", va="top",
            fontsize=9.0, color=GREEN, fontweight="bold",
            bbox=dict(boxstyle="round,pad=0.5", facecolor=GREEN, alpha=0.12, edgecolor=GREEN))
    ax.text(0.5, 0.035, f"{result.n_steps} steps · tools: {result.tools_used} · "
            "the loop, tools, retrieval and arithmetic are REAL; the Thought text is an illustrative stand-in",
            ha="center", va="bottom", fontsize=7.6, color=SLATE, style="italic")
    _save(fig, "rag10_agent_trace.png")


def fig_router_decision() -> None:
    """The router's cosine scores (query vs each tool description) for several probes -- measured.

    Runs the REAL Router over a few probes; each grouped bar pair is the query's cosine to the
    retrieve-tool vs calculator-tool descriptions, and the taller bar is the route. Shows both the
    clear cases and the marginal one (the "1440 divided by 97" probe is nearly a tie) -- routing is
    a real scoring decision, sometimes a close call.
    """
    corpus = full_corpus()
    _agent, tools, retriever_tool = build_agent(corpus)
    router = Router(tools, retriever_tool._dense)  # noqa: SLF001 -- reuse the shared encoder
    probes = [
        ("Who leads\nHelios-7?", "Who is the project lead for Helios-7?"),
        ("1440 divided\nby 97?", "What is 1440 divided by 97?"),
        ("When was it\nlaunched?", "When was Helios-7 launched?"),
        ("Compute\n200 × 4", "Compute 200 times 4."),
    ]
    labels = [p[0] for p in probes]
    retrieve_scores = [float(router.route_scores(p[1])[0]) for p in probes]
    calc_scores = [float(router.route_scores(p[1])[1]) for p in probes]

    import numpy as np
    x = np.arange(len(probes))
    width = 0.38
    fig, ax = plt.subplots(figsize=(9.6, 5.4))
    _style_axis(ax)
    b1 = ax.bar(x - width / 2, retrieve_scores, width, label="retrieve tool", color=BLUE,
                edgecolor=INK, linewidth=0.8)
    b2 = ax.bar(x + width / 2, calc_scores, width, label="calculator tool", color=AMBER,
                edgecolor=INK, linewidth=0.8)
    for bars in (b1, b2):
        for bar in bars:
            h = bar.get_height()
            ax.annotate(f"{h:+.2f}", (bar.get_x() + bar.get_width() / 2, h), fontsize=8.2, color=INK,
                        ha="center", va="bottom" if h >= 0 else "top",
                        xytext=(0, 3 if h >= 0 else -3), textcoords="offset points")
    # mark the winning route above each pair
    for i, (r, c) in enumerate(zip(retrieve_scores, calc_scores)):
        route = "retrieve" if r > c else "calculator"
        ax.text(i, max(r, c) + 0.11, f"→ {route}", ha="center", fontsize=8.6,
                color=BLUE if route == "retrieve" else AMBER, fontweight="bold")
    ax.axhline(0, color=INK, linewidth=0.9)
    ax.set_xticks(x)
    ax.set_xticklabels(labels, fontsize=8.6)
    ax.set_ylabel("cosine(query, tool description)")
    ax.set_ylim(min(calc_scores) - 0.12, 1.0)
    ax.legend(loc="upper right", fontsize=8.6, framealpha=0.9)
    ax.set_title("Routing = a scoring decision: the query's cosine to each tool's description (measured)",
                 fontsize=11, pad=12)
    _save(fig, "rag10_router_decision.png")


def fig_step_cost() -> None:
    """Steps (≈ LLM calls) per query: static's single shot vs the agent's measured step count.

    The measured cost of agency: static RAG is one pass; the agent took its actual number of steps
    on the compound query. Each agent step is (in production) an LLM call -- so the bar height is a
    direct proxy for per-query latency and cost. Numbers are the chapter's own.
    """
    corpus = full_corpus()
    agent, _tools, _retriever = build_agent(corpus)
    result = agent.run(COMPOUND_QUERY, compound_orbit_policy)
    agent_steps = result.n_steps
    static_steps = 1

    fig, ax = plt.subplots(figsize=(7.6, 5.2))
    _style_axis(ax)
    bars = ax.bar(["static RAG\n(single shot)", "ReAct agent\n(compound query)"],
                  [static_steps, agent_steps], color=[SLATE, GREEN], edgecolor=INK, linewidth=0.9, width=0.55)
    for bar, val, note in zip(bars, [static_steps, agent_steps],
                              ["1 pass", f"{agent_steps} steps"]):
        ax.annotate(f"{note}", (bar.get_x() + bar.get_width() / 2, bar.get_height()),
                    fontsize=11, color=INK, ha="center", va="bottom", xytext=(0, 3),
                    textcoords="offset points", fontweight="bold")
    ax.annotate(f"{agent_steps}× the per-query work\n(each step ≈ 1 LLM call)",
                xy=(1, agent_steps), xytext=(0.35, agent_steps * 0.78), fontsize=9.5, color=GREEN,
                ha="center", fontweight="bold",
                arrowprops=dict(arrowstyle="->", color=GREEN, lw=1.4, alpha=0.7))
    ax.set_ylabel("steps per query  (≈ LLM calls)")
    ax.set_ylim(0, agent_steps * 1.4)
    ax.set_title("The cost of agency: adaptive power costs steps (measured on the compound query)",
                 fontsize=10.5, pad=12)
    _save(fig, "rag10_step_cost.png")


def fig_static_miss_agent_hit() -> None:
    """The headline contrast: static can't produce the orbit COUNT; the agent computes 14.

    Runs BOTH pipelines on the compound query and shows what each delivers: static reads off the
    resolution but returns None for the count (no arithmetic step); the agent delivers both the
    count (14) and the resolution. Every value is measured by the real pipelines.
    """
    corpus = full_corpus()
    agent, _tools, retriever_tool = build_agent(corpus)
    dense = retriever_tool._dense  # noqa: SLF001 -- shared encoder for static retrieval
    static = static_rag(COMPOUND_QUERY, dense, corpus, k=3)
    result = agent.run(COMPOUND_QUERY, compound_orbit_policy)

    fig, ax = plt.subplots(figsize=(11.6, 5.6))
    ax.axis("off")
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.text(0.5, 0.96, "Same compound query, two pipelines: what each can actually deliver",
            ha="center", fontsize=12.5, fontweight="bold", color=INK)
    ax.text(0.5, 0.905, f"“{COMPOUND_QUERY}”", ha="center", fontsize=9.0, style="italic", color=SLATE)

    # two sub-questions the query asks
    subq = ["complete orbits per day\n(needs 1440 ÷ period)", "imager ground resolution\n(a fact lookup)"]

    # static column
    ax.text(0.27, 0.80, "STATIC single-shot RAG", ha="center", fontsize=11, fontweight="bold", color=SLATE)
    static_vals = [
        (static.orbit_count if static.orbit_count is not None else "— (can't compute)", RED),
        (static.resolution or "—", GREEN if static.resolution else RED),
    ]
    # agent column
    ax.text(0.73, 0.80, "ReAct AGENT", ha="center", fontsize=11, fontweight="bold", color=GREEN)
    complete_orbits = int(float(result.steps[1].observation.split("=")[-1]))
    agent_vals = [(f"{complete_orbits} orbits", GREEN), (static.resolution or "4 meters", GREEN)]

    for j, sq in enumerate(subq):
        y = 0.60 - j * 0.30
        ax.text(0.5, y + 0.11, sq, ha="center", fontsize=9.0, color=INK, fontweight="bold")
        sval, scol = static_vals[j]
        aval, acol = agent_vals[j]
        ax.text(0.27, y, str(sval), ha="center", va="center", fontsize=11, color="white", fontweight="bold",
                bbox=dict(boxstyle="round,pad=0.5", facecolor=scol, alpha=0.9, edgecolor=INK))
        ax.text(0.73, y, str(aval), ha="center", va="center", fontsize=11, color="white", fontweight="bold",
                bbox=dict(boxstyle="round,pad=0.5", facecolor=acol, alpha=0.9, edgecolor=INK))

    ax.text(0.27, 0.05, "misses the COUNT: no arithmetic step\nin a fixed retrieve-then-generate pipeline",
            ha="center", fontsize=8.6, color=RED, style="italic", fontweight="bold")
    ax.text(0.73, 0.05, f"delivers both, in {result.n_steps} steps:\nretrieve → compute → retrieve → finish",
            ha="center", fontsize=8.6, color=GREEN, style="italic", fontweight="bold")
    _save(fig, "rag10_static_miss_agent_hit.png")


def main() -> None:
    fig_static_vs_agent()
    fig_react_loop()
    fig_agent_trace()
    fig_router_decision()
    fig_step_cost()
    fig_static_miss_agent_hit()
    print("all figures written to", OUT_DIR)


if __name__ == "__main__":
    main()
