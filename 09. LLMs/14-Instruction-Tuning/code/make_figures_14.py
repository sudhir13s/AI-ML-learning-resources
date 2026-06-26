"""Reproducible figure generator for 14-Instruction-Tuning.

Produces every embedded PNG for the chapter from the SAME numbers used in the page and the
teaching notebook -- so the figures cannot silently drift from the prose. Run:

    python make_figures_14.py

Each figure is written to ../images/ at 150 dpi. The palette matches the chapter's Mermaid
diagrams (muted, white text on coloured fills). Two kinds of numbers appear:
  * MEASURED -- the multitask-vs-single-task held-out accuracy bars come straight from
    instruction_tuning.py's deterministic CPU run (100.0% vs 0.7%).
  * ILLUSTRATIVE -- the FLAN-style "accuracy vs number of instruction clusters" and the
    "diversity vs raw count" curves are shaped to match the published FLAN trends but are not
    re-derived from a training run here; they are labelled "illustrative" on the figure.

Verified on Python 3.12 / numpy 2.x / matplotlib 3.x.
"""

from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")  # headless backend: write files, never open a window
import matplotlib.pyplot as plt
import numpy as np

# ---- Palette (matches the chapter Mermaid classDefs) --------------------------------
BLUE = "#3A6B96"
PURPLE = "#5D4A8A"
GREEN = "#2E7A5A"
RED = "#8B3B4A"
SLATE = "#4A5B6E"
AMBER = "#7A6528"
NAVY = "#2A5B80"
INK = "#1C2530"
GRID = "#D4D9DF"

# ---- Numbers from the demo (instruction_tuning.py, deterministic CPU run) ------------
MULTI_HELDOUT_ACC = 1.000  # multitask model, SUBSTITUTE with UNSEEN keys
SINGLE_HELDOUT_ACC = 0.007  # single-task model, same held-out test
MULTI_INDIST_ACC = 1.000  # both master the in-distribution fixed-key task
SINGLE_INDIST_ACC = 1.000

OUT_DIR = Path(__file__).resolve().parent.parent.parent / "images"
DPI = 150


def _style_axis(ax: plt.Axes) -> None:
    """Consistent muted styling: light grid, no top/right spines, ink-coloured labels."""
    ax.grid(True, color=GRID, linewidth=0.8, zorder=0)
    ax.set_axisbelow(True)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    for side in ("left", "bottom"):
        ax.spines[side].set_color(SLATE)
    ax.tick_params(colors=INK)
    ax.xaxis.label.set_color(INK)
    ax.yaxis.label.set_color(INK)
    ax.title.set_color(INK)


def _save(fig: plt.Figure, name: str) -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    path = OUT_DIR / name
    fig.savefig(path, dpi=DPI, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    print(f"wrote {path.name}")


# =====================================================================================
# Figure 1 -- the FLAN-style scaling curve: zero-shot accuracy vs number of task clusters
# =====================================================================================
def fig_task_scaling() -> None:
    """Held-out zero-shot accuracy RISES with the number of distinct instruction-task clusters.

    This is the central empirical law of instruction tuning (FLAN, Wei et al. 2021, Fig. 5):
    each additional cluster of task types added to the tuning mix raises performance on tasks
    held OUT of the mix. The curve is illustrative (FLAN's exact axis is held-out cluster
    accuracy); the SHAPE -- monotone rising, decelerating -- is the published result.
    """
    n_clusters = np.array([1, 2, 4, 7, 10, 20, 40, 60])
    # A saturating rise: big early gains, diminishing returns -- the FLAN Fig.5 shape.
    acc = 38 + 22 * (1 - np.exp(-n_clusters / 9.0))
    fig, ax = plt.subplots(figsize=(8.2, 4.8))
    _style_axis(ax)
    ax.plot(n_clusters, acc, "-o", color=BLUE, linewidth=2.2, markersize=7, zorder=3)
    ax.axhline(38, color=RED, linestyle="--", linewidth=1.4, zorder=2,
               label="zero-shot with 0 clusters (no instruction tuning)")
    ax.set_xlabel("number of instruction-task clusters in the tuning mix")
    ax.set_ylabel("zero-shot accuracy on HELD-OUT tasks (%)")
    ax.set_ylim(34, 64)
    ax.set_title("More instruction-task diversity -> better zero-shot on UNSEEN tasks (illustrative; FLAN Fig. 5 shape)",
                 fontweight="bold", fontsize=10)
    ax.legend(frameon=False, loc="lower right")
    fig.tight_layout()
    _save(fig, "it_task_scaling.png")


# =====================================================================================
# Figure 2 -- the demo payoff: multitask vs single-task held-out generalization (MEASURED)
# =====================================================================================
def fig_multitask_vs_single() -> None:
    """The chapter's from-scratch result: diverse multitask tuning transfers; narrow tuning does not.

    Both models hit 100% on the in-distribution fixed-key SUBSTITUTE (capacity is NOT the issue).
    On SUBSTITUTE with UNSEEN keys, the multitask model -- which had to LEARN to read the key --
    scores 100%, while the single-task model, which memorized one mapping, collapses to ~1%.
    Numbers are MEASURED from instruction_tuning.py.
    """
    fig, axes = plt.subplots(1, 2, figsize=(11.2, 4.6), sharey=True)
    labels = ["single-task\n(one fixed key)", "multitask\n(diverse mix)"]
    colors = [RED, GREEN]

    ax = axes[0]
    _style_axis(ax)
    vals = [SINGLE_INDIST_ACC * 100, MULTI_INDIST_ACC * 100]
    bars = ax.bar(labels, vals, color=[SLATE, SLATE], zorder=3)
    for bar, v in zip(bars, vals):
        ax.text(bar.get_x() + bar.get_width() / 2, v + 1.5, f"{v:.0f}%",
                ha="center", va="bottom", fontsize=11, color=INK, fontweight="bold")
    ax.set_ylim(0, 112)
    ax.set_ylabel("exact-match accuracy (%)")
    ax.set_title("In-distribution (fixed key)\nboth succeed -- capacity is not the issue", fontweight="bold", fontsize=10)

    ax = axes[1]
    _style_axis(ax)
    vals = [SINGLE_HELDOUT_ACC * 100, MULTI_HELDOUT_ACC * 100]
    bars = ax.bar(labels, vals, color=colors, zorder=3)
    for bar, v in zip(bars, vals):
        ax.text(bar.get_x() + bar.get_width() / 2, v + 1.5, f"{v:.1f}%",
                ha="center", va="bottom", fontsize=11, color=INK, fontweight="bold")
    ax.set_ylim(0, 112)
    ax.set_title("ZERO-SHOT (unseen keys)\ndiversity transfers; memorization fails", fontweight="bold", fontsize=10)

    fig.suptitle("From-scratch demo: a diverse instruction mix yields the META-SKILL that transfers (measured)",
                 fontweight="bold", color=INK)
    fig.tight_layout()
    _save(fig, "it_multitask_vs_single.png")


# =====================================================================================
# Figure 3 -- one task, many phrasings -> one unified instruction format
# =====================================================================================
def fig_instruction_template() -> None:
    """A schematic: several natural-language PHRASINGS of the same task map to one rendered format.

    Instruction tuning trains on MULTIPLE templates per task so the model binds behaviour to the
    instruction's meaning, not one surface string. Drawn as boxes + arrows (no axes).
    """
    fig, ax = plt.subplots(figsize=(10.6, 4.8))
    ax.axis("off")
    phrasings = [
        '"Translate to French: <x>"',
        '"What is <x> in French?"',
        '"Render the following in French: <x>"',
    ]
    for i, text in enumerate(phrasings):
        y = 0.78 - i * 0.28
        ax.add_patch(plt.Rectangle((0.02, y - 0.07), 0.40, 0.16, facecolor=BLUE, edgecolor="none", zorder=2))
        ax.text(0.22, y, text, ha="center", va="center", color="#fff", fontsize=10, zorder=3)
        ax.annotate("", xy=(0.58, 0.50), xytext=(0.43, y),
                    arrowprops=dict(arrowstyle="-|>", color=SLATE, lw=1.6))
    ax.add_patch(plt.Rectangle((0.58, 0.36), 0.40, 0.28, facecolor=PURPLE, edgecolor="none", zorder=2))
    ax.text(0.78, 0.56, "Unified instruction format", ha="center", va="center", color="#fff",
            fontsize=10, fontweight="bold", zorder=3)
    ax.text(0.78, 0.46, "[ instruction tokens | input | target ]", ha="center", va="center",
            color="#fff", fontsize=9, zorder=3)
    ax.text(0.78, 0.40, "loss masked to the target", ha="center", va="center",
            color="#E8E0F0", fontsize=8, style="italic", zorder=3)
    ax.text(0.22, 0.92, "Many phrasings (templates) of ONE task", ha="center", fontsize=11,
            fontweight="bold", color=INK)
    ax.text(0.78, 0.70, "One task, one format", ha="center", fontsize=11, fontweight="bold", color=INK)
    ax.set_xlim(0, 1)
    ax.set_ylim(0.2, 1.0)
    fig.suptitle("Instruction templates: multiple phrasings per task force the model to read MEANING, not a surface string",
                 fontweight="bold", color=INK, fontsize=10)
    fig.tight_layout()
    _save(fig, "it_instruction_template.png")


# =====================================================================================
# Figure 4 -- diversity beats raw count (illustrative)
# =====================================================================================
def fig_diversity_vs_count() -> None:
    """Two curves at equal example budgets: spreading examples across MORE task types wins.

    Holding the number of training examples fixed, zero-shot held-out accuracy is higher when
    those examples span MANY task types than when they are concentrated in few -- "diversity >
    raw count" (FLAN / FLAN-T5, Chung et al. 2022). Illustrative shape, labelled as such.
    """
    examples = np.array([1, 2, 4, 8, 16, 32, 64])  # thousands of examples (same budget axis)
    diverse = 40 + 18 * (1 - np.exp(-examples / 8.0))  # spread across many task types
    concentrated = 40 + 6 * (1 - np.exp(-examples / 8.0))  # piled into a few task types
    fig, ax = plt.subplots(figsize=(8.2, 4.8))
    _style_axis(ax)
    ax.plot(examples, diverse, "-o", color=GREEN, linewidth=2.2, markersize=7,
            label="examples spread across MANY task types", zorder=3)
    ax.plot(examples, concentrated, "-s", color=AMBER, linewidth=2.2, markersize=7,
            label="same budget, FEW task types", zorder=3)
    ax.set_xlabel("instruction-tuning examples (thousands, fixed budget)")
    ax.set_ylabel("zero-shot accuracy on HELD-OUT tasks (%)")
    ax.set_ylim(36, 62)
    ax.set_title("Task DIVERSITY beats raw example count at a fixed budget (illustrative)",
                 fontweight="bold", fontsize=10)
    ax.legend(frameon=False, loc="lower right")
    fig.tight_layout()
    _save(fig, "it_diversity_vs_count.png")


# =====================================================================================
# Figure 5 -- the FLAN instruction-task taxonomy map (clusters of task types)
# =====================================================================================
def fig_task_taxonomy() -> None:
    """A treemap-style map of instruction-tuning task CLUSTERS (the FLAN Collection landscape).

    Instruction-tuning corpora group many datasets into task-type clusters: NLI, reading
    comprehension, summarization, translation, sentiment, QA, commonsense, coreference,
    paraphrase, struct-to-text. The MIX of clusters -- not any single one -- is what drives
    held-out generalization. Areas are illustrative (relative cluster sizes, not exact counts).
    """
    clusters = [
        ("Reading\ncomprehension", 16, BLUE),
        ("Question\nanswering", 14, PURPLE),
        ("Natural language\ninference", 11, GREEN),
        ("Sentiment", 9, AMBER),
        ("Summarization", 9, NAVY),
        ("Translation", 8, RED),
        ("Commonsense", 8, SLATE),
        ("Paraphrase", 7, BLUE),
        ("Coreference", 6, PURPLE),
        ("Struct-to-text", 6, GREEN),
        ("Misc.", 6, SLATE),
    ]
    fig, ax = plt.subplots(figsize=(10.8, 5.2))
    ax.axis("off")
    # Simple row-packed treemap: place rectangles left-to-right, wrapping rows.
    x, y, row_h = 0.0, 0.0, 0.5
    total = sum(c[1] for c in clusters)
    width_unit = 1.0 / (total / 2)  # two rows
    row_budget = total / 2
    used = 0.0
    for name, size, color in clusters:
        w = size * width_unit
        if used + size > row_budget + 1e-6 and y == 0.0:
            x, y, used = 0.0, row_h, 0.0  # wrap to second row
        ax.add_patch(plt.Rectangle((x, y), w - 0.006, row_h - 0.012, facecolor=color,
                                   edgecolor="white", linewidth=2, zorder=2))
        ax.text(x + w / 2, y + row_h / 2, name, ha="center", va="center",
                color="#fff", fontsize=9, fontweight="bold", zorder=3)
        x += w
        used += size
    ax.set_xlim(0, 1.02)
    ax.set_ylim(0, 1.0)
    fig.suptitle("The instruction-tuning task landscape: many task-type clusters (FLAN Collection; areas illustrative)",
                 fontweight="bold", color=INK, fontsize=11)
    fig.tight_layout()
    _save(fig, "it_task_taxonomy.png")


# =====================================================================================
# Figure 6 -- chain-of-thought in the mix lifts reasoning (FLAN-T5, illustrative)
# =====================================================================================
def fig_cot_in_mix() -> None:
    """Adding chain-of-thought data to the instruction mix lifts held-out reasoning accuracy.

    FLAN-T5 (Chung et al. 2022) showed that including CoT (step-by-step) examples in the tuning
    mix improves reasoning benchmarks AND does not hurt non-reasoning tasks. Two grouped bars;
    illustrative magnitudes matching the reported direction.
    """
    groups = ["reasoning\nbenchmarks", "direct (non-CoT)\nbenchmarks"]
    without_cot = [42.0, 56.0]
    with_cot = [51.0, 57.0]
    x = np.arange(len(groups))
    width = 0.36
    fig, ax = plt.subplots(figsize=(8.4, 4.8))
    _style_axis(ax)
    b1 = ax.bar(x - width / 2, without_cot, width, color=AMBER, label="instruction mix WITHOUT CoT data", zorder=3)
    b2 = ax.bar(x + width / 2, with_cot, width, color=GREEN, label="instruction mix WITH CoT data", zorder=3)
    for bars in (b1, b2):
        for bar in bars:
            v = bar.get_height()
            ax.text(bar.get_x() + bar.get_width() / 2, v + 0.6, f"{v:.0f}", ha="center",
                    va="bottom", fontsize=9, color=INK, fontweight="bold")
    ax.set_xticks(x)
    ax.set_xticklabels(groups)
    ax.set_ylabel("held-out accuracy (%)")
    ax.set_ylim(0, 68)
    ax.set_title("Including chain-of-thought data lifts reasoning, leaves direct tasks intact (FLAN-T5; illustrative)",
                 fontweight="bold", fontsize=10)
    ax.legend(frameon=False, loc="upper left")
    fig.tight_layout()
    _save(fig, "it_cot_in_mix.png")


# =====================================================================================
# Figure 7 -- the diversity axis pushed hard over time (instruction-corpus size, log scale)
# =====================================================================================
def fig_dataset_scaling() -> None:
    """Horizontal log-scale bars: instruction corpora grew along the DIVERSITY axis over time.

    Colour = build method (academic-recast / synthetic / human). The x-axis deliberately MIXES
    two units -- some bars count TASKS (FLAN, Super-NI, FLAN Collection) and some count EXAMPLES
    (Alpaca, Dolly, LIMA) -- so it is labelled "illustrative" and each bar's unit is annotated.
    The takeaway is the visible RISE in the number/diversity of instruction tasks over 2021->2023,
    reinforcing Law 1 (more, more-diverse tasks -> better zero-shot).
    """
    ACADEMIC, SYNTHETIC, HUMAN = BLUE, PURPLE, GREEN
    # (label, value, unit, build-method colour), ordered small->large for a clean log ladder.
    rows = [
        ("LIMA (2023)", 1_000, "examples", HUMAN),
        ("Dolly-15k (2023)", 15_000, "examples", HUMAN),
        ("Alpaca (2023)", 52_000, "examples", SYNTHETIC),
        ("FLAN (2021)", 62, "datasets", ACADEMIC),
        ("Super-NaturalInstructions (2022)", 1_616, "tasks", ACADEMIC),
        ("FLAN Collection (2022/23)", 1_836, "tasks", ACADEMIC),
    ]
    # sort by value so the log ladder is monotone
    rows = sorted(rows, key=lambda r: r[1])
    labels = [f"{r[0]}  ({r[2]})" for r in rows]
    values = [r[1] for r in rows]
    colors = [r[3] for r in rows]
    y = np.arange(len(rows))

    fig, ax = plt.subplots(figsize=(10.8, 5.0))
    _style_axis(ax)
    bars = ax.barh(y, values, color=colors, zorder=3)
    ax.set_xscale("log")
    ax.set_yticks(y)
    ax.set_yticklabels(labels)
    ax.set_xlabel("instruction-corpus size (log scale; TASKS or EXAMPLES — see each bar)")
    ax.set_xlim(50, 200_000)
    for bar, v in zip(bars, values):
        ax.text(v * 1.12, bar.get_y() + bar.get_height() / 2, f"{v:,}",
                va="center", ha="left", fontsize=9, color=INK, fontweight="bold")
    # a small legend mapping colour -> build method
    handles = [
        plt.Rectangle((0, 0), 1, 1, color=ACADEMIC),
        plt.Rectangle((0, 0), 1, 1, color=SYNTHETIC),
        plt.Rectangle((0, 0), 1, 1, color=HUMAN),
    ]
    ax.legend(handles, ["academic-recast", "synthetic (model-generated)", "human-written"],
              frameon=False, loc="lower right", fontsize=9)
    ax.set_title("The diversity axis, pushed hard (2021->2023): instruction corpora grew fast (illustrative; mixed units)",
                 fontweight="bold", fontsize=10)
    fig.tight_layout()
    _save(fig, "it_dataset_scaling.png")


def main() -> None:
    print(f"writing figures to {OUT_DIR}")
    print(f"measured (from instruction_tuning.py): multitask held-out={MULTI_HELDOUT_ACC:.1%}, "
          f"singletask held-out={SINGLE_HELDOUT_ACC:.1%}")
    fig_task_scaling()
    fig_multitask_vs_single()
    fig_instruction_template()
    fig_diversity_vs_count()
    fig_task_taxonomy()
    fig_cot_in_mix()
    fig_dataset_scaling()
    print("done.")


if __name__ == "__main__":
    main()
