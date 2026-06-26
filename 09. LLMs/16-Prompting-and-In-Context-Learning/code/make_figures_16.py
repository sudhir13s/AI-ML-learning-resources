"""Reproducible figure generator for 16-Prompting-and-In-Context-Learning.

Produces every embedded PNG for the chapter from the SAME numbers used on the page and in the
notebook -- so the figures cannot silently drift from the prose. Run:

    python make_figures_16.py

Each figure is written to ../../images/ at 150 dpi. The palette matches the chapter's Mermaid
diagrams (muted, white text on coloured fills). The measured figures recompute their numbers
from the actual demo (prompting_icl.py), never hardcoded from memory:
  - the few-shot vs zero-shot accuracy curve (k=0 ~0.03 -> k>=1 ~1.00),
  - the induction-head look-back attention row (peaks on the value after the key),
  - the order-sensitivity swing (fraction of conflicting-demo sequences that flip ~0.57).
The conceptual figures (prompt anatomy, zero/few-shot structure) are illustrative and labelled
as such in their captions on the page.

Verified on Python 3.12 / torch 2.x / matplotlib 3.x.
"""

from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")  # headless backend: write files, never open a window
import matplotlib.pyplot as plt
import numpy as np

# Reuse the EXACT demo machinery so figures and the script can never disagree.
from prompting_icl import (
    CUE_ID,
    SHOT_VALUES,
    accuracy_vs_shots,
    contextual_calibration,
    induction_lookback,
    order_sensitivity,
    train_model,
)

# ---- Palette (matches the chapter Mermaid classDefs) --------------------------------
BLUE = "#3A6B96"
PURPLE = "#5D4A8A"
GREEN = "#2E7A5A"
RED = "#8B3B4A"
SLATE = "#4A5B6E"
AMBER = "#7A6528"
NAVY = "#2A5B80"
INK = "#1C2530"  # near-black for axis text
GRID = "#D4D9DF"

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
# Figure 1 -- zero-shot vs few-shot prompt STRUCTURE (conceptual, illustrative)
# =====================================================================================
def fig_zero_vs_fewshot() -> None:
    """Side-by-side: a zero-shot prompt (instruction + input) vs a few-shot prompt
    (instruction + k demonstrations + input). The thing that changes is the CONTEXT."""
    fig, (ax0, ax1) = plt.subplots(1, 2, figsize=(11.5, 4.6))
    for ax in (ax0, ax1):
        ax.set_xlim(0, 10)
        ax.set_ylim(0, 10)
        ax.axis("off")

    def block(ax, y, height, color, label, sub):
        ax.add_patch(
            plt.Rectangle((0.4, y), 9.2, height, facecolor=color, edgecolor="white", lw=1.5)
        )
        ax.text(0.7, y + height / 2 + 0.22, label, color="white", fontsize=11,
                fontweight="bold", va="center")
        ax.text(0.7, y + height / 2 - 0.5, sub, color="white", fontsize=8.5,
                va="center", style="italic")

    # Zero-shot
    ax0.set_title("Zero-shot prompt", fontsize=13, color=INK, fontweight="bold")
    block(ax0, 7.0, 2.2, PURPLE, "INSTRUCTION", "Classify the sentiment as pos/neg.")
    block(ax0, 4.3, 2.2, BLUE, "INPUT", "Review: 'The plot dragged.'  Sentiment:")
    ax0.text(5.0, 2.8, "model continues -> 'neg'", ha="center", color=GREEN,
             fontsize=10.5, fontweight="bold")
    ax0.annotate("", xy=(5.0, 3.4), xytext=(5.0, 4.2),
                 arrowprops=dict(arrowstyle="-|>", color=GREEN, lw=2))

    # Few-shot
    ax1.set_title("Few-shot prompt (k = 2 demonstrations)", fontsize=13, color=INK,
                  fontweight="bold")
    block(ax1, 8.2, 1.5, PURPLE, "INSTRUCTION", "Classify the sentiment as pos/neg.")
    block(ax1, 6.3, 1.5, AMBER, "DEMO 1", "'Loved every minute.' -> pos")
    block(ax1, 4.4, 1.5, AMBER, "DEMO 2", "'A total waste.' -> neg")
    block(ax1, 2.5, 1.5, BLUE, "INPUT", "'The plot dragged.' ->")
    ax1.text(5.0, 1.4, "model infers the task FROM the demos -> 'neg'", ha="center",
             color=GREEN, fontsize=10, fontweight="bold")
    ax1.annotate("", xy=(5.0, 2.0), xytext=(5.0, 2.5),
                 arrowprops=dict(arrowstyle="-|>", color=GREEN, lw=2))
    fig.suptitle("Same model, no weight updates — only the context changes",
                 fontsize=11, color=SLATE, y=0.02)
    _save(fig, "icl_zero_vs_fewshot.png")


# =====================================================================================
# Figure 2 -- anatomy of a prompt (conceptual, illustrative)
# =====================================================================================
def fig_prompt_anatomy() -> None:
    """A stacked strip showing the building blocks practitioners assemble into a prompt."""
    parts = [
        ("System / role", "You are a careful financial analyst.", NAVY),
        ("Instruction", "Extract every dollar amount as JSON.", PURPLE),
        ("Demonstrations", "Input -> Output  (x k examples)", AMBER),
        ("Delimiters", '"""  ...text...  """  (fences inputs)', SLATE),
        ("Input", "The actual query / document to act on", BLUE),
        ("Output cue", "JSON:", GREEN),
    ]
    fig, ax = plt.subplots(figsize=(9.5, 5.2))
    ax.set_xlim(0, 10)
    ax.set_ylim(0, len(parts))
    ax.axis("off")
    for i, (label, sub, color) in enumerate(parts):
        y = len(parts) - 1 - i
        ax.add_patch(plt.Rectangle((0.3, y + 0.12), 9.4, 0.76, facecolor=color,
                                   edgecolor="white", lw=1.5))
        ax.text(0.6, y + 0.62, label, color="white", fontsize=11, fontweight="bold",
                va="center")
        ax.text(0.6, y + 0.30, sub, color="white", fontsize=8.8, va="center", style="italic")
    ax.set_title("Anatomy of a prompt — the levers you actually control",
                 fontsize=13, color=INK, fontweight="bold", pad=12)
    _save(fig, "icl_prompt_anatomy.png")


# =====================================================================================
# Figure 3 -- few-shot vs zero-shot accuracy curve (MEASURED from the demo)
# =====================================================================================
def fig_accuracy_curve(acc: dict[int, float]) -> None:
    """Next-token accuracy vs k in-context demonstrations: chance at k=0, then ICL switches on."""
    ks = list(SHOT_VALUES)
    ys = [acc[k] for k in ks]
    chance = 1.0 / 40  # 1 / N_SYMBOLS
    fig, ax = plt.subplots(figsize=(7.4, 4.8))
    _style_axis(ax)
    ax.plot(ks, ys, "-o", color=BLUE, lw=2.4, markersize=9, markerfacecolor=BLUE,
            markeredgecolor="white", zorder=3, label="measured accuracy")
    ax.axhline(chance, color=RED, ls="--", lw=1.8, label=f"chance (1/40 = {chance:.3f})")
    ax.annotate("zero-shot:\nnothing to copy\n-> ~chance", xy=(0, ys[0]), xytext=(0.5, 0.30),
                fontsize=9.5, color=RED,
                arrowprops=dict(arrowstyle="-|>", color=RED, lw=1.6))
    ax.annotate("one demonstration is\nenough -> task 'switches on'",
                xy=(1, ys[1]), xytext=(1.6, 0.62), fontsize=9.5, color=GREEN,
                arrowprops=dict(arrowstyle="-|>", color=GREEN, lw=1.6))
    ax.set_xlabel("k  (number of in-context demonstrations)")
    ax.set_ylabel("next-token accuracy")
    ax.set_ylim(-0.03, 1.08)
    ax.set_xticks(ks)
    ax.set_title("In-context learning curve: accuracy vs shots",
                 fontsize=12.5, color=INK, fontweight="bold")
    ax.legend(loc="center right", framealpha=0.95)
    _save(fig, "icl_accuracy_vs_shots.png")


# =====================================================================================
# Figure 4 -- induction-head look-back attention (MEASURED from the demo)
# =====================================================================================
def fig_induction_lookback(look: dict[str, object]) -> None:
    """Attention from the query position over the sequence; the bar on the value-after-key
    is the induction head copying."""
    attn = np.array(look["attn_from_query"], dtype=float)  # type: ignore[arg-type]
    ids = look["tokens"]  # type: ignore[assignment]
    vpos = int(look["lookback_value_pos"])  # type: ignore[arg-type]
    query_key = int(look["query_key"])  # type: ignore[arg-type]
    target = int(look["target"])  # type: ignore[arg-type]
    positions = np.arange(len(attn))
    labels = []
    for i, tid in enumerate(ids):  # type: ignore[arg-type]
        if tid == CUE_ID:
            labels.append("CUE")
        elif i == len(ids) - 1:  # type: ignore[arg-type]
            labels.append(f"{tid}\n(query)")
        else:
            labels.append(str(tid))
    colors = [SLATE] * len(attn)
    colors[vpos] = GREEN  # the looked-up value: the induction target
    fig, ax = plt.subplots(figsize=(9.8, 4.5))
    _style_axis(ax)
    ax.bar(positions, attn, color=colors, edgecolor="white", lw=1.0, zorder=3)
    ax.set_xticks(positions)
    ax.set_xticklabels(labels, fontsize=8.5)
    ax.set_xlabel("sequence position (key/value pairs, then CUE + query key)")
    ax.set_ylabel("attention weight from the query")
    ax.annotate(
        f"copies value {target}\n(the token AFTER the\nquery key's 1st occurrence)",
        xy=(vpos, attn[vpos]), xytext=(1.0, 0.66),
        fontsize=9.5, color=GREEN, fontweight="bold",
        arrowprops=dict(arrowstyle="-|>", color=GREEN, lw=1.8),
    )
    ax.set_title(
        f"Induction head: the query (key {query_key}) looks back and copies",
        fontsize=12.5, color=INK, fontweight="bold",
    )
    _save(fig, "icl_induction_lookback.png")


# =====================================================================================
# Figure 5 -- order sensitivity (MEASURED from the demo)
# =====================================================================================
def fig_order_sensitivity(sens: dict[str, float]) -> None:
    """Two bars: fraction of conflicting-demo sequences whose answer flips with order,
    and how often the model copies the LATER demo (recency) when it flips."""
    frac = sens["frac_order_sensitive"]
    recency = sens["recency_rate"]
    fig, ax = plt.subplots(figsize=(7.0, 4.6))
    _style_axis(ax)
    bars = ax.bar(
        ["answer FLIPS\nwith order", "copies the LATER\ndemo (recency)"],
        [frac, recency],
        color=[RED, AMBER], edgecolor="white", lw=1.4, width=0.62, zorder=3,
    )
    for b, v in zip(bars, [frac, recency]):
        ax.text(b.get_x() + b.get_width() / 2, v + 0.02, f"{v:.2f}",
                ha="center", color=INK, fontsize=12, fontweight="bold")
    ax.axhline(0.5, color=SLATE, ls=":", lw=1.4)
    ax.text(1.45, 0.51, "50% line", color=SLATE, fontsize=8.5, va="bottom")
    ax.set_ylim(0, 1.0)
    ax.set_ylabel("fraction of sequences")
    ax.set_title(
        "ICL is brittle: reordering the SAME demos swings the answer",
        fontsize=12, color=INK, fontweight="bold",
    )
    _save(fig, "icl_order_sensitivity.png")


# =====================================================================================
# Figure 6 -- content-free calibration: before vs after (left MEASURED, right illustrative)
# =====================================================================================
def fig_calibration(cal: dict[str, float]) -> None:
    """Two panels. LEFT (measured from the demo): the content-free probe's mass on the majority
    label collapses from a large bias to the uniform prior after calibration -- a calibrated
    prompt is unbiased on a content-free input. RIGHT (illustrative, Zhao et al. 2021): the
    downstream few-shot accuracy gain that this debiasing buys on real tasks."""
    before = cal["cf_majority_mass_before"]
    after = cal["cf_majority_mass_after"]
    uniform = cal["uniform_prior"]
    acc_before = cal["reported_acc_before"]
    acc_after = cal["reported_acc_after"]

    fig, (axL, axR) = plt.subplots(1, 2, figsize=(11.0, 4.7))
    _style_axis(axL)
    _style_axis(axR)

    # LEFT -- measured content-free bias before/after
    barsL = axL.bar(
        ["BEFORE\ncalibration", "AFTER\ncalibration"], [before, after],
        color=[RED, GREEN], edgecolor="white", lw=1.4, width=0.6, zorder=3,
    )
    for b, v in zip(barsL, [before, after]):
        axL.text(b.get_x() + b.get_width() / 2, v + 0.02, f"{v:.3f}",
                 ha="center", color=INK, fontsize=12, fontweight="bold")
    axL.axhline(uniform, color=SLATE, ls=":", lw=1.6)
    axL.text(0.5, uniform + 0.10, f"uniform prior (1/40 = {uniform:.3f})",
             color=SLATE, fontsize=8.5, va="bottom", ha="center")
    axL.set_ylim(0, 1.0)
    axL.set_ylabel("content-free probe:\nmass on the majority label")
    axL.set_title("Measured: the bias the probe reveals,\ndivided out to uniform",
                  fontsize=11, color=INK, fontweight="bold")

    # RIGHT -- illustrative downstream accuracy gain (Zhao et al. 2021)
    barsR = axR.bar(
        ["BEFORE\ncalibration", "AFTER\ncalibration"], [acc_before, acc_after],
        color=[SLATE, BLUE], edgecolor="white", lw=1.4, width=0.6, zorder=3,
    )
    for b, v in zip(barsR, [acc_before, acc_after]):
        axR.text(b.get_x() + b.get_width() / 2, v + 0.015, f"{v:.3f}",
                 ha="center", color=INK, fontsize=12, fontweight="bold")
    axR.annotate("", xy=(1, acc_after), xytext=(0, acc_before),
                 arrowprops=dict(arrowstyle="-|>", color=GREEN, lw=2.0))
    axR.set_ylim(0, 1.0)
    axR.set_ylabel("few-shot accuracy")
    axR.set_title("Illustrative (Zhao et al. 2021):\ndownstream accuracy gain",
                  fontsize=11, color=INK, fontweight="bold")

    fig.suptitle(
        "Content-free calibration: measure the prompt's bias on an empty input, divide it out",
        fontsize=12.5, color=INK, fontweight="bold", y=1.02,
    )
    _save(fig, "icl_calibration.png")


def main() -> None:
    print("training the tiny induction transformer (CPU, deterministic)...")
    model = train_model()
    acc = accuracy_vs_shots(model)
    look = induction_lookback(model)
    sens = order_sensitivity(model)
    cal = contextual_calibration(model)
    # Echo the numbers so a reader can confirm figures == page == .py at a glance.
    print(f"  few-shot curve: {{{', '.join(f'{k}:{acc[k]:.3f}' for k in SHOT_VALUES)}}}")
    print(f"  induction peak weight: {look['attn_from_query'][int(look['lookback_value_pos'])]:.2f}")  # type: ignore[index]
    print(f"  order-sensitive fraction: {sens['frac_order_sensitive']:.3f}  "
          f"recency: {sens['recency_rate']:.3f}")
    print(f"  calibration content-free mass: {cal['cf_majority_mass_before']:.3f} -> "
          f"{cal['cf_majority_mass_after']:.3f}")
    fig_zero_vs_fewshot()
    fig_prompt_anatomy()
    fig_accuracy_curve(acc)
    fig_induction_lookback(look)
    fig_order_sensitivity(sens)
    fig_calibration(cal)
    print("all figures written to", OUT_DIR)


if __name__ == "__main__":
    main()
