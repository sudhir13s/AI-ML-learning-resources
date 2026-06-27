"""Reproducible figure generator for 20-Hallucination-and-Alignment-Basics.

Produces every embedded PNG for the chapter from the SAME numbers used on the page and in the
notebook -- the constants and functions are IMPORTED from hallucination_alignment.py, so the
figures cannot silently drift from the prose or the demo. Run:

    python make_figures_20.py

Each figure is written to ../../images/ (the shared chapter image dir) at 150 dpi, every
filename prefixed `hall_`. The palette matches the chapter's Mermaid diagrams (muted, white
text on coloured fills).

Figures produced (all `hall_*`):
  hall_softmax_floor.png      -- the softmax always assigns SOME mass to every token (no zero option)
  hall_temperature_dists.png  -- the toy knowledge dist reshaped by T (supported vs distractors)
  hall_temp_vs_rate.png       -- unsupported-claim rate climbs with temperature
  hall_grounding_dists.png     -- ungrounded vs grounded next-token distribution
  hall_grounding_vs_rate.png   -- grounding lowers the unsupported-claim curve at every T
  hall_snowball.png            -- one early wrong token snowballs into a wrong continuation
  hall_coverage_accuracy.png   -- abstention: coverage falls, answered-accuracy rises with tau
  hall_reliability.png         -- reliability diagram (confidence vs accuracy) + ECE
  hall_helpful_harmless.png    -- the helpful-vs-harmless Pareto frontier

Verified on Python 3.12 / torch 2.x / numpy 2.x / matplotlib 3.x.
"""

from __future__ import annotations

import _pathsetup  # noqa: F401  (sys.path bootstrap for the moved generator)

from pathlib import Path

import matplotlib

matplotlib.use("Agg")  # headless backend: write files, never open a window
import matplotlib.pyplot as plt
import numpy as np

from hallucination_alignment import (
    ANSWER_VOCAB,
    GROUNDING_BOOST,
    PERMISSIVENESS_THRESHOLDS,
    SUPPORTED_IDX,
    TEMPS_HALLUCINATION,
    base_logits_tensor,
    coverage_accuracy_curve,
    expected_calibration_error,
    grounded_logits_tensor,
    hallucination_rate_curve,
    helpful_harmless_curve,
    make_calibration_data,
    make_request_population,
    reliability_bins,
    softmax_with_temperature,
)

# ---- Palette (matches the chapter Mermaid classDefs) --------------------------------------
BLUE = "#3A6B96"
PURPLE = "#5D4A8A"
GREEN = "#2E7A5A"
RED = "#8B3B4A"
SLATE = "#4A5B6E"
AMBER = "#7A6528"
NAVY = "#2A5B80"
INK = "#1C2530"  # near-black for axis text
GRID = "#D4D9DF"

OUT_DIR = Path(__file__).resolve().parent.parent / "09. LLMs" / "images"
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
    print(f"wrote {path}")


def _bar_colors(n: int) -> list[str]:
    """Supported token GREEN, every distractor RED -- the supported/unsupported split."""
    return [GREEN if i == SUPPORTED_IDX else RED for i in range(n)]


def fig_softmax_floor() -> None:
    """The softmax floor: even when the model 'knows' the answer, every wrong token keeps mass.

    Plot the T=1.0 distribution on a log-y axis so the tiny-but-nonzero probabilities on the
    distractor tokens are visible. The whole point: there is no 'attend to nothing / assign 0'
    option -- the residual mass on wrong tokens is the structural seed of hallucination.
    """
    logits = base_logits_tensor(device="cpu")
    probs = softmax_with_temperature(logits, 1.0).numpy()
    x = np.arange(len(ANSWER_VOCAB))
    fig, ax = plt.subplots(figsize=(8.6, 4.6))
    ax.bar(x, probs, color=_bar_colors(len(ANSWER_VOCAB)), edgecolor="white",
           linewidth=0.6, zorder=3)
    ax.set_yscale("log")
    ax.set_ylim(probs.min() * 0.5, 1.0)
    for i in range(len(ANSWER_VOCAB)):
        ax.text(x[i], probs[i] * 1.15, f"{probs[i]:.3f}", ha="center", va="bottom",
                fontsize=8, color=INK)
    ax.set_xticks(x)
    ax.set_xticklabels([f"{w}\n{'(supported)' if i == SUPPORTED_IDX else '(wrong)'}"
                        for i, w in enumerate(ANSWER_VOCAB)], fontsize=8.5)
    ax.set_ylabel("probability (log scale)")
    ax.set_title("The softmax floor: every wrong token keeps non-zero mass — there is no 'assign 0' option")
    _style_axis(ax)
    fig.tight_layout()
    _save(fig, "hall_softmax_floor.png")


def fig_temperature_dists() -> None:
    """The toy knowledge distribution reshaped by temperature T = 0.5 / 1.0 / 2.0.

    Supported token GREEN, distractors RED. As T rises the supported token's mass bleeds out to
    the distractors -- the visual cause of the rising unsupported-claim rate in the next figure.
    """
    temps = (0.5, 1.0, 2.0)
    colors_map = {0.5: BLUE, 1.0: PURPLE, 2.0: AMBER}  # unused per-bar; bars use supported/wrong
    logits = base_logits_tensor(device="cpu")
    x = np.arange(len(ANSWER_VOCAB))
    fig, axes = plt.subplots(1, 3, figsize=(13.0, 4.4), sharey=True)
    for ax, temp in zip(axes, temps):
        probs = softmax_with_temperature(logits, temp).numpy()
        supported_mass = probs[SUPPORTED_IDX]
        ax.bar(x, probs, color=_bar_colors(len(ANSWER_VOCAB)), edgecolor="white",
               linewidth=0.6, zorder=3)
        ax.set_title(f"T = {temp}   (supported mass {supported_mass:.2f})", fontsize=12,
                     color=colors_map[temp])
        ax.set_xticks(x)
        ax.set_xticklabels(ANSWER_VOCAB, rotation=45, ha="right", fontsize=8)
        ax.set_ylim(0, 1.02)
        _style_axis(ax)
    axes[0].set_ylabel("probability")
    fig.suptitle(
        "Raising temperature bleeds mass off the supported answer (green) onto wrong distractors (red)",
        fontsize=13, color=INK,
    )
    fig.tight_layout(rect=(0, 0, 1, 0.94))
    _save(fig, "hall_temperature_dists.png")


def fig_temp_vs_rate() -> None:
    """Unsupported-claim rate as a function of decoding temperature (ungrounded sampler)."""
    logits = base_logits_tensor(device="cpu")
    rates = hallucination_rate_curve(logits, TEMPS_HALLUCINATION, device="cpu")
    fig, ax = plt.subplots(figsize=(8.8, 4.8))
    ax.plot(TEMPS_HALLUCINATION, rates, "-o", color=RED, linewidth=2.6, markersize=8, zorder=4)
    for t, r in zip(TEMPS_HALLUCINATION, rates):
        ax.annotate(f"{r:.2f}", (t, r), textcoords="offset points", xytext=(0, 9),
                    ha="center", fontsize=9, color=INK)
    ax.set_xlabel("decoding temperature T")
    ax.set_ylabel("unsupported-claim rate")
    ax.set_ylim(0, max(rates) * 1.18)
    ax.set_title("Hotter decoding hallucinates more: unsupported-claim rate climbs with temperature")
    _style_axis(ax)
    fig.tight_layout()
    _save(fig, "hall_temp_vs_rate.png")


def fig_grounding_dists() -> None:
    """Ungrounded vs grounded next-token distribution at T = 1.0 (the effect of retrieval)."""
    ungrounded = softmax_with_temperature(base_logits_tensor(device="cpu"), 1.0).numpy()
    grounded = softmax_with_temperature(grounded_logits_tensor(device="cpu"), 1.0).numpy()
    x = np.arange(len(ANSWER_VOCAB))
    width = 0.4
    fig, ax = plt.subplots(figsize=(9.4, 4.8))
    ax.bar(x - width / 2, ungrounded, width, color=SLATE, edgecolor="white", linewidth=0.6,
           label="ungrounded (no retrieval)", zorder=3)
    ax.bar(x + width / 2, grounded, width, color=GREEN, edgecolor="white", linewidth=0.6,
           label=f"grounded (+{GROUNDING_BOOST:.0f} logit on supported)", zorder=3)
    ax.set_xticks(x)
    ax.set_xticklabels(ANSWER_VOCAB, rotation=45, ha="right")
    ax.set_ylabel("probability (T = 1.0)")
    ax.set_title("Retrieval grounding concentrates mass on the supported answer ('Austen')")
    ax.legend(frameon=False, fontsize=10)
    _style_axis(ax)
    fig.tight_layout()
    _save(fig, "hall_grounding_dists.png")


def fig_grounding_vs_rate() -> None:
    """Unsupported-claim rate vs temperature: ungrounded vs grounded, same axes."""
    base = base_logits_tensor(device="cpu")
    grounded = grounded_logits_tensor(device="cpu")
    rates_u = hallucination_rate_curve(base, TEMPS_HALLUCINATION, device="cpu")
    rates_g = hallucination_rate_curve(grounded, TEMPS_HALLUCINATION, device="cpu")
    fig, ax = plt.subplots(figsize=(8.8, 4.8))
    ax.plot(TEMPS_HALLUCINATION, rates_u, "-o", color=RED, linewidth=2.6, markersize=8,
            label="ungrounded", zorder=4)
    ax.plot(TEMPS_HALLUCINATION, rates_g, "-o", color=GREEN, linewidth=2.6, markersize=8,
            label="grounded (retrieval)", zorder=4)
    ax.fill_between(TEMPS_HALLUCINATION, rates_g, rates_u, color=AMBER, alpha=0.18, zorder=2,
                    label="hallucinations removed by grounding")
    ax.set_xlabel("decoding temperature T")
    ax.set_ylabel("unsupported-claim rate")
    ax.set_title("Grounding lowers the hallucination curve at every temperature")
    ax.legend(frameon=False, fontsize=10, loc="upper left")
    _style_axis(ax)
    fig.tight_layout()
    _save(fig, "hall_grounding_vs_rate.png")


def fig_snowball() -> None:
    """Snowballing: one early wrong token forces the rest of the answer to stay wrong.

    A toy two-step continuation. At step 1 the model picks an entity (right vs a fabricated one).
    At step 2 it must produce a fact ABOUT that entity. Conditioned on a wrong entity, the only
    *coherent* continuations are also wrong -- so an early error compounds rather than self-corrects.
    We show P(correct full answer) collapsing once the first token is wrong.
    """
    # P(first token correct) under two regimes: greedy-ish (low T) vs hot (high T).
    p_first_correct = np.array([0.95, 0.80, 0.60, 0.40])  # decreasing as T rises
    temps = np.array([0.5, 1.0, 1.5, 2.0])
    # If the first token is wrong, the model is now conditioned on a false premise; the chance it
    # recovers a globally correct answer is tiny (it confabulates consistently). p_recover small.
    p_recover_if_wrong = 0.05
    # P(correct answer) = P(first right) * P(stay right | first right)  +  P(first wrong)*p_recover
    p_stay_right = 0.9
    p_answer_correct = p_first_correct * p_stay_right + (1 - p_first_correct) * p_recover_if_wrong
    fig, ax = plt.subplots(figsize=(9.0, 4.8))
    width = 0.38
    x = np.arange(len(temps))
    ax.bar(x - width / 2, p_first_correct, width, color=BLUE, edgecolor="white", linewidth=0.6,
           label="P(first token correct)", zorder=3)
    ax.bar(x + width / 2, p_answer_correct, width, color=GREEN, edgecolor="white", linewidth=0.6,
           label="P(whole answer correct)", zorder=3)
    for i in range(len(temps)):
        ax.annotate(f"{p_first_correct[i]:.2f}", (x[i] - width / 2, p_first_correct[i]),
                    textcoords="offset points", xytext=(0, 5), ha="center", fontsize=8, color=INK)
        ax.annotate(f"{p_answer_correct[i]:.2f}", (x[i] + width / 2, p_answer_correct[i]),
                    textcoords="offset points", xytext=(0, 5), ha="center", fontsize=8, color=INK)
    ax.set_xticks(x)
    ax.set_xticklabels([f"T={t}" for t in temps])
    ax.set_ylabel("probability")
    ax.set_ylim(0, 1.05)
    ax.set_title("Snowballing: once the first token is wrong, the answer rarely recovers (illustrative)")
    ax.legend(frameon=False, fontsize=10)
    _style_axis(ax)
    fig.tight_layout()
    _save(fig, "hall_snowball.png")


def fig_coverage_accuracy() -> None:
    """Abstention trade: sweep the confidence threshold tau; coverage falls, accuracy rises."""
    confidence, correct = make_calibration_data()
    taus = np.linspace(0.5, 0.98, 40)
    cov, acc = coverage_accuracy_curve(confidence, correct, taus)
    base_acc = float(correct.mean())
    fig, ax = plt.subplots(figsize=(9.0, 4.8))
    ax.plot(taus, acc, color=GREEN, linewidth=2.8, label="answered accuracy", zorder=4)
    ax.plot(taus, cov, color=BLUE, linewidth=2.8, label="coverage (fraction answered)", zorder=4)
    ax.axhline(base_acc, color=SLATE, linestyle=":", linewidth=1.8,
               label=f"answer-everything accuracy = {base_acc:.2f}", zorder=2)
    ax.set_xlabel("abstention threshold  τ  (answer only if confidence ≥ τ)")
    ax.set_ylabel("rate")
    ax.set_ylim(0, 1.03)
    ax.set_title("Abstention buys accuracy with coverage: raise τ → fewer answers, but more reliable")
    ax.legend(frameon=True, framealpha=0.9, edgecolor="none", fontsize=9.5,
              loc="lower center", bbox_to_anchor=(0.5, 0.02))
    _style_axis(ax)
    fig.tight_layout()
    _save(fig, "hall_coverage_accuracy.png")


def fig_reliability() -> None:
    """Reliability diagram: per-confidence-bin accuracy vs the y=x calibrated line, plus ECE."""
    confidence, correct = make_calibration_data()
    centers, accs, counts = reliability_bins(confidence, correct, n_bins=10)
    ece = expected_calibration_error(confidence, correct, n_bins=10)
    valid = counts > 0
    fig, ax = plt.subplots(figsize=(7.6, 6.0))
    # Perfect-calibration reference line y = x.
    ax.plot([0, 1], [0, 1], color=SLATE, linestyle="--", linewidth=1.8,
            label="perfect calibration (accuracy = confidence)", zorder=2)
    ax.bar(centers[valid], accs[valid], width=0.09, color=BLUE, edgecolor="white",
           linewidth=0.6, alpha=0.9, label="observed accuracy per bin", zorder=3)
    # Draw the gap (confidence - accuracy) as a red overlay on each bar's top.
    for c, a, n in zip(centers[valid], accs[valid], counts[valid]):
        ax.plot([c, c], [a, c], color=RED, linewidth=2.2, zorder=4)
    ax.set_xlabel("confidence (model's max softmax probability)")
    ax.set_ylabel("accuracy (fraction correct in bin)")
    ax.set_xlim(0.45, 1.0)
    ax.set_ylim(0, 1.0)
    ax.set_title(f"Reliability diagram: bars below the line = over-confidence.  ECE = {ece:.3f}")
    ax.legend(frameon=False, fontsize=9.5, loc="upper left")
    _style_axis(ax)
    fig.tight_layout()
    _save(fig, "hall_reliability.png")


def fig_helpful_harmless() -> None:
    """The helpful-vs-harmless Pareto frontier traced by the single permissiveness knob r."""
    perceived_harm, is_harmful = make_request_population()
    helpful, harmless = helpful_harmless_curve(
        perceived_harm, is_harmful, PERMISSIVENESS_THRESHOLDS
    )
    fig, ax = plt.subplots(figsize=(7.8, 6.0))
    sc = ax.scatter(helpful, harmless, c=PERMISSIVENESS_THRESHOLDS, cmap="viridis",
                    s=55, zorder=4, edgecolor="white", linewidth=0.5)
    ax.plot(helpful, harmless, color=SLATE, linewidth=1.4, alpha=0.6, zorder=3)
    cbar = fig.colorbar(sc, ax=ax)
    cbar.set_label("permissiveness threshold r", color=INK)
    # Annotate the two extremes and a balanced point.
    ax.annotate("r→0: refuses all\n(safe, useless)", (helpful[0], harmless[0]),
                textcoords="offset points", xytext=(16, -28), fontsize=9, color=INK)
    ax.annotate("r→1: answers all\n(helpful, unsafe)", (helpful[-1], harmless[-1]),
                textcoords="offset points", xytext=(-46, 22), fontsize=9, color=INK)
    ax.set_xlabel("helpfulness  (benign requests answered)")
    ax.set_ylabel("harmlessness  (harmful requests refused)")
    ax.set_xlim(-0.03, 1.05)
    ax.set_ylim(-0.03, 1.08)
    ax.set_title("Helpful vs harmless: one scalar knob cannot reach the top-right corner")
    _style_axis(ax)
    fig.tight_layout()
    _save(fig, "hall_helpful_harmless.png")


def main() -> None:
    fig_softmax_floor()
    fig_temperature_dists()
    fig_temp_vs_rate()
    fig_grounding_dists()
    fig_grounding_vs_rate()
    fig_snowball()
    fig_coverage_accuracy()
    fig_reliability()
    fig_helpful_harmless()
    print("all figures written to", OUT_DIR)


if __name__ == "__main__":
    main()
