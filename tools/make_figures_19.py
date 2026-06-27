"""Reproducible figure generator for 19-LLM-Evaluation-and-Benchmarks.

Produces every embedded PNG for the chapter from the SAME numbers used on the page and in the
notebook -- the constants and functions are IMPORTED from llm_evaluation.py, so the figures
cannot silently drift from the prose or the demo. Run:

    python make_figures_19.py

Each figure is written to ../../images/ (the shared chapter image dir) at 150 dpi, every name
prefixed `eval_`. The palette matches the chapter's Mermaid diagrams (muted, white text on
coloured fills).

Figures produced:
  eval_perplexity_intuition.png   -- exp(CE): good vs bad model PPL + the PPL=exp(CE) curve
  eval_passk_curve.png            -- pass@k vs k for several c/n (the diminishing-returns curve)
  eval_passk_bias.png             -- unbiased vs naive (biased-low) pass@k estimator, swept over k
  eval_ece_reliability.png        -- reliability diagram: calibrated vs over-confident + ECE gap
  eval_elo_convergence.png        -- Elo ratings separating from a flat start to the true order
  eval_elo_logistic.png           -- Elo/Bradley-Terry expected-score logistic in the rating gap
  eval_judge_position_bias.png    -- flip-rate & first-position win-rate vs judge bias strength
  eval_benchmark_landscape.png    -- what each benchmark measures, on a knowledge<->reasoning map
  eval_contamination.png          -- how test-set contamination inflates a reported score
  eval_goodhart_saturation.png    -- benchmarks saturate: scores pile up at the ceiling over time

Verified on Python 3.12 / numpy 2.x / matplotlib 3.x.
"""

from __future__ import annotations

import _pathsetup  # noqa: F401  (sys.path bootstrap for the moved generator)

from pathlib import Path

import matplotlib

matplotlib.use("Agg")  # headless backend: write files, never open a window
import matplotlib.pyplot as plt
import numpy as np

from llm_evaluation import (
    BAD_MODEL_PROBS,
    ELO_BASE,
    ELO_MATCHES,
    ELO_SCALE,
    ELO_SEED,
    GOOD_MODEL_PROBS,
    HELD_OUT_IDS,
    PASS_C,
    PASS_N,
    PASS_NAIVE_SEED,
    PASS_NAIVE_TRIALS,
    TRUE_SKILL,
    cross_entropy_nats,
    expected_calibration_error,
    expected_score,
    make_calibration_data,
    measure_position_bias,
    pass_at_k,
    pass_at_k_naive_mc,
    perplexity,
    simulate_elo,
)

try:
    import torch

    _TORCH_OK = True
except ImportError:  # the figures need only numpy; torch is just for the perplexity tensors
    _TORCH_OK = False

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


def fig_perplexity_intuition() -> None:
    """Left: good vs bad model perplexity on the held-out sequence. Right: PPL = exp(CE)."""
    good = torch.tensor(GOOD_MODEL_PROBS)
    bad = torch.tensor(BAD_MODEL_PROBS)
    ce_good = cross_entropy_nats(good, HELD_OUT_IDS)
    ce_bad = cross_entropy_nats(bad, HELD_OUT_IDS)
    ppl_good = perplexity(good, HELD_OUT_IDS)
    ppl_bad = perplexity(bad, HELD_OUT_IDS)

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12.4, 4.6))

    # Left: the two models' perplexity as bars, annotated with their cross-entropy.
    labels = ["confident-correct\nmodel", "near-uniform\nmodel"]
    ppls = [ppl_good, ppl_bad]
    ces = [ce_good, ce_bad]
    colors = [GREEN, RED]
    bars = ax1.bar(labels, ppls, color=colors, edgecolor="white", linewidth=0.8, zorder=3)
    for bar, ppl, ce in zip(bars, ppls, ces):
        ax1.annotate(f"PPL {ppl:.2f}\nCE {ce:.2f} nats",
                     (bar.get_x() + bar.get_width() / 2, ppl),
                     textcoords="offset points", xytext=(0, 6), ha="center",
                     fontsize=10, color=INK)
    ax1.set_ylabel("perplexity (lower = better)")
    ax1.set_ylim(0, max(ppls) * 1.25)
    ax1.set_title("Lower cross-entropy → lower perplexity")
    _style_axis(ax1)

    # Right: the monotone map PPL = exp(CE); place both models on the curve.
    ce_grid = np.linspace(0.0, 2.0, 200)
    ax2.plot(ce_grid, np.exp(ce_grid), color=PURPLE, linewidth=2.6, zorder=3,
             label="PPL = exp(CE)")
    for ce, ppl, color, name in [
        (ce_good, ppl_good, GREEN, "good"), (ce_bad, ppl_bad, RED, "near-uniform")
    ]:
        ax2.plot(ce, ppl, "o", color=color, markersize=10, zorder=5)
        ax2.annotate(f"{name}", (ce, ppl), textcoords="offset points",
                     xytext=(8, -2), fontsize=10, color=INK)
    ax2.set_xlabel("cross-entropy CE (nats)")
    ax2.set_ylabel("perplexity = exp(CE)")
    ax2.set_title("Perplexity is just exp() of the LM loss")
    ax2.legend(frameon=False, fontsize=10, loc="upper left")
    _style_axis(ax2)

    fig.suptitle(
        "Perplexity: the effective branching factor a model hesitates between per token",
        fontsize=13, color=INK,
    )
    fig.tight_layout(rect=(0, 0, 1, 0.95))
    _save(fig, "eval_perplexity_intuition.png")


def fig_passk_curve() -> None:
    """pass@k vs k for several correct-fractions c/n -- the diminishing-returns shape."""
    n = 50
    k_values = np.arange(1, n + 1)
    fig, ax = plt.subplots(figsize=(9.4, 5.0))
    fractions = [(2, BLUE), (5, GREEN), (10, AMBER), (25, PURPLE)]
    for c, color in fractions:
        curve = [pass_at_k(n, c, int(k)) for k in k_values]
        ax.plot(k_values, curve, color=color, linewidth=2.6, zorder=3,
                label=f"c/n = {c}/{n} (per-sample pass {c / n:.0%})")
    ax.axhline(1.0, color=SLATE, linestyle=":", linewidth=1.6, zorder=2)
    ax.set_xlabel("k (number of samples allowed per problem)")
    ax.set_ylabel("pass@k  =  P(at least one of k passes)")
    ax.set_ylim(0, 1.05)
    ax.set_title("More attempts always help — but with sharply diminishing returns")
    ax.legend(frameon=False, fontsize=9.5, loc="lower right")
    _style_axis(ax)
    fig.tight_layout()
    _save(fig, "eval_passk_curve.png")


def fig_passk_bias() -> None:
    """Unbiased pass@k vs the naive with-replacement estimator, swept over k (bias is LOW)."""
    k_values = list(range(1, PASS_N + 1))
    unbiased = [pass_at_k(PASS_N, PASS_C, k) for k in k_values]
    naive = [
        pass_at_k_naive_mc(PASS_N, PASS_C, k, PASS_NAIVE_TRIALS, PASS_NAIVE_SEED)
        for k in k_values
    ]
    fig, ax = plt.subplots(figsize=(9.4, 5.0))
    ax.plot(k_values, unbiased, "-o", color=GREEN, linewidth=2.6, markersize=6, zorder=4,
            label="unbiased  1 − C(n−c,k)/C(n,k)")
    ax.plot(k_values, naive, "-s", color=RED, linewidth=2.4, markersize=6, zorder=3,
            label="naive (with replacement) — biased LOW")
    # Shade the bias gap so it reads as 'systematic under-count', not noise.
    ax.fill_between(k_values, naive, unbiased, color=RED, alpha=0.12, zorder=1)
    ax.set_xlabel("k")
    ax.set_ylabel("estimated pass@k")
    ax.set_ylim(0, 1.05)
    ax.set_title(f"Naive sampling under-estimates pass@k (n={PASS_N}, c={PASS_C})")
    ax.legend(frameon=False, fontsize=10, loc="lower right")
    _style_axis(ax)
    fig.tight_layout()
    _save(fig, "eval_passk_bias.png")


def fig_ece_reliability() -> None:
    """Reliability diagram: per-bin accuracy vs confidence for a calibrated & over-confident model."""
    conf_cal, corr_cal = make_calibration_data(6000, 0.0, 0)
    conf_over, corr_over = make_calibration_data(6000, 0.15, 0)
    ece_cal, bins_cal = expected_calibration_error(conf_cal, corr_cal)
    ece_over, bins_over = expected_calibration_error(conf_over, corr_over)

    fig, axes = plt.subplots(1, 2, figsize=(12.4, 5.0), sharey=True)
    for ax, (title, bins, ece, color) in zip(
        axes,
        [("Calibrated", bins_cal, ece_cal, GREEN),
         ("Over-confident", bins_over, ece_over, RED)],
    ):
        centers = [(b["lo"] + b["hi"]) / 2 for b in bins if b["count"] > 0]
        accs = [b["acc"] for b in bins if b["count"] > 0]
        confs = [b["conf"] for b in bins if b["count"] > 0]
        ax.plot([0, 1], [0, 1], color=SLATE, linestyle="--", linewidth=1.8, zorder=2,
                label="perfect calibration")
        ax.bar(centers, accs, width=0.085, color=color, edgecolor="white", linewidth=0.6,
               zorder=3, label="accuracy in bin")
        # The gap arrows: confidence above accuracy = over-confidence.
        for cen, acc, conf in zip(centers, accs, confs):
            ax.plot([cen, cen], [acc, conf], color=AMBER, linewidth=2.2, zorder=4)
        ax.set_xlim(0.45, 1.02)
        ax.set_ylim(0, 1.02)
        ax.set_xlabel("confidence (predicted P correct)")
        ax.set_title(f"{title}  —  ECE = {ece:.3f}")
        ax.legend(frameon=False, fontsize=9, loc="upper left")
        _style_axis(ax)
    axes[0].set_ylabel("empirical accuracy")
    fig.suptitle(
        "Reliability diagrams: bars on the diagonal = calibrated; bars below = over-confident (ECE = gap)",
        fontsize=12.5, color=INK,
    )
    fig.tight_layout(rect=(0, 0, 1, 0.94))
    _save(fig, "eval_ece_reliability.png")


def fig_elo_convergence() -> None:
    """Elo ratings separating from a flat start to the true ordering over simulated games."""
    _final, history = simulate_elo(TRUE_SKILL, ELO_MATCHES, ELO_SEED)
    names = list(TRUE_SKILL)
    colors = {"A": GREEN, "B": BLUE, "C": AMBER, "D": RED}
    steps = np.arange(len(history))
    fig, ax = plt.subplots(figsize=(10.4, 5.4))
    for name in names:
        series = [h[name] for h in history]
        ax.plot(steps, series, color=colors[name], linewidth=2.0, zorder=3,
                label=f"{name}  (true skill {TRUE_SKILL[name]:.0f})")
    ax.axhline(ELO_BASE, color=SLATE, linestyle=":", linewidth=1.4, zorder=2,
               label=f"flat start ({ELO_BASE:.0f})")
    ax.set_xlabel("matches played")
    ax.set_ylabel("Elo rating")
    ax.set_title("Elo separates equal-start agents into the true skill order\nfrom win/loss alone",
                 fontsize=12)
    ax.legend(frameon=False, fontsize=9, loc="upper left", bbox_to_anchor=(1.01, 1.0))
    _style_axis(ax)
    fig.tight_layout()
    _save(fig, "eval_elo_convergence.png")


def fig_elo_logistic() -> None:
    """The Bradley-Terry / Elo expected-score logistic in the rating gap."""
    gaps = np.linspace(-800, 800, 400)
    exp_scores = [expected_score(g, 0.0) for g in gaps]
    fig, ax = plt.subplots(figsize=(9.4, 5.0))
    ax.plot(gaps, exp_scores, color=PURPLE, linewidth=2.8, zorder=3)
    ax.axhline(0.5, color=SLATE, linestyle=":", linewidth=1.4, zorder=2)
    ax.axvline(0.0, color=SLATE, linestyle=":", linewidth=1.4, zorder=2)
    for gap in (-ELO_SCALE, ELO_SCALE):
        e = expected_score(gap, 0.0)
        ax.plot(gap, e, "o", color=AMBER, markersize=9, zorder=5)
        ax.annotate(f"gap {gap:+.0f}\nE = {e:.2f}", (gap, e), textcoords="offset points",
                    xytext=(10, -4 if gap > 0 else 10), fontsize=9.5, color=INK)
    ax.set_xlabel("rating gap  R_A − R_B")
    ax.set_ylabel("expected score  E_A  =  P(A beats B)")
    ax.set_title(f"Elo's expected score is a logistic: +{ELO_SCALE:.0f} points ≈ 10× odds")
    _style_axis(ax)
    fig.tight_layout()
    _save(fig, "eval_elo_logistic.png")


def fig_judge_position_bias() -> None:
    """Flip-rate and first-position win-rate vs the judge's position-bias strength."""
    biases = np.linspace(0.0, 0.4, 17)
    flip_rates = []
    first_winrates = []
    for b in biases:
        res = measure_position_bias(500, float(b), 0)
        flip_rates.append(res.flip_rate)
        first_winrates.append(res.first_pos_winrate)
    fig, ax = plt.subplots(figsize=(9.4, 5.0))
    ax.plot(biases, flip_rates, "-o", color=RED, linewidth=2.6, markersize=5, zorder=4,
            label="verdict flip-rate (swap the order → winner changes)")
    ax.plot(biases, first_winrates, "-s", color=BLUE, linewidth=2.6, markersize=5, zorder=3,
            label="first-position win-rate (0.5 = fair)")
    ax.axhline(0.5, color=SLATE, linestyle=":", linewidth=1.4, zorder=2)
    ax.set_xlabel("judge position-bias strength (thumb on the first answer)")
    ax.set_ylabel("rate")
    ax.set_ylim(0, 1.0)
    ax.set_title("A position-biased judge flips verdicts and favours whichever answer it sees first")
    ax.legend(frameon=False, fontsize=9.5, loc="upper left")
    _style_axis(ax)
    fig.tight_layout()
    _save(fig, "eval_judge_position_bias.png")


def fig_benchmark_landscape() -> None:
    """A knowledge<->reasoning x format map placing the major benchmarks; illustrative."""
    # (x = knowledge..reasoning, y = closed-form MCQ .. open-ended generation), illustrative
    # positions chosen to teach what each benchmark stresses, not to assert exact difficulty.
    # name -> (x, y, dot color, (label_dx, label_dy) offset in points, ha)
    benches = {
        "MMLU": (0.16, 0.14, BLUE, (10, 0), "left"),
        "ARC": (0.30, 0.22, BLUE, (10, 0), "left"),
        "HellaSwag": (0.42, 0.12, NAVY, (10, 0), "left"),
        "GSM8K": (0.74, 0.40, GREEN, (10, 0), "left"),
        "HumanEval (pass@k)": (0.86, 0.74, GREEN, (-12, 0), "right"),
        "BIG-bench": (0.52, 0.58, PURPLE, (10, 0), "left"),
        "MT-Bench (LLM-judge)": (0.46, 0.90, AMBER, (10, -2), "left"),
        "Chatbot Arena (human Elo)": (0.66, 0.96, RED, (-12, 4), "right"),
    }
    fig, ax = plt.subplots(figsize=(10.6, 6.4))
    for name, (x, y, color, (dx, dy), ha) in benches.items():
        ax.scatter(x, y, s=200, color=color, edgecolor="white", linewidth=1.2, zorder=3)
        ax.annotate(name, (x, y), textcoords="offset points", xytext=(dx, dy),
                    ha=ha, va="center", fontsize=10.5, color=INK, zorder=4,
                    fontweight="bold")
    ax.set_xlim(0, 1.05)
    ax.set_ylim(0, 1.08)
    ax.set_xlabel("← memorised knowledge          multi-step reasoning →", fontsize=11)
    ax.set_ylabel("← fixed multiple-choice          open-ended generation →", fontsize=11)
    ax.set_title("What each benchmark stresses (illustrative map, not a difficulty ranking)")
    ax.set_xticks([])
    ax.set_yticks([])
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    for side in ("left", "bottom"):
        ax.spines[side].set_color(SLATE)
    ax.title.set_color(INK)
    ax.xaxis.label.set_color(INK)
    ax.yaxis.label.set_color(INK)
    fig.tight_layout()
    _save(fig, "eval_benchmark_landscape.png")


def fig_contamination() -> None:
    """How test-set contamination inflates a reported score above true capability; illustrative."""
    # Two models with the SAME true capability; one had x% of the test set leak into training,
    # so it 'recalls' those items at ~100% instead of its true rate -> an inflated headline.
    true_acc = 0.55
    contam_fractions = np.linspace(0.0, 0.5, 11)
    # reported = (1 - f) * true_acc + f * 1.0  (leaked items answered ~perfectly)
    reported = (1 - contam_fractions) * true_acc + contam_fractions * 1.0
    fig, ax = plt.subplots(figsize=(9.4, 5.0))
    ax.plot(contam_fractions * 100, reported * 100, "-o", color=RED, linewidth=2.8,
            markersize=6, zorder=4, label="reported score (with leaked items)")
    ax.axhline(true_acc * 100, color=GREEN, linestyle="--", linewidth=2.2, zorder=3,
               label=f"true capability ({true_acc:.0%})")
    ax.fill_between(contam_fractions * 100, true_acc * 100, reported * 100,
                    color=RED, alpha=0.12, zorder=1)
    ax.set_xlabel("% of the test set leaked into training (contamination)")
    ax.set_ylabel("benchmark score (%)")
    ax.set_ylim(50, 100)
    ax.set_title("Contamination inflates the headline score above true ability")
    ax.legend(frameon=False, fontsize=10, loc="upper left")
    _style_axis(ax)
    fig.tight_layout()
    _save(fig, "eval_contamination.png")


def fig_goodhart_saturation() -> None:
    """Benchmarks saturate: as models improve, scores pile up near the ceiling; illustrative."""
    # Illustrative trajectories of several benchmarks toward their 100% ceiling over model
    # generations -- the qualitative 'saturation' story behind retiring old benchmarks.
    generations = np.arange(0, 8)
    curves = {
        "older benchmark\n(saturated)": (0.40, 0.95, BLUE),
        "mid benchmark": (0.25, 0.55, GREEN),
        "newer/harder\nbenchmark": (0.10, 0.30, PURPLE),
    }
    fig, ax = plt.subplots(figsize=(9.4, 5.0))
    for name, (start, rate, color) in curves.items():
        # a saturating curve: score -> 1 as generations grow, at speed `rate`
        scores = 1 - (1 - start) * np.exp(-rate * generations)
        ax.plot(generations, scores * 100, "-o", color=color, linewidth=2.6, markersize=6,
                zorder=3, label=name)
    ax.axhline(100, color=SLATE, linestyle=":", linewidth=1.6, zorder=2, label="ceiling (100%)")
    ax.axhspan(92, 100, color=AMBER, alpha=0.10, zorder=1)
    ax.annotate("saturation zone:\nbenchmark stops\ndiscriminating", (5.2, 88),
                fontsize=9, color=INK)
    ax.set_xlabel("model generation →")
    ax.set_ylabel("benchmark score (%)")
    ax.set_ylim(0, 105)
    ax.set_title("Benchmarks saturate: once everyone scores ~100%, the test loses signal")
    ax.legend(frameon=False, fontsize=9, loc="lower right")
    _style_axis(ax)
    fig.tight_layout()
    _save(fig, "eval_goodhart_saturation.png")


def main() -> None:
    if not _TORCH_OK:
        raise SystemExit("torch is required for the perplexity figure; install it first.")
    fig_perplexity_intuition()
    fig_passk_curve()
    fig_passk_bias()
    fig_ece_reliability()
    fig_elo_convergence()
    fig_elo_logistic()
    fig_judge_position_bias()
    fig_benchmark_landscape()
    fig_contamination()
    fig_goodhart_saturation()
    print("all figures written to", OUT_DIR)


if __name__ == "__main__":
    main()
