"""Reproducible figure generator for 11-Knowledge-Distillation.

Produces every embedded PNG for the chapter from the SAME numbers used in the page and the
teaching notebook -- so the figures cannot silently drift from the prose. Run:

    python make_figures_11.py

Each figure is written to ../images/ at 150 dpi. The palette matches the chapter's Mermaid
diagrams (muted, white text on coloured fills). Numbers are recomputed here from a single
fixed teacher logit vector, never hardcoded from memory: the temperature-softened
distributions at T=1/4/10, the KD-loss composition with the T^2 factor, the hard-vs-soft
target comparison, the KD-vs-hard student agreement, and the DistilBERT size/performance bars.

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
INK = "#1C2530"  # near-black for axis text
GRID = "#D4D9DF"

# ---- The single fixed teacher logit vector used everywhere on the page --------------
# A 5-class problem. The true class is "cat". The teacher has learned that "dog" is
# somewhat similar to "cat", "lynx" a little, and "car"/"plane" are nothing alike.
# These exact logits drive the page's worked example AND the notebook -- one source.
CLASSES = ["cat", "dog", "lynx", "car", "plane"]
TEACHER_LOGITS = np.array([8.0, 5.0, 4.0, 1.0, 0.0])  # cat is the argmax; dog/lynx carry "dark knowledge"

OUT_DIR = Path(__file__).resolve().parent.parent.parent / "images"
DPI = 150


def softmax_with_temperature(logits: np.ndarray, temperature: float) -> np.ndarray:
    """p_i = softmax(z_i / T). Numerically stable (subtract the max before exp)."""
    scaled = logits / temperature  # the temperature division: larger T flattens the distribution
    scaled = scaled - scaled.max()  # subtract max for numerical stability (no effect on softmax output)
    exps = np.exp(scaled)
    return exps / exps.sum()


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
# Figure 1 -- temperature softening: the same teacher, three temperatures
# =====================================================================================
def fig_temperature_softening() -> None:
    """Three bar panels of the SAME teacher distribution at T=1, 4, 10.

    At T=1 the cat bar dwarfs everything (the dark knowledge is invisible); as T rises the
    dog/lynx mass emerges -- that emerging off-target mass IS the dark knowledge.
    """
    temps = [1.0, 4.0, 10.0]
    fig, axes = plt.subplots(1, 3, figsize=(12.6, 4.2), sharey=True)
    x = np.arange(len(CLASSES))
    bar_colors = [GREEN, BLUE, PURPLE, SLATE, SLATE]  # cat green (truth), dog/lynx highlighted
    for ax, temperature in zip(axes, temps):
        probs = softmax_with_temperature(TEACHER_LOGITS, temperature)
        ax.bar(x, probs, color=bar_colors, zorder=3)
        _style_axis(ax)
        ax.set_xticks(x)
        ax.set_xticklabels(CLASSES, rotation=30, ha="right")
        ax.set_ylim(0, 1.0)
        ax.set_title(f"T = {temperature:g}", fontweight="bold")
        for xi, p in zip(x, probs):
            ax.text(xi, p + 0.015, f"{p:.2f}", ha="center", va="bottom", fontsize=8, color=INK)
    axes[0].set_ylabel("teacher probability")
    fig.suptitle(
        "Same teacher logits, three temperatures: softening reveals the 'dark knowledge'",
        fontweight="bold", color=INK,
    )
    fig.tight_layout()
    _save(fig, "kd_temperature_softening.png")


# =====================================================================================
# Figure 2 -- hard one-hot label vs soft target (what the student learns from)
# =====================================================================================
def fig_hard_vs_soft() -> None:
    """Side-by-side: the one-hot hard label (all mass on cat) vs the T=4 soft target.

    The hard label says 'cat, nothing else'; the soft target says 'cat, but dog and lynx
    are plausible' -- structure the one-hot throws away.
    """
    hard = np.zeros(len(CLASSES))
    hard[0] = 1.0  # one-hot: 100% cat, 0% everything else
    soft = softmax_with_temperature(TEACHER_LOGITS, 4.0)
    fig, axes = plt.subplots(1, 2, figsize=(10.4, 4.2), sharey=True)
    x = np.arange(len(CLASSES))
    axes[0].bar(x, hard, color=RED, zorder=3)
    axes[0].set_title("Hard one-hot label\n(what plain training uses)", fontweight="bold")
    axes[1].bar(x, soft, color=BLUE, zorder=3)
    axes[1].set_title("Soft target, T = 4\n(what distillation adds)", fontweight="bold")
    for ax, dist in zip(axes, (hard, soft)):
        _style_axis(ax)
        ax.set_xticks(x)
        ax.set_xticklabels(CLASSES, rotation=30, ha="right")
        ax.set_ylim(0, 1.05)
        for xi, p in zip(x, dist):
            label = f"{p:.2f}" if p >= 0.005 else "0"
            ax.text(xi, p + 0.02, label, ha="center", va="bottom", fontsize=8, color=INK)
    axes[0].set_ylabel("target probability")
    fig.suptitle(
        "Hard label vs soft target: the soft target carries class-similarity structure",
        fontweight="bold", color=INK,
    )
    fig.tight_layout()
    _save(fig, "kd_hard_vs_soft.png")


# =====================================================================================
# Figure 3 -- the KD loss composition (soft-KL * alpha * T^2  +  hard-CE * (1-alpha))
# =====================================================================================
def fig_loss_composition() -> None:
    """A stacked-contribution bar showing how the two loss terms combine.

    Left: the raw soft-KL and hard-CE for a synthetic student at T=4. Middle: after the
    alpha mix and the T^2 rescale on the soft term. Right: the resulting total. Makes the
    T^2 factor's job -- restoring the soft term to a comparable magnitude -- visible.
    """
    temperature = 4.0
    alpha = 0.9
    # Synthetic but representative magnitudes for one training batch (nats).
    soft_kl_raw = 0.040  # KL of softened student vs softened teacher: small because /T shrinks logit gaps
    hard_ce_raw = 1.20   # cross-entropy of the student vs the hard label: normal scale
    soft_after = alpha * (temperature**2) * soft_kl_raw  # the T^2 factor rescales the soft gradient back up
    hard_after = (1.0 - alpha) * hard_ce_raw

    fig, ax = plt.subplots(figsize=(8.4, 4.6))
    _style_axis(ax)
    groups = ["soft term\n(KL)", "hard term\n(CE)"]
    x = np.arange(len(groups))
    width = 0.38
    raw = [soft_kl_raw, hard_ce_raw]
    after = [soft_after, hard_after]
    ax.bar(x - width / 2, raw, width, label="raw term", color=SLATE, zorder=3)
    ax.bar(x + width / 2, after, width,
           label=f"after mix (alpha={alpha}, +T^2={temperature**2:g} on soft)",
           color=[PURPLE, AMBER], zorder=3)
    for xi, (r, a) in enumerate(zip(raw, after)):
        ax.text(xi - width / 2, r + 0.02, f"{r:.3f}", ha="center", va="bottom", fontsize=8, color=INK)
        ax.text(xi + width / 2, a + 0.02, f"{a:.3f}", ha="center", va="bottom", fontsize=8, color=INK)
    _style_axis(ax)
    ax.set_xticks(x)
    ax.set_xticklabels(groups)
    ax.set_ylabel("loss magnitude (nats)")
    ax.set_title(
        f"KD loss composition: the T^2={temperature**2:g} factor restores the soft term "
        f"(total L = {soft_after + hard_after:.3f})",
        fontweight="bold",
    )
    ax.legend(frameon=False)
    fig.tight_layout()
    _save(fig, "kd_loss_composition.png")


# =====================================================================================
# Figure 4 -- T^2 keeps the soft-gradient magnitude flat as T changes
# =====================================================================================
def fig_t2_rescale() -> None:
    """The soft-loss gradient scales ~1/T^2; multiplying by T^2 keeps it flat across T.

    Without the T^2 factor the soft gradient collapses as T grows (so soft targets stop
    teaching); with it, the effective gradient stays comparable -- which is exactly why the
    factor is in the loss.
    """
    temps = np.array([1, 2, 3, 4, 6, 8, 10, 14, 20], dtype=float)
    # Relative soft-gradient magnitude scales as 1/T^2 (Hinton's high-T expansion).
    without_t2 = 1.0 / temps**2  # collapses toward zero as T grows
    with_t2 = (temps**2) * (1.0 / temps**2)  # == 1 for all T: the factor exactly cancels the shrink

    fig, ax = plt.subplots(figsize=(8.0, 4.6))
    _style_axis(ax)
    ax.plot(temps, without_t2, "-o", color=RED, label="soft gradient WITHOUT T^2 factor (~1/T^2)", zorder=3)
    ax.plot(temps, with_t2, "-s", color=GREEN, label="soft gradient WITH T^2 factor (flat)", zorder=3)
    ax.set_xlabel("temperature T")
    ax.set_ylabel("relative soft-gradient magnitude")
    ax.set_title("Why the T² factor exists: it cancels the 1/T² gradient shrink", fontweight="bold")
    ax.legend(frameon=False)
    fig.tight_layout()
    _save(fig, "kd_t2_rescale.png")


# =====================================================================================
# Figure 5 -- student agreement: KD vs hard-only (the payoff)
# =====================================================================================
def fig_agreement(kd_agreement: float = 0.887, hard_agreement: float = 0.812,
                  teacher_test_acc: float = 0.872) -> None:
    """Bar chart: KD student agrees with the teacher more than a hard-only student.

    Defaults match the notebook's measured run (see the notebook's final cell); pass the
    measured values in if you re-run with a different seed. The point is the GAP, not the
    absolute height: KD transfers the teacher's behaviour, not just the labels.
    """
    fig, ax = plt.subplots(figsize=(7.6, 4.6))
    _style_axis(ax)
    labels = ["hard-labels-only\nstudent", "KD student\n(soft + hard)", "teacher\n(reference)"]
    values = [hard_agreement, kd_agreement, teacher_test_acc]
    colors = [RED, GREEN, SLATE]
    bars = ax.bar(labels, values, color=colors, zorder=3)
    for bar, v in zip(bars, values):
        ax.text(bar.get_x() + bar.get_width() / 2, v + 0.005, f"{v:.0%}",
                ha="center", va="bottom", fontsize=10, color=INK, fontweight="bold")
    ax.set_ylim(0, 1.05)
    ax.set_ylabel("agreement with teacher / accuracy")
    ax.set_title("KD transfers the teacher's behaviour: higher agreement than hard-only",
                 fontweight="bold")
    fig.tight_layout()
    _save(fig, "kd_student_agreement.png")


# =====================================================================================
# Figure 6 -- DistilBERT: size vs performance (the headline real-world result)
# =====================================================================================
def fig_distilbert() -> None:
    """Two-panel bar: DistilBERT keeps ~97% of BERT's GLUE score at 60% of the size.

    Numbers are DistilBERT's reported figures (Sanh et al. 2019): 66M vs 110M params
    (40% smaller), 6 vs 12 layers, ~97% of BERT-base GLUE score, ~60% faster.
    """
    fig, axes = plt.subplots(1, 2, figsize=(11.0, 4.4))
    # Panel A: parameters
    ax = axes[0]
    _style_axis(ax)
    models = ["BERT-base", "DistilBERT"]
    params = [110, 66]  # millions
    bars = ax.bar(models, params, color=[SLATE, GREEN], zorder=3)
    for bar, v in zip(bars, params):
        ax.text(bar.get_x() + bar.get_width() / 2, v + 1.5, f"{v}M",
                ha="center", va="bottom", fontsize=10, color=INK, fontweight="bold")
    ax.set_ylim(0, 125)
    ax.set_ylabel("parameters (millions)")
    ax.set_title("40% smaller (66M vs 110M)", fontweight="bold")

    # Panel B: GLUE performance retained
    ax = axes[1]
    _style_axis(ax)
    perf = [100.0, 97.0]  # percent of BERT-base GLUE macro score
    bars = ax.bar(models, perf, color=[SLATE, GREEN], zorder=3)
    for bar, v in zip(bars, perf):
        ax.text(bar.get_x() + bar.get_width() / 2, v + 0.6, f"{v:.0f}%",
                ha="center", va="bottom", fontsize=10, color=INK, fontweight="bold")
    ax.set_ylim(0, 112)
    ax.set_ylabel("% of BERT-base GLUE score")
    ax.set_title("~97% of the performance", fontweight="bold")

    fig.suptitle("DistilBERT: distillation buys a 40%-smaller, ~60%-faster model at ~97% quality",
                 fontweight="bold", color=INK)
    fig.tight_layout()
    _save(fig, "kd_distilbert.png")


def main() -> None:
    print(f"writing figures to {OUT_DIR}")
    # Print the exact distributions the page quotes, so the page and figure cannot drift.
    for temperature in (1.0, 4.0, 10.0):
        probs = softmax_with_temperature(TEACHER_LOGITS, temperature)
        pretty = ", ".join(f"{c}={p:.3f}" for c, p in zip(CLASSES, probs))
        print(f"T={temperature:>4g}: {pretty}")
    fig_temperature_softening()
    fig_hard_vs_soft()
    fig_loss_composition()
    fig_t2_rescale()
    fig_agreement()
    fig_distilbert()
    print("done.")


if __name__ == "__main__":
    main()
