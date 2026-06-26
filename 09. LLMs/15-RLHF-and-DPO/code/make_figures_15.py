"""Reproducible figure generator for 15-RLHF-and-DPO.

Produces every embedded PNG for the chapter from the SAME numbers used on the page and in the
notebook -- so the figures cannot silently drift from the prose. Run:

    python make_figures_15.py

Each figure is written to ../../images/ (the shared chapter image dir) at 150 dpi. The palette
matches the chapter's Mermaid diagrams (muted, white text on coloured fills). The two analytic
curves (Bradley-Terry, DPO margin) are computed inline from the same sigma/-log sigma the page
defines; the over-optimization curve is illustrative (labelled as such on the page); and the
measured DPO-update figure imports ``run_toy_dpo`` from ``rlhf_dpo.py`` so the plotted curve is
literally the run the script asserts on -- one seeded source of truth.

Figures produced:
  rlhf_bradley_terry.png   -- P(chosen>rejected)=sigma(gap) and the per-pair loss -log sigma(gap)
  rlhf_overoptimization.png-- proxy reward keeps rising while true quality peaks then falls (KL leash)
  dpo_margin.png           -- DPO loss -log sigma(margin) and P=sigma(margin) vs the implicit margin
  dpo_update.png           -- the MEASURED toy DPO run from rlhf_dpo.py: chosen up, rejected down

Verified on Python 3.12 / torch 2.x / matplotlib 3.x.
"""

from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")  # headless backend: write files, never open a window
import matplotlib.pyplot as plt
import numpy as np

from rlhf_dpo import BETA, REF_LOGPROB, run_toy_dpo

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

# Figures live in the SHARED chapter images dir (09. LLMs/images/), matching the KV-Cache
# exemplar -- so the page references them as ../images/<name>.png.
OUT_DIR = Path(__file__).resolve().parent.parent.parent / "images"
DPI = 150


def _sigmoid(x: np.ndarray) -> np.ndarray:
    """Numerically stable logistic sigma(x) for the figure curves."""
    return np.where(x >= 0, 1.0 / (1.0 + np.exp(-x)), np.exp(x) / (1.0 + np.exp(x)))


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


def fig_bradley_terry() -> None:
    """P(chosen>rejected) = sigma(reward gap) and the per-pair loss -log sigma(gap)."""
    gap = np.linspace(-6.5, 6.5, 400)
    prob = _sigmoid(gap)  # P(chosen preferred) -- the Bradley-Terry probability
    loss = -np.log(_sigmoid(gap))  # -log sigma(gap) -- the per-pair Bradley-Terry loss

    fig, ax_left = plt.subplots(figsize=(8.4, 4.8))
    ax_left.plot(gap, prob, color=BLUE, linewidth=2.6, label="P(chosen ≻ rejected) = σ(gap)")
    ax_left.set_xlabel("reward gap  r(chosen) − r(rejected)")
    ax_left.set_ylabel("P(chosen ≻ rejected)", color=BLUE)
    ax_left.tick_params(axis="y", colors=BLUE)
    ax_left.axvline(0.0, color=SLATE, linewidth=1.0, linestyle=":", zorder=1)
    _style_axis(ax_left)

    ax_right = ax_left.twinx()
    ax_right.plot(
        gap, loss, color=PURPLE, linewidth=2.4, linestyle="--", label="per-pair loss = −log σ(gap)"
    )
    ax_right.set_ylabel("Bradley-Terry loss  −log σ(gap)", color=PURPLE)
    ax_right.tick_params(axis="y", colors=PURPLE)
    ax_right.spines["top"].set_visible(False)

    # Annotate the two anchor points the page works by hand: gap 0 -> 0.5, gap +3 -> 0.95.
    ax_left.scatter([0.0], [0.5], color=RED, s=55, zorder=4)
    ax_left.annotate(
        "gap 0 → 0.5\n(coin flip, no opinion)",
        xy=(0.0, 0.5), xytext=(1.0, 0.40), color=RED, fontsize=9,
        arrowprops=dict(arrowstyle="->", color=RED, lw=1.1),
    )
    ax_left.scatter([3.0], [_sigmoid(np.array([3.0]))[0]], color=GREEN, s=55, zorder=4)
    ax_left.annotate(
        "gap +3 → 0.95\n(confident, low loss)",
        xy=(3.0, _sigmoid(np.array([3.0]))[0]), xytext=(-1.0, 0.90), color=GREEN, fontsize=9,
        arrowprops=dict(arrowstyle="->", color=GREEN, lw=1.1),
    )

    lines = ax_left.get_lines()[:1] + ax_right.get_lines()[:1]
    ax_left.legend(lines, [ln.get_label() for ln in lines], loc="center left", frameon=False, fontsize=9)
    ax_left.set_title("The Bradley-Terry reward model: a sigmoid of the reward gap", fontweight="bold")
    _save(fig, "rlhf_bradley_terry.png")


def fig_overoptimization() -> None:
    """Illustrative: proxy reward keeps rising while true quality peaks then falls past the leash."""
    kl = np.linspace(0.0, 10.0, 400)
    # Proxy reward: monotone-increasing, saturating (the reward model keeps "improving").
    proxy = 3.6 * (1.0 - np.exp(-kl / 2.6))
    # True quality: rises with the proxy early, then turns over -- Goodhart over-optimization.
    true_quality = 3.6 * (1.0 - np.exp(-kl / 2.6)) - 0.085 * kl**1.55
    sweet_spot = float(kl[np.argmax(true_quality)])  # the peak of true quality = where to stop

    fig, ax = plt.subplots(figsize=(8.4, 4.8))
    ax.plot(kl, proxy, color=RED, linewidth=2.6, label="Proxy reward (what the reward model says)")
    ax.plot(kl, true_quality, color=GREEN, linewidth=2.6, label="True quality (actual human preference)")
    ax.fill_between(
        kl, true_quality, proxy, where=(kl >= sweet_spot), color=RED, alpha=0.10, zorder=0
    )
    ax.axvline(sweet_spot, color=SLATE, linewidth=1.4, linestyle="--", zorder=1)
    peak_q = float(np.max(true_quality))
    ax.scatter([sweet_spot], [peak_q], color=AMBER, s=70, zorder=4, edgecolor=INK, linewidth=0.8)
    ax.annotate(
        "sweet spot:\nstop here (KL leash)",
        xy=(sweet_spot, peak_q), xytext=(sweet_spot + 0.6, peak_q - 0.7),
        color=NAVY, fontsize=10, fontweight="bold",
        arrowprops=dict(arrowstyle="->", color=NAVY, lw=1.1),
    )
    ax.annotate(
        "reward hacking:\nproxy ↑ but quality ↓",
        xy=(8.0, 3.2), xytext=(5.4, 3.35), color=RED, fontsize=10, fontweight="bold",
        arrowprops=dict(arrowstyle="->", color=RED, lw=1.1),
    )
    ax.set_xlabel("KL divergence from the reference model  (how far the policy drifts)")
    ax.set_ylabel("reward")
    ax.set_ylim(0, 3.7)
    ax.legend(loc="lower right", frameon=False, fontsize=9)
    _style_axis(ax)
    ax.set_title("Reward over-optimization: why RLHF needs a KL leash", fontweight="bold")
    _save(fig, "rlhf_overoptimization.png")


def fig_dpo_margin() -> None:
    """DPO loss -log sigma(margin) and P=sigma(margin) as functions of the implicit-reward margin."""
    margin = np.linspace(-4.0, 4.0, 400)
    loss = -np.log(_sigmoid(margin))  # the DPO per-pair loss
    prob = _sigmoid(margin)  # P(chosen ≻ rejected) under the implicit reward

    fig, ax_left = plt.subplots(figsize=(8.4, 4.8))
    ax_left.plot(margin, loss, color=PURPLE, linewidth=2.6, label="DPO loss = −log σ(margin)")
    ax_left.set_xlabel("implicit-reward margin  β·[ (logπ/π_ref)$_{chosen}$ − (logπ/π_ref)$_{rejected}$ ]")
    ax_left.set_ylabel("DPO loss", color=PURPLE)
    ax_left.tick_params(axis="y", colors=PURPLE)
    ax_left.axvline(0.0, color=SLATE, linewidth=1.0, linestyle=":", zorder=1)
    _style_axis(ax_left)

    ax_right = ax_left.twinx()
    ax_right.plot(
        margin, prob, color=GREEN, linewidth=2.4, linestyle="--", label="P(chosen ≻ rejected) = σ(margin)"
    )
    ax_right.set_ylabel("P(chosen ≻ rejected)", color=GREEN)
    ax_right.tick_params(axis="y", colors=GREEN)
    ax_right.spines["top"].set_visible(False)

    ax_left.annotate(
        "init: policy = reference\nmargin 0, loss = log 2",
        xy=(0.0, -np.log(0.5)), xytext=(0.2, 2.4), color=SLATE, fontsize=9,
    )

    lines = ax_left.get_lines()[:1] + ax_right.get_lines()[:1]
    ax_left.legend(lines, [ln.get_label() for ln in lines], loc="upper center", frameon=False, fontsize=9)
    ax_left.set_title("DPO directly raises the preferred response's probability", fontweight="bold")
    _save(fig, "dpo_margin.png")


def fig_dpo_update() -> None:
    """The MEASURED toy DPO run from rlhf_dpo.py: chosen up, rejected down, margin grows, loss falls."""
    history = run_toy_dpo()  # the exact seeded run the script asserts on -- one source of truth
    steps = np.asarray(history["step"])
    chosen = np.asarray(history["chosen"])
    rejected = np.asarray(history["rejected"])
    margin = np.asarray(history["margin"])
    loss = np.asarray(history["loss"])

    fig, (ax_lp, ax_m) = plt.subplots(1, 2, figsize=(12.4, 4.8))

    # Left: the two policy log-probs separating from the shared reference start.
    ax_lp.plot(steps, chosen, color=GREEN, linewidth=2.6, label="log π(chosen)  — rises")
    ax_lp.plot(steps, rejected, color=RED, linewidth=2.6, label="log π(rejected)  — falls")
    ax_lp.axhline(REF_LOGPROB, color=SLATE, linewidth=1.0, linestyle=":", zorder=1)
    ax_lp.annotate(
        f"both start at the\nreference ({REF_LOGPROB:.1f})",
        xy=(2, REF_LOGPROB), xytext=(8, REF_LOGPROB + 1.2), color=SLATE, fontsize=9,
    )
    ax_lp.set_xlabel("DPO training step")
    ax_lp.set_ylabel("log-probability")
    ax_lp.legend(loc="center right", frameon=False, fontsize=9)
    _style_axis(ax_lp)
    ax_lp.set_title("DPO splits chosen from rejected", fontweight="bold")

    # Right: the implicit-reward margin grows while the loss decays.
    ax_m.plot(steps, margin, color=PURPLE, linewidth=2.6, label="implicit-reward margin  β·Δlog(π/π_ref)")
    ax_m.set_xlabel("DPO training step")
    ax_m.set_ylabel("margin", color=PURPLE)
    ax_m.tick_params(axis="y", colors=PURPLE)
    _style_axis(ax_m)

    ax_loss = ax_m.twinx()
    ax_loss.plot(steps, loss, color=AMBER, linewidth=2.4, linestyle="--", label="DPO loss = −log σ(margin)")
    ax_loss.set_ylabel("DPO loss", color=AMBER)
    ax_loss.tick_params(axis="y", colors=AMBER)
    ax_loss.spines["top"].set_visible(False)

    lines = ax_m.get_lines()[:1] + ax_loss.get_lines()[:1]
    ax_m.legend(lines, [ln.get_label() for ln in lines], loc="center right", frameon=False, fontsize=9)
    ax_m.set_title("Margin grows, loss falls", fontweight="bold")

    fig.suptitle(
        f"A measured DPO update: chosen up, rejected down, margin grows  (β={BETA})",
        fontweight="bold", fontsize=13,
    )
    fig.tight_layout(rect=(0, 0, 1, 0.96))
    _save(fig, "dpo_update.png")


def main() -> None:
    fig_bradley_terry()
    fig_overoptimization()
    fig_dpo_margin()
    fig_dpo_update()
    print("all figures written to", OUT_DIR)


if __name__ == "__main__":
    main()
