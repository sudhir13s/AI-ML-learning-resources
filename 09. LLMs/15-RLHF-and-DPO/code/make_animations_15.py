"""Animated (GIF) figure generator for 15-RLHF-and-DPO.

Companion to ``make_figures_15.py``. Where that script renders the STATIC PNGs, this one renders
the animated GIF(s) -- the same measured numbers, brought to life so a reader watches the update
happen step by step rather than reading the final state off a static plot.

    python make_animations_15.py

GIFs are written to ../../images/ (the shared chapter image dir) and use matplotlib's PillowWriter
(no ffmpeg required). The palette, styling, and source data are imported from the existing figure
generator and ``rlhf_dpo.py`` -- so the animation cannot drift from the static figure or the prose:
it is literally the same seeded ``run_toy_dpo`` run, frame by frame.

Animations produced:
  dpo_update.gif -- the measured toy DPO run: log pi(chosen) rises, log pi(rejected) falls (both
                    from the shared reference), while the implicit-reward margin grows and the DPO
                    loss decays. The animated twin of dpo_update.png.

Verified on Python 3.12 / matplotlib 3.x / Pillow (PillowWriter).
"""

from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")  # headless: render frames to a file, never open a window
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.animation import FuncAnimation, PillowWriter

# Reuse the EXACT palette + axis styling from the static generator (one source of truth), and the
# same seeded DPO run the static figure and the page's prose are built from.
from make_figures_15 import AMBER, GREEN, INK, NAVY, PURPLE, RED, SLATE, _style_axis
from rlhf_dpo import BETA, REF_LOGPROB, run_toy_dpo

OUT_DIR = Path(__file__).resolve().parent.parent.parent / "images"
DPI = 95            # keep the GIF a sensible size for a repo asset
FPS = 16
FRAME_STRIDE = 2    # animate every 2nd training step (61 frames) -- smooth but compact
HOLD_FRAMES = 14    # linger on the final state so the end is readable


def make_dpo_update_gif() -> None:
    """Animate the measured toy DPO run: chosen up, rejected down, margin grows, loss falls."""
    history = run_toy_dpo()  # the exact seeded run -- identical to dpo_update.png
    steps = np.asarray(history["step"], dtype=float)
    chosen = np.asarray(history["chosen"], dtype=float)
    rejected = np.asarray(history["rejected"], dtype=float)
    margin = np.asarray(history["margin"], dtype=float)
    loss = np.asarray(history["loss"], dtype=float)
    n = len(steps)

    # Frame schedule: reveal the run in strides, then hold the final frame so it's readable.
    reveal = list(range(1, n + 1, FRAME_STRIDE))
    if reveal[-1] != n:
        reveal.append(n)
    frames = reveal + [n] * HOLD_FRAMES

    fig, (ax_lp, ax_m) = plt.subplots(1, 2, figsize=(11.4, 4.6))

    # ---- Left panel: the two policy log-probs separating from the shared reference ------------
    ax_lp.axhline(REF_LOGPROB, color=SLATE, linewidth=1.0, linestyle=":", zorder=1)
    ax_lp.annotate(
        f"both start at the\nreference ({REF_LOGPROB:.1f})",
        xy=(2, REF_LOGPROB), xytext=(int(steps[-1] * 0.30), REF_LOGPROB + 1.4),
        color=SLATE, fontsize=9,
    )
    (line_ch,) = ax_lp.plot([], [], color=GREEN, linewidth=2.6, label="log π(chosen)  — rises")
    (line_rj,) = ax_lp.plot([], [], color=RED, linewidth=2.6, label="log π(rejected)  — falls")
    (dot_ch,) = ax_lp.plot([], [], "o", color=GREEN, markersize=6, zorder=5)
    (dot_rj,) = ax_lp.plot([], [], "o", color=RED, markersize=6, zorder=5)
    ax_lp.set_xlim(0, steps[-1])
    ax_lp.set_ylim(rejected.min() - 0.6, chosen.max() + 0.6)
    ax_lp.set_xlabel("DPO training step")
    ax_lp.set_ylabel("log-probability")
    ax_lp.legend(loc="center right", frameon=False, fontsize=9)
    _style_axis(ax_lp)
    ax_lp.set_title("DPO splits chosen from rejected", fontweight="bold")

    # ---- Right panel: implicit-reward margin grows (left axis) while DPO loss decays (right) ---
    (line_m,) = ax_m.plot([], [], color=PURPLE, linewidth=2.6,
                          label="implicit-reward margin  β·Δlog(π/π_ref)")
    (dot_m,) = ax_m.plot([], [], "o", color=PURPLE, markersize=6, zorder=5)
    ax_m.set_xlim(0, steps[-1])
    ax_m.set_ylim(0, margin.max() * 1.08)
    ax_m.set_xlabel("DPO training step")
    ax_m.set_ylabel("margin", color=PURPLE)
    ax_m.tick_params(axis="y", colors=PURPLE)
    _style_axis(ax_m)
    ax_m.set_title("Margin grows, loss falls", fontweight="bold")

    ax_loss = ax_m.twinx()
    (line_loss,) = ax_loss.plot([], [], color=AMBER, linewidth=2.4, linestyle="--",
                                label="DPO loss = −log σ(margin)")
    (dot_loss,) = ax_loss.plot([], [], "o", color=AMBER, markersize=6, zorder=5)
    ax_loss.set_ylim(0, loss.max() * 1.08)
    ax_loss.set_ylabel("DPO loss", color=AMBER)
    ax_loss.tick_params(axis="y", colors=AMBER)
    ax_loss.spines["top"].set_visible(False)

    lines = ax_m.get_lines()[:1] + ax_loss.get_lines()[:1]
    ax_m.legend(lines, [ln.get_label() for ln in lines], loc="center right", frameon=False, fontsize=9)

    # A live step counter, top-centre, so the reader can see progress.
    step_text = fig.text(0.5, 0.015, "", ha="center", color=INK, fontsize=10, fontweight="bold")
    title = fig.suptitle(
        f"A measured DPO update, watched happening: chosen up, rejected down  (β={BETA})",
        fontweight="bold", fontsize=13,
    )

    def update(k: int):
        x = steps[:k]
        line_ch.set_data(x, chosen[:k])
        line_rj.set_data(x, rejected[:k])
        line_m.set_data(x, margin[:k])
        line_loss.set_data(x, loss[:k])
        j = k - 1  # leading-point index
        dot_ch.set_data([steps[j]], [chosen[j]])
        dot_rj.set_data([steps[j]], [rejected[j]])
        dot_m.set_data([steps[j]], [margin[j]])
        dot_loss.set_data([steps[j]], [loss[j]])
        step_text.set_text(
            f"step {int(steps[j]):>3}/{int(steps[-1])}    "
            f"margin {margin[j]:.2f}    loss {loss[j]:.3f}"
        )
        return line_ch, line_rj, line_m, line_loss, dot_ch, dot_rj, dot_m, dot_loss, step_text

    fig.tight_layout(rect=(0, 0.05, 1, 0.94))
    anim = FuncAnimation(fig, update, frames=frames, interval=1000 / FPS, blit=False)

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    out = OUT_DIR / "dpo_update.gif"
    anim.save(out, writer=PillowWriter(fps=FPS), dpi=DPI)
    plt.close(fig)
    print(f"wrote {out}")


def make_overoptimization_gif() -> None:
    """Animate reward over-optimization: the policy drifts from the reference; the proxy reward
    keeps rising while true quality peaks at the KL-leash sweet spot and then falls (reward
    hacking). Same illustrative curves as ``rlhf_overoptimization.png``, brought to life."""
    kl = np.linspace(0.0, 10.0, 170)  # the animation grid (policy's KL drift from the reference)
    proxy = 3.6 * (1.0 - np.exp(-kl / 2.6))               # reward model keeps "improving"
    true_q = 3.6 * (1.0 - np.exp(-kl / 2.6)) - 0.085 * kl**1.55  # true quality turns over (Goodhart)
    sweet_i = int(np.argmax(true_q))
    sweet_kl = float(kl[sweet_i])
    peak_q = float(true_q[sweet_i])

    reveal = list(range(2, len(kl) + 1, 2))
    if reveal[-1] != len(kl):
        reveal.append(len(kl))
    # Hold at the sweet spot (the lesson) and again at the end (the danger).
    frames = (
        [k for k in reveal if k <= sweet_i + 1]
        + [sweet_i + 1] * 12
        + [k for k in reveal if k > sweet_i + 1]
        + [len(kl)] * 14
    )

    fig, ax = plt.subplots(figsize=(9.2, 5.0))
    ax.set_xlim(0, 10)
    ax.set_ylim(0, 3.7)
    ax.set_xlabel("KL divergence from the reference model  (how far the policy has drifted)")
    ax.set_ylabel("reward")
    _style_axis(ax)
    ax.set_title("Reward over-optimization: why RLHF needs a KL leash", fontweight="bold")

    (line_proxy,) = ax.plot([], [], color=RED, linewidth=2.6,
                            label="Proxy reward (what the reward model says)")
    (line_true,) = ax.plot([], [], color=GREEN, linewidth=2.6,
                           label="True quality (actual human preference)")
    (dot_proxy,) = ax.plot([], [], "o", color=RED, markersize=7, zorder=6)
    (dot_true,) = ax.plot([], [], "o", color=GREEN, markersize=7, zorder=6)
    leash = ax.axvline(sweet_kl, color=SLATE, linewidth=1.4, linestyle="--", zorder=1, alpha=0.0)
    star = ax.scatter([sweet_kl], [peak_q], color=AMBER, s=90, zorder=7,
                      edgecolor=INK, linewidth=0.8, alpha=0.0)
    ax.legend(loc="lower right", frameon=False, fontsize=9)
    label = ax.text(0.30, 3.45, "", fontsize=10.5, fontweight="bold", color=NAVY)
    fill_holder: list = []  # the reward-hacking shading (recreated each frame)

    def update(k: int):
        line_proxy.set_data(kl[:k], proxy[:k])
        line_true.set_data(kl[:k], true_q[:k])
        j = k - 1
        dot_proxy.set_data([kl[j]], [proxy[j]])
        dot_true.set_data([kl[j]], [true_q[j]])

        for coll in fill_holder:  # clear previous shading
            coll.remove()
        fill_holder.clear()

        if j < sweet_i:
            leash.set_alpha(0.0)
            star.set_alpha(0.0)
            label.set_text("policy drifting from the reference →\nboth proxy and true quality rising")
            label.set_color(GREEN)
        elif j == sweet_i:
            leash.set_alpha(1.0)
            star.set_alpha(1.0)
            label.set_text("KL leash → stop here ✓\ntrue quality is at its peak")
            label.set_color(NAVY)
        else:  # drifted past the leash: reward hacking
            leash.set_alpha(1.0)
            star.set_alpha(1.0)
            coll = ax.fill_between(kl[sweet_i : k], true_q[sweet_i : k], proxy[sweet_i : k],
                                   color=RED, alpha=0.13, zorder=0)
            fill_holder.append(coll)
            label.set_text("no leash → reward hacking\nproxy still ↑ but true quality ↓")
            label.set_color(RED)
        return line_proxy, line_true, dot_proxy, dot_true, leash, star, label

    fig.tight_layout()
    anim = FuncAnimation(fig, update, frames=frames, interval=1000 / FPS, blit=False)
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    out = OUT_DIR / "rlhf_overoptimization.gif"
    anim.save(out, writer=PillowWriter(fps=FPS), dpi=DPI)
    plt.close(fig)
    print(f"wrote {out}")


def main() -> None:
    make_dpo_update_gif()
    make_overoptimization_gif()
    print("all animations written to", OUT_DIR)


if __name__ == "__main__":
    main()
