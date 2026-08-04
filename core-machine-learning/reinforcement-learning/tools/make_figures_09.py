"""Figure generator for 09-Policy-Gradients-REINFORCE — every figure is driven by the REAL measured run.

One measured experiment (``run_experiment`` in ``reinforce.py``) drives every figure below, so nothing
quantitative is hand-typed: the real CartPole learning curves (with and without a baseline, across seeds), the
measured gradient-variance reduction, the score-function proof (REINFORCE estimate vs analytic vs finite-diff and
its Monte-Carlo convergence), and the trained-vs-untrained policy behaviour all come from the same executed
pipeline the chapter and notebook use.

Writes muted-palette PNGs to the shared domain image dir (../images/) with prefix ``rl09_``:

  rl09_learning_curve.png    -- the headline with-baseline REINFORCE learning curve on real CartPole-v1 climbing
                                to the solved threshold (475) — real convergence, measured.
  rl09_baseline_variance.png -- the KEY practical lesson: (a) with-baseline vs no-baseline learning curves
                                (mean +/- std across seeds); (b) the measured order-of-magnitude gradient-variance
                                reduction from subtracting the mean-return baseline.
  rl09_score_function.png    -- the PROOF: (a) the REINFORCE score-function gradient == the analytic gradient ==
                                a finite-difference of J on a tractable bandit; (b) the MC estimate's error
                                shrinking ~1/sqrt(N).
  rl09_policy_behavior.png   -- what the learned policy DOES: the pole angle stays bounded across a full 500-step
                                episode under the trained policy, versus an untrained policy that topples in a
                                few steps.

    python make_figures_09.py

Verified on Python 3.12 / matplotlib 3.10 / numpy 2.4 / torch 2.12 / gymnasium 1.3.
"""

from __future__ import annotations

import sys
from pathlib import Path

# This generator lives in ``08. Reinforcement_Learning/tools/``; the chapter module it demonstrates stays in that
# chapter's ``code/`` folder. Put that folder on sys.path so the ``reinforce`` import resolves.
_CHAPTER_CODE = Path(__file__).resolve().parent.parent / "09-Policy-Gradients-REINFORCE" / "code"
sys.path.insert(0, str(_CHAPTER_CODE))

import matplotlib  # noqa: E402  (imported after the sys.path insert above, which must run first)

matplotlib.use("Agg")  # headless: render straight to PNG, never open a window
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

from reinforce import (  # noqa: E402
    PolicyNetwork,
    collect_rollout,
    make_env,
    run_experiment,
)

# ---- Palette (matches the chapter's muted Mermaid classDefs) ------------------------------------
BLUE = "#3A6B96"  # data / reference lines
PURPLE = "#5D4A8A"  # process / loss
GREEN = "#2E7A5A"  # good / with-baseline / solved
RED = "#8B3B4A"  # penalty / no-baseline / failure
AMBER = "#7A6528"  # highlight
SLATE = "#4A5B6E"  # neutral / raw traces
INK = "#1C2530"  # labels
GRID = "#D4D9DF"  # gridlines

OUT_DIR = Path(__file__).resolve().parent.parent / "images"
DPI = 120
IMG_PREFIX = "rl09_"


def _style_axis(ax: plt.Axes) -> None:
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


def _stack_curves(results: list, window: int) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Rolling-mean each seed's return curve, then return (x, mean-over-seeds, std-over-seeds) on the common length."""
    rolled = [r.rolling_mean for r in results]
    length = min(len(c) for c in rolled)
    stacked = np.stack([c[:length] for c in rolled])
    x = np.arange(length) + window
    return x, stacked.mean(axis=0), stacked.std(axis=0)


# ================================================================================================
# Figure 1: the headline learning curve (with baseline) climbing to the solved threshold
# ================================================================================================


def fig_learning_curve(exp) -> None:
    result = exp.with_baseline[0]  # the headline seed
    returns = result.episode_returns
    rolling = result.rolling_mean
    window = len(returns) - len(rolling) + 1

    fig, ax = plt.subplots(figsize=(8.6, 4.7))
    _style_axis(ax)
    ax.plot(np.arange(len(returns)), returns, color=SLATE, alpha=0.20, linewidth=0.7,
            label="per-episode return (raw)")
    ax.plot(np.arange(len(rolling)) + window, rolling, color=GREEN, linewidth=2.3,
            label=f"rolling mean ({window} episodes)")
    ax.axhline(exp.reward_threshold, color=BLUE, linestyle="--", linewidth=1.5,
               label=f"solved threshold = {exp.reward_threshold:.0f}")
    ax.axhline(exp.max_return, color=SLATE, linestyle=":", linewidth=1.1,
               label=f"environment cap = {exp.max_return:.0f}")
    if result.solved_episode is not None:
        ax.axvline(result.solved_episode, color=AMBER, linestyle="-.", linewidth=1.3,
                   label=f"first solved @ episode {result.solved_episode}")
    ax.set_xlabel("training episode")
    ax.set_ylabel("episode return (steps balanced)")
    ax.set_ylim(0, exp.max_return * 1.05)
    ax.legend(fontsize=8.5, frameon=False, loc="lower right")
    ax.set_title(f"REINFORCE with a baseline solves real {exp.env_label} — a policy trained from scratch climbs "
                 f"to the solved threshold", fontsize=10.3, color=INK)
    fig.tight_layout()
    _save(fig, f"{IMG_PREFIX}learning_curve.png")


# ================================================================================================
# Figure 2: the baseline lesson — with vs without baseline curves + the gradient-variance reduction
# ================================================================================================


def fig_baseline_variance(exp) -> None:
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(11.4, 4.7))
    window = exp.with_baseline[0].episode_returns.shape[0] - exp.with_baseline[0].rolling_mean.shape[0] + 1

    x_w, mean_w, std_w = _stack_curves(exp.with_baseline, window)
    x_n, mean_n, std_n = _stack_curves(exp.no_baseline, window)

    _style_axis(ax1)
    ax1.plot(x_w, mean_w, color=GREEN, linewidth=2.3, label="with mean baseline")
    ax1.fill_between(x_w, mean_w - std_w, mean_w + std_w, color=GREEN, alpha=0.15)
    ax1.plot(x_n, mean_n, color=RED, linewidth=2.3, label="no baseline (plain REINFORCE)")
    ax1.fill_between(x_n, mean_n - std_n, mean_n + std_n, color=RED, alpha=0.13)
    ax1.axhline(exp.reward_threshold, color=BLUE, linestyle="--", linewidth=1.4,
                label=f"solved = {exp.reward_threshold:.0f}")
    ax1.set_xlabel("training episode")
    ax1.set_ylabel("rolling-mean return (mean ± std over seeds)")
    ax1.set_ylim(0, exp.max_return * 1.05)
    ax1.legend(fontsize=8.5, frameon=False, loc="upper left")
    ax1.set_title(f"(a) the baseline speeds and steadies learning\n(mean ± std over {len(exp.seeds)} seeds)",
                  fontsize=10, color=INK)

    gv = exp.grad_var
    _style_axis(ax2)
    bars = ax2.bar(["no baseline\n$A_t = G_t$", "mean baseline\n$A_t = G_t - \\bar{G}$"],
                   [gv.var_no_baseline, gv.var_with_baseline], color=[RED, GREEN], width=0.55)
    ax2.set_ylabel("mean per-parameter gradient variance")
    ax2.set_title(f"(b) subtracting the baseline cuts gradient\nvariance {gv.reduction_factor:.1f}× "
                  f"({gv.n_rollouts} rollouts, fixed policy)", fontsize=10, color=INK)
    for bar, val in zip(bars, [gv.var_no_baseline, gv.var_with_baseline]):
        ax2.text(bar.get_x() + bar.get_width() / 2, val, f"{val:.4f}", ha="center", va="bottom",
                 fontsize=9, color=INK, fontweight="bold")
    ax2.set_ylim(0, gv.var_no_baseline * 1.25)

    fig.suptitle("The single most important trick in REINFORCE: a state-only baseline cuts the gradient's "
                 "variance without adding bias — measured", fontsize=10.6, color=INK, y=1.02)
    fig.tight_layout()
    _save(fig, f"{IMG_PREFIX}baseline_variance.png")


# ================================================================================================
# Figure 3: the score-function proof — REINFORCE == analytic == finite-diff, and MC convergence
# ================================================================================================


def fig_score_function(exp) -> None:
    sp = exp.score_proof
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(11.4, 4.6))

    _style_axis(ax1)
    idx = np.arange(len(sp.analytic))
    w = 0.26
    ax1.bar(idx - w, sp.analytic, width=w, color=BLUE, label="analytic ∇J (exact)")
    ax1.bar(idx, sp.reinforce_mc, width=w, color=GREEN,
            label=f"REINFORCE MC (N={sp.n_samples:,})")
    ax1.bar(idx + w, sp.finite_diff, width=w, color=AMBER, label="finite-difference of J")
    ax1.axhline(0, color=SLATE, linewidth=0.8)
    ax1.set_xticks(idx)
    ax1.set_xticklabels([f"∂J/∂θ$_{i}$" for i in idx])
    ax1.set_ylabel("gradient component")
    ax1.legend(fontsize=8.5, frameon=False)
    ax1.set_title(f"(a) three independent gradients agree\nmax|MC−analytic| = {sp.mc_error:.1e}, "
                  f"max|FD−analytic| = {sp.fd_error:.1e}", fontsize=10, color=INK)

    _style_axis(ax2)
    ax2.loglog(sp.convergence_ns, sp.convergence, "o-", color=GREEN, linewidth=2.0, markersize=5,
               label="max|MC − analytic|")
    ref = sp.convergence[0] * np.sqrt(sp.convergence_ns[0]) / np.sqrt(sp.convergence_ns)
    ax2.loglog(sp.convergence_ns, ref, "--", color=SLATE, linewidth=1.3, label=r"$\propto 1/\sqrt{N}$ reference")
    ax2.set_xlabel("Monte-Carlo samples N")
    ax2.set_ylabel("max abs error vs analytic gradient")
    ax2.legend(fontsize=8.5, frameon=False)
    ax2.set_title("(b) the estimator is unbiased: error shrinks\nlike 1/√N as samples grow",
                  fontsize=10, color=INK)

    fig.suptitle("The policy-gradient theorem, verified end to end: the score-function estimator equals the exact "
                 "gradient (the 'real thing' proof)", fontsize=10.5, color=INK, y=1.02)
    fig.tight_layout()
    _save(fig, f"{IMG_PREFIX}score_function.png")


# ================================================================================================
# Figure 4: what the learned policy does — pole angle stays bounded (trained) vs topples (untrained)
# ================================================================================================


def fig_policy_behavior(exp) -> None:
    spec = make_env()
    trained = exp.with_baseline[0].policy
    untrained = PolicyNetwork(spec.n_obs, spec.n_actions)  # fresh, random weights

    roll_trained = collect_rollout(trained, spec.env, seed=99)
    roll_untrained = collect_rollout(untrained, make_env().env, seed=99)

    fig, ax = plt.subplots(figsize=(8.6, 4.5))
    _style_axis(ax)
    deg = 180.0 / np.pi
    ax.plot(np.array(roll_untrained.pole_angles) * deg, color=RED, linewidth=2.0,
            label=f"untrained policy (topples in {len(roll_untrained.pole_angles)} steps)")
    ax.plot(np.array(roll_trained.pole_angles) * deg, color=GREEN, linewidth=2.0,
            label=f"trained REINFORCE policy (balances {len(roll_trained.pole_angles)} steps)")
    ax.axhline(12, color=SLATE, linestyle="--", linewidth=1.1, label="±12° failure limit")
    ax.axhline(-12, color=SLATE, linestyle="--", linewidth=1.1)
    ax.set_xlabel("timestep within the episode")
    ax.set_ylabel("pole angle (degrees from vertical)")
    ax.legend(fontsize=8.5, frameon=False, loc="lower left")
    ax.set_title("What the policy learned: keep the pole near vertical. The trained policy holds the angle in a "
                 "tight band for the whole episode", fontsize=10.0, color=INK)
    fig.tight_layout()
    _save(fig, f"{IMG_PREFIX}policy_behavior.png")


def main() -> None:
    exp = run_experiment()
    fig_learning_curve(exp)
    fig_baseline_variance(exp)
    fig_score_function(exp)
    fig_policy_behavior(exp)
    # guard against silent drift: the proven relationships the figures show must hold
    assert exp.score_proof.mc_error < 5e-3
    assert exp.grad_var.reduction_factor > 2.0
    print("all figures written to", OUT_DIR)


if __name__ == "__main__":
    main()
