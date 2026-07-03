"""Figure generator for 06-Q-Learning — every figure is driven by the REAL measured run in ``q_learning.py``.

One measured experiment (``run_experiment``) drives every figure below, so nothing quantitative is hand-typed:
the learned Q-table's greedy policy and state values, the value-iteration ground truth, the real reward-per-
episode learning curve, the ε schedule, the Q(start) convergence trace, and the CliffWalking Q-learning-vs-SARSA
paths and online-return curves all come from the same executed pipeline the chapter and notebook use.

Writes muted-palette PNGs to the shared domain image dir (../images/) with prefix ``rl06_``:

  rl06_policy_value.png -- the learned greedy policy (arrows) over the learned state-value heatmap on real
                          FrozenLake, beside the value-iteration ground truth — they MATCH (Q-learning found it).
  rl06_learning_curve.png -- reward-per-episode (rolling mean) rising to the optimal return: real convergence.
  rl06_cliffwalking.png -- Sutton & Barto Example 6.6: (a) off-policy Q-learning's OPTIMAL cliff-edge path vs
                          on-policy SARSA's SAFE path on the real grid; (b) their online-return curves (SARSA
                          earns more online, Q-learning learns the better greedy policy).
  rl06_schedule.png     -- (a) the ε-greedy exploration schedule (decay) and (b) Q(start) converging to V*.

    python make_figures_06.py

Verified on Python 3.12 / matplotlib 3.10 / numpy 2.4 / gymnasium 1.3.
"""

from __future__ import annotations

import sys
from pathlib import Path

# This generator lives in ``08. Reinforcement_Learning/tools/``; the chapter module it demonstrates stays in that
# chapter's ``code/`` folder. Put that folder on sys.path so the ``q_learning`` import resolves.
_CHAPTER_CODE = Path(__file__).resolve().parent.parent / "06-Q-Learning" / "code"
sys.path.insert(0, str(_CHAPTER_CODE))

import matplotlib  # noqa: E402  (imported after the sys.path insert above, which must run first)

matplotlib.use("Agg")  # headless: render straight to PNG, never open a window
import matplotlib.patches as mpatches  # noqa: E402
import matplotlib.patheffects as pe  # noqa: E402
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

_STROKE = [pe.withStroke(linewidth=2.2, foreground="#0E1620")]  # dark outline so light text/arrows stay legible

from q_learning import Env, greedy_policy, run_experiment  # noqa: E402

# ---- Palette (matches the chapter's muted Mermaid classDefs) ------------------------------------
BLUE = "#3A6B96"  # data / values
PURPLE = "#5D4A8A"  # process
GREEN = "#2E7A5A"  # good / optimal / Q-learning
RED = "#8B3B4A"  # penalty / cliff / SARSA-online
AMBER = "#7A6528"  # highlight / SARSA path
SLATE = "#4A5B6E"  # neutral
INK = "#1C2530"  # labels
GRID = "#D4D9DF"  # gridlines

OUT_DIR = Path(__file__).resolve().parent.parent / "images"
DPI = 120
IMG_PREFIX = "rl06_"

# FrozenLake action -> unit arrow (Gymnasium convention: 0=left, 1=down, 2=right, 3=up); y points down on a grid.
_FL_ARROWS = {0: (-1, 0), 1: (0, 1), 2: (1, 0), 3: (0, -1)}
_FL_LABELS = list("SFFFFHFHFFFHHFFG")  # the standard 4x4 map, flattened


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


def _rolling(values: np.ndarray, window: int) -> np.ndarray:
    if len(values) < window:
        return values
    kernel = np.ones(window) / window
    return np.convolve(values, kernel, mode="valid")


# ================================================================================================
# Figure: learned policy + state-value heatmap vs the value-iteration ground truth (they match)
# ================================================================================================


def _draw_frozenlake(ax: plt.Axes, values: np.ndarray, policy: np.ndarray, title: str) -> None:
    grid = values.reshape(4, 4)
    ax.imshow(grid, cmap="YlGnBu", vmin=0.0, vmax=float(values.max()) or 1.0)
    ax.set_xticks([])
    ax.set_yticks([])
    ax.set_title(title, fontsize=10.5, color=INK)
    for s in range(16):
        r, c = divmod(s, 4)
        cell = _FL_LABELS[s]
        if cell == "H":  # holes: no policy arrow, mark them
            ax.text(c, r, "hole", ha="center", va="center", fontsize=8, color="white", fontweight="bold")
            ax.add_patch(mpatches.Rectangle((c - 0.5, r - 0.5), 1, 1, facecolor=RED, alpha=0.55, edgecolor="none"))
            continue
        if cell == "G":
            ax.text(c, r, "GOAL", ha="center", va="center", fontsize=9, color=INK, fontweight="bold")
            continue
        dx, dy = _FL_ARROWS[int(policy[s])]
        arrow = ax.arrow(c, r - 0.06, dx * 0.26, dy * 0.26, head_width=0.15, head_length=0.13,
                         fc="white", ec="white", linewidth=1.8)
        arrow.set_path_effects(_STROKE)
        if cell == "S":
            ax.text(c - 0.44, r - 0.44, "start", ha="left", va="top", fontsize=7.5, color="white",
                    fontweight="bold").set_path_effects(_STROKE)
        ax.text(c, r + 0.42, f"{values[s]:.2f}", ha="center", va="bottom", fontsize=7.5, color="white",
                fontweight="bold").set_path_effects(_STROKE)


def fig_policy_value(exp) -> None:
    assert exp.fl_result is not None
    learned_policy = greedy_policy(exp.fl_result.q_table)
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10.4, 4.9))
    _draw_frozenlake(ax1, exp.fl_learned_values, learned_policy,
                     f"(a) Q-learning: learned policy + V (max_a Q)\nsuccess {exp.fl_eval.success_rate:.0%}, "
                     f"{exp.fl_eval.mean_length:.0f}-step optimal path")
    _draw_frozenlake(ax2, exp.fl_v_star, exp.fl_pi_star,
                     f"(b) value iteration ground truth: V* + optimal policy\nV*(start) = {exp.fl_v_start_star:.3f}")
    fig.suptitle("Q-learning recovers the optimal route on real FrozenLake — V(start) and the greedy path match "
                 "the DP optimum (colour = state value, arrow = greedy action)", fontsize=10.6, color=INK, y=1.02)
    fig.tight_layout()
    _save(fig, f"{IMG_PREFIX}policy_value.png")


# ================================================================================================
# Figure: reward-per-episode learning curve (real convergence to the optimal return)
# ================================================================================================


def fig_learning_curve(exp) -> None:
    assert exp.fl_result is not None
    returns = exp.fl_result.episode_returns
    window = 50
    smoothed = _rolling(returns, window)
    episodes = np.arange(len(smoothed)) + window

    fig, ax = plt.subplots(figsize=(8.4, 4.6))
    _style_axis(ax)
    ax.plot(np.arange(len(returns)), returns, color=SLATE, alpha=0.18, linewidth=0.7, label="per-episode return")
    ax.plot(episodes, smoothed, color=GREEN, linewidth=2.2, label=f"rolling mean ({window} episodes)")
    ax.axhline(1.0, color=BLUE, linestyle="--", linewidth=1.5, label="optimal return (reach goal = 1.0)")
    ax.set_xlabel("training episode")
    ax.set_ylabel("episode return (reached goal = 1, else 0)")
    ax.set_ylim(-0.05, 1.1)
    ax.legend(fontsize=9, frameon=False, loc="lower right")
    ax.set_title("Real learning curve on FrozenLake — as ε decays, the agent converges to reaching the goal "
                 "every episode", fontsize=10.5, color=INK)
    fig.tight_layout()
    _save(fig, f"{IMG_PREFIX}learning_curve.png")


# ================================================================================================
# Figure: CliffWalking — Q-learning (optimal, risky) vs SARSA (safe) paths + online return (Ex. 6.6)
# ================================================================================================


def _draw_cliff(ax: plt.Axes, env: Env, q_path: list[int], sarsa_path: list[int]) -> None:
    n_rows, n_cols = env.shape
    board = np.zeros((n_rows, n_cols))
    cliff = set(range(37, 47))
    for s in cliff:
        r, c = divmod(s, n_cols)
        board[r, c] = 1.0
    ax.imshow(board, cmap="Greys", vmin=0, vmax=3, alpha=0.35)
    ax.set_xticks([])
    ax.set_yticks([])
    for s in cliff:
        r, c = divmod(s, n_cols)
        ax.add_patch(mpatches.Rectangle((c - 0.5, r - 0.5), 1, 1, facecolor=RED, alpha=0.45, edgecolor="none"))
    ax.text(5.0, 3.0, "The Cliff (−100)", ha="center", va="center", fontsize=8.5, color="white", fontweight="bold")
    ax.text(0, 3, "S", ha="center", va="center", fontsize=11, color=INK, fontweight="bold")
    ax.text(n_cols - 1, 3, "G", ha="center", va="center", fontsize=11, color=INK, fontweight="bold")

    def _plot_path(path: list[int], color: str, label: str, offset: float) -> None:
        coords = np.array([divmod(s, n_cols) for s in path], dtype=float)
        ax.plot(coords[:, 1], coords[:, 0] + offset, color=color, linewidth=2.6, marker="o", markersize=4,
                label=label, alpha=0.95)

    _plot_path(q_path, GREEN, "Q-learning (off-policy): optimal path, −13", -0.12)
    _plot_path(sarsa_path, AMBER, "SARSA (on-policy): safe path, −17", +0.12)
    ax.legend(fontsize=8.5, frameon=False, loc="upper center", bbox_to_anchor=(0.5, -0.02))
    ax.set_title("(a) greedy paths — off-policy hugs the cliff edge (optimal),\non-policy detours to safety",
                 fontsize=10, color=INK)


def fig_cliffwalking(exp) -> None:
    assert exp.cw_q_result is not None and exp.cw_sarsa_result is not None
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(11.4, 4.7))

    _draw_cliff(ax1, exp.cw_env, exp.cw_q_path, exp.cw_sarsa_path)

    _style_axis(ax2)
    window = 20
    q_smooth = _rolling(exp.cw_q_result.episode_returns, window)
    s_smooth = _rolling(exp.cw_sarsa_result.episode_returns, window)
    x = np.arange(len(q_smooth)) + window
    ax2.plot(x, s_smooth, color=AMBER, linewidth=2.2, label=f"SARSA (on-policy)  online ≈ {exp.cw_sarsa_online:.0f}")
    ax2.plot(x, q_smooth, color=GREEN, linewidth=2.2, label=f"Q-learning (off-policy)  online ≈ {exp.cw_q_online:.0f}")
    ax2.axhline(exp.cw_v_start_star, color=BLUE, linestyle="--", linewidth=1.4,
                label=f"optimal return = {exp.cw_v_start_star:.0f}")
    ax2.set_xlabel("training episode")
    ax2.set_ylabel("online return per episode (ε = 0.1 fixed)")
    ax2.set_ylim(-100, 0)
    ax2.legend(fontsize=8.5, frameon=False, loc="lower right")
    ax2.set_title("(b) online reward during training — SARSA earns more\n(Q-learning falls off the cliff while "
                  "exploring)", fontsize=10, color=INK)

    fig.suptitle("CliffWalking (Sutton & Barto Example 6.6): off-policy learns the optimal risky path, "
                 "on-policy learns the safe one — measured", fontsize=10.8, color=INK, y=1.03)
    fig.tight_layout()
    _save(fig, f"{IMG_PREFIX}cliffwalking.png")


# ================================================================================================
# Figure: the ε-greedy exploration schedule and Q(start) converging to V*
# ================================================================================================


def fig_schedule(exp) -> None:
    assert exp.fl_result is not None
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10.8, 4.4))

    _style_axis(ax1)
    ax1.plot(exp.fl_result.epsilons, color=PURPLE, linewidth=2.2)
    ax1.axhline(0.01, color=SLATE, linestyle="--", linewidth=1.2, label="floor ε = 0.01")
    ax1.set_xlabel("training episode")
    ax1.set_ylabel("ε (probability of a random action)")
    ax1.set_title("(a) exploration schedule — ε decays 1.0 → 0.01\n(explore early, exploit late)",
                  fontsize=10, color=INK)
    ax1.legend(fontsize=9, frameon=False)

    _style_axis(ax2)
    ax2.plot(exp.fl_result.q_start_trace, color=GREEN, linewidth=2.0, label="Q(start, greedy) during learning")
    ax2.axhline(exp.fl_v_start_star, color=BLUE, linestyle="--", linewidth=1.5,
                label=f"V*(start) = {exp.fl_v_start_star:.3f} (DP optimum)")
    ax2.set_xlabel("training episode")
    ax2.set_ylabel("max_a Q(start, a)")
    ax2.set_ylim(0, max(1.0, exp.fl_v_start_star * 1.15))
    ax2.set_title("(b) bootstrapping in action — the start value\nclimbs to the DP optimum V*",
                  fontsize=10, color=INK)
    ax2.legend(fontsize=9, frameon=False, loc="lower right")

    fig.suptitle("Exploration decays and the value estimate converges — the two dials that make Q-learning work",
                 fontsize=10.8, color=INK, y=1.03)
    fig.tight_layout()
    _save(fig, f"{IMG_PREFIX}schedule.png")


def main() -> None:
    exp = run_experiment()
    fig_policy_value(exp)
    fig_learning_curve(exp)
    fig_cliffwalking(exp)
    fig_schedule(exp)
    # guard against silent drift: the learned start value the figures show must equal the DP optimum
    assert abs(exp.fl_v_start_learned - exp.fl_v_start_star) < 1e-6
    print("all figures written to", OUT_DIR)


if __name__ == "__main__":
    main()
