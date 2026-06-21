"""Loss-functions concept-page diagrams (muted palette, parallel matplotlib scale).

Two figures for 05. Deep_Learning/concepts/04-Loss-Functions.md:
  1. loss_regression.png -- MSE vs MAE vs Huber as a function of the error: MSE's
     quadratic blow-up (outlier-sensitive) vs MAE's linear vs Huber's hybrid.
  2. loss_crossentropy.png -- the cross-entropy penalty -log(p) on the TRUE class:
     near-zero when confident-correct, exploding when confident-wrong.
"""
import os, matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

OUT = os.path.join(os.path.dirname(__file__), "..", "05. Deep_Learning", "concepts", "images")
OUT = os.path.abspath(OUT)
os.makedirs(OUT, exist_ok=True)

BLUE, PURPLE, GREEN, RED, SLATE, AMBER, NAVY = (
    "#3A6B96", "#5D4A8A", "#2E7A5A", "#8B3B4A", "#4A5B6E", "#7A6528", "#2A5B80")
plt.rcParams.update({"font.size": 11, "font.family": "DejaVu Sans"})


def _despine(ax):
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)


def loss_regression():
    e = np.linspace(-3, 3, 400)
    mse = e ** 2
    mae = np.abs(e)
    delta = 1.0
    huber = np.where(np.abs(e) <= delta, 0.5 * e ** 2, delta * (np.abs(e) - 0.5 * delta))
    fig, ax = plt.subplots(figsize=(8.4, 5.0))
    ax.plot(e, mse, color=RED, lw=2.4, label="MSE  $e^2$  (quadratic — outlier-sensitive)")
    ax.plot(e, mae, color=BLUE, lw=2.4, label="MAE  $|e|$  (linear — robust, kink at 0)")
    ax.plot(e, huber, color=GREEN, lw=2.6, ls="--", label="Huber ($\\delta{=}1$)  (quadratic near 0, linear far)")
    ax.axvline(0, color=SLATE, ls=":", lw=1)
    ax.set_xlabel("prediction error  e = ŷ − y"); ax.set_ylabel("loss")
    ax.set_title("Regression losses: how hard each penalizes a given error",
                 fontsize=13.5, fontweight="bold")
    ax.set_ylim(-0.2, 6); ax.legend(frameon=False, fontsize=9.5, loc="upper center"); _despine(ax)
    fig.tight_layout(); fig.savefig(f"{OUT}/loss_regression.png", dpi=150, bbox_inches="tight")
    plt.close(fig); print("wrote loss_regression.png")


def loss_crossentropy():
    p = np.linspace(0.001, 1.0, 400)
    ce = -np.log(p)
    fig, ax = plt.subplots(figsize=(8.4, 5.0))
    ax.plot(p, ce, color=PURPLE, lw=2.6)
    ax.fill_between(p, ce, 0, where=(p < 0.5), color=RED, alpha=0.12)
    ax.fill_between(p, ce, 0, where=(p >= 0.5), color=GREEN, alpha=0.12)
    ax.annotate("confident & WRONG\n→ huge penalty", xy=(0.05, -np.log(0.05)), xytext=(0.22, 3.6),
                fontsize=10, color=RED, fontweight="bold",
                arrowprops=dict(arrowstyle="->", color=RED))
    ax.annotate("confident & right\n→ ~0 loss", xy=(0.95, -np.log(0.95)), xytext=(0.55, 1.1),
                fontsize=10, color=GREEN, fontweight="bold",
                arrowprops=dict(arrowstyle="->", color=GREEN))
    ax.scatter([0.1, 0.5, 0.9], [-np.log(0.1), -np.log(0.5), -np.log(0.9)],
               color=NAVY, zorder=5, s=45)
    for pv in (0.1, 0.5, 0.9):
        ax.annotate(f"p={pv}: {-np.log(pv):.2f}", (pv, -np.log(pv)),
                    textcoords="offset points", xytext=(8, 6), fontsize=8.5, color=NAVY)
    ax.set_xlabel("probability the model assigned to the TRUE class,  p"); ax.set_ylabel("cross-entropy loss  −log p")
    ax.set_title("Cross-entropy punishes confident-and-wrong, rewards confident-and-right",
                 fontsize=12.5, fontweight="bold")
    ax.set_ylim(-0.2, 5); _despine(ax)
    fig.tight_layout(); fig.savefig(f"{OUT}/loss_crossentropy.png", dpi=150, bbox_inches="tight")
    plt.close(fig); print("wrote loss_crossentropy.png")


if __name__ == "__main__":
    loss_regression()
    loss_crossentropy()
    print("OUT:", OUT)
