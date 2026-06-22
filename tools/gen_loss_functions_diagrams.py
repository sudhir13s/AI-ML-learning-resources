"""Loss-functions concept-page diagrams (muted palette, parallel matplotlib scale).

Five MEASURED figures for 05. Deep_Learning/concepts/04-Loss-Functions.md. Every
curve is computed numerically (no hand-drawn shapes), so the picture *is* the math:

  1. loss_regression_family.png -- regression losses vs residual e = ŷ − y:
     MSE parabola vs MAE V vs Huber hybrid vs log-cosh. Annotated with the
     outlier-sensitivity gap at a large residual.
  2. loss_classification_margin.png -- classification losses vs the margin/score
     of the TRUE class: 0–1 (step), hinge, logistic/cross-entropy (softplus form),
     and focal at gamma=2. Shows how each surrogate upper-bounds the 0–1 loss.
  3. loss_focal_downweighting.png -- focal loss −(1−p_t)^γ log p_t vs p_t for
     gamma in {0,1,2,5}; gamma=0 is plain cross-entropy. Annotated with the
     measured down-weighting factor on an easy (p_t=0.9) example.
  4. loss_crossentropy.png -- the cross-entropy penalty −log(p) on the TRUE class:
     near-zero when confident-correct, exploding when confident-wrong.
  5. loss_grad_ce_vs_mse.png -- |dL/dz| of cross-entropy vs MSE for a single
     sigmoid output as a function of the logit z, with the true label = 1.
     CE gradient stays |p−1|; MSE gradient collapses where the model is
     confidently wrong (the saturation that stalls training).

Run:  ~/.uv/envs/ml-py312/bin/python3 tools/gen_loss_functions_diagrams.py
"""
import os
import matplotlib
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


def regression_family():
    """MSE vs MAE vs Huber vs log-cosh, all measured."""
    e = np.linspace(-3, 3, 600)
    mse = e ** 2
    mae = np.abs(e)
    delta = 1.0
    huber = np.where(np.abs(e) <= delta,
                     0.5 * e ** 2,
                     delta * (np.abs(e) - 0.5 * delta))
    logcosh = np.log(np.cosh(e))
    fig, ax = plt.subplots(figsize=(8.6, 5.2))
    ax.plot(e, mse, color=RED, lw=2.6, label=r"MSE  $e^2$  (quadratic — outlier-sensitive)")
    ax.plot(e, mae, color=BLUE, lw=2.4, label=r"MAE  $|e|$  (linear — robust, kink at 0)")
    ax.plot(e, huber, color=GREEN, lw=2.8, ls="--",
            label=r"Huber ($\delta{=}1$)  (quadratic near 0, linear far)")
    ax.plot(e, logcosh, color=AMBER, lw=2.0, ls=":",
            label=r"log-cosh  $\log\cosh e$  (smooth Huber-like)")
    ax.axvline(0, color=SLATE, ls=":", lw=1)
    # measured annotation: gap at e = 3 (placed lower-left, clear of the legend)
    ax.scatter([3, 3], [9, 2.5], color=NAVY, s=40, zorder=6)
    ax.annotate("at e=3:  MSE=9  vs  Huber=2.5\n→ MSE chases the outlier 3.6× harder",
                xy=(3, 9), xytext=(-2.9, 5.6), fontsize=9.5, color=NAVY, fontweight="bold",
                arrowprops=dict(arrowstyle="->", color=NAVY))
    ax.set_xlabel(r"prediction error  $e = \hat y - y$")
    ax.set_ylabel("loss")
    ax.set_title("Regression losses: how hard each penalizes a given residual",
                 fontsize=13.5, fontweight="bold")
    ax.set_ylim(-0.2, 9.8)
    ax.legend(frameon=False, fontsize=9.0, loc="upper center", ncol=1,
              bbox_to_anchor=(0.62, 1.0))
    _despine(ax)
    fig.tight_layout()
    fig.savefig(f"{OUT}/loss_regression_family.png", dpi=150, bbox_inches="tight")
    plt.close(fig)
    print("wrote loss_regression_family.png")


def classification_margin():
    """Surrogate classification losses vs the true-class score m, all measured."""
    m = np.linspace(-3, 3, 600)
    zero_one = (m <= 0).astype(float)                      # step: wrong if m<=0
    hinge = np.maximum(0.0, 1.0 - m)                        # SVM hinge
    logistic = np.log1p(np.exp(-m)) / np.log(2.0)           # log-loss in bits (passes (0,1))
    pt = 1.0 / (1.0 + np.exp(-m))                           # treat m as a logit -> p_true
    gamma = 2.0
    focal = (1.0 - pt) ** gamma * (-np.log(pt)) / np.log(2.0)
    fig, ax = plt.subplots(figsize=(8.6, 5.2))
    ax.plot(m, zero_one, color=SLATE, lw=2.0, label="0–1 loss (what we'd love, not differentiable)")
    ax.plot(m, hinge, color=BLUE, lw=2.4, label=r"hinge  $\max(0,\,1-m)$  (SVM margin)")
    ax.plot(m, logistic, color=PURPLE, lw=2.6,
            label="logistic / cross-entropy (smooth upper bound)")
    ax.plot(m, focal, color=RED, lw=2.4, ls="--",
            label=r"focal ($\gamma{=}2$) — eases off on easy positives")
    ax.axvline(0, color=SLATE, ls=":", lw=1)
    ax.annotate("margin: correct →", xy=(2.2, 0.05), xytext=(1.2, 1.4),
                fontsize=9.5, color=GREEN, fontweight="bold",
                arrowprops=dict(arrowstyle="->", color=GREEN))
    ax.set_xlabel(r"true-class score / margin  $m$  (positive = correct, confident)")
    ax.set_ylabel("loss")
    ax.set_title("Classification losses are smooth surrogates for the 0–1 loss",
                 fontsize=13.0, fontweight="bold")
    ax.set_ylim(-0.1, 3.2)
    ax.legend(frameon=False, fontsize=9.0, loc="upper right")
    _despine(ax)
    fig.tight_layout()
    fig.savefig(f"{OUT}/loss_classification_margin.png", dpi=150, bbox_inches="tight")
    plt.close(fig)
    print("wrote loss_classification_margin.png")


def focal_downweighting():
    """Focal loss vs p_t for several gamma, all measured."""
    pt = np.linspace(0.01, 1.0, 600)
    ce = -np.log(pt)
    fig, ax = plt.subplots(figsize=(8.6, 5.2))
    colors = {0: PURPLE, 1: BLUE, 2: GREEN, 5: RED}
    for g, c in colors.items():
        loss = (1.0 - pt) ** g * ce
        lbl = (r"$\gamma=0$  (= cross-entropy)" if g == 0
               else rf"$\gamma={g}$")
        ax.plot(pt, loss, color=c, lw=2.6 if g == 0 else 2.2, label=lbl)
    # measured down-weighting at an easy example p_t = 0.9
    pe = 0.9
    ce_e = -np.log(pe)
    f2 = (1 - pe) ** 2 * ce_e
    ax.scatter([pe, pe], [ce_e, f2], color=NAVY, s=42, zorder=6)
    ax.annotate(f"easy example p_t=0.9:\nCE={ce_e:.3f},  γ=2 focal={f2:.4f}\n→ {ce_e/f2:.0f}× down-weighted",
                xy=(pe, ce_e), xytext=(0.34, 1.9), fontsize=9.5, color=NAVY, fontweight="bold",
                arrowprops=dict(arrowstyle="->", color=NAVY))
    ax.set_xlabel(r"$p_t$ — probability the model gave the TRUE class")
    ax.set_ylabel(r"loss  $(1-p_t)^\gamma\,(-\log p_t)$")
    ax.set_title("Focal loss down-weights easy, well-classified examples",
                 fontsize=13.0, fontweight="bold")
    ax.set_ylim(-0.1, 3.0)
    ax.set_xlim(0, 1.0)
    ax.legend(frameon=False, fontsize=10, loc="upper right", title="modulating exponent")
    _despine(ax)
    fig.tight_layout()
    fig.savefig(f"{OUT}/loss_focal_downweighting.png", dpi=150, bbox_inches="tight")
    plt.close(fig)
    print("wrote loss_focal_downweighting.png")


def crossentropy_penalty():
    """The cross-entropy penalty −log p on the TRUE class. Measured."""
    p = np.linspace(0.001, 1.0, 600)
    ce = -np.log(p)
    fig, ax = plt.subplots(figsize=(8.6, 5.2))
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
    ax.set_xlabel("probability the model assigned to the TRUE class,  p")
    ax.set_ylabel("cross-entropy loss  −log p")
    ax.set_title("Cross-entropy punishes confident-and-wrong, rewards confident-and-right",
                 fontsize=12.5, fontweight="bold")
    ax.set_ylim(-0.2, 5)
    _despine(ax)
    fig.tight_layout()
    fig.savefig(f"{OUT}/loss_crossentropy.png", dpi=150, bbox_inches="tight")
    plt.close(fig)
    print("wrote loss_crossentropy.png")


def grad_ce_vs_mse():
    """|dL/dz| for CE vs MSE on one sigmoid output, true label = 1. Measured."""
    z = np.linspace(-6, 6, 600)
    p = 1.0 / (1.0 + np.exp(-z))
    y = 1.0
    # cross-entropy (BCE) gradient wrt logit: p - y
    grad_ce = np.abs(p - y)
    # MSE on sigmoid output, L = 0.5(p-y)^2 ; dL/dz = (p-y)*p*(1-p)
    grad_mse = np.abs((p - y) * p * (1 - p))
    fig, ax = plt.subplots(figsize=(8.6, 5.2))
    ax.plot(z, grad_ce, color=GREEN, lw=2.8, label=r"cross-entropy:  $|\partial L/\partial z| = |p-y|$")
    ax.plot(z, grad_mse, color=RED, lw=2.6, ls="--",
            label=r"MSE on sigmoid:  $|(p-y)\,p(1-p)|$")
    # confidently-wrong region: z very negative but y=1
    ax.axvspan(-6, -3, color=RED, alpha=0.08)
    ax.annotate("confidently WRONG (z≪0, y=1):\nCE gradient ≈ 1 (strong)\nMSE gradient → 0 (vanishes)",
                xy=(-5, 0.02), xytext=(-2.3, 0.62), fontsize=9.5, color=RED, fontweight="bold",
                arrowprops=dict(arrowstyle="->", color=RED))
    ax.scatter([-5], [np.abs(1/(1+np.exp(5)) - 1)], color=GREEN, s=40, zorder=6)
    ax.scatter([-5], [np.abs((1/(1+np.exp(5)) - 1) * (1/(1+np.exp(5))) * (1 - 1/(1+np.exp(5))))],
               color=RED, s=40, zorder=6)
    ax.set_xlabel(r"logit  $z$  (true label $y=1$; $p=\sigma(z)$)")
    ax.set_ylabel(r"gradient magnitude  $|\partial L/\partial z|$")
    ax.set_title("Why not MSE on a classifier: its gradient saturates exactly when wrong",
                 fontsize=12.5, fontweight="bold")
    ax.set_ylim(-0.03, 1.05)
    ax.legend(frameon=False, fontsize=9.5, loc="upper right")
    _despine(ax)
    fig.tight_layout()
    fig.savefig(f"{OUT}/loss_grad_ce_vs_mse.png", dpi=150, bbox_inches="tight")
    plt.close(fig)
    print("wrote loss_grad_ce_vs_mse.png")


if __name__ == "__main__":
    regression_family()
    classification_margin()
    focal_downweighting()
    crossentropy_penalty()
    grad_ce_vs_mse()
    print("OUT:", OUT)
