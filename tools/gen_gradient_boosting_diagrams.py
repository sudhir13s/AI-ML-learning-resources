"""Gradient-boosting concept-page diagrams (muted palette, parallel scale). REAL fits.

Figures for 03. Supervised_Learning/concepts/10-Gradient-Boosting-XGBoost.md:
  1. gb_stages.png    -- boosting fits a 1D curve progressively: the ensemble after
     1, 5, and 50 shallow trees converges to the true function (each tree fits the
     residual of the last).
  2. gb_lr.png        -- test error vs number of trees for two learning rates: a large LR
     overfits fast, a small LR (shrinkage) generalizes better but needs more trees.
  3. gb_residuals.png -- the additive model unrolled: top row = the tree fit to the current
     residual at rounds 1/2/3; bottom row = the residual SHRINKING each round (the negative
     gradient driven toward zero).
  4. gb_vs_bagging.png -- train AND test error vs number of trees, boosting vs a bagged forest:
     boosting's train error keeps falling (test eventually rises -> CAN overfit); bagging
     plateaus (never overfits from tree count).
  5. gb_xgb_gain.png  -- the XGBoost leaf/split picture: optimal leaf weight w* = -G/(H+lambda)
     and the split gain, computed from real per-sample g_i, h_i numbers.
"""
import os, matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from sklearn.tree import DecisionTreeRegressor

OUT = os.path.join(os.path.dirname(__file__), "..", "03. Supervised_Learning", "concepts", "images")
OUT = os.path.abspath(OUT)
os.makedirs(OUT, exist_ok=True)

BLUE, PURPLE, GREEN, RED, SLATE, AMBER, NAVY = (
    "#3A6B96", "#5D4A8A", "#2E7A5A", "#8B3B4A", "#4A5B6E", "#7A6528", "#2A5B80")
plt.rcParams.update({"font.size": 11, "font.family": "DejaVu Sans"})


def _despine(ax):
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)


def _boost(x, y, n_trees, lr=0.3, depth=3):
    """Gradient boosting from scratch (MSE loss): yield the ensemble prediction."""
    F = np.full_like(y, y.mean()); trees = []
    snaps = {}
    for m in range(1, n_trees + 1):
        r = y - F                                   # pseudo-residual = -gradient of MSE
        t = DecisionTreeRegressor(max_depth=depth, random_state=0).fit(x[:, None], r)
        F = F + lr * t.predict(x[:, None]); trees.append(t)
        snaps[m] = F.copy()
    return snaps


def gb_stages():
    rng = np.random.default_rng(0)
    x = np.sort(rng.uniform(-3, 3, 120)); y = np.sin(1.5 * x) + rng.normal(0, 0.18, len(x))
    snaps = _boost(x, y, 50, lr=0.3)
    fig, axes = plt.subplots(1, 3, figsize=(13.2, 4.2), sharey=True)
    for ax, m in zip(axes, [1, 5, 50]):
        ax.scatter(x, y, color=SLATE, s=14, alpha=0.5)
        ax.plot(x, np.sin(1.5 * x), color=GREEN, lw=2, ls="--", label="true function")
        ax.plot(x, snaps[m], color=RED, lw=2.4, label=f"ensemble ({m} tree{'s' if m>1 else ''})")
        ax.set_title(f"after {m} tree{'s' if m>1 else ''}", fontsize=12, fontweight="bold")
        ax.set_xlabel("x"); ax.legend(frameon=False, fontsize=8.5, loc="upper center"); _despine(ax)
    axes[0].set_ylabel("y")
    fig.suptitle("Gradient boosting: each tree fits the residual; the ensemble converges to the signal",
                 fontsize=13.5, fontweight="bold", y=1.04)
    fig.tight_layout(); fig.savefig(f"{OUT}/gb_stages.png", dpi=150, bbox_inches="tight")
    plt.close(fig); print("wrote gb_stages.png")


def gb_lr():
    rng = np.random.default_rng(1)
    x = np.sort(rng.uniform(-3, 3, 150)); y = np.sin(1.5 * x) + rng.normal(0, 0.25, len(x))
    xte = np.sort(rng.uniform(-3, 3, 400)); yte = np.sin(1.5 * xte)        # clean test signal
    fig, ax = plt.subplots(figsize=(8.6, 5.0))
    N = 120
    for lr, col, lab in [(1.0, RED, "learning rate = 1.0 (fast → overfits)"),
                         (0.1, GREEN, "learning rate = 0.1 (shrinkage → generalizes)")]:
        F = np.full_like(y, y.mean()); Fte = np.full_like(yte, y.mean()); errs = []
        for m in range(N):
            r = y - F
            t = DecisionTreeRegressor(max_depth=3, random_state=0).fit(x[:, None], r)
            F = F + lr * t.predict(x[:, None]); Fte = Fte + lr * t.predict(xte[:, None])
            errs.append(np.mean((yte - Fte) ** 2))
        ax.plot(range(1, N + 1), errs, color=col, lw=2.3, label=lab)
        best = int(np.argmin(errs)) + 1
        ax.scatter([best], [min(errs)], color=col, s=45, zorder=5)
    ax.set_xlabel("number of trees (boosting rounds)"); ax.set_ylabel("test error (MSE)")
    ax.set_title("Shrinkage: a small learning rate needs more trees but generalizes better",
                 fontsize=12, fontweight="bold")
    ax.legend(frameon=False, fontsize=9.5); _despine(ax); ax.set_ylim(0, 0.35)
    fig.tight_layout(); fig.savefig(f"{OUT}/gb_lr.png", dpi=150, bbox_inches="tight")
    plt.close(fig); print("wrote gb_lr.png")


def gb_residuals():
    """Unroll the additive model: each tree fits the CURRENT residual; the residual shrinks."""
    rng = np.random.default_rng(3)
    x = np.sort(rng.uniform(-3, 3, 80)); y = np.sin(1.5 * x) + rng.normal(0, 0.12, len(x))
    lr = 0.6
    F = np.full_like(y, y.mean())
    fig, axes = plt.subplots(2, 3, figsize=(13.2, 7.0), sharex=True)
    for j in range(3):                                   # rounds 1, 2, 3
        r = y - F                                        # current pseudo-residual = -gradient
        t = DecisionTreeRegressor(max_depth=2, random_state=0).fit(x[:, None], r)
        hpred = t.predict(x[:, None])
        # top: the tree fit to the current residual
        ax = axes[0, j]
        ax.scatter(x, r, color=AMBER, s=16, alpha=0.6, label="residual $r=y-F$")
        ax.plot(x, hpred, color=PURPLE, lw=2.6, label=f"tree $h_{{{j+1}}}$ fit to $r$")
        ax.axhline(0, color=SLATE, lw=0.8, ls=":")
        ax.set_title(f"round {j+1}: fit tree to residual", fontsize=12, fontweight="bold")
        ax.legend(frameon=False, fontsize=8.5, loc="upper center"); _despine(ax)
        if j == 0: ax.set_ylabel("residual value")
        F = F + lr * hpred                               # additive update
        # bottom: the residual AFTER this round (shrinking toward zero)
        ax2 = axes[1, j]
        new_r = y - F
        ax2.scatter(x, new_r, color=GREEN, s=16, alpha=0.6)
        ax2.axhline(0, color=SLATE, lw=0.8, ls=":")
        ax2.set_title(f"residual after round {j+1}  (RMS={np.sqrt((new_r**2).mean()):.3f})",
                      fontsize=11.5, fontweight="bold")
        ax2.set_xlabel("x"); ax2.set_ylim(-1.25, 1.25); _despine(ax2)
        if j == 0: ax2.set_ylabel("remaining residual")
    fig.suptitle("The additive model unrolled: each tree fits the residual; the residual shrinks each round",
                 fontsize=13.5, fontweight="bold", y=0.99)
    fig.tight_layout(); fig.savefig(f"{OUT}/gb_residuals.png", dpi=150, bbox_inches="tight")
    plt.close(fig); print("wrote gb_residuals.png")


def gb_vs_bagging():
    """Boosting vs a bagged forest: train AND test error vs tree count."""
    from sklearn.ensemble import BaggingRegressor
    rng = np.random.default_rng(7)
    x = np.sort(rng.uniform(-3, 3, 120)); y = np.sin(1.5 * x) + rng.normal(0, 0.30, len(x))
    xte = np.sort(rng.uniform(-3, 3, 500)); yte = np.sin(1.5 * xte)        # clean test signal
    N = 200
    # boosting (lr=0.3, shallow): track train + test error per round
    F = np.full_like(y, y.mean()); Fte = np.full_like(yte, y.mean())
    btr, bte = [], []
    for m in range(N):
        r = y - F
        t = DecisionTreeRegressor(max_depth=3, random_state=0).fit(x[:, None], r)
        F = F + 0.3 * t.predict(x[:, None]); Fte = Fte + 0.3 * t.predict(xte[:, None])
        btr.append(np.mean((y - F) ** 2)); bte.append(np.mean((yte - Fte) ** 2))
    # bagging: a forest of deep trees, error vs number of trees
    gtr, gte = [], []
    for m in range(1, N + 1, 5):
        bag = BaggingRegressor(DecisionTreeRegressor(max_depth=8),
                               n_estimators=m, random_state=0).fit(x[:, None], y)
        gtr.append(np.mean((y - bag.predict(x[:, None])) ** 2))
        gte.append(np.mean((yte - bag.predict(xte[:, None])) ** 2))
    fig, ax = plt.subplots(figsize=(9.2, 5.2))
    ax.plot(range(1, N + 1), btr, color=RED, lw=2.2, label="boosting — TRAIN error (keeps falling)")
    ax.plot(range(1, N + 1), bte, color=RED, lw=2.2, ls="--", label="boosting — TEST error (eventually rises: overfits)")
    ax.plot(range(1, N + 1, 5), gtr, color=NAVY, lw=2.2, label="bagged forest — TRAIN error")
    ax.plot(range(1, N + 1, 5), gte, color=NAVY, lw=2.2, ls="--", label="bagged forest — TEST error (plateaus: no overfit)")
    ax.set_xlabel("number of trees"); ax.set_ylabel("error (MSE)")
    ax.set_title("Boosting can overfit from tree count; a bagged forest plateaus",
                 fontsize=12, fontweight="bold")
    ax.legend(frameon=False, fontsize=9); _despine(ax); ax.set_ylim(0, 0.30)
    fig.tight_layout(); fig.savefig(f"{OUT}/gb_vs_bagging.png", dpi=150, bbox_inches="tight")
    plt.close(fig); print("wrote gb_vs_bagging.png")


def gb_xgb_gain():
    """XGBoost leaf weight w* = -G/(H+lambda) and split gain, from real g_i, h_i numbers."""
    # five samples sit at a node; a candidate split sends 1,2 left and 3,4,5 right.
    g = np.array([-0.8, -0.6,  0.5,  0.7,  0.9])         # gradients g_i
    h = np.array([ 1.0,  1.0,  1.0,  1.0,  1.0])         # hessians h_i (=1 for MSE)
    lam, gamma = 1.0, 0.0
    left, right = [0, 1], [2, 3, 4]
    GL, HL = g[left].sum(), h[left].sum()
    GR, HR = g[right].sum(), h[right].sum()
    G, H = g.sum(), h.sum()
    wL, wR, wP = -GL / (HL + lam), -GR / (HR + lam), -G / (H + lam)
    gain = 0.5 * (GL**2 / (HL + lam) + GR**2 / (HR + lam) - G**2 / (H + lam)) - gamma

    fig, axes = plt.subplots(1, 2, figsize=(13.0, 5.0))
    # left panel: the node, the split, the leaf weights
    ax = axes[0]; ax.axis("off")
    ax.set_title("Optimal leaf weight  $w^*_j = -G_j/(H_j+\\lambda)$",
                 fontsize=12.5, fontweight="bold", pad=12)
    def box(x0, y0, w0, h0, color, lines):
        ax.add_patch(plt.Rectangle((x0, y0), w0, h0, facecolor=color, edgecolor="white", lw=2))
        ax.text(x0 + w0/2, y0 + h0/2, "\n".join(lines), ha="center", va="center",
                color="white", fontsize=10.5, fontweight="bold")
    box(0.30, 0.74, 0.40, 0.20, SLATE,
        [f"parent node (5 samples)", f"G={G:+.1f}, H={H:.0f}", f"$w^*$={wP:+.2f}"])
    box(0.04, 0.30, 0.40, 0.22, BLUE,
        ["left leaf  {x1,x2}", f"G_L={GL:+.1f}, H_L={HL:.0f}", f"$w^*_L$={wL:+.2f}"])
    box(0.56, 0.30, 0.40, 0.22, GREEN,
        ["right leaf  {x3,x4,x5}", f"G_R={GR:+.1f}, H_R={HR:.0f}", f"$w^*_R$={wR:+.2f}"])
    ax.annotate("", xy=(0.24, 0.52), xytext=(0.45, 0.74),
                arrowprops=dict(arrowstyle="-|>", color=BLUE, lw=2.2))
    ax.annotate("", xy=(0.76, 0.52), xytext=(0.55, 0.74),
                arrowprops=dict(arrowstyle="-|>", color=GREEN, lw=2.2))
    ax.text(0.30, 0.66, "split", ha="center", color=AMBER, fontsize=10, fontweight="bold")
    ax.set_xlim(0, 1); ax.set_ylim(0.2, 1.0)

    # right panel: the gain formula with numbers
    ax2 = axes[1]; ax2.axis("off")
    ax2.set_title("Split gain  $=\\frac{1}{2}\\left[\\frac{G_L^2}{H_L+\\lambda}+\\frac{G_R^2}{H_R+\\lambda}-\\frac{G^2}{H+\\lambda}\\right]-\\gamma$",
                  fontsize=12, fontweight="bold", pad=12)
    terms = [
        ("left term", f"$G_L^2/(H_L+\\lambda) = ({GL:.1f})^2/({HL:.0f}+{lam:.0f}) = {GL**2/(HL+lam):.3f}$"),
        ("right term", f"$G_R^2/(H_R+\\lambda) = ({GR:.1f})^2/({HR:.0f}+{lam:.0f}) = {GR**2/(HR+lam):.3f}$"),
        ("parent term", f"$G^2/(H+\\lambda) = ({G:.1f})^2/({H:.0f}+{lam:.0f}) = {G**2/(H+lam):.3f}$"),
        ("gain", f"$\\frac{{1}}{{2}}[{GL**2/(HL+lam):.3f}+{GR**2/(HR+lam):.3f}-{G**2/(H+lam):.3f}]-{gamma:.0f} = {gain:.3f}$"),
    ]
    ys = [0.80, 0.62, 0.44, 0.20]
    cols = [BLUE, GREEN, SLATE, RED]
    for (lab, expr), yy, cc in zip(terms, ys, cols):
        ax2.add_patch(plt.Rectangle((0.02, yy - 0.06), 0.96, 0.12, facecolor=cc, edgecolor="white", lw=1.5))
        ax2.text(0.06, yy, lab, ha="left", va="center", color="white", fontsize=10, fontweight="bold")
        ax2.text(0.40, yy, expr, ha="left", va="center", color="white", fontsize=10.5)
    ax2.text(0.5, 0.05, "gain > 0  →  the split lowers the regularized loss, so XGBoost keeps it",
             ha="center", color=GREEN, fontsize=10, fontweight="bold")
    ax2.set_xlim(0, 1); ax2.set_ylim(0, 0.92)
    fig.suptitle("XGBoost's second-order objective: leaf weight and split gain from g and h",
                 fontsize=13.5, fontweight="bold", y=1.02)
    fig.tight_layout(); fig.savefig(f"{OUT}/gb_xgb_gain.png", dpi=150, bbox_inches="tight")
    plt.close(fig); print("wrote gb_xgb_gain.png")


if __name__ == "__main__":
    gb_stages()
    gb_lr()
    gb_residuals()
    gb_vs_bagging()
    gb_xgb_gain()
    print("OUT:", OUT)
