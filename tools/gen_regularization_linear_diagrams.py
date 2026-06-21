"""Regularization for linear models — concept-page diagrams (muted palette). REAL measured.

Four figures for 03. Supervised_Learning/concepts/03-Regularization-Linear-Models.md:
  1. reglin_constraint_geometry.png -- the L1 diamond vs L2 circle constraint regions with
     elliptical least-squares loss contours. The Lasso solution lands on a CORNER of the diamond
     (a coefficient is exactly zero) -> sparsity; the Ridge solution touches the circle off-axis
     (both coefficients small but non-zero). This is the canonical "why L1 is sparse" picture.
  2. reglin_paths.png -- the regularization PATH: coefficients vs lambda for Ridge and Lasso
     side by side on the same standardized data. Ridge shrinks all coefficients smoothly; Lasso
     drives them to EXACTLY zero one by one (measured with sklearn).
  3. reglin_soft_threshold.png -- the soft-thresholding operator: input coordinate vs Lasso output,
     the dead zone [-lambda, lambda] mapped to zero and the shrink-by-lambda elsewhere, contrasted
     with Ridge's pure multiplicative shrink (a line through the origin with slope < 1).
  4. reglin_error_vs_lambda.png -- train vs test (CV) error as lambda sweeps: the classic U where
     too little regularization overfits (low train / high test) and too much underfits (both high).
     The sweet spot and the one-standard-error rule are marked (measured on a noisy linear problem).
"""
import os, matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import Polygon, Circle
from sklearn.linear_model import Ridge, Lasso
from sklearn.model_selection import cross_val_score
from sklearn.preprocessing import StandardScaler

OUT = os.path.join(os.path.dirname(__file__), "..", "03. Supervised_Learning", "concepts", "images")
OUT = os.path.abspath(OUT)
os.makedirs(OUT, exist_ok=True)

BLUE, PURPLE, GREEN, RED, SLATE, AMBER, NAVY = (
    "#3A6B96", "#5D4A8A", "#2E7A5A", "#8B3B4A", "#4A5B6E", "#7A6528", "#2A5B80")
plt.rcParams.update({"font.size": 11, "font.family": "DejaVu Sans"})


def _despine(ax):
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)


# ---------------------------------------------------------------------------
# 1. Constraint geometry: L1 diamond vs L2 circle, with loss contours.
# ---------------------------------------------------------------------------
def constraint_geometry():
    # An OLS objective (w - b)^T A (w - b) with a correlated (tilted) Hessian A and
    # an unconstrained optimum b far from the origin, so the constraint binds.
    b = np.array([2.6, 1.1])                       # unconstrained OLS optimum
    A = np.array([[1.0, 0.78], [0.78, 1.0]])       # correlated features -> tilted ellipses

    def loss(W1, W2):
        d1, d2 = W1 - b[0], W2 - b[1]
        return A[0, 0]*d1*d1 + 2*A[0, 1]*d1*d2 + A[1, 1]*d2*d2

    def solve_constrained(kind, t):
        # brute-force the minimizer of the loss on / inside the budget |.|<=t (fine grid).
        g = np.linspace(-0.2, 3.4, 700)
        G1, G2 = np.meshgrid(g, g)
        if kind == "l1":
            mask = (np.abs(G1) + np.abs(G2)) <= t
        else:
            mask = (G1**2 + G2**2) <= t**2
        L = loss(G1, G2)
        L[~mask] = np.inf
        idx = np.unravel_index(np.argmin(L), L.shape)
        return G1[idx], G2[idx]

    t = 1.6
    w_l1 = solve_constrained("l1", t)
    w_l2 = solve_constrained("l2", t)

    fig, axes = plt.subplots(1, 2, figsize=(11.0, 5.4))
    g = np.linspace(-0.6, 3.8, 400)
    G1, G2 = np.meshgrid(g, g)
    L = loss(G1, G2)
    levels = np.array([0.25, 1.0, 2.25, 4.0, 6.25, 9.0])

    for ax, kind, wsol, title in [
        (axes[0], "l1", w_l1, "Lasso (L1): budget is a DIAMOND"),
        (axes[1], "l2", w_l2, "Ridge (L2): budget is a CIRCLE"),
    ]:
        ax.contour(G1, G2, L, levels=levels, colors=SLATE, linewidths=0.9, alpha=0.65)
        if kind == "l1":
            diamond = Polygon([(t, 0), (0, t), (-t, 0), (0, -t)], closed=True,
                              facecolor=BLUE, edgecolor=NAVY, alpha=0.30, lw=2.2)
            ax.add_patch(diamond)
        else:
            circ = Circle((0, 0), t, facecolor=PURPLE, edgecolor="#4D3A7A", alpha=0.28, lw=2.2)
            ax.add_patch(circ)
        # unconstrained OLS optimum
        ax.plot(*b, marker="x", ms=11, mew=3, color=RED)
        ax.annotate("OLS optimum\n(unconstrained)", b, textcoords="offset points",
                    xytext=(8, 6), fontsize=9, color=RED, fontweight="bold")
        # constrained solution
        ax.plot(*wsol, marker="o", ms=10, color=GREEN, mec="#1E6A4A", mew=1.8, zorder=5)
        sol_lbl = (f"solution\n(w₁={wsol[0]:.2f}, w₂={wsol[1]:.2f})")
        ax.annotate(sol_lbl, wsol, textcoords="offset points", xytext=(-92, -34),
                    fontsize=9, color=GREEN, fontweight="bold")
        ax.axhline(0, color="#888", lw=0.8); ax.axvline(0, color="#888", lw=0.8)
        ax.set_xlim(-0.7, 3.8); ax.set_ylim(-1.0, 2.4)
        ax.set_aspect("equal"); _despine(ax)
        ax.set_xlabel("w₁"); ax.set_ylabel("w₂")
        ax.set_title(title, fontsize=11.5, fontweight="bold")

    # Annotate WHY: the diamond's corner sits on an axis -> a coefficient is exactly 0.
    axes[0].annotate("corner on the axis →\nw₂ = 0 exactly (SPARSE)", (0, t),
                     textcoords="offset points", xytext=(10, -2), fontsize=9,
                     color=BLUE, fontweight="bold")
    axes[1].annotate("smooth boundary →\nboth small, neither 0", (0, t),
                     textcoords="offset points", xytext=(8, -2), fontsize=9,
                     color=PURPLE, fontweight="bold")
    fig.suptitle("Why L1 is sparse and L2 is not: the constraint region the loss contour first touches",
                 fontsize=12.5, fontweight="bold", y=0.99)
    fig.tight_layout(rect=[0, 0, 1, 0.96])
    fig.savefig(f"{OUT}/reglin_constraint_geometry.png", dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"wrote reglin_constraint_geometry.png  | L1 sol={w_l1}  L2 sol={w_l2}")
    return w_l1, w_l2


# ---------------------------------------------------------------------------
# 2. Regularization paths: coefficients vs lambda for Ridge and Lasso.
# ---------------------------------------------------------------------------
def _path_data(seed=0, n=120, p=8):
    rng = np.random.default_rng(seed)
    X = rng.normal(size=(n, p))
    X[:, 1] = X[:, 0] + 0.05 * rng.normal(size=n)     # near-duplicate of feature 0 (collinear)
    true_w = np.array([3.0, 0.0, -2.0, 0.0, 1.5, 0.0, 0.0, 0.8])
    y = X @ true_w + rng.normal(0, 1.0, n)
    X = StandardScaler().fit_transform(X)
    y = y - y.mean()
    return X, y, true_w


def reg_paths():
    X, y, _ = _path_data()
    p = X.shape[1]
    alphas = np.logspace(-2, 2.4, 60)
    ridge_coefs = np.array([Ridge(alpha=a).fit(X, y).coef_ for a in alphas])
    lasso_coefs = np.array([Lasso(alpha=a, max_iter=20000).fit(X, y).coef_ for a in alphas])

    fig, axes = plt.subplots(1, 2, figsize=(11.0, 5.0), sharey=True)
    palette = [BLUE, NAVY, GREEN, PURPLE, AMBER, SLATE, RED, "#1E6A4A"]
    for ax, coefs, title, note in [
        (axes[0], ridge_coefs, "Ridge (L2): coefficients SHRINK smoothly",
         "none ever reaches exactly 0"),
        (axes[1], lasso_coefs, "Lasso (L1): coefficients hit EXACTLY 0",
         "features drop out one by one"),
    ]:
        for j in range(p):
            ax.plot(alphas, coefs[:, j], color=palette[j], lw=2.0, label=f"w{j}")
        ax.set_xscale("log"); ax.axhline(0, color="#888", lw=0.8, ls=":")
        ax.set_xlabel("λ  (regularization strength, log scale)")
        ax.set_title(title, fontsize=11.5, fontweight="bold")
        ax.text(0.5, -0.22, note, transform=ax.transAxes, ha="center",
                fontsize=9.5, color=SLATE, style="italic")
        _despine(ax)
    axes[0].set_ylabel("coefficient value")
    axes[1].legend(frameon=False, fontsize=8, ncol=2, loc="upper right")
    fig.suptitle("The regularization path: how each coefficient moves as λ grows",
                 fontsize=12.5, fontweight="bold", y=1.0)
    fig.tight_layout(rect=[0, 0.02, 1, 0.96])
    fig.savefig(f"{OUT}/reglin_paths.png", dpi=150, bbox_inches="tight")
    plt.close(fig)
    # report how many lasso coefs are zero at a mid lambda
    mid = Lasso(alpha=1.0, max_iter=20000).fit(X, y).coef_
    print(f"wrote reglin_paths.png  | Lasso @alpha=1.0 zeros: {int(np.sum(np.abs(mid) < 1e-8))}/{p}")


# ---------------------------------------------------------------------------
# 3. Soft-thresholding operator vs Ridge shrink.
# ---------------------------------------------------------------------------
def soft_threshold():
    lam = 1.0
    z = np.linspace(-4, 4, 400)
    soft = np.sign(z) * np.maximum(np.abs(z) - lam, 0.0)     # Lasso prox
    ridge = z / (1 + lam)                                    # Ridge multiplicative shrink

    fig, ax = plt.subplots(figsize=(8.4, 5.2))
    ax.plot(z, z, color=SLATE, lw=1.4, ls="--", label="identity (no penalty)")
    ax.plot(z, ridge, color=PURPLE, lw=2.4, label="Ridge: z/(1+λ)  (shrink toward 0)")
    ax.plot(z, soft, color=BLUE, lw=2.6, label="Lasso: soft-threshold Sλ(z)")
    # shade the dead zone
    ax.axvspan(-lam, lam, color=GREEN, alpha=0.12)
    ax.text(0, -3.3, "dead zone [−λ, λ]\n→ output exactly 0", ha="center", fontsize=9.5,
            color=GREEN, fontweight="bold")
    ax.axhline(0, color="#888", lw=0.8); ax.axvline(0, color="#888", lw=0.8)
    ax.set_xlabel("input  z  (the OLS coordinate)")
    ax.set_ylabel("regularized coefficient")
    ax.set_title("Soft-thresholding: Lasso zeroes a band; Ridge only scales",
                 fontsize=12, fontweight="bold")
    ax.legend(frameon=False, fontsize=9.5, loc="upper left")
    _despine(ax); ax.set_xlim(-4, 4); ax.set_ylim(-4, 4)
    fig.tight_layout(); fig.savefig(f"{OUT}/reglin_soft_threshold.png", dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"wrote reglin_soft_threshold.png  | Sλ(0.6)={np.sign(0.6)*max(abs(0.6)-lam,0):.2f}  "
          f"Sλ(2.5)={np.sign(2.5)*max(abs(2.5)-lam,0):.2f}")


# ---------------------------------------------------------------------------
# 4. Train vs test (CV) error vs lambda: the U-curve and the sweet spot.
# ---------------------------------------------------------------------------
def error_vs_lambda():
    rng = np.random.default_rng(3)
    n, p = 60, 40                       # p close to n -> overfitting without regularization
    X = rng.normal(size=(n, p))
    true_w = np.zeros(p); true_w[:6] = rng.normal(0, 2, 6)
    y = X @ true_w + rng.normal(0, 2.0, n)
    X = StandardScaler().fit_transform(X); y = y - y.mean()

    alphas = np.logspace(-2, 3.2, 40)
    train_mse, cv_mean, cv_std = [], [], []
    for a in alphas:
        m = Ridge(alpha=a)
        m.fit(X, y)
        train_mse.append(np.mean((y - m.predict(X))**2))
        sc = cross_val_score(Ridge(alpha=a), X, y, cv=5, scoring="neg_mean_squared_error")
        cv_mean.append(-sc.mean()); cv_std.append(sc.std())
    train_mse, cv_mean, cv_std = map(np.array, (train_mse, cv_mean, cv_std))

    best = int(np.argmin(cv_mean))
    # one-standard-error rule: largest lambda whose CV error <= min + 1 SE
    thresh = cv_mean[best] + cv_std[best]
    ose_candidates = np.where(cv_mean <= thresh)[0]
    ose = ose_candidates[ose_candidates >= best]
    ose_idx = ose[-1] if len(ose) else best

    fig, ax = plt.subplots(figsize=(8.6, 5.2))
    ax.plot(alphas, train_mse, color=BLUE, lw=2.4, marker="o", ms=3, label="train error")
    ax.plot(alphas, cv_mean, color=RED, lw=2.4, marker="s", ms=3, label="test (5-fold CV) error")
    ax.fill_between(alphas, cv_mean - cv_std, cv_mean + cv_std, color=RED, alpha=0.12)
    ax.axvline(alphas[best], color=GREEN, lw=1.8, ls="--")
    ax.annotate("best λ\n(min CV error)", (alphas[best], cv_mean[best]),
                textcoords="offset points", xytext=(10, 28), fontsize=9.5,
                color=GREEN, fontweight="bold")
    ax.axvline(alphas[ose_idx], color=AMBER, lw=1.8, ls=":")
    ax.annotate("1-SE rule λ\n(simpler model)", (alphas[ose_idx], cv_mean[ose_idx]),
                textcoords="offset points", xytext=(8, -42), fontsize=9.5,
                color=AMBER, fontweight="bold")
    ax.set_xscale("log"); ax.set_xlabel("λ  (regularization strength, log scale)")
    ax.set_ylabel("mean squared error")
    ax.text(alphas[1], train_mse.max()*0.9, "overfit\n(low train,\nhigh test)",
            fontsize=9, color=SLATE, ha="left")
    ax.text(alphas[-3], cv_mean[-1]*0.96, "underfit\n(both high)",
            fontsize=9, color=SLATE, ha="right")
    ax.set_title("Choosing λ: the U-shaped test error and its sweet spot",
                 fontsize=12, fontweight="bold")
    ax.legend(frameon=False, fontsize=9.5, loc="upper center"); _despine(ax)
    fig.tight_layout(); fig.savefig(f"{OUT}/reglin_error_vs_lambda.png", dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"wrote reglin_error_vs_lambda.png  | best λ={alphas[best]:.3f}  1-SE λ={alphas[ose_idx]:.3f}")


if __name__ == "__main__":
    constraint_geometry()
    reg_paths()
    soft_threshold()
    error_vs_lambda()
    print("OUT:", OUT)
