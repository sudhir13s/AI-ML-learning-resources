"""Logistic-regression concept-page diagrams (muted palette, parallel scale).

Four figures for 03. Supervised_Learning/concepts/02-Logistic-Regression.md:
  1. logreg_sigmoid.png -- the sigmoid maps the linear score z = w.x+b to a
     probability in (0,1); threshold at 0.5 (z=0) splits the classes.
  2. logreg_boundary.png -- a model fit from scratch on two 2D blobs: the LINEAR
     decision boundary with the sigmoid probability shaded behind it.
  3. logreg_loss_surface.png -- why log-loss not MSE: the log-loss surface over a
     weight is CONVEX (one bowl) while MSE-with-sigmoid is non-convex (flat plateaus
     + a local kink), and the MSE gradient vanishes where log-loss's stays strong.
  4. logreg_calibration.png -- reliability diagram: REAL sklearn calibration_curve
     showing logistic regression hugging the diagonal (well-calibrated) vs an
     over-confident Naive Bayes -- the calibration contrast.
"""
import os, matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

OUT = os.path.join(os.path.dirname(__file__), "..", "03. Supervised_Learning", "concepts", "images")
OUT = os.path.abspath(OUT)
os.makedirs(OUT, exist_ok=True)

BLUE, PURPLE, GREEN, RED, SLATE, AMBER, NAVY = (
    "#3A6B96", "#5D4A8A", "#2E7A5A", "#8B3B4A", "#4A5B6E", "#7A6528", "#2A5B80")
plt.rcParams.update({"font.size": 11, "font.family": "DejaVu Sans"})
sig = lambda z: 1 / (1 + np.exp(-z))


def _despine(ax):
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)


def logreg_sigmoid():
    z = np.linspace(-6, 6, 400)
    fig, ax = plt.subplots(figsize=(8.6, 5.0))
    ax.plot(z, sig(z), color=PURPLE, lw=3)
    ax.axhline(0.5, color=SLATE, ls=":", lw=1.3); ax.axvline(0, color=SLATE, ls=":", lw=1.3)
    ax.fill_between(z, 0, 1, where=(z > 0), color=GREEN, alpha=0.07)
    ax.fill_between(z, 0, 1, where=(z < 0), color=BLUE, alpha=0.07)
    ax.text(3.2, 0.18, "z > 0 → p > 0.5\npredict class 1", color=GREEN, fontsize=10, fontweight="bold", ha="center")
    ax.text(-3.2, 0.82, "z < 0 → p < 0.5\npredict class 0", color=BLUE, fontsize=10, fontweight="bold", ha="center")
    ax.scatter([0], [0.5], color=RED, s=70, zorder=6)
    ax.annotate("decision threshold\np = 0.5 at z = 0", (0, 0.5), textcoords="offset points",
                xytext=(20, -38), fontsize=9.5, color=RED, fontweight="bold",
                arrowprops=dict(arrowstyle="->", color=RED))
    ax.set_xlabel("linear score  z = w·x + b"); ax.set_ylabel("probability  p = σ(z)")
    ax.set_title("Logistic regression: squash a linear score into a probability",
                 fontsize=13, fontweight="bold")
    ax.set_ylim(-0.03, 1.03); _despine(ax)
    fig.tight_layout(); fig.savefig(f"{OUT}/logreg_sigmoid.png", dpi=150, bbox_inches="tight")
    plt.close(fig); print("wrote logreg_sigmoid.png")


def logreg_boundary():
    rng = np.random.default_rng(1)
    X0 = rng.normal([-1.3, -0.6], 0.85, (120, 2))
    X1 = rng.normal([1.3, 0.8], 0.85, (120, 2))
    X = np.vstack([X0, X1]); y = np.concatenate([np.zeros(120), np.ones(120)])
    Xb = np.c_[np.ones(len(X)), X]                      # bias column
    w = np.zeros(3)
    for _ in range(3000):                               # gradient descent on log-loss
        p = sig(Xb @ w); w -= 0.05 * Xb.T @ (p - y) / len(y)
    # probability field
    xx, yy = np.meshgrid(np.linspace(-4, 4, 300), np.linspace(-3.5, 3.5, 300))
    grid = np.c_[np.ones(xx.size), xx.ravel(), yy.ravel()]
    P = sig(grid @ w).reshape(xx.shape)
    fig, ax = plt.subplots(figsize=(8.2, 5.6))
    cf = ax.contourf(xx, yy, P, levels=20, cmap="RdBu_r", alpha=0.45, vmin=0, vmax=1)
    ax.contour(xx, yy, P, levels=[0.5], colors=[SLATE], linewidths=2.4)        # the boundary
    ax.scatter(X0[:, 0], X0[:, 1], color=BLUE, s=22, edgecolor="white", lw=0.4, label="class 0")
    ax.scatter(X1[:, 0], X1[:, 1], color=RED, s=22, edgecolor="white", lw=0.4, label="class 1")
    ax.text(0.05, 3.0, "decision boundary (p = 0.5) is a straight line:  w·x + b = 0",
            fontsize=9.5, color=SLATE, fontweight="bold", ha="center")
    cbar = fig.colorbar(cf, ax=ax, fraction=0.046); cbar.set_label("P(class 1)")
    ax.set_xlabel("feature 1"); ax.set_ylabel("feature 2")
    ax.set_title("The decision boundary is linear; the sigmoid shades the confidence",
                 fontsize=12.5, fontweight="bold")
    ax.legend(loc="lower right", frameon=True, fontsize=9); _despine(ax)
    fig.tight_layout(); fig.savefig(f"{OUT}/logreg_boundary.png", dpi=150, bbox_inches="tight")
    plt.close(fig); print("wrote logreg_boundary.png")


def logreg_loss_surface():
    """Why log-loss, not MSE. Left: the two loss surfaces over a single weight w for a
    confident-wrong setup -- log-loss is one convex bowl; MSE+sigmoid has flat plateaus
    (a non-convex shape). Right: the per-example gradient magnitude vs the score z for
    y=1 -- MSE's gradient collapses toward 0 exactly where the model is confidently
    WRONG (z very negative), while log-loss keeps a strong corrective signal."""
    X = np.array([-2., -1., 1., 2.]); y = np.array([0., 0., 1., 1.])
    ws = np.linspace(-4, 6, 400)

    def logloss(w):
        p = np.clip(sig(w * X), 1e-9, 1 - 1e-9)
        return -(y*np.log(p) + (1-y)*np.log(1-p)).mean()

    def mseloss(w):
        return ((sig(w * X) - y) ** 2).mean()

    ll = np.array([logloss(w) for w in ws]); ms = np.array([mseloss(w) for w in ws])
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(11.4, 4.8))

    ax1.plot(ws, ll, color=GREEN, lw=2.8, label="log-loss (convex)")
    ax1.plot(ws, ms, color=RED, lw=2.8, label="MSE + sigmoid (non-convex)")
    ax1.axvline(ws[ll.argmin()], color=GREEN, ls=":", lw=1.2)
    ax1.set_xlabel("weight  w"); ax1.set_ylabel("loss")
    ax1.set_title("Log-loss is a single convex bowl;\nMSE+sigmoid flattens into plateaus",
                  fontsize=11.5, fontweight="bold")
    ax1.legend(frameon=False, fontsize=9.5, loc="upper right"); _despine(ax1)

    z = np.linspace(-6, 6, 400); p = sig(z)
    g_ll = np.abs(p - 1.0)                       # |d/dz cross-entropy|, y=1
    g_ms = np.abs((p - 1.0) * p * (1 - p))       # |d/dz of (p-y)^2|, y=1
    ax2.plot(z, g_ll, color=GREEN, lw=2.8, label="|gradient| log-loss")
    ax2.plot(z, g_ms, color=RED, lw=2.8, label="|gradient| MSE+sigmoid")
    ax2.axvspan(-6, -2, color=BLUE, alpha=0.06)
    ax2.text(-5.7, 0.7, "confidently WRONG\n(y=1 but z≪0):\nMSE gradient ≈ 0,\nlog-loss stays strong",
             fontsize=8.8, color=RED, fontweight="bold", va="center")
    ax2.set_xlabel("linear score  z = w·x + b   (true label y = 1)")
    ax2.set_ylabel("gradient magnitude  |∂L/∂z|")
    ax2.set_title("MSE's gradient vanishes on confident-wrong\npredictions; log-loss's does not",
                  fontsize=11.5, fontweight="bold")
    ax2.legend(frameon=False, fontsize=9.5, loc="upper right"); _despine(ax2)

    fig.tight_layout(); fig.savefig(f"{OUT}/logreg_loss_surface.png", dpi=150, bbox_inches="tight")
    plt.close(fig); print("wrote logreg_loss_surface.png")


def logreg_calibration():
    """Logistic regression IS well-calibrated. Reliability diagram (REAL sklearn
    calibration_curve): predicted probability vs actual fraction positive. Logistic
    regression hugs the diagonal; an over-confident Naive Bayes (correlated features
    double-counted) bows away from it -- the calibration contrast worth showing."""
    from sklearn.linear_model import LogisticRegression
    from sklearn.naive_bayes import GaussianNB
    from sklearn.calibration import calibration_curve
    from sklearn.datasets import make_classification
    from sklearn.model_selection import train_test_split
    X, y = make_classification(n_samples=6000, n_features=20, n_informative=5,
                               n_redundant=10, random_state=0)
    Xtr, Xte, ytr, yte = train_test_split(X, y, test_size=0.5, random_state=0)
    fig, ax = plt.subplots(figsize=(7.6, 6.0))
    ax.plot([0, 1], [0, 1], color=SLATE, ls="--", lw=1.5, label="perfectly calibrated")
    for model, col, lab in [
        (LogisticRegression(max_iter=2000), GREEN, "Logistic Regression (well-calibrated)"),
        (GaussianNB(), RED, "Naive Bayes (over-confident)")]:
        p = model.fit(Xtr, ytr).predict_proba(Xte)[:, 1]
        frac, mean_pred = calibration_curve(yte, p, n_bins=10)
        ax.plot(mean_pred, frac, "o-", color=col, lw=2.4, ms=5, label=lab)
    ax.text(0.40, 0.10,
            "Logistic regression's curve\nsits on the diagonal: a predicted\n0.8 really is ~80% positive.",
            fontsize=9, color=GREEN, fontweight="bold")
    ax.set_xlabel("predicted probability"); ax.set_ylabel("actual fraction positive")
    ax.set_title("Logistic regression is well-calibrated (it optimizes log-loss directly)",
                 fontsize=11.5, fontweight="bold")
    ax.legend(frameon=False, fontsize=9.5, loc="upper left"); _despine(ax)
    ax.set_xlim(0, 1); ax.set_ylim(0, 1)
    fig.tight_layout(); fig.savefig(f"{OUT}/logreg_calibration.png", dpi=150, bbox_inches="tight")
    plt.close(fig); print("wrote logreg_calibration.png")


if __name__ == "__main__":
    logreg_sigmoid()
    logreg_boundary()
    logreg_loss_surface()
    logreg_calibration()
    print("OUT:", OUT)
