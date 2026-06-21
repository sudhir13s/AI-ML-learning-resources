"""Bias-variance concept-page diagrams (muted palette, parallel matplotlib scale).

Four figures for 03. Supervised_Learning/concepts/12-Bias-Variance-Tradeoff.md:
  1. bv_dartboard.png -- the classic 2x2 analogy: bias (off-centre) x variance
     (scattered), shown as dart patterns on a target.
  2. bv_ucurve.png -- a REAL bootstrap bias-variance decomposition: sweep
     polynomial degree, measure bias^2, variance, and total test error -> the
     U-shaped curve with the sweet spot.
  3. bv_knn.png -- kNN bias/variance as a function of k (the 1/k story): small k
     = low bias / high variance, large k = high bias / low variance, measured by
     bootstrap on a regression target.
  4. bv_double_descent.png -- the modern over-parameterized regime: classical U
     up to the interpolation threshold, then a SECOND descent past it (Belkin
     2019), measured with random-feature ridgeless regression.
"""
import os, matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Circle
import numpy as np

OUT = os.path.join(os.path.dirname(__file__), "..", "03. Supervised_Learning", "concepts", "images")
OUT = os.path.abspath(OUT)
os.makedirs(OUT, exist_ok=True)

BLUE, PURPLE, GREEN, RED, SLATE, AMBER, NAVY = (
    "#3A6B96", "#5D4A8A", "#2E7A5A", "#8B3B4A", "#4A5B6E", "#7A6528", "#2A5B80")
plt.rcParams.update({"font.size": 11, "font.family": "DejaVu Sans"})


def _despine(ax):
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)


def bv_dartboard():
    rng = np.random.default_rng(1)
    fig, axes = plt.subplots(2, 2, figsize=(8.4, 8.4))
    configs = [
        (axes[0, 0], (0, 0), 0.10, "low bias · low variance", GREEN, "the goal"),
        (axes[0, 1], (0, 0), 0.45, "low bias · high variance", AMBER, "overfit: right on average, scattered"),
        (axes[1, 0], (0.7, 0.4), 0.10, "high bias · low variance", BLUE, "underfit: tight but off-target"),
        (axes[1, 1], (0.7, 0.4), 0.45, "high bias · high variance", RED, "worst of both"),
    ]
    for ax, center, spread, title, col, sub in configs:
        for r, c in [(1.0, "#e9ecf0"), (0.66, "#dde2e8"), (0.33, "#cfd6de")]:
            ax.add_patch(Circle((0, 0), r, facecolor=c, edgecolor="white", zorder=1))
        ax.add_patch(Circle((0, 0), 0.08, facecolor=SLATE, zorder=2))
        pts = rng.normal(center, spread, (12, 2))
        ax.scatter(pts[:, 0], pts[:, 1], color=col, s=55, zorder=4, edgecolor="white")
        ax.set_xlim(-1.1, 1.1); ax.set_ylim(-1.1, 1.1); ax.set_aspect("equal"); ax.axis("off")
        ax.set_title(title, fontsize=11.5, fontweight="bold", color=col)
        ax.text(0, -1.28, sub, ha="center", fontsize=8.8, color=SLATE)
    fig.suptitle("Bias vs variance: the dartboard (bullseye = the true answer)",
                 fontsize=13.5, fontweight="bold", y=0.99)
    fig.tight_layout(rect=[0, 0, 1, 0.97]); fig.savefig(f"{OUT}/bv_dartboard.png", dpi=150, bbox_inches="tight")
    plt.close(fig); print("wrote bv_dartboard.png")


def bv_ucurve():
    rng = np.random.default_rng(0)
    f = lambda x: np.sin(1.5 * x)                       # true function
    noise = 0.25
    xt = np.linspace(-3, 3, 60)                          # test points (full range: edges overfit)
    degrees = list(range(1, 13))
    bias2, var, total = [], [], []
    for d in degrees:
        preds = []
        for _ in range(150):                            # bootstrap training sets
            xtr = rng.uniform(-3, 3, 18); ytr = f(xtr) + rng.normal(0, noise, 18)  # few points -> high-degree overfits
            coef = np.polyfit(xtr, ytr, d)
            preds.append(np.polyval(coef, xt))
        preds = np.array(preds)                          # (runs, test)
        mean_pred = preds.mean(0)
        bias2.append(((mean_pred - f(xt)) ** 2).mean())
        var.append(preds.var(0).mean())
        total.append(bias2[-1] + var[-1] + noise ** 2)
    fig, ax = plt.subplots(figsize=(8.6, 5.2))
    YTOP = 1.2
    ax.set_ylim(0, YTOP)                                # set limits BEFORE annotations
    ax.plot(degrees, np.clip(bias2, 0, YTOP), color=BLUE, lw=2.4, marker="o", ms=4, label="bias²  (down with complexity)")
    ax.plot(degrees, np.clip(var, 0, YTOP), color=RED, lw=2.4, marker="s", ms=4, label="variance  (up with complexity)")
    ax.plot(degrees, np.clip(total, 0, YTOP), color=PURPLE, lw=2.8, marker="^", ms=4, label="total test error (= bias² + var + noise)")
    ax.axhline(noise ** 2, color=SLATE, ls=":", lw=1.3)
    ax.text(8.2, noise**2 + 0.02, "irreducible noise σ²", fontsize=8.5, color=SLATE)
    best = degrees[int(np.argmin(total))]
    ax.axvline(best, color=GREEN, ls="--", lw=2)
    ax.text(best + 0.15, YTOP * 0.82, f"sweet spot\n(degree {best})", color=GREEN, fontsize=9.5, fontweight="bold")
    ax.set_xlabel("model complexity (polynomial degree)"); ax.set_ylabel("error")
    ax.set_title("The bias–variance trade-off, measured by bootstrap",
                 fontsize=13, fontweight="bold")
    ax.legend(frameon=False, fontsize=9.5); _despine(ax)
    fig.tight_layout(); fig.savefig(f"{OUT}/bv_ucurve.png", dpi=150)
    plt.close(fig); print("wrote bv_ucurve.png")


def bv_knn():
    """kNN bias/variance vs k -- the 1/k complexity story, measured by bootstrap.

    Small k -> flexible -> low bias, high variance.
    Large k -> smooth  -> high bias,  low variance.
    The effective complexity of kNN is n/k, so the x-axis runs k from large
    (left, simple) to small (right, complex) to read like the U-curve.
    """
    rng = np.random.default_rng(3)
    f = lambda x: np.sin(1.5 * x)
    noise = 0.25
    xt = np.linspace(-2.6, 2.6, 50)[:, None]            # test grid
    fx = f(xt[:, 0])
    ks = [40, 25, 18, 12, 9, 7, 5, 4, 3, 2, 1]          # large k (simple) -> small k (complex)
    bias2, var, total = [], [], []
    n = 80
    for k in ks:
        preds = []
        for _ in range(160):
            xtr = rng.uniform(-2.6, 2.6, n); ytr = f(xtr) + rng.normal(0, noise, n)
            order = np.argsort(np.abs(xtr[None, :] - xt), axis=1)[:, :k]   # k nearest by |x-x'|
            preds.append(ytr[order].mean(1))
        preds = np.array(preds)
        bias2.append(((preds.mean(0) - fx) ** 2).mean())
        var.append(preds.var(0).mean())
        total.append(bias2[-1] + var[-1] + noise ** 2)
    x = np.arange(len(ks))
    fig, ax = plt.subplots(figsize=(8.6, 5.2))
    YTOP = 0.9
    ax.set_ylim(0, YTOP)
    ax.plot(x, np.clip(bias2, 0, YTOP), color=BLUE, lw=2.4, marker="o", ms=5, label="bias²  (rises as k grows)")
    ax.plot(x, np.clip(var, 0, YTOP), color=RED, lw=2.4, marker="s", ms=5, label="variance  (rises as k shrinks)")
    ax.plot(x, np.clip(total, 0, YTOP), color=PURPLE, lw=2.8, marker="^", ms=5, label="total test error")
    ax.axhline(noise ** 2, color=SLATE, ls=":", lw=1.3)
    ax.text(len(ks) - 3.4, noise**2 + 0.015, "irreducible noise σ²", fontsize=8.5, color=SLATE)
    best = int(np.argmin(total))
    ax.axvline(best, color=GREEN, ls="--", lw=2)
    ax.text(best + 0.15, YTOP * 0.8, f"sweet spot\n(k = {ks[best]})", color=GREEN, fontsize=9.5, fontweight="bold")
    ax.set_xticks(x); ax.set_xticklabels(ks)
    ax.set_xlabel("k  (large k = simple / smooth  →  small k = complex / flexible)")
    ax.set_ylabel("error")
    ax.set_title("kNN: complexity is 1/k — bias and variance trade off with k",
                 fontsize=13, fontweight="bold")
    ax.legend(frameon=False, fontsize=9.5, loc="upper center"); _despine(ax)
    fig.tight_layout(); fig.savefig(f"{OUT}/bv_knn.png", dpi=150)
    plt.close(fig); print("wrote bv_knn.png")


def bv_double_descent():
    """Double descent (Belkin 2019), measured with a linear random-feature model.

    Teacher–student random-features regression (the canonical setup that exhibits
    double descent cleanly). A fixed pool of D random features defines the truth
    y = Phi_full @ w_star + noise. A student fits the FIRST P of those features
    with the least-norm (ridgeless pseudoinverse) solution, n training points
    fixed. Sweep P:
      - P < n  : classical regime, U-shaped error;
      - P = n  : interpolation threshold, the variance spike;
      - P > n  : over-parameterized, the least-norm solution shrinks and test
                 error DESCENDS AGAIN -> the second descent.
    """
    rng = np.random.default_rng(0)
    n = 50                                              # training points -> threshold at P = n
    D = 600                                             # full feature pool (the "truth" basis)
    d_in = 30
    noise = 0.30

    Wpool = rng.standard_normal((d_in, D)) / np.sqrt(d_in)
    bpool = rng.uniform(0, 2 * np.pi, D)
    w_star = rng.standard_normal(D) / np.sqrt(D)        # teacher weights over the full pool

    def feats(X, P):                                    # first P random ReLU-ish features
        return np.maximum(X @ Wpool[:, :P] + bpool[:P], 0.0)

    Xtr = rng.standard_normal((n, d_in)); Xte = rng.standard_normal((600, d_in))
    full_tr = np.maximum(Xtr @ Wpool + bpool, 0.0)
    full_te = np.maximum(Xte @ Wpool + bpool, 0.0)
    ytr_clean = full_tr @ w_star; yte = full_te @ w_star

    Ps = sorted(set(list(range(2, 50, 3)) + [n - 2, n - 1, n, n + 1, n + 2]
                     + list(range(54, 600, 18)) + [599]))
    test_err = []
    for P in Ps:
        errs = []
        for _ in range(40):                             # average over noise draws
            ytr = ytr_clean + rng.normal(0, noise, n)
            Phi = feats(Xtr, P)
            w = np.linalg.pinv(Phi) @ ytr               # least-norm / ridgeless solution
            pred = feats(Xte, P) @ w
            errs.append(np.mean((pred - yte) ** 2))
        test_err.append(np.median(errs))                # median: robust to the spike's heavy tail
    Ps = np.array(Ps); test_err = np.array(test_err)
    fig, ax = plt.subplots(figsize=(8.6, 5.2))
    YTOP = max(2.0, np.percentile(test_err[Ps < n], 95) * 1.4)
    ax.set_ylim(0, YTOP)
    ax.plot(Ps / n, np.clip(test_err, 0, YTOP), color=PURPLE, lw=2.6, marker="o", ms=3.5,
            label="test error (random-feature ridgeless)")
    ax.axvline(1.0, color=RED, ls="--", lw=2)
    ax.text(1.08, YTOP * 0.88, "interpolation\nthreshold (P = n)", color=RED, fontsize=9, fontweight="bold")
    ax.annotate("classical\nU-curve", xy=(0.45, YTOP * 0.34), color=BLUE, fontsize=9.5, fontweight="bold", ha="center")
    ax.annotate("modern\nsecond descent", xy=(7.5, YTOP * 0.22), color=GREEN, fontsize=9.5, fontweight="bold", ha="center")
    ax.set_xlabel("model complexity  (parameters P / training points n)")
    ax.set_ylabel("test error")
    ax.set_title("Double descent: test error descends again past interpolation",
                 fontsize=13, fontweight="bold")
    ax.legend(frameon=False, fontsize=9.5); _despine(ax)
    fig.tight_layout(); fig.savefig(f"{OUT}/bv_double_descent.png", dpi=150)
    plt.close(fig); print("wrote bv_double_descent.png")


if __name__ == "__main__":
    bv_dartboard()
    bv_ucurve()
    bv_knn()
    bv_double_descent()
    print("OUT:", OUT)
