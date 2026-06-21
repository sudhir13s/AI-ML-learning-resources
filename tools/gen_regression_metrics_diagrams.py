"""Regression-metrics concept-page diagrams (muted palette, parallel scale).

Four figures for 03. Supervised_Learning/concepts/15-Regression-Metrics.md:
  1. rm_loss_curves.png -- MSE vs MAE vs Huber as a function of the residual r:
     quadratic (MSE) vs linear (MAE) vs hybrid (Huber, quadratic near 0 / linear far).
  2. rm_outlier.png     -- the outlier-sensitivity demo: a single large outlier moves
     RMSE far more than MAE (measured bars, before vs after).
  3. rm_r2.png          -- R^2 visualized: the TSS=ESS+RSS decomposition, with the
     mean line (SS_tot) vs the fitted line (SS_res) and the residuals to each.
  4. rm_quantile.png    -- quantile / pinball loss asymmetry for a few tau (the kink
     at r=0 with slopes tau and (1-tau)), and where its minimum lands.

All values are MEASURED (computed in-script), not hand-drawn. ylim is set before any
annotation so axes never blow up.
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


def _despine(ax):
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)


def loss_curves():
    """MSE (quadratic) vs MAE (linear) vs Huber (hybrid) as a function of residual r."""
    r = np.linspace(-4, 4, 400)
    delta = 1.0
    mse = 0.5 * r ** 2                      # 1/2 r^2 so slope at large r is comparable
    mae = np.abs(r)
    huber = np.where(np.abs(r) <= delta,
                     0.5 * r ** 2,
                     delta * (np.abs(r) - 0.5 * delta))
    fig, ax = plt.subplots(figsize=(8.6, 5.4))
    ax.plot(r, mse, color=BLUE, lw=2.8, label=r"MSE  $\frac{1}{2}r^2$  (quadratic)")
    ax.plot(r, mae, color=GREEN, lw=2.8, label=r"MAE  $|r|$  (linear, robust)")
    ax.plot(r, huber, color=AMBER, lw=2.8, ls="--",
            label=r"Huber ($\delta{=}1$): quadratic near 0, linear far")
    ax.axvline(delta, color=SLATE, ls=":", lw=1.2)
    ax.axvline(-delta, color=SLATE, ls=":", lw=1.2)
    ax.set_ylim(0, 9)                       # set BEFORE annotating
    ax.set_xlim(-4, 4)
    ax.text(delta + 0.08, 4.3, r"$|r|{=}\delta$:" + "\nHuber switches\nto linear",
            fontsize=8.5, color=SLATE)
    ax.annotate("MSE explodes\non outliers", (3.4, 0.5 * 3.4 ** 2), xytext=(2.1, 8.1),
                fontsize=9.5, color=BLUE, fontweight="bold",
                arrowprops=dict(arrowstyle="->", color=BLUE, lw=1.4))
    ax.annotate("MAE grows\nlinearly", (3.6, 3.6), xytext=(-1.5, 4.9),
                fontsize=9.5, color=GREEN, fontweight="bold",
                arrowprops=dict(arrowstyle="->", color=GREEN, lw=1.4))
    ax.set_xlabel(r"residual  $r = y - \hat{y}$")
    ax.set_ylabel("loss contribution of one point")
    ax.set_title("How each metric penalizes an error: square vs absolute vs hybrid",
                 fontsize=13, fontweight="bold")
    ax.legend(loc="upper center", fontsize=9.5, framealpha=0.95)
    _despine(ax)
    fig.tight_layout(); fig.savefig(f"{OUT}/rm_loss_curves.png", dpi=150, bbox_inches="tight")
    plt.close(fig); print("wrote rm_loss_curves.png")


def outlier_demo():
    """One outlier moves RMSE far more than MAE — measured bars, before vs after."""
    errs_clean = np.array([2.0, -1.0, 3.0, -2.0, 1.0])          # |errors| = 2,1,3,2,1
    errs_out = errs_clean.copy(); errs_out[2] = 20.0            # one big outlier
    def mae(e): return np.mean(np.abs(e))
    def rmse(e): return np.sqrt(np.mean(e ** 2))
    vals = {
        "MAE\n(clean)": (mae(errs_clean), GREEN),
        "RMSE\n(clean)": (rmse(errs_clean), BLUE),
        "MAE\n(+ 1 outlier)": (mae(errs_out), GREEN),
        "RMSE\n(+ 1 outlier)": (rmse(errs_out), RED),
    }
    fig, ax = plt.subplots(figsize=(8.8, 5.4))
    xs = np.arange(len(vals))
    heights = [v for v, _ in vals.values()]
    cols = [c for _, c in vals.values()]
    bars = ax.bar(xs, heights, color=cols, alpha=0.9, edgecolor="white", lw=1.5, width=0.62)
    ax.set_ylim(0, max(heights) * 1.22)                         # set BEFORE annotating
    for b, h in zip(bars, heights):
        ax.text(b.get_x() + b.get_width() / 2, h + max(heights) * 0.02,
                f"{h:.2f}", ha="center", fontsize=11, fontweight="bold")
    ax.set_xticks(xs); ax.set_xticklabels(vals.keys(), fontsize=10)
    ax.set_ylabel("error metric value")
    mae_jump = mae(errs_out) / mae(errs_clean)
    rmse_jump = rmse(errs_out) / rmse(errs_clean)
    ax.set_title(f"One outlier: RMSE jumps {rmse_jump:.1f}x, MAE only {mae_jump:.1f}x",
                 fontsize=13, fontweight="bold")
    ax.text(0.5, max(heights) * 1.10,
            "same 5 errors;\nthe 3rd becomes 20", ha="center", fontsize=9.5,
            color=SLATE, style="italic")
    _despine(ax)
    fig.tight_layout(); fig.savefig(f"{OUT}/rm_outlier.png", dpi=150, bbox_inches="tight")
    plt.close(fig); print("wrote rm_outlier.png")


def r2_decomposition():
    """R^2 visualized: SS_tot (to the mean) vs SS_res (to the fit), TSS=ESS+RSS."""
    rng = np.random.default_rng(3)
    x = np.linspace(1, 9, 12)
    y = 1.2 * x + 2 + rng.normal(0, 2.2, x.size)
    # OLS fit
    A = np.vstack([x, np.ones_like(x)]).T
    beta, _, _, _ = np.linalg.lstsq(A, y, rcond=None)
    yhat = A @ beta
    ybar = y.mean()
    ss_res = np.sum((y - yhat) ** 2)
    ss_tot = np.sum((y - ybar) ** 2)
    r2 = 1 - ss_res / ss_tot

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12.0, 5.2), sharey=True)
    # Left: residuals to the mean (SS_tot)
    ax1.axhline(ybar, color=PURPLE, lw=2.4, label=f"mean $\\bar y$ = {ybar:.1f}")
    for xi, yi in zip(x, y):
        ax1.plot([xi, xi], [ybar, yi], color=RED, lw=1.6, alpha=0.8)
    ax1.scatter(x, y, color=BLUE, s=46, zorder=3, edgecolor="white")
    ax1.set_title(f"$SS_{{tot}}$: spread around the mean = {ss_tot:.1f}",
                  fontsize=12, fontweight="bold", color=PURPLE)
    ax1.set_xlabel("x"); ax1.set_ylabel("y"); ax1.legend(fontsize=9.5, loc="upper left")
    _despine(ax1)
    # Right: residuals to the fit (SS_res)
    ax2.plot(x, yhat, color=GREEN, lw=2.4, label="fitted line $\\hat y$")
    for xi, yi, yhi in zip(x, y, yhat):
        ax2.plot([xi, xi], [yhi, yi], color=RED, lw=1.6, alpha=0.8)
    ax2.scatter(x, y, color=BLUE, s=46, zorder=3, edgecolor="white")
    ax2.set_title(f"$SS_{{res}}$: spread around the fit = {ss_res:.1f}",
                  fontsize=12, fontweight="bold", color=GREEN)
    ax2.set_xlabel("x"); ax2.legend(fontsize=9.5, loc="upper left")
    _despine(ax2)
    fig.suptitle(f"$R^2 = 1 - SS_{{res}}/SS_{{tot}} = 1 - {ss_res:.1f}/{ss_tot:.1f} = {r2:.2f}$"
                 "   (fraction of variance the model explains)",
                 fontsize=13.5, fontweight="bold", y=1.02)
    fig.tight_layout(); fig.savefig(f"{OUT}/rm_r2.png", dpi=150, bbox_inches="tight")
    plt.close(fig); print(f"wrote rm_r2.png  (R2={r2:.4f})")


def quantile_loss():
    """Pinball / quantile loss asymmetry for a few tau, as a function of residual."""
    r = np.linspace(-4, 4, 400)
    def pinball(r, tau):
        return np.where(r >= 0, tau * r, (tau - 1) * r)   # r = y - yhat
    fig, ax = plt.subplots(figsize=(8.6, 5.4))
    for tau, col in [(0.1, BLUE), (0.5, GREEN), (0.9, RED)]:
        ax.plot(r, pinball(r, tau), color=col, lw=2.8, label=f"$\\tau$ = {tau}")
    ax.axvline(0, color=SLATE, ls=":", lw=1.2)
    ax.set_ylim(0, 4.0)                                   # set BEFORE annotating
    ax.set_xlim(-4, 4)
    ax.annotate(r"$\tau{=}0.9$, $r>0$:" + "\nslope = $\\tau$ = 0.9", (3.2, 0.9 * 3.2),
                xytext=(1.0, 3.4), fontsize=9.0, color=RED, fontweight="bold",
                arrowprops=dict(arrowstyle="->", color=RED, lw=1.3))
    # over-predict side of tau=0.9: slope = 1-tau = 0.1 (shallow), point on the BLUE-ish low line
    ax.annotate(r"$\tau{=}0.9$, $r<0$:" + "\nslope = $1-\\tau$ = 0.1", (-3.0, (1 - 0.9) * 3.0),
                xytext=(-3.9, 1.1), fontsize=9.0, color=SLATE, fontweight="bold",
                arrowprops=dict(arrowstyle="->", color=SLATE, lw=1.3))
    ax.set_xlabel(r"residual  $r = y - \hat{y}$")
    ax.set_ylabel("pinball loss of one point")
    ax.set_title(r"Quantile (pinball) loss: asymmetric slopes pin $\hat y$ to the $\tau$-quantile",
                 fontsize=12.5, fontweight="bold")
    ax.legend(loc="upper center", fontsize=10, framealpha=0.95)
    _despine(ax)
    fig.tight_layout(); fig.savefig(f"{OUT}/rm_quantile.png", dpi=150, bbox_inches="tight")
    plt.close(fig); print("wrote rm_quantile.png")


if __name__ == "__main__":
    loss_curves()
    outlier_demo()
    r2_decomposition()
    quantile_loss()
    print("OUT:", OUT)
