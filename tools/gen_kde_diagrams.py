"""Kernel Density Estimation concept-page diagrams (muted palette, parallel matplotlib scale).

Four MEASURED visuals for 04. Unsupervised_Learning/concepts/10-Kernel-Density-Estimation.md:
  1. kde_construction.png  -- the CONSTRUCTION: an individual Gaussian bump dropped at each
     data point, summed (and divided by n) into the smooth estimate. Annotated.
  2. kde_bandwidth.png     -- the BANDWIDTH knob: undersmoothed (spiky) vs oversmoothed (flat)
     vs ~optimal, all against the true density. The bias-variance trade-off, made visible.
  3. kde_vs_histogram.png  -- KDE vs HISTOGRAM on the same sample: bin-edge dependence and
     discontinuity of the histogram vs the smooth, edge-free KDE.
  4. kde_kernels.png       -- KERNEL SHAPE barely matters: Gaussian / Epanechnikov / tophat at
     the SAME bandwidth give nearly the same estimate (and the kernel shapes themselves).

All curves are computed from data (np.random) — nothing is hand-drawn. Run with:
  /Users/sudhirsingh/.uv/envs/ml-py312/bin/python3 tools/gen_kde_diagrams.py
"""
import os, matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

OUT = os.path.join(os.path.dirname(__file__), "..", "04. Unsupervised_Learning", "concepts", "images")
OUT = os.path.abspath(OUT)
os.makedirs(OUT, exist_ok=True)

BLUE, PURPLE, GREEN, RED, SLATE, AMBER, NAVY = (
    "#3A6B96", "#5D4A8A", "#2E7A5A", "#8B3B4A", "#4A5B6E", "#7A6528", "#2A5B80")
plt.rcParams.update({"font.size": 11, "font.family": "DejaVu Sans"})


def _despine(ax):
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)


# ---------- shared kernels -----------------------------------------------------
def gaussian_kernel(u):
    return np.exp(-0.5 * u ** 2) / np.sqrt(2 * np.pi)


def epanechnikov_kernel(u):
    out = np.where(np.abs(u) <= 1, 0.75 * (1 - u ** 2), 0.0)
    return out


def tophat_kernel(u):
    return np.where(np.abs(u) <= 1, 0.5, 0.0)


def kde_eval(grid, data, h, kernel=gaussian_kernel):
    """f_hat_h(x) = (1/(n h)) sum_i K((x - x_i)/h), evaluated on grid."""
    n = len(data)
    u = (grid[:, None] - data[None, :]) / h          # (G, n)
    return kernel(u).sum(axis=1) / (n * h)


# ---------- 1. Construction: bumps summing to the estimate ---------------------
def construction():
    rng = np.random.default_rng(7)
    data = np.array([-2.1, -1.3, 0.4, 1.0, 2.6])     # n=5, fixed for a clean teaching picture
    n, h = len(data), 0.7
    grid = np.linspace(-5, 5, 800)
    est = kde_eval(grid, data, h)

    fig, ax = plt.subplots(figsize=(8.8, 4.9))
    # individual bumps: each is (1/(n h)) K((x-x_i)/h) so they literally sum to est
    for i, xi in enumerate(data):
        bump = gaussian_kernel((grid - xi) / h) / (n * h)
        ax.plot(grid, bump, color=PURPLE, lw=1.3, alpha=0.65,
                label="individual kernels  (1/nh)·K((x−xᵢ)/h)" if i == 0 else None)
        ax.fill_between(grid, bump, color=PURPLE, alpha=0.08)
    ax.plot(grid, est, color=GREEN, lw=3.0, label="KDE estimate  f̂(x) = Σ bumps")
    ax.plot(data, np.zeros_like(data), "o", color=BLUE, ms=9, zorder=5,
            markeredgecolor="white", label="data points xᵢ")
    for xi in data:
        ax.vlines(xi, 0, gaussian_kernel(0) / (n * h), color=BLUE, lw=0.8, alpha=0.5)
    ax.annotate("drop a smooth bump\nat each data point…", (data[2], gaussian_kernel(0) / (n * h)),
                textcoords="offset points", xytext=(28, 18), fontsize=9.5,
                color=PURPLE, fontweight="bold",
                arrowprops=dict(arrowstyle="->", color=PURPLE))
    ax.annotate("…then average them\ninto a smooth density", (-1.6, est[np.argmin(np.abs(grid + 1.6))]),
                textcoords="offset points", xytext=(-150, 24), fontsize=9.5,
                color=GREEN, fontweight="bold",
                arrowprops=dict(arrowstyle="->", color=GREEN))
    ax.set_xlabel("x"); ax.set_ylabel("density")
    ax.set_title("KDE construction: a kernel bump per point, averaged\n(n=5, Gaussian kernel, h=0.7)",
                 fontsize=14, fontweight="bold")
    ax.legend(loc="upper right", frameon=False, fontsize=9)
    ax.set_xlim(-5, 5); ax.set_ylim(0, None); _despine(ax)
    fig.tight_layout(); fig.savefig(f"{OUT}/kde_construction.png", dpi=150, bbox_inches="tight")
    plt.close(fig); print("wrote kde_construction.png")


# ---------- 2. Bandwidth: under / over / ~optimal vs truth ---------------------
def bandwidth():
    rng = np.random.default_rng(3)
    # a bimodal truth: mixture of two Gaussians
    n = 200
    comp = rng.random(n) < 0.5
    data = np.where(comp, rng.normal(-2.0, 0.8, n), rng.normal(2.0, 1.0, n))
    grid = np.linspace(-6, 6, 800)

    def true_pdf(x):
        g = lambda x, m, s: np.exp(-0.5 * ((x - m) / s) ** 2) / (s * np.sqrt(2 * np.pi))
        return 0.5 * g(x, -2.0, 0.8) + 0.5 * g(x, 2.0, 1.0)

    # Silverman's rule for reference
    sd = data.std(ddof=1)
    iqr = np.subtract(*np.percentile(data, [75, 25]))
    A = min(sd, iqr / 1.349)
    h_silver = 0.9 * A * n ** (-1 / 5)

    fig, ax = plt.subplots(figsize=(8.8, 4.9))
    ax.plot(grid, true_pdf(grid), color=SLATE, lw=2.6, ls="--", label="true density")
    ax.plot(grid, kde_eval(grid, data, 0.12), color=RED, lw=1.8,
            label="h=0.12  undersmoothed (spiky, high variance)")
    ax.plot(grid, kde_eval(grid, data, h_silver), color=GREEN, lw=2.6,
            label=f"h={h_silver:.2f}  Silverman (≈ just right)")
    ax.plot(grid, kde_eval(grid, data, 1.4), color=AMBER, lw=2.0,
            label="h=1.40  oversmoothed (flat, high bias)")
    ax.plot(data, np.full_like(data, -0.004), "|", color=BLUE, ms=8, alpha=0.5)
    ax.set_xlabel("x"); ax.set_ylabel("density")
    ax.set_title("Bandwidth is the bias–variance knob (bimodal truth, n=200)",
                 fontsize=14, fontweight="bold")
    ax.legend(loc="upper right", frameon=False, fontsize=8.5)
    ax.set_xlim(-6, 6); ax.set_ylim(-0.01, None); _despine(ax)
    fig.tight_layout(); fig.savefig(f"{OUT}/kde_bandwidth.png", dpi=150, bbox_inches="tight")
    plt.close(fig); print("wrote kde_bandwidth.png")


# ---------- 3. KDE vs histogram (bin-edge dependence) --------------------------
def kde_vs_histogram():
    rng = np.random.default_rng(11)
    n = 150
    data = np.concatenate([rng.normal(-1.5, 0.7, n // 2), rng.normal(1.8, 0.9, n // 2)])
    grid = np.linspace(-5, 5, 800)

    def true_pdf(x):
        g = lambda x, m, s: np.exp(-0.5 * ((x - m) / s) ** 2) / (s * np.sqrt(2 * np.pi))
        return 0.5 * g(x, -1.5, 0.7) + 0.5 * g(x, 1.8, 0.9)

    sd = data.std(ddof=1)
    iqr = np.subtract(*np.percentile(data, [75, 25]))
    h = 0.9 * min(sd, iqr / 1.349) * n ** (-1 / 5)

    fig, axes = plt.subplots(1, 2, figsize=(11.2, 4.6), sharey=True)
    # two histograms differing only by a SHIFTED bin origin -> different shapes
    for ax, shift, title in ((axes[0], 0.0, "histogram, bins start at −5.0"),
                             (axes[1], 0.25, "histogram, bins shifted by +0.25")):
        edges = np.arange(-5 + shift, 5 + shift + 0.6, 0.6)
        ax.hist(data, bins=edges, density=True, color=SLATE, alpha=0.45,
                edgecolor="white", label="histogram")
        ax.plot(grid, kde_eval(grid, data, h), color=GREEN, lw=2.8, label=f"KDE (h={h:.2f})")
        ax.plot(grid, true_pdf(grid), color=RED, lw=1.8, ls="--", label="true density")
        ax.plot(data, np.full_like(data, -0.005), "|", color=BLUE, ms=7, alpha=0.45)
        ax.set_title(title, fontsize=12, fontweight="bold")
        ax.set_xlabel("x"); _despine(ax); ax.set_xlim(-5, 5)
    axes[0].set_ylabel("density")
    axes[0].legend(loc="upper left", frameon=False, fontsize=8.5)
    fig.suptitle("Histogram bins jump with the bin origin; the KDE is smooth and edge-free",
                 fontsize=14, fontweight="bold")
    fig.tight_layout(rect=[0, 0, 1, 0.95])
    fig.savefig(f"{OUT}/kde_vs_histogram.png", dpi=150, bbox_inches="tight")
    plt.close(fig); print("wrote kde_vs_histogram.png")


# ---------- 4. Kernel shape barely matters at the same h ----------------------
def kernels():
    rng = np.random.default_rng(5)
    n = 120
    data = rng.normal(0, 1, n) + rng.normal(3, 0.6, n) * (rng.random(n) < 0.4)
    grid = np.linspace(-4, 6, 800)
    h = 0.6

    fig, axes = plt.subplots(1, 2, figsize=(11.2, 4.6))
    # (a) the three kernel shapes
    u = np.linspace(-2.2, 2.2, 400)
    axes[0].plot(u, gaussian_kernel(u), color=BLUE, lw=2.6, label="Gaussian")
    axes[0].plot(u, epanechnikov_kernel(u), color=GREEN, lw=2.6, label="Epanechnikov")
    axes[0].plot(u, tophat_kernel(u), color=AMBER, lw=2.6, label="tophat (uniform)")
    axes[0].set_title("the kernel shapes K(u)", fontsize=12, fontweight="bold")
    axes[0].set_xlabel("u = (x − xᵢ)/h"); axes[0].set_ylabel("K(u)")
    axes[0].legend(loc="upper right", frameon=False, fontsize=9); _despine(axes[0])

    # (b) the resulting estimates at the SAME h -> nearly identical
    axes[1].plot(grid, kde_eval(grid, data, h, gaussian_kernel), color=BLUE, lw=2.6, label="Gaussian KDE")
    axes[1].plot(grid, kde_eval(grid, data, h, epanechnikov_kernel), color=GREEN, lw=2.2, ls="--", label="Epanechnikov KDE")
    axes[1].plot(grid, kde_eval(grid, data, h, tophat_kernel), color=AMBER, lw=1.8, ls=":", label="tophat KDE")
    axes[1].plot(data, np.full_like(data, -0.004), "|", color=SLATE, ms=7, alpha=0.5)
    axes[1].set_title(f"same h={h} → nearly the same estimate", fontsize=12, fontweight="bold")
    axes[1].set_xlabel("x"); axes[1].set_ylabel("density")
    axes[1].legend(loc="upper right", frameon=False, fontsize=9); _despine(axes[1])

    fig.suptitle("Kernel SHAPE matters far less than the bandwidth", fontsize=14, fontweight="bold")
    fig.tight_layout(rect=[0, 0, 1, 0.95])
    fig.savefig(f"{OUT}/kde_kernels.png", dpi=150, bbox_inches="tight")
    plt.close(fig); print("wrote kde_kernels.png")


if __name__ == "__main__":
    construction()
    bandwidth()
    kde_vs_histogram()
    kernels()
    print("OUT:", OUT)
