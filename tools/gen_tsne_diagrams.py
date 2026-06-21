"""t-SNE concept-page diagrams (muted palette, parallel scale). REAL measured.

Four figures for 04. Unsupervised_Learning/concepts/07-t-SNE.md:
  1. tsne_vs_pca.png      -- the same digits subset under PCA (2-D) and t-SNE (2-D)
     side by side: PCA's overlapping smear vs t-SNE's separated clusters. This is
     the headline "why t-SNE for visualization" figure.
  2. tsne_perplexity.png  -- the SAME data embedded at perplexity 5 / 30 / 50, so
     you SEE that perplexity reshapes the picture (the distill.pub lesson).
  3. tsne_crowding.png     -- Gaussian vs Student-t (df=1) tail: the heavy tail of
     the t-distribution is what lets moderate high-D distances map farther apart in
     low-D, curing the crowding problem. Annotated with the tail-mass ratio.
  4. tsne_neighbor_kernel.png -- the high-D Gaussian affinity kernel for three
     perplexities (small/medium/large sigma) over distance, showing how perplexity
     = effective neighborhood width.

All numbers are computed here so the page can quote them exactly. Run with:
  ~/.uv/envs/ml-py312/bin/python3 tools/gen_tsne_diagrams.py
"""
import os, matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from sklearn.datasets import load_digits, make_swiss_roll
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE

OUT = os.path.join(os.path.dirname(__file__), "..", "04. Unsupervised_Learning", "concepts", "images")
OUT = os.path.abspath(OUT)
os.makedirs(OUT, exist_ok=True)

BLUE, PURPLE, GREEN, RED, SLATE, AMBER, NAVY = (
    "#3A6B96", "#5D4A8A", "#2E7A5A", "#8B3B4A", "#4A5B6E", "#7A6528", "#2A5B80")
# 10 distinguishable muted colors for the 10 digit classes.
DIGIT_COLORS = ["#3A6B96", "#2E7A5A", "#8B3B4A", "#5D4A8A", "#7A6528",
                "#2A5B80", "#4A5B6E", "#B5651D", "#107896", "#9A4C95"]
plt.rcParams.update({"font.size": 11, "font.family": "DejaVu Sans"})


def _despine(ax):
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)


def _scatter_embedding(ax, Y, y, title):
    for d in range(10):
        m = y == d
        ax.scatter(Y[m, 0], Y[m, 1], s=8, c=DIGIT_COLORS[d], label=str(d),
                   alpha=0.75, linewidths=0)
    ax.set_title(title, fontsize=13, fontweight="bold")
    ax.set_xticks([]); ax.set_yticks([])
    for s in ax.spines.values():
        s.set_edgecolor("#999")


def vs_pca():
    """PCA vs t-SNE on the same digits subset — overlap vs separation."""
    digits = load_digits()
    X, y = digits.data, digits.target  # 1797 x 64
    # PCA to 2-D (linear).
    Ypca = PCA(n_components=2, random_state=0).fit_transform(X)
    # t-SNE to 2-D (non-linear). PCA-init for determinism + speed.
    ts = TSNE(n_components=2, perplexity=30, init="pca",
              learning_rate="auto", random_state=42, max_iter=1000)
    Ytsne = ts.fit_transform(X)
    print(f"[vs_pca] digits: {X.shape[0]} points x {X.shape[1]} dims, 10 classes")
    print(f"[vs_pca] final KL divergence (t-SNE) = {ts.kl_divergence_:.4f}")

    fig, axes = plt.subplots(1, 2, figsize=(12, 5.6))
    _scatter_embedding(axes[0], Ypca, y, "PCA (linear) — clusters overlap")
    _scatter_embedding(axes[1], Ytsne, y, "t-SNE (non-linear) — clusters separate")
    handles, labels = axes[1].get_legend_handles_labels()
    fig.legend(handles, labels, loc="center right", title="digit",
               fontsize=9, title_fontsize=10, markerscale=1.6,
               frameon=False, bbox_to_anchor=(1.0, 0.5))
    fig.suptitle("Same 1,797 handwritten digits (64-D), projected to 2-D",
                 fontsize=14, fontweight="bold")
    fig.tight_layout(rect=[0, 0, 0.93, 0.96])
    p = os.path.join(OUT, "tsne_vs_pca.png")
    fig.savefig(p, dpi=130, bbox_inches="tight"); plt.close(fig)
    print("wrote", p)


def perplexity_panels():
    """Same digits, three perplexities — the picture changes with perplexity."""
    digits = load_digits()
    X, y = digits.data, digits.target
    perps = [5, 30, 50]
    fig, axes = plt.subplots(1, 3, figsize=(15, 5.2))
    for ax, perp in zip(axes, perps):
        ts = TSNE(n_components=2, perplexity=perp, init="pca",
                  learning_rate="auto", random_state=42, max_iter=1000)
        Y = ts.fit_transform(X)
        _scatter_embedding(ax, Y, y, f"perplexity = {perp}  (KL={ts.kl_divergence_:.2f})")
        print(f"[perplexity] perp={perp:>2}  KL={ts.kl_divergence_:.4f}")
    fig.suptitle("Perplexity reshapes the map — same data, three settings",
                 fontsize=14, fontweight="bold")
    fig.tight_layout(rect=[0, 0, 1, 0.95])
    p = os.path.join(OUT, "tsne_perplexity.png")
    fig.savefig(p, dpi=130, bbox_inches="tight"); plt.close(fig)
    print("wrote", p)


def crowding():
    """Gaussian vs Student-t(df=1) tail: why the heavy tail fixes crowding."""
    d = np.linspace(0, 6, 600)
    # Both peaked at 0, normalized to 1 there for a fair tail comparison.
    gauss = np.exp(-(d ** 2) / 2.0)            # unnormalized Gaussian kernel
    student = 1.0 / (1.0 + d ** 2)             # Student-t with 1 dof (Cauchy), q-kernel

    fig, ax = plt.subplots(figsize=(9.5, 5.6))
    ax.plot(d, gauss, color=BLUE, lw=2.6, label="Gaussian  exp(-d²/2)  (high-D affinity)")
    ax.plot(d, student, color=RED, lw=2.6, label="Student-t, 1 dof  (1+d²)⁻¹  (low-D affinity)")
    ax.fill_between(d, gauss, student, where=(student > gauss), color=RED, alpha=0.12)

    # Annotate the tail mass at a moderate distance d=3.
    d0 = 3.0
    g0 = float(np.exp(-(d0 ** 2) / 2.0))
    s0 = float(1.0 / (1.0 + d0 ** 2))
    ax.axvline(d0, color=SLATE, ls="--", lw=1.2)
    ax.annotate(f"at d=3:\n  Gaussian = {g0:.4f}\n  Student-t = {s0:.3f}\n  ratio ≈ {s0/g0:,.0f}×",
                xy=(d0, s0), xytext=(3.4, 0.55), fontsize=10.5,
                bbox=dict(boxstyle="round,pad=0.4", fc="#f2efe6", ec=AMBER),
                arrowprops=dict(arrowstyle="->", color=AMBER, lw=1.4))
    print(f"[crowding] at d=3.0  gauss={g0:.6f}  student={s0:.6f}  ratio={s0/g0:.1f}x")
    ax.set_xlabel("pairwise distance  d", fontsize=12)
    ax.set_ylabel("affinity (un-normalized)", fontsize=12)
    ax.set_title("Heavy tail cures crowding: the t-distribution keeps mass at moderate distance",
                 fontsize=13.5, fontweight="bold")
    ax.legend(fontsize=10.5, frameon=False, loc="upper right")
    ax.set_ylim(0, 1.05); ax.set_xlim(0, 6)
    _despine(ax)
    fig.tight_layout()
    p = os.path.join(OUT, "tsne_crowding.png")
    fig.savefig(p, dpi=130, bbox_inches="tight"); plt.close(fig)
    print("wrote", p)


def neighbor_kernel():
    """High-D Gaussian affinity vs distance for three sigmas = three perplexities."""
    d = np.linspace(0, 6, 600)
    sigmas = [(0.5, GREEN, "small σ — tight neighborhood (low perplexity)"),
              (1.2, BLUE, "medium σ"),
              (2.5, RED, "large σ — broad neighborhood (high perplexity)")]
    fig, ax = plt.subplots(figsize=(9.5, 5.6))
    for sig, col, lab in sigmas:
        k = np.exp(-(d ** 2) / (2 * sig ** 2))
        ax.plot(d, k, color=col, lw=2.6, label=lab)
    ax.set_xlabel("distance from the center point  x_i", fontsize=12)
    ax.set_ylabel("p_{j|i}  (un-normalized Gaussian affinity)", fontsize=12)
    ax.set_title("Perplexity sets σ_i: how far the Gaussian neighborhood reaches",
                 fontsize=13.5, fontweight="bold")
    ax.legend(fontsize=10.5, frameon=False, loc="upper right")
    ax.set_ylim(0, 1.05); ax.set_xlim(0, 6)
    _despine(ax)
    fig.tight_layout()
    p = os.path.join(OUT, "tsne_neighbor_kernel.png")
    fig.savefig(p, dpi=130, bbox_inches="tight"); plt.close(fig)
    print("wrote", p)


def swiss_roll():
    """The manifold motivation: PCA can't unroll a Swiss roll, t-SNE separates the
    sheet's strands by neighborhood. Color = position ALONG the rolled sheet."""
    X, color = make_swiss_roll(n_samples=1500, noise=0.05, random_state=0)
    Ypca = PCA(n_components=2, random_state=0).fit_transform(X)
    Ytsne = TSNE(n_components=2, perplexity=30, init="pca",
                 learning_rate="auto", random_state=0, max_iter=1000).fit_transform(X)

    fig = plt.figure(figsize=(15, 4.8))
    ax0 = fig.add_subplot(1, 3, 1, projection="3d")
    ax0.scatter(X[:, 0], X[:, 1], X[:, 2], c=color, cmap="viridis", s=7, alpha=0.8)
    ax0.set_title("Swiss roll in 3-D\n(color = position along the sheet)",
                  fontsize=12.5, fontweight="bold")
    ax0.set_xticks([]); ax0.set_yticks([]); ax0.set_zticks([])
    for ax, Y, t in ((fig.add_subplot(1, 3, 2), Ypca, "PCA (linear) — strands overlap"),
                     (fig.add_subplot(1, 3, 3), Ytsne, "t-SNE — strands pulled apart")):
        ax.scatter(Y[:, 0], Y[:, 1], c=color, cmap="viridis", s=7, alpha=0.85,
                   linewidths=0)
        ax.set_title(t, fontsize=12.5, fontweight="bold")
        ax.set_xticks([]); ax.set_yticks([])
        for s in ax.spines.values():
            s.set_edgecolor("#999")
    fig.suptitle("Why linear fails on a curved manifold", fontsize=14, fontweight="bold")
    fig.tight_layout(rect=[0, 0, 1, 0.94])
    p = os.path.join(OUT, "tsne_swiss_roll.png")
    fig.savefig(p, dpi=130, bbox_inches="tight"); plt.close(fig)
    print("wrote", p)


if __name__ == "__main__":
    vs_pca()
    perplexity_panels()
    crowding()
    neighbor_kernel()
    swiss_roll()
    print("\nAll t-SNE diagrams written to", OUT)
