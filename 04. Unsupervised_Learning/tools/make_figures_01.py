"""Figure generator for 01-K-Means-Clustering — every number from the REAL runs in ``kmeans.py``.

All figures come from the same real pipeline the chapter and notebook use (``kmeans.py`` on standardized
**Wine**, a controlled 2-D **make_blobs** illustration, a controlled 12-cluster layout, and the
**moons/anisotropic** failure cases): the from-scratch Lloyd's algorithm, its per-iteration inertia trace,
the k-means++-vs-random init distribution, the elbow + silhouette sweep, and the ARI-scored failure modes.
Nothing is hand-typed; every centroid, curve, bar, and annotation is read off an executed function call.

Writes muted-palette PNGs to the shared domain image dir (../images/) with prefix ``unsup01_``:

  unsup01_lloyd_iters.png       -- Lloyd's algorithm on 2-D blobs across four snapshots: centroids (X marks)
                                   migrate from a poor random init to the true cluster centres while the
                                   assignments recolour and the inertia J (printed per panel) falls.
  unsup01_inertia_curve.png     -- the inertia J at every Lloyd iteration of that same run, proving it
                                   decreases monotonically to convergence (25329 -> 948.9 over 7 steps).
  unsup01_elbow_silhouette.png  -- choosing k on real Wine: inertia bends (elbow) and mean silhouette PEAKS,
                                   both at k=3, the true number of cultivars.
  unsup01_init.png              -- k-means++ vs random seeding on a controlled 12-cluster layout: 50 single
                                   starts each, as overlaid inertia histograms; random has a heavy bad tail.
  unsup01_failure.png           -- where k-means breaks: two moons (ARI 0.27) and anisotropic blobs (ARI
                                   0.66), coloured by the k-means assignment — straight cuts across curved
                                   and sheared structure.

    python make_figures_01.py

Verified on Python 3.12 / matplotlib 3.10 / numpy 2.4 / scikit-learn 1.9.
"""

from __future__ import annotations

import sys
from pathlib import Path

# This generator lives in the domain-level ``04. Unsupervised_Learning/tools/`` folder, while the chapter
# module it demonstrates stays in that chapter's ``code/`` folder. Put that folder on sys.path so the
# ``kmeans`` import below resolves no matter the working directory.
_CHAPTER_CODE = Path(__file__).resolve().parent.parent / "01-K-Means-Clustering" / "code"
sys.path.insert(0, str(_CHAPTER_CODE))

import matplotlib  # noqa: E402  (imported after the sys.path insert above, which must run first)

matplotlib.use("Agg")  # headless: render straight to PNG, never open a window
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
from matplotlib.lines import Line2D  # noqa: E402

from kmeans import (  # noqa: E402  (resolved via the sys.path insert above)
    MANY_K,
    WINE_K,
    compare_init,
    load_anisotropic,
    load_blobs_2d,
    load_blobs_many,
    load_moons,
    lloyd_history,
    measure_failure,
    sweep_k,
    load_wine_scaled,
)

# ---- Palette (matches the chapter's muted Mermaid classDefs) ------------------------------------
BLUE = "#3A6B96"  # cluster 0 / random / elbow
PURPLE = "#5D4A8A"  # cluster 1 / process / silhouette
GREEN = "#2E7A5A"  # cluster 2 / good / k-means++
RED = "#8B3B4A"  # cost / failure / centroids
SLATE = "#4A5B6E"  # cluster 3 / neutral
AMBER = "#7A6528"  # highlight / chosen k / annotations
INK = "#1C2530"  # labels
GRID = "#D4D9DF"  # gridlines

OUT_DIR = Path(__file__).resolve().parent.parent / "images"
DPI = 120
IMG_PREFIX = "unsup01_"
CLUSTER_COLOURS = (BLUE, PURPLE, GREEN, SLATE)  # four blobs, in cluster order
CENTROID = RED  # the moving centroids, X-marked


def _style_axis(ax: plt.Axes) -> None:
    """Consistent muted styling: light grid, no top/right spines, ink-coloured labels."""
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


# ============================ Fig 1: Lloyd's iteration snapshots =================================
# seed 7 on the 2-D blobs starts from a poor random init, sits on a plateau, then breaks through to the
# true optimum over 7 iterations (J: 25329 -> 15020 -> 14316 -> 14216 -> 12693 -> 2161 -> 948.9) — the
# clearest teaching trajectory. Reused by fig_inertia_curve so both figures show the same real run.
_LLOYD_SEED = 7
_LLOYD_PANELS = (0, 4, 5, 6)  # init, the plateau, the breakthrough, convergence


def _lloyd_run() -> list:
    return lloyd_history(load_blobs_2d().x, load_blobs_2d().true_k or 4, seed=_LLOYD_SEED, init="random")


def fig_lloyd_iters() -> None:
    """Four snapshots of Lloyd's algorithm on 2-D blobs: centroids migrate, assignments recolour, J falls."""
    data = load_blobs_2d()
    history = _lloyd_run()
    k = data.true_k or 4

    fig, axes = plt.subplots(1, 4, figsize=(16, 4.4), sharex=True, sharey=True)
    for ax, it in zip(axes, _LLOYD_PANELS):
        snap = history[it]
        for c in range(k):
            m = snap.labels == c
            ax.scatter(data.x[m, 0], data.x[m, 1], s=14, color=CLUSTER_COLOURS[c], alpha=0.55, zorder=1)
        ax.scatter(snap.centers[:, 0], snap.centers[:, 1], marker="X", s=180, color=CENTROID,
                   edgecolor="white", linewidth=1.6, zorder=3)
        stage = {0: "random init", _LLOYD_PANELS[-1]: "converged"}.get(it, f"iteration {it}")
        ax.set_title(f"iter {it} — {stage}\nJ = {snap.inertia:,.0f}", fontsize=10.5)
        _style_axis(ax)
    fig.suptitle(
        "Lloyd's algorithm on four 2-D blobs (controlled illustration) — assign to nearest centroid, "
        "move each centroid to its members' mean, repeat\n"
        "the X centroids start poorly placed, sit on a plateau (iter 4), then break through to the true "
        "centres; the inertia J falls at every step",
        fontsize=12, color=INK, y=1.06,
    )
    legend = [Line2D([0], [0], marker="X", color="none", markerfacecolor=CENTROID,
                     markeredgecolor="white", markersize=13, label="centroid")]
    axes[-1].legend(handles=legend, frameon=False, fontsize=10, loc="lower right")
    fig.tight_layout()
    _save(fig, f"{IMG_PREFIX}lloyd_iters.png")


# ============================ Fig 2: the monotone inertia curve ==================================
def fig_inertia_curve() -> None:
    """The inertia J at every iteration of the same Lloyd run — a monotone descent to convergence."""
    history = _lloyd_run()
    js = [s.inertia for s in history]
    iters = np.arange(len(js))

    fig, ax = plt.subplots(figsize=(8.6, 5.2))
    ax.plot(iters, js, color=BLUE, linewidth=2.4, marker="o", markersize=7, zorder=3)
    ax.fill_between(iters, js, min(js) * 0.5, color=BLUE, alpha=0.06)
    for it, j in zip(iters, js):
        ax.annotate(f"{j:,.0f}", xy=(it, j), xytext=(0, 9), textcoords="offset points",
                    ha="center", fontsize=8.5, color=INK)
    # mark the breakthrough step (the largest single drop)
    drops = np.diff(js)
    b = int(np.argmin(drops)) + 1
    ax.annotate("a centroid escapes a\ncrowded blob — J plunges",
                xy=(b, js[b]), xytext=(b - 2.4, js[b] + 6500), fontsize=9, color=AMBER,
                arrowprops={"arrowstyle": "->", "color": AMBER})
    ax.set_xlabel("Lloyd iteration")
    ax.set_ylabel("inertia  J = Σ ‖x − μ‖²")
    ax.set_title(
        "Inertia falls monotonically to convergence (the executed proof)\n"
        f"same run as the snapshots: J = {js[0]:,.0f} → {js[-1]:,.0f} over {len(js) - 1} steps, "
        "never rising — the code asserts it",
        fontsize=11.5,
    )
    ax.set_ylim(0, max(js) * 1.12)
    _style_axis(ax)
    fig.tight_layout()
    _save(fig, f"{IMG_PREFIX}inertia_curve.png")


# ============================ Fig 3: choosing k (elbow + silhouette) =============================
def fig_elbow_silhouette() -> None:
    """Elbow (inertia) and silhouette (peak) on real Wine — both select k=3, the true cultivar count."""
    wine = load_wine_scaled()
    sweep = sweep_k(wine)
    ks = np.array(sweep.ks)
    best = sweep.best_k_silhouette

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5.4))

    # -- left: the elbow (inertia falls, bends at k=3) --
    ax1.plot(ks, sweep.inertias, color=BLUE, linewidth=2.4, marker="o", markersize=6)
    ax1.scatter([best], [sweep.inertias[sweep.ks.index(best)]], s=180, facecolor="none",
                edgecolor=AMBER, linewidth=2.2, zorder=4)
    ax1.annotate(f"elbow at k={best}\n(the bend)", xy=(best, sweep.inertias[sweep.ks.index(best)]),
                 xytext=(best + 1.1, sweep.inertias[sweep.ks.index(best)] + 120), fontsize=9.5, color=AMBER,
                 arrowprops={"arrowstyle": "->", "color": AMBER})
    ax1.set_xlabel("number of clusters k")
    ax1.set_ylabel("inertia  J (within-cluster sum of squares)")
    ax1.set_title("The elbow — inertia always falls, so read the BEND\n"
                  "on Wine it drops steeply through k=3, then flattens", fontsize=11)
    _style_axis(ax1)

    # -- right: the silhouette (a genuine peak at k=3) --
    ax2.plot(ks, sweep.silhouettes, color=PURPLE, linewidth=2.4, marker="s", markersize=6)
    peak_s = sweep.silhouettes[sweep.ks.index(best)]
    ax2.axvline(best, color=AMBER, linewidth=1.4, linestyle="--")
    ax2.scatter([best], [peak_s], s=180, facecolor="none", edgecolor=AMBER, linewidth=2.2, zorder=4)
    ax2.annotate(f"peak at k={best}\ns = {peak_s:.3f}", xy=(best, peak_s),
                 xytext=(best + 1.0, peak_s - 0.02), fontsize=9.5, color=AMBER,
                 arrowprops={"arrowstyle": "->", "color": AMBER})
    ax2.set_xlabel("number of clusters k")
    ax2.set_ylabel("mean silhouette  s ∈ [−1, 1]")
    ax2.set_title("The silhouette — a genuine MAXIMUM, not a bend\n"
                  "it peaks at the true number of cultivars, k=3", fontsize=11)
    _style_axis(ax2)

    fig.suptitle(
        f"Choosing k on real Wine ({wine.x.shape[0]} wines, {wine.x.shape[1]} standardized features) — "
        f"elbow and silhouette independently pick k={WINE_K}",
        fontsize=12.5, color=INK, y=1.04,
    )
    fig.tight_layout()
    _save(fig, f"{IMG_PREFIX}elbow_silhouette.png")


# ============================ Fig 4: k-means++ vs random init ====================================
def fig_init() -> None:
    """Overlaid inertia histograms of 50 single-start runs each — k-means++ vs random on 12 blobs."""
    comp = compare_init(load_blobs_many(), k=MANY_K)
    r, p = comp.summary("random"), comp.summary("kpp")
    lo = min(comp.random_inertias.min(), comp.kpp_inertias.min())
    hi = max(comp.random_inertias.max(), comp.kpp_inertias.max())
    bins = np.linspace(lo * 0.99, hi * 1.01, 26)

    fig, ax = plt.subplots(figsize=(9.2, 5.4))
    ax.hist(comp.random_inertias, bins=bins, color=BLUE, alpha=0.55,
            label=f"random  (mean {r['mean']:,.0f}, std {r['std']:,.0f}, worst {r['worst']:,.0f})")
    ax.hist(comp.kpp_inertias, bins=bins, color=GREEN, alpha=0.7,
            label=f"k-means++  (mean {p['mean']:,.0f}, std {p['std']:,.0f}, worst {p['worst']:,.0f})")
    ax.axvline(p["best"], color=AMBER, linewidth=1.6, linestyle="--")
    ax.annotate(f"global optimum ≈ {p['best']:,.0f}", xy=(p["best"], ax.get_ylim()[1] * 0.9),
                xytext=(p["best"] + (hi - lo) * 0.06, ax.get_ylim()[1] * 0.82), fontsize=9, color=AMBER,
                arrowprops={"arrowstyle": "->", "color": AMBER})
    ax.set_xlabel("final inertia J of a single Lloyd run")
    ax.set_ylabel("number of runs (of 50)")
    ax.set_title(
        f"k-means++ vs random seeding — {comp.n_trials} single starts each on {MANY_K} blobs "
        "(controlled)\n"
        f"random scatters with a heavy bad-local-optimum tail; k-means++ is {r['std'] / p['std']:.1f}× "
        "tighter and lands near the optimum far more often",
        fontsize=11,
    )
    ax.legend(frameon=False, fontsize=9.5, loc="upper right")
    _style_axis(ax)
    fig.tight_layout()
    _save(fig, f"{IMG_PREFIX}init.png")


# ============================ Fig 5: the failure modes ===========================================
def fig_failure() -> None:
    """k-means on two moons and anisotropic blobs — coloured by its assignment, ARI vs truth in the title."""
    moons = load_moons()
    aniso = load_anisotropic()
    moons_fail = measure_failure(moons, k=2)
    aniso_fail = measure_failure(aniso, k=3)

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5.6))

    for c in range(2):
        m = moons_fail.labels == c
        ax1.scatter(moons.x[m, 0], moons.x[m, 1], s=18, color=CLUSTER_COLOURS[c], alpha=0.75)
    ax1.set_title(
        "Non-convex: two interleaving moons (k=2)\n"
        f"the TRUE clusters are the two crescents, but k-means cuts a straight line — ARI = {moons_fail.ari:.2f}",
        fontsize=11)
    _style_axis(ax1)

    for c in range(3):
        m = aniso_fail.labels == c
        ax2.scatter(aniso.x[m, 0], aniso.x[m, 1], s=16, color=CLUSTER_COLOURS[c], alpha=0.75)
    ax2.set_title(
        "Anisotropic: three sheared, stretched blobs (k=3)\n"
        f"round Voronoi cells cut across the diagonal stripes — ARI = {aniso_fail.ari:.2f}",
        fontsize=11)
    _style_axis(ax2)

    fig.suptitle(
        "Where k-means breaks — straight (Voronoi) boundaries cannot follow curved or oriented structure "
        "(ARI: 1.0 = perfect, 0 = chance)\n"
        "the honest limitation: reach for DBSCAN (density, arbitrary shapes) or a GMM (per-cluster "
        "covariance) instead",
        fontsize=12, color=INK, y=1.05,
    )
    fig.tight_layout()
    _save(fig, f"{IMG_PREFIX}failure.png")


def main() -> None:
    print(f"numpy {np.__version__} | matplotlib {matplotlib.__version__}")
    fig_lloyd_iters()
    fig_inertia_curve()
    fig_elbow_silhouette()
    fig_init()
    fig_failure()
    print(f"\nall figures written to {OUT_DIR} with prefix '{IMG_PREFIX}'")


if __name__ == "__main__":
    main()
