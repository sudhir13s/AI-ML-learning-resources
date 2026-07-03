"""Generate the step-by-step teaching notebook (01-K-Means-Clustering.ipynb).

The notebook mirrors ``kmeans.py`` one measurement at a time, so a learner can open it, run every cell live,
and *see* how k-means works on real data — the inertia objective, the assign and update steps, why the mean
is the optimal centre, k-means++ seeding, Lloyd's loop with its monotone-decreasing inertia, the animation
of centroids migrating, the verification against scikit-learn on real Wine, the k-means++-vs-random init
distribution, choosing k by elbow + silhouette, and the two failure modes scored by ARI. Each numbered step
has a short markdown lead-in (the intuition) followed by ONE focused code cell with real output. This
generator writes the .ipynb; a separate nbconvert pass executes it headless so the outputs are embedded.

    python build_notebook_01.py         # writes the .ipynb (unexecuted) into the chapter's code/
    python -m nbconvert --to notebook --execute --inplace \
        "../01-K-Means-Clustering/code/01-K-Means-Clustering.ipynb"

This generator lives in the domain-level ``04. Unsupervised_Learning/tools/`` folder; the notebook it writes
(and the module it mirrors) stay in the chapter's ``code/`` folder. Kept as a generator (not a hand-edited
.ipynb) so the notebook and the module stay in lockstep.
"""

from __future__ import annotations

import json
from pathlib import Path

_CHAPTER_CODE = Path(__file__).resolve().parent.parent / "01-K-Means-Clustering" / "code"
NB_PATH = _CHAPTER_CODE / "01-K-Means-Clustering.ipynb"

_CELL_ID = 0


def _next_id() -> str:
    """Stable, sequential cell id (silences nbformat's MissingIDFieldWarning)."""
    global _CELL_ID
    _CELL_ID += 1
    return f"cell-{_CELL_ID:02d}"


def md(source: str) -> dict:
    return {"cell_type": "markdown", "id": _next_id(), "metadata": {}, "source": source.splitlines(keepends=True)}


def code(source: str) -> dict:
    return {
        "cell_type": "code",
        "id": _next_id(),
        "execution_count": None,
        "metadata": {},
        "outputs": [],
        "source": source.splitlines(keepends=True),
    }


CELLS: list[dict] = []


def add_md(source: str) -> None:
    CELLS.append(md(source))


def add_code(source: str) -> None:
    CELLS.append(code(source))


# ============================ Title ============================================================
add_md(
    "# K-Means Clustering — a step-by-step, runnable notebook\n"
    "\n"
    "You are handed a pile of points with **no labels** and asked to find the natural groups. K-means makes "
    "one decisive bet — *a good cluster is a compact round blob around a centre* — and turns it into a loop "
    "any child could run: **drop k flags, send each point to its nearest flag, slide each flag to the middle "
    "of the points that gathered around it, repeat until nothing moves.** That loop is **Lloyd's algorithm**, "
    "and it provably lowers a single objective (the within-cluster sum of squares) at every step.\n"
    "\n"
    "This notebook builds that, one measurement at a time, on **real scikit-learn data** (no download): "
    "**Wine** (178 wines, 13 chemical features, 3 cultivars), standardized, carries the load-bearing claims — "
    "the from-scratch/scikit-learn match, the init comparison, and the choice of *k*. A **controlled 2-D "
    "make_blobs** layout is used *only* to make Lloyd's iteration visible (you cannot watch centroids move in "
    "13 dimensions), and **two moons + sheared blobs** show the honest failure modes. Every function used "
    "here lives in `kmeans.py`, imported so the notebook and the module can never drift apart.\n"
    "\n"
    "By the end you will have **measured**, not just been told:\n"
    "\n"
    "1. the **inertia** objective J, and that the mean is the **optimal** centre;\n"
    "2. Lloyd's loop, with its inertia **falling monotonically** to convergence;\n"
    "3. the from-scratch result **matching scikit-learn** on real Wine (same inertia, ARI = 1.0);\n"
    "4. **k-means++** beating random seeding as a measured distribution;\n"
    "5. **choosing k** by the elbow and the silhouette (both pick k=3 on Wine);\n"
    "6. the **failure modes** — moons and anisotropic blobs — quantified by a low ARI.\n"
    "\n"
    "Everything runs on CPU in a couple of seconds, seeded for reproducibility."
)

# ---- Step 0: setup ----
add_md(
    "## Step 0 — Setup and version banner\n"
    "\n"
    "We import the real functions from the chapter module (so this notebook uses the *exact same code* the "
    "figures and the page use) and print the library versions the results were produced on."
)
add_code(
    "import numpy as np\n"
    "import matplotlib.pyplot as plt\n"
    "import sklearn\n"
    "\n"
    "from kmeans import (\n"
    "    assign, inertia, kpp_init, lloyd_history, KMeansScratch,\n"
    "    verify_against_sklearn, compare_init, sweep_k, measure_failure,\n"
    "    load_wine_scaled, load_blobs_2d, load_blobs_many, load_moons, load_anisotropic,\n"
    "    WINE_K, MANY_K,\n"
    ")\n"
    "\n"
    "print(f'numpy {np.__version__} | scikit-learn {sklearn.__version__}')\n"
    "wine = load_wine_scaled()\n"
    "blobs = load_blobs_2d()\n"
    "print(f'Wine : {wine.x.shape[0]} wines x {wine.x.shape[1]} standardized features, '\n"
    "      f'true k = {wine.true_k}')\n"
    "print(f'Blobs: {blobs.x.shape[0]} 2-D points, {blobs.true_k} controlled clusters (for the animation)')"
)

# ---- Step 1: the objective ----
add_md(
    "## Step 1 — The objective: inertia (within-cluster sum of squares)\n"
    "\n"
    "K-means minimizes one number — the **inertia**, the total squared distance from each point to its "
    "cluster's centre:\n"
    "\n"
    "$$J = \\sum_{i=1}^{n} \\lVert x_i - \\mu_{c_i} \\rVert^2$$\n"
    "\n"
    "Small $J$ means tight, compact clusters. Let's compute it for a tiny hand-checkable case: four points "
    "in two obvious pairs, with the centres at the pair midpoints."
)
add_code(
    "X = np.array([[0.0, 0.0], [0.0, 1.0], [8.0, 8.0], [9.0, 8.0]])\n"
    "centers = np.array([[0.0, 0.5], [8.5, 8.0]])   # midpoints of the two pairs\n"
    "labels = np.array([0, 0, 1, 1])\n"
    "print(f'inertia J = {inertia(X, labels, centers):.2f}')\n"
    "# by hand: left pair each 0.5 from (0,0.5) -> 0.25+0.25; right pair each 0.5 from (8.5,8) -> 0.25+0.25\n"
    "print('by hand   = 0.25 + 0.25 + 0.25 + 0.25 = 1.00  (matches)')"
)

# ---- Step 2: assign step ----
add_md(
    "## Step 2 — The assign step (E-step): each point to its nearest centroid\n"
    "\n"
    "With the centres fixed, the **best** label for a point is trivially its nearest centre — that is the "
    "assignment that minimizes its term in $J$. `assign` computes the full point-to-centre squared-distance "
    "matrix and takes the `argmin`."
)
add_code(
    "seed_centers = kpp_init(blobs.x, blobs.true_k, np.random.default_rng(0))\n"
    "lab = assign(blobs.x, seed_centers)\n"
    "print(f'{blobs.x.shape[0]} points assigned to {blobs.true_k} centroids')\n"
    "print(f'cluster sizes: {np.bincount(lab)}')\n"
    "print('each point now belongs to whichever centroid is closest — a Voronoi partition.')"
)

# ---- Step 3: update step + the mean is optimal ----
add_md(
    "## Step 3 — The update step (M-step): the mean is the optimal centre\n"
    "\n"
    "With the labels fixed, what is the best centre for a cluster? Minimize $\\sum_{i} \\lVert x_i - \\mu "
    "\\rVert^2$: set the gradient to zero, $\\sum_i -2(x_i - \\mu) = 0$, and out falls "
    "$\\mu = \\frac{1}{|C|}\\sum_i x_i$ — the **arithmetic mean**. So 'move each centroid to the mean of its "
    "members' is not a heuristic; it is provably optimal. Let's confirm the mean beats any nudge away from "
    "it."
)
add_code(
    "cluster = blobs.x[lab == 0]\n"
    "mu = cluster.mean(axis=0)\n"
    "def sse(c):\n"
    "    return float(((cluster - c) ** 2).sum())\n"
    "print(f'SSE at the mean {np.round(mu, 3)} : {sse(mu):.2f}')\n"
    "for nudge in ([0.5, 0.0], [0.0, 0.5], [-0.5, -0.5]):\n"
    "    print(f'SSE at mean + {nudge} : {sse(mu + np.array(nudge)):.2f}  (higher)')\n"
    "print('\\nAny move away from the mean raises the within-cluster SSE — the mean is the minimizer.')"
)

# ---- Step 4: k-means++ ----
add_md(
    "## Step 4 — k-means++ seeding: spread the seeds with D² sampling\n"
    "\n"
    "A bad random init can strand two seeds in one blob and leave another unseeded — a local optimum Lloyd's "
    "cannot escape. **k-means++** picks the first centre at random, then each next centre with probability "
    "proportional to $D(x)^2$ (squared distance to the nearest chosen centre) — biasing hard toward "
    "far-apart seeds while staying randomized. Here are four seeds it places on the blobs."
)
add_code(
    "seeds = kpp_init(blobs.x, blobs.true_k, np.random.default_rng(1))\n"
    "plt.figure(figsize=(6.5, 5))\n"
    "plt.scatter(blobs.x[:, 0], blobs.x[:, 1], s=12, color='#B7C2CE', alpha=0.7)\n"
    "plt.scatter(seeds[:, 0], seeds[:, 1], marker='X', s=200, color='#8B3B4A',\n"
    "            edgecolor='white', linewidth=1.6, label='k-means++ seeds')\n"
    "plt.title('k-means++ places one seed per blob — spread out on purpose')\n"
    "plt.legend()\n"
    "plt.tight_layout()\n"
    "plt.show()\n"
    "print('Each of the four blobs gets a seed: k-means++ avoids the clumped starts random init suffers.')"
)

# ---- Step 5: Lloyd's + monotone inertia ----
add_md(
    "## Step 5 — Lloyd's algorithm and the monotone-decreasing inertia\n"
    "\n"
    "Alternate the two steps until the assignments stop changing. **Neither step can raise $J$** (assign "
    "picks the nearest centre; update picks the optimal mean), so $J$ falls monotonically — and since it is "
    "bounded below by 0 with finitely many possible assignments, the loop **must converge**. `lloyd_history` "
    "records $J$ at every iteration; watch it fall from a deliberately poor random start."
)
add_code(
    "history = lloyd_history(blobs.x, blobs.true_k, seed=7, init='random')\n"
    "js = [s.inertia for s in history]\n"
    "for it, j in enumerate(js):\n"
    "    print(f'  iter {it}: J = {j:9.1f}')\n"
    "assert all(js[i + 1] <= js[i] + 1e-9 for i in range(len(js) - 1)), 'J must never rise'\n"
    "print(f'\\nconverged in {len(js) - 1} steps; J fell {js[0]:.0f} -> {js[-1]:.0f}, never rising.')"
)

# ---- Step 6: the animation snapshots ----
add_md(
    "## Step 6 — See it converge: centroids migrating\n"
    "\n"
    "The same run, drawn. The X centroids start poorly placed (two share the top blob), sit on a plateau, "
    "then one **breaks through** to the empty blob and the inertia plunges. This is the whole algorithm in "
    "one picture."
)
add_code(
    "panels = [0, 4, 5, len(history) - 1]\n"
    "colours = ['#3A6B96', '#5D4A8A', '#2E7A5A', '#4A5B6E']\n"
    "fig, axes = plt.subplots(1, 4, figsize=(15, 4), sharex=True, sharey=True)\n"
    "for ax, it in zip(axes, panels):\n"
    "    snap = history[it]\n"
    "    for c in range(blobs.true_k):\n"
    "        m = snap.labels == c\n"
    "        ax.scatter(blobs.x[m, 0], blobs.x[m, 1], s=10, color=colours[c], alpha=0.5)\n"
    "    ax.scatter(snap.centers[:, 0], snap.centers[:, 1], marker='X', s=140,\n"
    "               color='#8B3B4A', edgecolor='white', linewidth=1.4)\n"
    "    ax.set_title(f'iter {it}:  J = {snap.inertia:,.0f}')\n"
    "plt.tight_layout()\n"
    "plt.show()"
)

# ---- Step 7: the Voronoi partition ----
add_md(
    "## Step 7 — What k-means computes: a Voronoi partition of convex cells\n"
    "\n"
    "The assign step sends each point to its nearest centroid, so the boundary between two clusters is the "
    "set of points **equidistant** from their centres — a straight line (in 2-D) or hyperplane. The result "
    "is a **Voronoi diagram**: $k$ convex polygonal cells, one per centroid, with flat boundaries. Fit our "
    "`KMeansScratch` on the blobs and paint its prediction over a dense grid — the straight cell walls are "
    "the whole reason k-means cannot bend around curved structure (Steps 10–11)."
)
add_code(
    "km = KMeansScratch(n_clusters=blobs.true_k, init='k-means++', n_init=10, seed=0).fit(blobs.x)\n"
    "x0 = np.linspace(blobs.x[:, 0].min() - 1, blobs.x[:, 0].max() + 1, 300)\n"
    "x1 = np.linspace(blobs.x[:, 1].min() - 1, blobs.x[:, 1].max() + 1, 300)\n"
    "gx, gy = np.meshgrid(x0, x1)\n"
    "regions = assign(np.column_stack([gx.ravel(), gy.ravel()]), km.cluster_centers_).reshape(gx.shape)\n"
    "colours = ['#3A6B96', '#5D4A8A', '#2E7A5A', '#4A5B6E']\n"
    "plt.figure(figsize=(6.8, 5.4))\n"
    "plt.contourf(gx, gy, regions, alpha=0.18, levels=[-0.5, 0.5, 1.5, 2.5, 3.5], colors=colours)\n"
    "plt.scatter(blobs.x[:, 0], blobs.x[:, 1], s=10, c=[colours[c] for c in km.labels_], alpha=0.6)\n"
    "plt.scatter(km.cluster_centers_[:, 0], km.cluster_centers_[:, 1], marker='X', s=180,\n"
    "            color='#8B3B4A', edgecolor='white', linewidth=1.6)\n"
    "plt.title('Voronoi cells: straight boundaries, convex regions (from-scratch k-means)')\n"
    "plt.tight_layout()\n"
    "plt.show()\n"
    "print(f'final inertia = {km.inertia_:.1f}  |  {km.n_iter_} iterations to converge')\n"
    "print('Every boundary is a straight line — k-means can only carve CONVEX cells.')"
)

# ---- Step 8: verify against sklearn ----
add_md(
    "## Step 8 — Is it the real thing? Verify against scikit-learn on real Wine\n"
    "\n"
    "Now the real dataset. Cluster standardized **Wine** with our `KMeansScratch` and scikit-learn's "
    "`KMeans` (same k, n_init, seed). Because **cluster labels are arbitrary** (our 'cluster 0' need not be "
    "sklearn's 'cluster 0'), we do not compare labels directly — we compare the **inertia** (the objective "
    "value) and the **adjusted Rand index** (agreement up to a permutation). A match on both is the honest "
    "proof our Lloyd's is genuine."
)
add_code(
    "match = verify_against_sklearn(wine, k=WINE_K)\n"
    "print(f'from-scratch inertia : {match.scratch_inertia:.3f}')\n"
    "print(f'scikit-learn inertia : {match.sklearn_inertia:.3f}')\n"
    "print(f'adjusted Rand index  : {match.ari:.3f}   (1.0 = identical partition up to a permutation)')\n"
    "assert abs(match.scratch_inertia - match.sklearn_inertia) < 1.0\n"
    "assert match.ari > 0.99\n"
    "print('\\n=> same inertia, same partition: the from-scratch algorithm is the real thing.')"
)

# ---- Step 9: init comparison ----
add_md(
    "## Step 9 — k-means++ vs random, as a measured distribution\n"
    "\n"
    "Initialization barely matters on cleanly-separated data (random almost always finds Wine's 3 clusters), "
    "but its cost grows with the number of clusters. On a controlled **12-blob** layout, run 50 single "
    "starts each way and histogram the final inertia: random scatters with a heavy bad-local-optimum tail, "
    "k-means++ piles near the global optimum."
)
add_code(
    "comp = compare_init(load_blobs_many(), k=MANY_K)\n"
    "r = comp.summary('random')\n"
    "p = comp.summary('kpp')\n"
    "print(f\"random    : mean {r['mean']:.0f}  std {r['std']:.0f}  best {r['best']:.0f}  worst {r['worst']:.0f}\")\n"
    "print(f\"k-means++ : mean {p['mean']:.0f}  std {p['std']:.0f}  best {p['best']:.0f}  worst {p['worst']:.0f}\")\n"
    "bins = np.linspace(comp.kpp_inertias.min() * 0.99, comp.random_inertias.max() * 1.01, 24)\n"
    "plt.figure(figsize=(8.5, 4.6))\n"
    "plt.hist(comp.random_inertias, bins=bins, color='#3A6B96', alpha=0.55, label='random')\n"
    "plt.hist(comp.kpp_inertias, bins=bins, color='#2E7A5A', alpha=0.7, label='k-means++')\n"
    "plt.axvline(p['best'], color='#7A6528', ls='--', label=f\"optimum ~ {p['best']:.0f}\")\n"
    "plt.xlabel('final inertia of a single run')\n"
    "plt.ylabel('runs (of 50)')\n"
    "plt.title(f\"k-means++ is {r['std']/p['std']:.1f}x tighter (12 blobs)\")\n"
    "plt.legend()\n"
    "plt.tight_layout()\n"
    "plt.show()"
)

# ---- Step 10: choosing k ----
add_md(
    "## Step 10 — Choosing k on real Wine: elbow + silhouette\n"
    "\n"
    "K needs to be chosen. Inertia **always** falls as k grows, so you can only read its **bend** (the "
    "elbow). The **silhouette** — for each point, (nearest-other-cluster distance − own-cluster "
    "distance) / max — has a genuine **peak**. Sweep k on Wine: both point to k=3, the true number of "
    "cultivars."
)
add_code(
    "sweep = sweep_k(wine)\n"
    "print(f'{\"k\":>3}{\"inertia\":>11}{\"silhouette\":>13}')\n"
    "for k, jj, ss in zip(sweep.ks, sweep.inertias, sweep.silhouettes):\n"
    "    mark = '  <- peak' if k == sweep.best_k_silhouette else ''\n"
    "    print(f'{k:>3}{jj:>11.1f}{ss:>13.3f}{mark}')\n"
    "fig, (a1, a2) = plt.subplots(1, 2, figsize=(12, 4.4))\n"
    "a1.plot(sweep.ks, sweep.inertias, 'o-', color='#3A6B96', lw=2.2)\n"
    "a1.set_title('elbow (inertia)')\n"
    "a1.set_xlabel('k')\n"
    "a1.set_ylabel('inertia')\n"
    "a2.plot(sweep.ks, sweep.silhouettes, 's-', color='#5D4A8A', lw=2.2)\n"
    "a2.set_title('silhouette (peak)')\n"
    "a2.axvline(sweep.best_k_silhouette, color='#7A6528', ls='--')\n"
    "a2.set_xlabel('k')\n"
    "a2.set_ylabel('s')\n"
    "plt.tight_layout()\n"
    "plt.show()\n"
    "print(f'\\nsilhouette peaks at k={sweep.best_k_silhouette} — the true number of Wine cultivars.')"
)

# ---- Step 11: failure — moons ----
add_md(
    "## Step 11 — Where it breaks (1): non-convex moons\n"
    "\n"
    "K-means draws **straight** (Voronoi) boundaries, so it cannot represent a crescent. On two interleaving "
    "moons the true clusters are the two arcs, but k-means slices a line straight through both. The adjusted "
    "Rand index against the true labels quantifies the failure (≈ 0 = chance, 1 = perfect)."
)
add_code(
    "moons = load_moons()\n"
    "fail = measure_failure(moons, k=2)\n"
    "plt.figure(figsize=(6.5, 4.6))\n"
    "for c in range(2):\n"
    "    m = fail.labels == c\n"
    "    plt.scatter(moons.x[m, 0], moons.x[m, 1], s=16, color=['#3A6B96', '#5D4A8A'][c], alpha=0.75)\n"
    "plt.title(f'k-means on two moons — ARI vs truth = {fail.ari:.2f} (fails)')\n"
    "plt.tight_layout()\n"
    "plt.show()\n"
    "assert fail.ari < 0.5\n"
    "print(f'ARI = {fail.ari:.3f}: k-means split each moon in half instead of separating the two arcs.')"
)

# ---- Step 12: failure — anisotropic ----
add_md(
    "## Step 12 — Where it breaks (2): anisotropic (sheared) blobs\n"
    "\n"
    "Squared Euclidean distance treats every direction equally, so k-means expects **round** clusters. Three "
    "diagonally-stretched blobs have orientation the algorithm ignores — its round Voronoi cells cut across "
    "the stripes. Again, ARI measures the damage."
)
add_code(
    "aniso = load_anisotropic()\n"
    "fail_a = measure_failure(aniso, k=3)\n"
    "plt.figure(figsize=(6.5, 4.6))\n"
    "for c in range(3):\n"
    "    m = fail_a.labels == c\n"
    "    plt.scatter(aniso.x[m, 0], aniso.x[m, 1], s=14,\n"
    "                color=['#3A6B96', '#5D4A8A', '#2E7A5A'][c], alpha=0.75)\n"
    "plt.title(f'k-means on sheared blobs — ARI vs truth = {fail_a.ari:.2f}')\n"
    "plt.tight_layout()\n"
    "plt.show()\n"
    "print(f'ARI = {fail_a.ari:.3f}: better than the moons, but the round cells still cut across the stripes.')\n"
    "print('A Gaussian Mixture Model, which gives each cluster its own covariance, recovers these.')"
)

# ---- Recap ----
add_md(
    "## Recap\n"
    "\n"
    "On real scikit-learn data (plus clearly-labelled controlled illustrations), one runnable notebook built "
    "the whole of k-means:\n"
    "\n"
    "| Step | What we measured | The takeaway |\n"
    "|---|---|---|\n"
    "| 1–3 | inertia; assign; update | J = Σ‖x−μ‖²; nearest-centre labels; the **mean** is the optimal centre |\n"
    "| 4–6 | k-means++; Lloyd's; the animation | spread seeds ∝ D²; **J falls monotonically** to convergence |\n"
    "| 7 | the Voronoi partition | straight boundaries, **convex** cells — the source of every failure mode |\n"
    "| 8 | verify vs scikit-learn (Wine) | **same inertia, ARI = 1.0** — the from-scratch loop is genuine |\n"
    "| 9 | k-means++ vs random (12 blobs) | k-means++ is **~2.5× tighter**; the advantage grows with k |\n"
    "| 10 | choosing k (Wine) | inertia **bends**, silhouette **peaks** — both at the true k=3 |\n"
    "| 11–12 | moons; anisotropic blobs | **low ARI** — straight cuts fail curved/sheared structure |\n"
    "\n"
    "**K-means partitions points by minimizing the within-cluster sum of squares; Lloyd's algorithm "
    "alternates assign and update, each step provably lowering J, so it converges — to a *local* optimum "
    "(the global one is NP-hard), which k-means++ and restarts make a good one.** It assumes spherical, "
    "equal-size clusters — exactly the hard limit of a Gaussian mixture, and the reason the next chapters "
    "reach for **DBSCAN**, **GMMs**, and **spectral clustering**."
)


def main() -> None:
    nb = {
        "cells": CELLS,
        "metadata": {
            "kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
            "language_info": {"name": "python", "version": "3.12"},
        },
        "nbformat": 4,
        "nbformat_minor": 5,
    }
    NB_PATH.write_text(json.dumps(nb, indent=1, ensure_ascii=False), encoding="utf-8")
    print(f"wrote {NB_PATH} with {len(CELLS)} cells "
          f"({sum(c['cell_type'] == 'code' for c in CELLS)} code, "
          f"{sum(c['cell_type'] == 'markdown' for c in CELLS)} markdown)")


if __name__ == "__main__":
    main()
