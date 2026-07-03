"""K-Means from scratch on REAL data, VERIFIED against scikit-learn — the module.

This is not a toy. Every number the chapter, the figures, and the notebook show is produced here from a
real pipeline (``numpy`` + ``scikit-learn`` + ``matplotlib``) on datasets that ship with scikit-learn (no
download) plus two clearly-labelled *controlled* generators used only for visualisation:

  * **Wine** (178 wines, 13 chemical measurements, 3 cultivars) — standardized (z-scored) — is the real,
    higher-dimensional dataset. It carries the load-bearing claims: the from-scratch/scikit-learn inertia
    match, the k-means++-vs-random init comparison, and the elbow + silhouette choice of *k*. Standardizing
    matters because k-means uses Euclidean distance, so an unscaled feature on a large numeric range (e.g.
    ``proline`` in the hundreds) would dominate the objective — see chapter 02, Feature Scaling.

  * **make_blobs (2-D, controlled)** — four well-separated round Gaussian blobs, generated with a fixed
    seed *purely so Lloyd's iteration is visible*: you cannot watch centroids migrate in 13 dimensions. It
    is labelled "controlled illustration" everywhere it appears; the real measured behaviour lives on Wine.

  * **make_moons + an anisotropic (sheared) blob layout** — the honest failure modes. Two interleaving
    crescents and three diagonally-stretched clusters that k-means provably cannot recover, measured by a
    low adjusted Rand index against the known structure. The forward link to DBSCAN and GMMs.

What this module measures (all real, all reproducible from the seed):

  * **K-means grown from scratch.** ``KMeansScratch`` implements Lloyd's algorithm — ``k-means++`` and
    uniform-random seeding, the assign step (each point to its nearest centroid), the update step (each
    centroid to the mean of its members), the empty-cluster reseed, and ``n_init`` restarts keeping the
    lowest inertia — plus a per-iteration inertia trace that is *asserted* to be monotonically
    non-increasing. That assertion is the convergence proof, executed.

  * **The from-scratch result VERIFIED against scikit-learn.** Same ``k``, same ``n_init``, same seed, on
    standardized Wine: our best inertia equals ``sklearn.cluster.KMeans``'s to a tiny tolerance, and the two
    partitions agree up to a label permutation (adjusted Rand index ~1.0). Because cluster labels are
    arbitrary we compare *inertia* and *ARI*, never raw label equality — the honest proof.

  * **k-means++ vs random seeding, as a measured distribution.** 50 single-start runs each way on a controlled
    12-cluster layout (init matters more as k grows; on well-separated few-cluster data like Wine it barely
    moves): random seeding scatters (some starts trapped in bad local optima), k-means++ piles tightly at the
    global optimum. The O(log k) guarantee showing up as best/worst/spread numbers.

  * **Choosing k on real data.** An inertia sweep (the elbow) and a mean-silhouette sweep on Wine: inertia
    bends and silhouette *peaks* at the true number of cultivars, k=3.

  * **The failure modes, quantified.** k-means on two moons and on anisotropic blobs, scored by adjusted
    Rand index against the true labels — low, because straight Voronoi cuts cannot follow curved or
    sheared structure.

Everything is seeded and CPU-only; runs standalone in a couple of seconds::

    python kmeans.py
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
from numpy.typing import NDArray
from sklearn.cluster import KMeans
from sklearn.datasets import load_wine, make_blobs, make_moons
from sklearn.metrics import adjusted_rand_score, silhouette_score
from sklearn.preprocessing import StandardScaler

# ---- Constants (hoisted; no magic numbers inline) ----------------------------------------------
RANDOM_STATE = 42  # the one seed fixing every dataset and every estimator's randomness
WINE_K = 3  # Wine's true number of cultivars; also the k the elbow + silhouette recover
BLOB_K = 4  # the controlled 2-D illustration uses four well-separated blobs
MANY_K = 12  # the controlled many-cluster layout for the init comparison (init matters more as k grows)
N_INIT = 10  # restarts kept for the best-of comparison (matches scikit-learn's modern default intent)
MAX_ITER = 100  # hard cap on Lloyd iterations (convergence is far faster in practice)
INIT_TRIALS = 50  # single-start runs per seeding scheme in the init comparison
K_SWEEP = (2, 3, 4, 5, 6, 7)  # the k values probed for the elbow and silhouette
INERTIA_MATCH_TOL = 1.0  # from-scratch and scikit-learn inertia must agree within this (z-scored units)
ARI_MATCH_FLOOR = 0.99  # from-scratch and scikit-learn partitions must agree up to a permutation
TIE_TOL = 1e-9  # numerical slack when asserting inertia never *rises*


# ============================ 1. real (and controlled) data =====================================
@dataclass
class Dataset:
    """A feature matrix, optional true labels, and human-readable names."""

    x: NDArray[np.float64]
    y: NDArray[np.int64] | None  # true labels when known (for ARI); None for unlabelled data
    feature_names: list[str]
    name: str
    true_k: int | None = None  # the known number of groups, where it exists


def load_wine_scaled() -> Dataset:
    """Wine standardized to zero mean / unit variance — the real, 13-D clustering problem.

    K-means minimizes squared Euclidean distance, so every feature must contribute on a comparable scale;
    an unscaled ``proline`` (values in the hundreds) would otherwise swamp features like ``hue`` (values
    near 1) and the clustering would track proline alone. Standardizing first is not optional for k-means.
    """
    data = load_wine()
    x = StandardScaler().fit_transform(data.data)
    return Dataset(
        x=x.astype(np.float64),
        y=data.target.astype(np.int64),
        feature_names=list(data.feature_names),
        name="Wine (13 features, standardized)",
        true_k=WINE_K,
    )


def load_blobs_2d(*, seed: int = RANDOM_STATE) -> Dataset:
    """Four well-separated 2-D Gaussian blobs — a CONTROLLED illustration so Lloyd's loop is visible.

    This is synthetic on purpose: you cannot watch centroids migrate in Wine's 13 dimensions, so the
    iteration animation and the monotone-inertia curve run on clean 2-D blobs. Every claim these figures
    make (J falls each step; the algorithm converges) is a property of the algorithm, not of this data.
    """
    x, y = make_blobs(n_samples=500, centers=BLOB_K, cluster_std=1.0, random_state=seed)
    return Dataset(
        x=x.astype(np.float64),
        y=y.astype(np.int64),
        feature_names=["x", "y"],
        name="make_blobs (2-D, controlled illustration)",
        true_k=BLOB_K,
    )


def load_blobs_many(*, seed: int = RANDOM_STATE) -> Dataset:
    """Twelve 2-D Gaussian blobs — a CONTROLLED many-cluster layout for the init comparison.

    Initialization barely matters when a few clusters are cleanly separated (as in Wine): random seeding
    almost always finds them. Its cost shows up as the number of clusters grows, because a random start is
    then far more likely to strand two seeds in one blob and leave another blob unseeded — a bad local
    optimum Lloyd's cannot escape. Twelve blobs make that gap visible; the advantage is real but scales
    with k, which is the honest framing.
    """
    x, y = make_blobs(n_samples=1500, centers=MANY_K, cluster_std=1.4, random_state=seed)
    return Dataset(
        x=x.astype(np.float64),
        y=y.astype(np.int64),
        feature_names=["x", "y"],
        name="make_blobs (12 clusters, controlled)",
        true_k=MANY_K,
    )


def load_moons(*, seed: int = 0) -> Dataset:
    """Two interleaving crescents — a real non-convex structure k-means cannot represent."""
    x, y = make_moons(n_samples=400, noise=0.06, random_state=seed)
    return Dataset(
        x=x.astype(np.float64),
        y=y.astype(np.int64),
        feature_names=["x", "y"],
        name="two moons (non-convex)",
        true_k=2,
    )


def load_anisotropic(*, seed: int = 170) -> Dataset:
    """Three diagonally-stretched (sheared) blobs — clusters with orientation k-means ignores."""
    x, y = make_blobs(n_samples=600, centers=3, cluster_std=0.9, random_state=seed)
    shear = np.array([[0.60, -0.63], [-0.40, 0.85]])  # a fixed linear transform that elongates the blobs
    return Dataset(
        x=(x @ shear).astype(np.float64),
        y=y.astype(np.int64),
        feature_names=["x", "y"],
        name="anisotropic blobs (sheared)",
        true_k=3,
    )


# ============================ 2. the core operations ============================================
def pairwise_sq_dists(x: NDArray[np.float64], centers: NDArray[np.float64]) -> NDArray[np.float64]:
    """Squared Euclidean distance from every point to every centre: an ``(n, k)`` matrix.

    Uses the expansion ``||x - c||^2 = ||x||^2 - 2 x·c + ||c||^2`` — one matrix product instead of an
    explicit per-pair loop, which is what makes the assign step ``O(n·k·d)`` and fast. Tiny negative values
    from floating-point cancellation are clipped to zero.
    """
    x_sq = np.sum(x**2, axis=1, keepdims=True)  # (n, 1)
    c_sq = np.sum(centers**2, axis=1, keepdims=True).T  # (1, k)
    cross = x @ centers.T  # (n, k)
    return np.maximum(x_sq - 2.0 * cross + c_sq, 0.0)


def assign(x: NDArray[np.float64], centers: NDArray[np.float64]) -> NDArray[np.int64]:
    """The assign step (E-step): label each point by its nearest centroid (argmin of squared distance)."""
    return pairwise_sq_dists(x, centers).argmin(axis=1).astype(np.int64)


def update(
    x: NDArray[np.float64], labels: NDArray[np.int64], k: int, old_centers: NDArray[np.float64]
) -> NDArray[np.float64]:
    """The update step (M-step): move each centroid to the mean of its assigned points.

    The mean is the *provably optimal* centre for the squared-distance objective once memberships are fixed
    (set the gradient of ``sum ||x - mu||^2`` to zero → ``mu = mean``). An **empty** cluster has no mean
    (0/0); we keep its previous centre so the algorithm cannot crash — scikit-learn instead reseeds it onto
    the point farthest from its centre, which :func:`KMeansScratch._reseed_empty` also does during fitting.
    """
    new_centers = old_centers.copy()
    for c in range(k):
        members = x[labels == c]
        if members.size:
            new_centers[c] = members.mean(axis=0)
    return new_centers


def inertia(x: NDArray[np.float64], labels: NDArray[np.int64], centers: NDArray[np.float64]) -> float:
    """Within-cluster sum of squares J = sum_i ||x_i - mu_{c_i}||^2 — the objective k-means minimizes."""
    diff = x - centers[labels]
    return float(np.sum(diff * diff))


def kpp_init(x: NDArray[np.float64], k: int, rng: np.random.Generator) -> NDArray[np.float64]:
    """k-means++ seeding: first centre uniform, each next sampled with probability proportional to D^2.

    ``D(x)`` is the distance from ``x`` to the nearest already-chosen centre; sampling proportional to
    ``D(x)^2`` biases hard toward points far from every current centre (the ones a clumped init would
    strand) while staying randomized, so a lone outlier cannot always hijack a seed. Squared weighting — not
    plain ``D`` — is what aligns the seeding with the squared objective and powers the O(log k) guarantee.

    This is the **greedy** variant scikit-learn uses: at each step it draws ``n_local_trials`` candidate
    points (each ∝ D^2) and keeps the one that lowers the total potential (sum of squared nearest-centre
    distances) the most, instead of committing to a single draw. Sampling several candidates and keeping the
    best removes the tail of unlucky single draws, which is why greedy k-means++ finds the global optimum so
    consistently in the init comparison. ``n_local_trials = 2 + floor(ln k)`` is scikit-learn's rule.
    """
    n = x.shape[0]
    n_local_trials = 2 + int(np.log(k))
    first = int(rng.integers(n))
    centers = [x[first]]
    closest_sq = pairwise_sq_dists(x, x[first : first + 1]).ravel()
    for _ in range(k - 1):
        total = closest_sq.sum()
        probs = closest_sq / total if total > 0 else np.full(n, 1.0 / n)
        candidates = rng.choice(n, size=n_local_trials, p=probs)  # several ∝ D^2 draws
        # For each candidate, the potential if we added it: sum of min(current closest, dist-to-candidate).
        cand_sq = pairwise_sq_dists(x, x[candidates])  # (n, n_local_trials)
        new_potentials = np.minimum(closest_sq[:, None], cand_sq).sum(axis=0)
        best = candidates[int(new_potentials.argmin())]  # keep the candidate that lowers J the most
        centers.append(x[best])
        closest_sq = np.minimum(closest_sq, pairwise_sq_dists(x, x[best : best + 1]).ravel())
    return np.array(centers)


def random_init(x: NDArray[np.float64], k: int, rng: np.random.Generator) -> NDArray[np.float64]:
    """Naive seeding: k distinct data points chosen uniformly at random (the baseline k-means++ beats)."""
    idx = rng.choice(x.shape[0], size=k, replace=False)
    return x[idx].copy()


# ============================ 3. k-means, from scratch =========================================
@dataclass
class FitResult:
    """The outcome of one Lloyd run: labels, final centres, inertia, and the per-iteration trace."""

    labels: NDArray[np.int64]
    centers: NDArray[np.float64]
    inertia: float
    trace: list[float] = field(default_factory=list)  # J at each iteration (must be non-increasing)
    n_iter: int = 0


class KMeansScratch:
    """Lloyd's algorithm, from scratch: k-means++/random seeding, assign/update, empty-cluster reseed, n_init.

    A single ``_lloyd`` run alternates the two cheap steps until the assignments stop changing, recording
    the inertia at every iteration and *asserting* it never rises — the executed convergence proof. ``fit``
    wraps that in ``n_init`` independent restarts and keeps the one with the lowest inertia, the standard
    defence against Lloyd's landing in a bad local optimum. Verified against scikit-learn in
    :func:`verify_against_sklearn`.
    """

    def __init__(
        self,
        *,
        n_clusters: int,
        init: str = "k-means++",
        n_init: int = N_INIT,
        max_iter: int = MAX_ITER,
        seed: int = RANDOM_STATE,
    ) -> None:
        if init not in {"k-means++", "random"}:
            raise ValueError(f"init must be 'k-means++' or 'random', got {init!r}")
        self.n_clusters = n_clusters
        self.init = init
        self.n_init = n_init
        self.max_iter = max_iter
        self.seed = seed
        self.labels_: NDArray[np.int64] | None = None
        self.cluster_centers_: NDArray[np.float64] | None = None
        self.inertia_: float = np.inf
        self.trace_: list[float] = []
        self.n_iter_: int = 0

    def _seed(self, x: NDArray[np.float64], rng: np.random.Generator) -> NDArray[np.float64]:
        if self.init == "k-means++":
            return kpp_init(x, self.n_clusters, rng)
        return random_init(x, self.n_clusters, rng)

    def _reseed_empty(
        self,
        x: NDArray[np.float64],
        labels: NDArray[np.int64],
        centers: NDArray[np.float64],
    ) -> NDArray[np.float64]:
        """Reseed any empty cluster onto the point currently farthest from its own centre (as sklearn does).

        An empty cluster contributes nothing to J while the worst-fit point inflates it, so moving the
        orphaned centre onto that point is the greedy way to spend it — and it stops the update step from
        ever facing a 0/0 mean.
        """
        centers = centers.copy()
        sq = pairwise_sq_dists(x, centers)
        for c in range(self.n_clusters):
            if not np.any(labels == c):
                worst = int(sq[np.arange(x.shape[0]), labels].argmax())  # point with the largest own-dist
                centers[c] = x[worst]
        return centers

    def _lloyd(self, x: NDArray[np.float64], rng: np.random.Generator) -> FitResult:
        """One Lloyd run from a fresh seed: assign/update until stable, tracking monotone-decreasing J."""
        centers = self._seed(x, rng)
        labels = assign(x, centers)
        centers = self._reseed_empty(x, labels, centers)
        trace: list[float] = []
        prev = np.inf
        for it in range(self.max_iter):
            j = inertia(x, labels, centers)
            trace.append(j)
            assert j <= prev + TIE_TOL, f"inertia rose ({prev} -> {j}) — Lloyd's must be monotone"
            new_centers = update(x, labels, self.n_clusters, centers)
            new_centers = self._reseed_empty(x, assign(x, new_centers), new_centers)
            new_labels = assign(x, new_centers)
            if np.array_equal(new_labels, labels) and np.allclose(new_centers, centers):
                return FitResult(labels=labels, centers=centers, inertia=j, trace=trace, n_iter=it + 1)
            labels, centers, prev = new_labels, new_centers, j
        return FitResult(
            labels=labels, centers=centers, inertia=inertia(x, labels, centers), trace=trace,
            n_iter=self.max_iter,
        )

    def fit(self, x: NDArray[np.float64]) -> KMeansScratch:
        """Run ``n_init`` seeded Lloyd restarts and keep the lowest-inertia result (the sklearn strategy)."""
        best: FitResult | None = None
        for i in range(self.n_init):
            rng = np.random.default_rng(self.seed + i)
            result = self._lloyd(x, rng)
            if best is None or result.inertia < best.inertia:
                best = result
        assert best is not None
        self.labels_ = best.labels
        self.cluster_centers_ = best.centers
        self.inertia_ = best.inertia
        self.trace_ = best.trace
        self.n_iter_ = best.n_iter
        return self

    def single_run_inertia(self, x: NDArray[np.float64], seed: int) -> float:
        """Inertia from ONE seeded Lloyd run (no restarts) — the atom of the init-comparison distribution."""
        return self._lloyd(x, np.random.default_rng(seed)).inertia


@dataclass
class Snapshot:
    """The state of Lloyd's algorithm at one iteration — for the step-by-step animation figure."""

    centers: NDArray[np.float64]  # centroid positions BEFORE this iteration's update
    labels: NDArray[np.int64]  # assignments made against those centroids
    inertia: float  # J at this state


def lloyd_history(
    x: NDArray[np.float64], k: int, *, seed: int = 0, init: str = "random"
) -> list[Snapshot]:
    """Record (centres, labels, J) at every Lloyd iteration — the animation data.

    Returns one :class:`Snapshot` per iteration (assignments against the current centres, then the centres
    move). The inertia strictly falls from snapshot to snapshot until it converges, which is exactly what
    the iteration figure and the monotone-J curve display. Used only for visualisation; ``init="random"``
    starts the centroids poorly so their migration across several steps is visible, whereas a k-means++
    seed on clean blobs converges in one or two steps.
    """
    rng = np.random.default_rng(seed)
    centers = kpp_init(x, k, rng) if init == "k-means++" else random_init(x, k, rng)
    labels = assign(x, centers)
    history: list[Snapshot] = []
    for _ in range(MAX_ITER):
        j = inertia(x, labels, centers)
        history.append(Snapshot(centers=centers.copy(), labels=labels.copy(), inertia=j))
        new_centers = update(x, labels, k, centers)
        new_labels = assign(x, new_centers)
        if np.array_equal(new_labels, labels) and np.allclose(new_centers, centers):
            break
        labels, centers = new_labels, new_centers
    return history


# ============================ 4. verify from-scratch == scikit-learn ============================
@dataclass
class MatchReport:
    """The from-scratch vs scikit-learn comparison on identical data, k, and seed."""

    scratch_inertia: float
    sklearn_inertia: float
    ari: float  # adjusted Rand index between the two partitions (1.0 = identical up to a permutation)
    k: int
    dataset: str


def verify_against_sklearn(data: Dataset, *, k: int) -> MatchReport:
    """Cluster with our KMeansScratch and scikit-learn's KMeans (same k, n_init, seed); compare inertia + ARI.

    Cluster *labels are arbitrary* — our "cluster 0" has no reason to be sklearn's "cluster 0" — so raw
    label equality is meaningless. The honest proof that our Lloyd's is the real thing is that the two reach
    the **same inertia** (the objective value) and the **same partition up to a permutation**, which the
    adjusted Rand index measures (chance-corrected, permutation-invariant).
    """
    scratch = KMeansScratch(n_clusters=k, init="k-means++", n_init=N_INIT, seed=RANDOM_STATE).fit(data.x)
    sk = KMeans(n_clusters=k, init="k-means++", n_init=N_INIT, random_state=RANDOM_STATE).fit(data.x)
    assert scratch.labels_ is not None
    return MatchReport(
        scratch_inertia=scratch.inertia_,
        sklearn_inertia=float(sk.inertia_),
        ari=float(adjusted_rand_score(scratch.labels_, sk.labels_)),
        k=k,
        dataset=data.name,
    )


# ============================ 5. k-means++ vs random, measured ==================================
@dataclass
class InitComparison:
    """Distribution of single-start inertia for random vs k-means++ seeding on a real dataset."""

    random_inertias: NDArray[np.float64]
    kpp_inertias: NDArray[np.float64]
    n_trials: int
    dataset: str
    k: int

    def summary(self, which: str) -> dict[str, float]:
        arr = self.random_inertias if which == "random" else self.kpp_inertias
        return {"mean": float(arr.mean()), "std": float(arr.std()),
                "best": float(arr.min()), "worst": float(arr.max())}


def compare_init(data: Dataset, *, k: int, n_trials: int = INIT_TRIALS) -> InitComparison:
    """Run ``n_trials`` single-start Lloyd runs each way and collect the final inertia of every one.

    A single start with random seeding often lands in a poor local optimum, so its inertia distribution is
    wide with a heavy upper tail; k-means++ spreads the seeds and lands near the global optimum almost
    every time, so its distribution is tight and low. This is the O(log k) guarantee made empirical.
    """
    random_js = np.array([
        KMeansScratch(n_clusters=k, init="random", n_init=1).single_run_inertia(data.x, seed=s)
        for s in range(n_trials)
    ])
    kpp_js = np.array([
        KMeansScratch(n_clusters=k, init="k-means++", n_init=1).single_run_inertia(data.x, seed=s)
        for s in range(n_trials)
    ])
    return InitComparison(
        random_inertias=random_js, kpp_inertias=kpp_js, n_trials=n_trials, dataset=data.name, k=k
    )


# ============================ 6. choosing k: elbow + silhouette =================================
@dataclass
class KSweep:
    """Inertia (elbow) and mean silhouette across a range of k on a real dataset."""

    ks: list[int]
    inertias: list[float]
    silhouettes: list[float]
    best_k_silhouette: int  # the k that MAXIMIZES the mean silhouette
    dataset: str


def sweep_k(data: Dataset, *, ks: tuple[int, ...] = K_SWEEP) -> KSweep:
    """Cluster at each k and record inertia and mean silhouette; the silhouette's peak is the chosen k.

    Inertia falls monotonically with k (more centres always fit tighter), so it can only be read as a
    *bend* — the elbow. The mean silhouette instead has a genuine **maximum**: it rewards clusters that are
    both internally tight and well separated, and collapses when k over- or under-splits the data.
    """
    inertias: list[float] = []
    silhouettes: list[float] = []
    for k in ks:
        km = KMeans(n_clusters=k, n_init=N_INIT, random_state=RANDOM_STATE).fit(data.x)
        inertias.append(float(km.inertia_))
        silhouettes.append(float(silhouette_score(data.x, km.labels_)))
    best_i = int(np.argmax(silhouettes))
    return KSweep(
        ks=list(ks), inertias=inertias, silhouettes=silhouettes,
        best_k_silhouette=ks[best_i], dataset=data.name,
    )


# ============================ 7. the failure modes, quantified ==================================
@dataclass
class FailureReport:
    """k-means on a structure it cannot represent, scored by ARI against the true labels."""

    labels: NDArray[np.int64]
    ari: float  # adjusted Rand index vs the true structure (near 0 = failed to recover it)
    k: int
    dataset: str


def measure_failure(data: Dataset, *, k: int) -> FailureReport:
    """Run k-means and score it against the KNOWN structure with the adjusted Rand index.

    Both moons and the sheared blobs have labels we know, so ARI directly quantifies the failure: k-means
    draws straight (Voronoi) boundaries, which cannot follow a crescent or a diagonal stripe, so it scores
    far below the ~1.0 it reaches on round, separated blobs.
    """
    assert data.y is not None
    km = KMeans(n_clusters=k, n_init=N_INIT, random_state=RANDOM_STATE).fit(data.x)
    return FailureReport(
        labels=km.labels_.astype(np.int64),
        ari=float(adjusted_rand_score(data.y, km.labels_)),
        k=k,
        dataset=data.name,
    )


# ============================ 8. run it all: the printed proof ==================================
def main() -> None:
    """Run every measured experiment and cross-check, printing the results the chapter cites."""
    import sklearn

    print(f"numpy {np.__version__} | scikit-learn {sklearn.__version__}\n")

    wine = load_wine_scaled()
    blobs = load_blobs_2d()

    # ---- 1. Lloyd's on the controlled 2-D blobs: inertia falls every iteration ----
    print("=== 1. Lloyd's from a k-means++ seed on 2-D blobs (J must never rise) ===")
    run = KMeansScratch(n_clusters=BLOB_K, init="k-means++", n_init=1, seed=0)._lloyd(
        blobs.x, np.random.default_rng(0)
    )
    for it, j in enumerate(run.trace):
        print(f"  iter {it:2d}: J = {j:10.1f}")
    print(f"  converged in {run.n_iter} iterations; monotone decrease held (asserted).\n")

    # ---- 2. from-scratch == scikit-learn on real Wine ----
    match = verify_against_sklearn(wine, k=WINE_K)
    print(f"=== 2. Verify from-scratch == scikit-learn on {match.dataset} (k={match.k}) ===")
    print(f"  from-scratch inertia : {match.scratch_inertia:.3f}")
    print(f"  scikit-learn inertia : {match.sklearn_inertia:.3f}")
    print(f"  adjusted Rand index  : {match.ari:.3f}  (1.0 = same partition up to a permutation)")
    if abs(match.scratch_inertia - match.sklearn_inertia) > INERTIA_MATCH_TOL:
        raise AssertionError("from-scratch and scikit-learn inertia must match on Wine")
    if match.ari < ARI_MATCH_FLOOR:
        raise AssertionError("from-scratch and scikit-learn partitions must agree up to a permutation")
    print("  -> same inertia, same partition: the from-scratch Lloyd's is the real thing.\n")

    # ---- 3. k-means++ beats random, as a measured distribution (many-cluster layout) ----
    comp = compare_init(load_blobs_many(), k=MANY_K)
    r, p = comp.summary("random"), comp.summary("kpp")
    print(f"=== 3. k-means++ vs random seeding on {comp.dataset}, {comp.n_trials} single starts each ===")
    print(f"  {'init':<12}{'mean J':>10}{'std':>9}{'best J':>10}{'worst J':>10}")
    print(f"  {'random':<12}{r['mean']:>10.1f}{r['std']:>9.1f}{r['best']:>10.1f}{r['worst']:>10.1f}")
    print(f"  {'k-means++':<12}{p['mean']:>10.1f}{p['std']:>9.1f}{p['best']:>10.1f}{p['worst']:>10.1f}")
    print(f"  -> k-means++ is {r['std'] / p['std']:.1f}x tighter and reaches the global optimum "
          f"({p['best']:.1f}) far more often; the advantage grows with k.\n")

    # ---- 4. choosing k on real Wine: elbow + silhouette ----
    sweep = sweep_k(wine)
    print(f"=== 4. Choosing k on {sweep.dataset}: inertia (elbow) + silhouette (peak) ===")
    print(f"  {'k':>4}{'inertia':>12}{'silhouette':>13}")
    for k, jj, ss in zip(sweep.ks, sweep.inertias, sweep.silhouettes):
        mark = "   <- peak" if k == sweep.best_k_silhouette else ""
        print(f"  {k:>4}{jj:>12.1f}{ss:>13.3f}{mark}")
    print(f"  -> silhouette peaks at k={sweep.best_k_silhouette}, the true number of Wine cultivars.")
    wine_recovery = measure_failure(wine, k=WINE_K)  # ARI of the k=3 clustering vs the real cultivar labels
    print(f"  adjusted Rand index at k={WINE_K} vs the true cultivars: {wine_recovery.ari:.3f}  "
          "(unsupervised, yet close to the labels)\n")
    if sweep.best_k_silhouette != wine.true_k:
        raise AssertionError("silhouette should select the true number of Wine cultivars (k=3)")
    if wine_recovery.ari < 0.85:
        raise AssertionError("k-means on Wine should recover the cultivars well (ARI ~0.90)")

    # ---- 5. the failure modes, quantified ----
    moons = measure_failure(load_moons(), k=2)
    aniso = measure_failure(load_anisotropic(), k=3)
    print("=== 5. Where k-means breaks (ARI vs the true structure; ~1.0 = perfect) ===")
    print(f"  two moons (k=2)        : ARI = {moons.ari:.3f}  (non-convex — straight cuts split each moon)")
    print(f"  anisotropic blobs (k=3): ARI = {aniso.ari:.3f}  (sheared — round cells cut across the stripes)")
    print("  -> low ARI is the honest signal to reach for DBSCAN (density) or a GMM (per-cluster shape).")
    if moons.ari > 0.5:
        raise AssertionError("k-means should FAIL on the two moons (low ARI)")


if __name__ == "__main__":
    main()
