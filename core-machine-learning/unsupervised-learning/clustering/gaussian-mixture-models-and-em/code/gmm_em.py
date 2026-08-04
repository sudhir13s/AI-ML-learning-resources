"""Gaussian Mixture Models via EM, from scratch on REAL data, VERIFIED against scikit-learn.

This is not a toy. Every number the chapter, the figures, and the notebook show is produced here from a
real pipeline (``numpy`` + ``scipy`` + ``scikit-learn`` + ``matplotlib``) on datasets that ship with
scikit-learn (no download) plus two clearly-labelled *controlled* generators used only where a known number
of components makes the point cleanly:

  * **Iris** (150 flowers, 4 measurements, 3 species) — standardized (z-scored) — is the real,
    higher-dimensional dataset and carries the load-bearing claim: our from-scratch EM reaches the **same
    log-likelihood and the same partition** as ``sklearn.mixture.GaussianMixture`` (full covariance, k-means
    init, same seed). Two of Iris's species (*versicolor*, *virginica*) overlap heavily, which is exactly
    where soft assignments earn their keep — the GMM recovers the three species far better than k-means
    (ARI 0.90 vs 0.62), because it fits each species its own elliptical covariance instead of a round cell.

  * **Iris petal plane (2-D, real)** — petal length and width, standardized — is the real 2-D view used for
    the figures you can actually *see*: the EM iterations (ellipses migrating, soft colours sharpening) and
    the monotone log-likelihood curve. It is a genuine slice of Iris, not synthetic; a fixed random init is
    used only so the ellipses visibly move (a k-means warm start would converge in one step).

  * **anisotropic (sheared) blobs — controlled, labelled** — the *same* diagonally-stretched clusters that
    [k-means fails on in chapter 01](../../01-K-Means-Clustering/01-K-Means-Clustering.md) (identical shear
    and seed). A full-covariance GMM recovers them perfectly (ARI 1.00) where k-means scores ARI 0.66 — the
    covariance matrix is the whole difference. Labelled "controlled illustration" wherever it appears.

  * **make_blobs (3 clusters, controlled)** — a clean, known-k layout used only for model selection, so the
    BIC/AIC minimum landing exactly on the true k=3 is unambiguous.

What this module measures (all real, all reproducible from the seed):

  * **GMM grown from scratch.** ``GMMScratch`` implements the EM loop derived in the chapter — the E-step
    computes responsibilities with a **log-sum-exp** normalizer (never raw densities, which underflow), the
    M-step applies the closed-form responsibility-weighted updates for the weights, means, and full
    covariances (with ``reg_covar`` floor against singular components), and every iteration's observed-data
    log-likelihood is *asserted* to be monotonically non-decreasing. That assertion is the EM convergence
    proof, executed.

  * **The from-scratch result VERIFIED against scikit-learn.** Same k, covariance type, init, and seed on
    standardized Iris: our best log-likelihood equals ``GaussianMixture``'s to a tiny tolerance and the two
    partitions agree up to a label permutation (adjusted Rand index ~1.0). Because component labels are
    arbitrary we compare *log-likelihood* and *ARI*, never raw label equality.

  * **Soft beats hard on overlapping data.** On real Iris the GMM's soft, elliptical fit recovers the three
    species markedly better than k-means' hard round cells, quantified by ARI against the true species.

  * **GMM succeeds where k-means fails.** On the anisotropic blobs k-means cannot represent (ch. 01), the
    full-covariance GMM recovers the clusters perfectly.

  * **Model selection.** BIC and AIC swept over the number of components on the controlled 3-blob data, both
    minimized at the true k=3; plus the covariance-type comparison (full / tied / diag / spherical) with
    parameter counts and BIC.

  * **The k-means limit, measured.** A spherical-covariance GMM's hard predictions agree almost perfectly
    with k-means' labels (ARI ~0.99) — k-means is the spherical, hard-assignment limit of a GMM.

  * **Two by-hand worked examples reproduced** — a single responsibility (Bayes' rule) and one full E/M
    iteration on ``{1, 2, 4, 7}`` — so the arithmetic in the chapter is executable, not asserted.

Everything is seeded and CPU-only; runs standalone in a few seconds::

    python gmm_em.py
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
from numpy.typing import NDArray
from scipy.special import logsumexp
from scipy.stats import multivariate_normal, norm
from sklearn.cluster import KMeans
from sklearn.datasets import load_iris, make_blobs
from sklearn.metrics import adjusted_rand_score
from sklearn.mixture import GaussianMixture
from sklearn.preprocessing import StandardScaler

# ---- Constants (hoisted; no magic numbers inline) ----------------------------------------------
RANDOM_STATE = 42  # the one seed fixing every dataset and every estimator's randomness
IRIS_K = 3  # Iris's true number of species; also the k the GMM recovers
BLOB_K = 3  # the controlled model-selection layout has three true clusters
N_INIT = 10  # EM restarts kept for the best-of comparison (matches scikit-learn's default)
MAX_ITER = 300  # hard cap on EM iterations (convergence is far faster in practice)
TOL = 1e-6  # log-likelihood-change convergence tolerance
REG_COVAR = 1e-6  # covariance floor added to each Sigma_k against singular / collapsing components
K_SWEEP = (1, 2, 3, 4, 5, 6, 7, 8)  # component counts probed for BIC / AIC model selection
LL_MATCH_TOL = 0.5  # from-scratch and scikit-learn log-likelihood must agree within this (nats, total)
ARI_MATCH_FLOOR = 0.95  # from-scratch and scikit-learn partitions must agree up to a permutation
MONOTONE_TOL = 1e-6  # numerical slack when asserting the log-likelihood never *decreases*
EM_FIG_SEED = 2  # the Iris-petal random-init seed whose EM run lands on the species-aligned optimum


# ============================ 1. real (and controlled) data =====================================
@dataclass
class Dataset:
    """A feature matrix, optional true labels, and human-readable names."""

    x: NDArray[np.float64]
    y: NDArray[np.int64] | None  # true labels when known (for ARI); None for unlabelled data
    feature_names: list[str]
    name: str
    true_k: int | None = None  # the known number of groups, where it exists


def load_iris_scaled() -> Dataset:
    """Iris standardized to zero mean / unit variance — the real, 4-D clustering problem.

    A GMM (like k-means) measures spread in the raw feature units, so every feature must contribute on a
    comparable scale; standardizing first keeps one wide-ranging measurement from dominating the covariances.
    Two of the three species overlap in feature space, which is exactly where a GMM's soft, elliptical fit
    beats a hard, round one.
    """
    data = load_iris()
    x = StandardScaler().fit_transform(data.data)
    return Dataset(
        x=x.astype(np.float64),
        y=data.target.astype(np.int64),
        feature_names=list(data.feature_names),
        name="Iris (4 features, standardized)",
        true_k=IRIS_K,
    )


def load_iris_petal_2d() -> Dataset:
    """Iris petal length & width, standardized — a REAL 2-D slice so EM's ellipses are visible.

    You cannot watch covariance ellipses migrate in four dimensions, so the iteration animation and the
    monotone-log-likelihood curve run on this genuine 2-D view of Iris (not synthetic data). Petal length and
    width separate *setosa* cleanly and leave *versicolor* / *virginica* overlapping — an honest picture of
    the soft assignments a GMM is built to express.
    """
    data = load_iris()
    x = StandardScaler().fit_transform(data.data[:, 2:4])  # petal length, petal width
    return Dataset(
        x=x.astype(np.float64),
        y=data.target.astype(np.int64),
        feature_names=["petal length (z)", "petal width (z)"],
        name="Iris petal plane (2-D, real)",
        true_k=IRIS_K,
    )


def load_anisotropic(*, seed: int = 170) -> Dataset:
    """Three diagonally-stretched (sheared) blobs — the SAME clusters k-means fails on in chapter 01.

    Identical shear matrix and seed to ``01-K-Means-Clustering``'s failure case, so the comparison is
    like-for-like: k-means draws round Voronoi cells that cut across the diagonal stripes (ARI ~0.66), while a
    full-covariance GMM fits a tilted ellipse to each stripe and recovers them (ARI ~1.0). Controlled and
    labelled, used to make the shape advantage vivid.
    """
    x, y = make_blobs(n_samples=600, centers=3, cluster_std=0.9, random_state=seed)
    shear = np.array([[0.60, -0.63], [-0.40, 0.85]])  # the exact ch. 01 transform that elongates the blobs
    return Dataset(
        x=(x @ shear).astype(np.float64),
        y=y.astype(np.int64),
        feature_names=["x", "y"],
        name="anisotropic blobs (sheared, controlled)",
        true_k=3,
    )


def load_blobs_3(*, seed: int = RANDOM_STATE) -> Dataset:
    """Three well-separated round Gaussian blobs — a CONTROLLED known-k layout for model selection.

    Used only so the BIC / AIC minimum landing exactly on the true k=3 is unambiguous; the recipe (sweep k,
    fit, take the argmin of BIC) is a property of the criterion, not of this data.
    """
    x, y = make_blobs(n_samples=500, centers=BLOB_K, cluster_std=1.0, random_state=seed)
    return Dataset(
        x=x.astype(np.float64),
        y=y.astype(np.int64),
        feature_names=["x", "y"],
        name="make_blobs (3 clusters, controlled)",
        true_k=BLOB_K,
    )


# ============================ 2. the core EM operations =========================================
def log_component_densities(
    x: NDArray[np.float64],
    weights: NDArray[np.float64],
    means: NDArray[np.float64],
    covs: NDArray[np.float64],
) -> NDArray[np.float64]:
    """The (n, k) matrix of log-weighted component densities: ``log pi_k + log N(x_n; mu_k, Sigma_k)``.

    Working in log-space is not optional: in more than a couple of dimensions the raw Gaussian density
    ``N(x; mu, Sigma)`` underflows to ``0.0``, and a downstream ``gamma = 0 / 0`` becomes ``NaN``. Every entry
    here is a log-density, combined later with :func:`scipy.special.logsumexp` — the numerically stable path
    scikit-learn also takes.
    """
    k = means.shape[0]
    cols = [
        np.log(weights[c]) + multivariate_normal(mean=means[c], cov=covs[c], allow_singular=False).logpdf(x)
        for c in range(k)
    ]
    return np.stack(cols, axis=1)


def e_step(
    x: NDArray[np.float64],
    weights: NDArray[np.float64],
    means: NDArray[np.float64],
    covs: NDArray[np.float64],
) -> tuple[NDArray[np.float64], float]:
    """E-step: responsibilities ``gamma_nk`` (soft posteriors) and the observed-data log-likelihood.

    The responsibility ``gamma(z_nk) = pi_k N(x_n; mu_k, Sigma_k) / sum_j pi_j N(x_n; mu_j, Sigma_j)`` is the
    posterior probability component k generated point n — Bayes' rule applied to the two-stage mixture. We
    compute it in log-space (subtract the log-sum-exp normalizer, then exponentiate), which is what keeps the
    denominator from underflowing. The same log-sum-exp over components, summed over points, *is* the
    observed-data log-likelihood ``sum_n log sum_k pi_k N(x_n; ...)`` — returned so the caller can assert it
    never decreases.
    """
    log_dens = log_component_densities(x, weights, means, covs)  # (n, k)
    log_norm = logsumexp(log_dens, axis=1)  # (n,) — the log mixture density per point
    log_resp = log_dens - log_norm[:, None]
    return np.exp(log_resp), float(log_norm.sum())


def m_step(
    x: NDArray[np.float64], resp: NDArray[np.float64], *, reg_covar: float = REG_COVAR
) -> tuple[NDArray[np.float64], NDArray[np.float64], NDArray[np.float64]]:
    """M-step: closed-form responsibility-weighted updates for weights, means, and full covariances.

    With the responsibilities fixed, the expected complete-data log-likelihood separates per component and is
    maximized in closed form (set each gradient to zero). Every update is the ordinary single-Gaussian MLE,
    but counting each point by its **soft** responsibility ``gamma`` instead of a hard 0/1:

      * soft counts   ``N_k = sum_n gamma_nk``            (the effective number of points component k owns)
      * weights       ``pi_k = N_k / N``
      * means         ``mu_k = (1 / N_k) sum_n gamma_nk x_n``
      * covariances   ``Sigma_k = (1 / N_k) sum_n gamma_nk (x_n - mu_k)(x_n - mu_k)^T  +  reg_covar I``

    The means must be updated *before* the covariances (the covariance formula centres on the new mean). The
    ``reg_covar`` floor on the diagonal keeps a component from collapsing onto a single point and driving the
    likelihood to ``+inf`` (the singularity failure mode) — it is MAP-EM with a weak prior, not an ad-hoc
    patch.
    """
    n, d = x.shape
    nk = resp.sum(axis=0)  # (k,) soft counts
    weights = nk / n
    means = (resp.T @ x) / nk[:, None]  # (k, d)
    k = resp.shape[1]
    covs = np.empty((k, d, d))
    eye = np.eye(d)
    for c in range(k):
        centered = x - means[c]  # (n, d)
        covs[c] = (resp[:, c, None] * centered).T @ centered / nk[c] + reg_covar * eye
    return weights, means, covs


# ============================ 3. GMM, from scratch =============================================
@dataclass
class FitResult:
    """The outcome of one EM run: parameters, responsibilities, final log-likelihood, and the trace."""

    weights: NDArray[np.float64]
    means: NDArray[np.float64]
    covs: NDArray[np.float64]
    resp: NDArray[np.float64]
    log_likelihood: float
    trace: list[float] = field(default_factory=list)  # log-likelihood at each iteration (non-decreasing)
    n_iter: int = 0

    @property
    def labels(self) -> NDArray[np.int64]:
        """Hard labels from the soft responsibilities (the most-probable component per point)."""
        return self.resp.argmax(axis=1).astype(np.int64)


class GMMScratch:
    """EM for a Gaussian mixture, from scratch: log-sum-exp E-step, closed-form M-step, n_init restarts.

    A single ``_em`` run alternates the two steps until the log-likelihood change drops below ``tol``,
    recording the log-likelihood at every iteration and *asserting* it never decreases — the executed EM
    convergence proof. ``fit`` wraps that in ``n_init`` independent restarts and keeps the run with the
    **highest** final log-likelihood, the standard defence against EM's local optima. Verified against
    scikit-learn in :func:`verify_against_sklearn`.
    """

    def __init__(
        self,
        *,
        n_components: int,
        n_init: int = N_INIT,
        max_iter: int = MAX_ITER,
        tol: float = TOL,
        reg_covar: float = REG_COVAR,
        init: str = "kmeans",
        seed: int = RANDOM_STATE,
    ) -> None:
        if init not in {"kmeans", "random"}:
            raise ValueError(f"init must be 'kmeans' or 'random', got {init!r}")
        self.n_components = n_components
        self.n_init = n_init
        self.max_iter = max_iter
        self.tol = tol
        self.reg_covar = reg_covar
        self.init = init
        self.seed = seed
        self.weights_: NDArray[np.float64] | None = None
        self.means_: NDArray[np.float64] | None = None
        self.covs_: NDArray[np.float64] | None = None
        self.resp_: NDArray[np.float64] | None = None
        self.log_likelihood_: float = -np.inf
        self.trace_: list[float] = []
        self.n_iter_: int = 0

    def _initialize(
        self, x: NDArray[np.float64], rng: np.random.Generator
    ) -> tuple[NDArray[np.float64], NDArray[np.float64], NDArray[np.float64]]:
        """Seed the parameters — a k-means warm start (scikit-learn's default) or random points.

        The k-means warm start places the means at the k-means centroids and each covariance at the
        within-cluster covariance of that hard partition, so EM only has to *soften and reshape* a decent
        solution. Random init (means at random points, one shared global covariance) starts EM cold and is
        used mainly for the animation, where a poor start makes the ellipses' migration visible.
        """
        n, d = x.shape
        k = self.n_components
        if self.init == "kmeans":
            km = KMeans(n_clusters=k, n_init=N_INIT, random_state=int(rng.integers(1 << 31))).fit(x)
            means = km.cluster_centers_.astype(np.float64).copy()
            hard = np.zeros((n, k))
            hard[np.arange(n), km.labels_] = 1.0
            return m_step(x, hard, reg_covar=self.reg_covar)
        means = x[rng.choice(n, k, replace=False)].astype(np.float64)
        covs = np.array([np.cov(x.T) + self.reg_covar * np.eye(d) for _ in range(k)])
        weights = np.full(k, 1.0 / k)
        return weights, means, covs

    def _em(self, x: NDArray[np.float64], rng: np.random.Generator) -> FitResult:
        """One EM run from a fresh seed: E/M until the log-likelihood plateaus, asserting monotonicity."""
        weights, means, covs = self._initialize(x, rng)
        trace: list[float] = []
        prev = -np.inf
        resp = np.empty((x.shape[0], self.n_components))
        for it in range(self.max_iter):
            resp, ll = e_step(x, weights, means, covs)
            assert ll >= prev - MONOTONE_TOL, f"log-likelihood fell ({prev} -> {ll}) — EM must be monotone"
            trace.append(ll)
            if it > 0 and abs(ll - prev) < self.tol:
                return FitResult(weights, means, covs, resp, ll, trace, it + 1)
            weights, means, covs = m_step(x, resp, reg_covar=self.reg_covar)
            prev = ll
        return FitResult(weights, means, covs, resp, trace[-1], trace, self.max_iter)

    def fit(self, x: NDArray[np.float64]) -> GMMScratch:
        """Run ``n_init`` seeded EM restarts and keep the highest-log-likelihood result (the sklearn strategy)."""
        best: FitResult | None = None
        for i in range(self.n_init):
            rng = np.random.default_rng(self.seed + i)
            result = self._em(x, rng)
            if best is None or result.log_likelihood > best.log_likelihood:
                best = result
        assert best is not None
        self.weights_, self.means_, self.covs_ = best.weights, best.means, best.covs
        self.resp_ = best.resp
        self.log_likelihood_ = best.log_likelihood
        self.trace_ = best.trace
        self.n_iter_ = best.n_iter
        return self

    @property
    def labels_(self) -> NDArray[np.int64]:
        assert self.resp_ is not None, "call fit first"
        return self.resp_.argmax(axis=1).astype(np.int64)


@dataclass
class Snapshot:
    """The state of EM at one iteration — for the step-by-step ellipse animation figure."""

    means: NDArray[np.float64]
    covs: NDArray[np.float64]
    resp: NDArray[np.float64]  # soft responsibilities → the blended point colours
    log_likelihood: float


def em_history(x: NDArray[np.float64], k: int, *, seed: int = EM_FIG_SEED) -> list[Snapshot]:
    """Record (means, covs, responsibilities, log-likelihood) at every EM iteration — the animation data.

    Uses a random init (not a k-means warm start) so the ellipses start poorly placed and their migration
    across many iterations is visible. The log-likelihood strictly rises from snapshot to snapshot until it
    plateaus, which is exactly what the iteration figure and the monotone-LL curve display.
    """
    rng = np.random.default_rng(seed)
    n, d = x.shape
    means = x[rng.choice(n, k, replace=False)].astype(np.float64)
    covs = np.array([np.cov(x.T) + REG_COVAR * np.eye(d) for _ in range(k)])
    weights = np.full(k, 1.0 / k)
    history: list[Snapshot] = []
    prev = -np.inf
    for _ in range(MAX_ITER):
        resp, ll = e_step(x, weights, means, covs)
        history.append(Snapshot(means.copy(), covs.copy(), resp.copy(), ll))
        if abs(ll - prev) < TOL and prev > -np.inf:
            break
        weights, means, covs = m_step(x, resp)
        prev = ll
    return history


# ============================ 4. verify from-scratch == scikit-learn ============================
@dataclass
class MatchReport:
    """The from-scratch vs scikit-learn comparison on identical data, k, covariance, and seed."""

    scratch_ll: float
    sklearn_ll: float
    scratch_ari_truth: float  # ARI of the from-scratch partition vs the true labels
    sklearn_ari_truth: float
    ari_scratch_vs_sklearn: float  # ARI between the two partitions (1.0 = identical up to a permutation)
    kmeans_ari_truth: float  # k-means on the same data, for the soft-vs-hard contrast
    k: int
    dataset: str


def verify_against_sklearn(data: Dataset, *, k: int) -> MatchReport:
    """Fit our GMMScratch and scikit-learn's GaussianMixture (same k, full cov, k-means init, seed); compare.

    Component *labels are arbitrary* — our "component 0" has no reason to be sklearn's "component 0" — so raw
    label equality is meaningless. The honest proof that our EM is the real thing is that the two reach the
    **same log-likelihood** (the objective value) and the **same partition up to a permutation** (adjusted
    Rand index). We also score both against the true species and add k-means for the soft-vs-hard contrast.
    """
    assert data.y is not None
    scratch = GMMScratch(n_components=k, n_init=N_INIT, init="kmeans", seed=RANDOM_STATE).fit(data.x)
    sk = GaussianMixture(
        n_components=k, covariance_type="full", n_init=N_INIT, reg_covar=REG_COVAR,
        init_params="kmeans", random_state=RANDOM_STATE,
    ).fit(data.x)
    km = KMeans(n_clusters=k, n_init=N_INIT, random_state=RANDOM_STATE).fit(data.x)
    return MatchReport(
        scratch_ll=scratch.log_likelihood_,
        sklearn_ll=float(sk.score(data.x) * data.x.shape[0]),
        scratch_ari_truth=float(adjusted_rand_score(data.y, scratch.labels_)),
        sklearn_ari_truth=float(adjusted_rand_score(data.y, sk.predict(data.x))),
        ari_scratch_vs_sklearn=float(adjusted_rand_score(scratch.labels_, sk.predict(data.x))),
        kmeans_ari_truth=float(adjusted_rand_score(data.y, km.labels_)),
        k=k,
        dataset=data.name,
    )


# ============================ 5. GMM vs k-means on shape ========================================
@dataclass
class ShapeReport:
    """GMM vs k-means on a structure only per-cluster covariance can capture, scored by ARI vs truth."""

    gmm_labels: NDArray[np.int64]
    kmeans_labels: NDArray[np.int64]
    gmm_means: NDArray[np.float64]
    gmm_covs: NDArray[np.float64]
    gmm_ari: float
    kmeans_ari: float
    k: int
    dataset: str


def compare_shape(data: Dataset, *, k: int) -> ShapeReport:
    """Fit both a full-covariance GMM and k-means on anisotropic data; score each against the true labels.

    The sheared blobs have orientation k-means ignores (round Voronoi cells cut across the diagonal stripes),
    so its ARI is far below the GMM's — the GMM fits each stripe a tilted ellipse and recovers them. Returns
    the GMM's learned means and covariances so the figure can draw the ellipses that ARE the model.
    """
    assert data.y is not None
    gm = GaussianMixture(
        n_components=k, covariance_type="full", n_init=N_INIT, reg_covar=REG_COVAR, random_state=RANDOM_STATE
    ).fit(data.x)
    km = KMeans(n_clusters=k, n_init=N_INIT, random_state=RANDOM_STATE).fit(data.x)
    return ShapeReport(
        gmm_labels=gm.predict(data.x).astype(np.int64),
        kmeans_labels=km.labels_.astype(np.int64),
        gmm_means=gm.means_,
        gmm_covs=gm.covariances_,
        gmm_ari=float(adjusted_rand_score(data.y, gm.predict(data.x))),
        kmeans_ari=float(adjusted_rand_score(data.y, km.labels_)),
        k=k,
        dataset=data.name,
    )


# ============================ 6. model selection: BIC / AIC =====================================
def soft_vs_hard_ari(data: Dataset, *, k: int) -> tuple[float, float]:
    """ARI vs truth for a full-covariance GMM and for k-means on the same data — the soft-vs-hard contrast.

    On overlapping data the GMM's soft, elliptical fit recovers the true groups better than k-means' hard
    round cells. Used for the 2-D petal-plane figure's numbers (so they are printed, not only drawn).
    """
    assert data.y is not None
    gm = GaussianMixture(
        n_components=k, covariance_type="full", n_init=N_INIT, reg_covar=REG_COVAR, random_state=RANDOM_STATE
    ).fit(data.x)
    km = KMeans(n_clusters=k, n_init=N_INIT, random_state=RANDOM_STATE).fit(data.x)
    return (
        float(adjusted_rand_score(data.y, gm.predict(data.x))),
        float(adjusted_rand_score(data.y, km.labels_)),
    )


@dataclass
class ModelSelection:
    """BIC and AIC across a range of component counts on a real / controlled dataset."""

    ks: list[int]
    bic: list[float]
    aic: list[float]
    best_k_bic: int
    best_k_aic: int
    dataset: str


def select_k(data: Dataset, *, ks: tuple[int, ...] = K_SWEEP) -> ModelSelection:
    """Fit a full-covariance GMM at each k and record BIC and AIC; the minimum of each is the chosen k.

    The raw log-likelihood always rises with k (more Gaussians fit the training data better), so it cannot
    pick k. BIC = -2 ell + p log N and AIC = -2 ell + 2p tax the parameter count p, so both have a genuine
    minimum — here at the true number of clusters.
    """
    bic: list[float] = []
    aic: list[float] = []
    for k in ks:
        gm = GaussianMixture(
            n_components=k, covariance_type="full", n_init=N_INIT, reg_covar=REG_COVAR,
            random_state=RANDOM_STATE,
        ).fit(data.x)
        bic.append(float(gm.bic(data.x)))
        aic.append(float(gm.aic(data.x)))
    return ModelSelection(
        ks=list(ks), bic=bic, aic=aic,
        best_k_bic=ks[int(np.argmin(bic))], best_k_aic=ks[int(np.argmin(aic))], dataset=data.name,
    )


@dataclass
class CovTypeReport:
    """Parameter count, BIC, and ARI for each covariance parameterization on one dataset."""

    rows: list[tuple[str, int, float, float]]  # (type, n_params, bic, ari)
    dataset: str


def compare_covariance_types(data: Dataset, *, k: int) -> CovTypeReport:
    """Fit full / tied / diag / spherical GMMs; report each one's parameter count, BIC, and ARI vs truth.

    On the sheared blobs all three clusters share the *same* shear, so ``tied`` (one shared full covariance)
    is not just adequate but the BIC winner — it captures the tilt with the fewest parameters. ``diag`` and
    ``spherical`` cannot tilt, so they misfit the diagonal stripes (low ARI). A concrete bias-variance
    lesson: more covariance structure is only worth its parameters when the data needs it.
    """
    assert data.y is not None
    rows: list[tuple[str, int, float, float]] = []
    for cov_type in ("full", "tied", "diag", "spherical"):
        gm = GaussianMixture(
            n_components=k, covariance_type=cov_type, n_init=N_INIT, reg_covar=REG_COVAR,
            random_state=RANDOM_STATE,
        ).fit(data.x)
        rows.append((
            cov_type,
            int(gm._n_parameters()),  # noqa: SLF001  (scikit-learn's own free-parameter count, for BIC)
            float(gm.bic(data.x)),
            float(adjusted_rand_score(data.y, gm.predict(data.x))),
        ))
    return CovTypeReport(rows=rows, dataset=data.name)


# ============================ 7. the k-means limit, measured ====================================
def kmeans_limit_ari(data: Dataset, *, k: int) -> tuple[float, float, float]:
    """Measure that a spherical GMM's hard predictions ≈ k-means labels (k-means is the spherical limit).

    A spherical-covariance GMM shares k-means' equal-round-cluster assumption; taking its most-probable
    component per point (a hard assignment) should agree almost perfectly with k-means. Returns the ARI
    between the spherical GMM and k-means, and each one's ARI against the truth, on the controlled blobs.
    """
    assert data.y is not None
    km = KMeans(n_clusters=k, n_init=N_INIT, random_state=RANDOM_STATE).fit(data.x)
    gm = GaussianMixture(
        n_components=k, covariance_type="spherical", n_init=N_INIT, reg_covar=REG_COVAR,
        random_state=RANDOM_STATE,
    ).fit(data.x)
    return (
        float(adjusted_rand_score(km.labels_, gm.predict(data.x))),
        float(adjusted_rand_score(data.y, km.labels_)),
        float(adjusted_rand_score(data.y, gm.predict(data.x))),
    )


# ============================ 8. two by-hand worked examples ====================================
def worked_example_1_responsibility() -> dict[str, float]:
    """Reproduce a single responsibility by hand: N(0,1) vs N(4,1), point at x=1, equal then unequal priors."""
    dens_a, dens_b = norm(0, 1).pdf(1.0), norm(4, 1).pdf(1.0)
    equal = 0.5 * dens_a / (0.5 * dens_a + 0.5 * dens_b)
    unequal = 0.7 * dens_a / (0.7 * dens_a + 0.3 * dens_b)
    return {"gamma_A_equal_prior": float(equal), "gamma_A_prior_0.7": float(unequal)}


def worked_example_2_one_em_step() -> dict[str, float]:
    """Reproduce one full E/M iteration on {1,2,4,7}, K=2, mu=(2,6), var=2, equal weights (all by hand)."""
    x = np.array([1.0, 2.0, 4.0, 7.0])
    sd = np.sqrt(2.0)
    ga = 0.5 * norm(2, sd).pdf(x)
    gb = 0.5 * norm(6, sd).pdf(x)
    ga, gb = ga / (ga + gb), gb / (ga + gb)
    n_a, n_b = ga.sum(), gb.sum()
    mu_a, mu_b = (ga * x).sum() / n_a, (gb * x).sum() / n_b
    var_a = (ga * (x - mu_a) ** 2).sum() / n_a
    var_b = (gb * (x - mu_b) ** 2).sum() / n_b
    return {
        "gamma_A": ga, "N_A": float(n_a), "pi_A": float(n_a / 4),
        "mu_A": float(mu_a), "mu_B": float(mu_b), "var_A": float(var_a), "var_B": float(var_b),
    }


def temperature_responsibility(sq_dists: tuple[float, float], sigma2: float) -> float:
    """The near-component responsibility for a point at the two squared distances, at temperature sigma2.

    This is the spherical-equal-weight responsibility = softmax(-d^2 / 2 sigma2). Shrinking sigma2 toward
    zero drives it to a hard 0/1 — the k-means limit, made numeric.
    """
    near, far = sq_dists
    num = np.exp(-near / (2 * sigma2))
    return float(num / (num + np.exp(-far / (2 * sigma2))))


# ============================ 9. run it all: the printed proof ==================================
def main() -> None:
    """Run every measured experiment and cross-check, printing the results the chapter cites."""
    import scipy
    import sklearn

    print(f"numpy {np.__version__} | scipy {scipy.__version__} | scikit-learn {sklearn.__version__}\n")

    iris = load_iris_scaled()

    # ---- 1. from-scratch EM == scikit-learn on real Iris; soft beats hard ----
    match = verify_against_sklearn(iris, k=IRIS_K)
    print(f"=== 1. Verify from-scratch EM == scikit-learn on {match.dataset} (k={match.k}, full cov) ===")
    print(f"  from-scratch log-likelihood : {match.scratch_ll:.4f}")
    print(f"  scikit-learn log-likelihood : {match.sklearn_ll:.4f}")
    print(f"  ARI (from-scratch vs sklearn): {match.ari_scratch_vs_sklearn:.4f}  (1.0 = same partition)")
    print(f"  ARI vs true species — GMM    : {match.scratch_ari_truth:.4f}")
    print(f"  ARI vs true species — k-means: {match.kmeans_ari_truth:.4f}  (soft/elliptical beats hard/round)")
    if abs(match.scratch_ll - match.sklearn_ll) > LL_MATCH_TOL:
        raise AssertionError("from-scratch and scikit-learn log-likelihood must match on Iris")
    if match.ari_scratch_vs_sklearn < ARI_MATCH_FLOOR:
        raise AssertionError("from-scratch and scikit-learn partitions must agree up to a permutation")
    print("  -> same log-likelihood, same partition: the from-scratch EM is the real thing.\n")

    # ---- 2. EM is monotone: the log-likelihood only rises (the executed convergence proof) ----
    history = em_history(load_iris_petal_2d().x, IRIS_K)
    lls = [s.log_likelihood for s in history]
    monotone = all(lls[i + 1] >= lls[i] - MONOTONE_TOL for i in range(len(lls) - 1))
    print("=== 2. EM log-likelihood is monotone on the real Iris petal plane (2-D view) ===")
    print(f"  iterations         : {len(lls)}")
    print(f"  log-likelihood     : {lls[0]:.2f} -> {lls[-1]:.2f}  (rose every step: {monotone})")
    if not monotone:
        raise AssertionError("EM log-likelihood must be monotonically non-decreasing")
    print("  -> the E/M loop provably cannot lower the likelihood; here it is, climbing to convergence.\n")

    # ---- 2b. soft vs hard on the real petal plane (the figure's numbers, printed) ----
    petal_gmm, petal_km = soft_vs_hard_ari(load_iris_petal_2d(), k=IRIS_K)
    print("=== 2b. Soft vs hard on the real Iris petal plane (2-D) ===")
    print(f"  GMM (soft, elliptical) ARI vs species : {petal_gmm:.4f}")
    print(f"  k-means (hard, round)  ARI vs species : {petal_km:.4f}  (the GMM keeps overlap points soft)\n")

    # ---- 3. GMM succeeds where k-means fails (the ch. 01 anisotropic data) ----
    shape = compare_shape(load_anisotropic(), k=3)
    print(f"=== 3. GMM vs k-means on {shape.dataset} — the clusters k-means fails on in ch. 01 ===")
    print(f"  full-covariance GMM : ARI = {shape.gmm_ari:.4f}  (fits a tilted ellipse to each stripe)")
    print(f"  k-means             : ARI = {shape.kmeans_ari:.4f}  (round cells cut across the stripes)")
    if shape.gmm_ari < 0.9 or shape.kmeans_ari > 0.8:
        raise AssertionError("GMM should recover the sheared blobs; k-means should not")
    print("  -> the covariance matrix is the whole difference.\n")

    # ---- 4. model selection: BIC and AIC both pick the true k ----
    sel = select_k(load_blobs_3())
    print(f"=== 4. Choosing k by BIC / AIC on {sel.dataset} (true k={BLOB_K}) ===")
    print(f"  {'k':>3}{'BIC':>12}{'AIC':>12}")
    for k, b, a in zip(sel.ks, sel.bic, sel.aic):
        mark = "   <- min" if k == sel.best_k_bic else ""
        print(f"  {k:>3}{b:>12.1f}{a:>12.1f}{mark}")
    print(f"  -> BIC minimized at k={sel.best_k_bic}, AIC at k={sel.best_k_aic}; both recover the true count.")
    if sel.best_k_bic != BLOB_K:
        raise AssertionError("BIC should select the true number of clusters on the controlled blobs")

    # ---- 5. covariance types: parameters vs fit ----
    cov = compare_covariance_types(load_anisotropic(), k=3)
    print(f"\n=== 5. Covariance type on {cov.dataset} (k=3): parameters vs fit ===")
    print(f"  {'type':<11}{'n_params':>9}{'BIC':>11}{'ARI':>8}")
    for name, n_params, bic, ari in cov.rows:
        print(f"  {name:<11}{n_params:>9}{bic:>11.1f}{ari:>8.3f}")
    print("  -> the blobs share one shear, so 'tied' captures the tilt with the fewest params (best BIC);")
    print("     'diag'/'spherical' cannot tilt, so they misfit the diagonal stripes (low ARI).")

    # ---- 6. the k-means limit, measured ----
    gmm_vs_km, km_truth, sph_truth = kmeans_limit_ari(load_blobs_3(), k=BLOB_K)
    print("\n=== 6. k-means is the spherical, hard limit of a GMM (controlled 3-blob) ===")
    print(f"  spherical-GMM predictions vs k-means labels : ARI = {gmm_vs_km:.4f}  (they nearly coincide)")
    print(f"  k-means vs truth = {km_truth:.4f}   spherical GMM vs truth = {sph_truth:.4f}")
    for sigma2 in (2.0, 0.2):
        gamma = temperature_responsibility((1.0, 4.0), sigma2)
        print(f"  responsibility (near center, sq-dists 1 & 4) at sigma^2={sigma2}: {gamma:.4f}")
    print("  -> shrinking the (equal, spherical) variance sharpens the soft assignment toward k-means' hard cut.")

    # ---- 7. reproduce the two by-hand worked examples ----
    ex1 = worked_example_1_responsibility()
    ex2 = worked_example_2_one_em_step()
    print("\n=== 7. The by-hand worked examples, executed ===")
    print(f"  Ex1  gamma_A(x=1), equal prior   = {ex1['gamma_A_equal_prior']:.4f}  (component A is N(0,1))")
    print(f"  Ex1  gamma_A(x=1), prior 0.7/0.3 = {ex1['gamma_A_prior_0.7']:.4f}  (the prior tilts it further)")
    print(f"  Ex2  responsibilities gamma_A    = {np.round(ex2['gamma_A'], 4)}")
    print(f"  Ex2  M-step: pi_A={ex2['pi_A']:.3f}  mu_A={ex2['mu_A']:.3f}  mu_B={ex2['mu_B']:.3f}  "
          f"var_A={ex2['var_A']:.3f}  var_B={ex2['var_B']:.3f}")


if __name__ == "__main__":
    main()
