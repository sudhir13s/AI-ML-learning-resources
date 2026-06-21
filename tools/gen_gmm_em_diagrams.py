"""Gaussian-Mixture-Model + EM concept-page diagrams (muted palette, parallel scale).

Four MEASURED figures for 04. Unsupervised_Learning/concepts/04-Gaussian-Mixture-Models-and-EM.md:
  1. gmm_em_iterations.png -- the EM loop in action: soft responsibilities + covariance
     ellipses at iter 0 (init), an early iter, and convergence. REAL EM run by hand.
  2. gmm_vs_kmeans.png    -- on elongated/anisotropic clusters GMM (full covariance) captures
     the tilted ellipses while k-means cuts straight through them. REAL sklearn fits, ARI measured.
  3. gmm_loglik.png       -- the observed-data log-likelihood, MEASURED per EM iteration,
     climbing monotonically to a plateau (the convergence guarantee, seen).
  4. gmm_bic.png          -- BIC vs number of components, MEASURED on 3-cluster data; the dip
     at k=3 selects the right model order.

Run with:  /Users/sudhirsingh/.uv/envs/ml-py312/bin/python3 tools/gen_gmm_em_diagrams.py
"""
import os, matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import Ellipse
from matplotlib.colors import ListedColormap
from scipy.stats import multivariate_normal
from sklearn.mixture import GaussianMixture
from sklearn.cluster import KMeans
from sklearn.datasets import make_blobs
from sklearn.metrics import adjusted_rand_score

OUT = os.path.join(os.path.dirname(__file__), "..", "04. Unsupervised_Learning", "concepts", "images")
OUT = os.path.abspath(OUT)
os.makedirs(OUT, exist_ok=True)

BLUE, PURPLE, GREEN, RED, SLATE, AMBER, NAVY = (
    "#3A6B96", "#5D4A8A", "#2E7A5A", "#8B3B4A", "#4A5B6E", "#7A6528", "#2A5B80")
plt.rcParams.update({"font.size": 11, "font.family": "DejaVu Sans"})
COMP = [BLUE, RED, GREEN]


def _despine(ax):
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)


def _ellipse(ax, mu, cov, color, nsig=2.0, lw=2.0, alpha=0.9):
    """Draw an n-sigma covariance ellipse from the eigendecomposition of cov."""
    vals, vecs = np.linalg.eigh(cov)
    order = vals.argsort()[::-1]
    vals, vecs = vals[order], vecs[:, order]
    angle = np.degrees(np.arctan2(vecs[1, 0], vecs[0, 0]))
    width, height = 2 * nsig * np.sqrt(vals)
    e = Ellipse(mu, width, height, angle=angle, fill=False, edgecolor=color, lw=lw, alpha=alpha)
    ax.add_patch(e)


# ----- a tiny from-scratch EM so we can snapshot exact iterations -----
def em_steps(X, K, iters, seed=0):
    """Yield (mu, cov, pi, resp) BEFORE each M-step so we can snapshot the trajectory."""
    rng = np.random.default_rng(seed)
    n, d = X.shape
    # init: random points as means, global covariance, uniform weights
    mu = X[rng.choice(n, K, replace=False)].astype(float)
    cov = np.array([np.cov(X.T) for _ in range(K)])
    pi = np.full(K, 1.0 / K)
    snaps = []
    for it in range(iters):
        # E-step: responsibilities
        r = np.stack([pi[k] * multivariate_normal(mu[k], cov[k], allow_singular=True).pdf(X)
                      for k in range(K)], axis=1)
        r /= r.sum(1, keepdims=True)
        snaps.append((mu.copy(), cov.copy(), pi.copy(), r.copy()))
        # M-step
        Nk = r.sum(0)
        mu = (r.T @ X) / Nk[:, None]
        cov = np.array([(r[:, k:k+1] * (X - mu[k])).T @ (X - mu[k]) / Nk[k]
                        + 1e-6 * np.eye(d) for k in range(K)])
        pi = Nk / n
    return snaps


def gmm_em_iterations():
    # anisotropic, slightly overlapping clusters so the soft assignment is visible
    rng = np.random.default_rng(3)
    n = 150
    A = rng.multivariate_normal([0, 0], [[1.0, 0.0], [0.0, 1.0]], n)
    B = rng.multivariate_normal([3.5, 3.0], [[1.4, 0.9], [0.9, 1.0]], n)
    C = rng.multivariate_normal([-1.0, 4.0], [[0.6, -0.4], [-0.4, 1.2]], n)
    X = np.vstack([A, B, C])
    snaps = em_steps(X, 3, iters=25, seed=1)
    picks = [0, 2, 24]  # init, a few iters in, converged
    titles = ["Iteration 0 (initialization)", "Iteration 2 (E/M underway)", "Iteration 25 (converged)"]
    fig, axes = plt.subplots(1, 3, figsize=(15, 5.0))
    for ax, p, t in zip(axes, picks, titles):
        mu, cov, pi, r = snaps[p]
        # color each point by its responsibility-weighted soft membership (RGB blend)
        rgb = np.zeros((len(X), 3))
        base = np.array([[0.227, 0.42, 0.588],    # BLUE
                         [0.545, 0.231, 0.29],    # RED
                         [0.18, 0.478, 0.353]])   # GREEN
        rgb = r @ base
        ax.scatter(X[:, 0], X[:, 1], c=rgb, s=16, edgecolor="white", lw=0.2)
        for k in range(3):
            _ellipse(ax, mu[k], cov[k], COMP[k], nsig=2.0, lw=2.2)
            ax.plot(*mu[k], marker="X", color=COMP[k], ms=12, mec="white", mew=1.2)
        ax.set_title(t, fontsize=11.5, fontweight="bold")
        ax.set_xlabel("feature 1"); ax.set_ylabel("feature 2")
        ax.set_xlim(-4, 7); ax.set_ylim(-4, 8); _despine(ax)
    fig.suptitle("EM for a GMM: soft responsibilities (point color) and covariance ellipses converge",
                 fontsize=13, fontweight="bold", y=1.02)
    fig.tight_layout(); fig.savefig(f"{OUT}/gmm_em_iterations.png", dpi=150, bbox_inches="tight")
    plt.close(fig)
    print("wrote gmm_em_iterations.png")


def gmm_vs_kmeans():
    # strongly anisotropic / elongated clusters (a sheared transform) — k-means' weak spot
    rng = np.random.default_rng(0)
    X0, y = make_blobs(n_samples=600, centers=3, cluster_std=0.7, random_state=0)
    transform = np.array([[0.60, -0.63], [-0.40, 0.85]])
    X = X0 @ transform
    km = KMeans(n_clusters=3, n_init=10, random_state=0).fit(X)
    gm = GaussianMixture(n_components=3, covariance_type="full", n_init=5, random_state=0).fit(X)
    ari_km = adjusted_rand_score(y, km.labels_)
    ari_gm = adjusted_rand_score(y, gm.predict(X))

    fig, (axk, axg) = plt.subplots(1, 2, figsize=(13, 5.4))
    cmap = ListedColormap(COMP)
    # k-means
    axk.scatter(X[:, 0], X[:, 1], c=km.labels_, cmap=cmap, s=14, edgecolor="white", lw=0.2, alpha=0.85)
    axk.scatter(km.cluster_centers_[:, 0], km.cluster_centers_[:, 1], marker="X", c="black", s=120,
                edgecolor="white", lw=1.4, zorder=5)
    axk.set_title(f"k-means (spherical, hard): ARI = {ari_km:.2f}", fontsize=11.5, fontweight="bold")
    # gmm
    axg.scatter(X[:, 0], X[:, 1], c=gm.predict(X), cmap=cmap, s=14, edgecolor="white", lw=0.2, alpha=0.85)
    for k in range(3):
        _ellipse(axg, gm.means_[k], gm.covariances_[k], "black", nsig=2.0, lw=2.0, alpha=0.85)
        axg.plot(*gm.means_[k], marker="X", color="black", ms=11, mec="white", mew=1.2)
    axg.set_title(f"GMM (full covariance, soft): ARI = {ari_gm:.2f}", fontsize=11.5, fontweight="bold")
    for ax in (axk, axg):
        ax.set_xlabel("feature 1"); ax.set_ylabel("feature 2"); _despine(ax)
    fig.suptitle("Elongated clusters: GMM's full covariance fits the tilt; k-means cuts straight through",
                 fontsize=13, fontweight="bold", y=1.0)
    fig.tight_layout(); fig.savefig(f"{OUT}/gmm_vs_kmeans.png", dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"wrote gmm_vs_kmeans.png  (ARI: kmeans={ari_km:.3f}, gmm={ari_gm:.3f})")


def gmm_loglik():
    # measure the observed-data log-likelihood per EM iteration with a from-scratch EM,
    # computing log p(X|theta) = sum_n log sum_k pi_k N(x_n; mu_k, Sigma_k) each iteration
    rng = np.random.default_rng(2)
    X, _ = make_blobs(n_samples=500, centers=4, cluster_std=1.0, random_state=4)
    n, d = X.shape
    K = 4
    rg = np.random.default_rng(7)
    mu = X[rg.choice(n, K, replace=False)].astype(float)
    cov = np.array([np.cov(X.T) for _ in range(K)])
    pi = np.full(K, 1.0 / K)
    lls = []
    for _ in range(30):
        # component densities -> per-point mixture density -> observed-data log-likelihood
        dens = np.stack([pi[k] * multivariate_normal(mu[k], cov[k], allow_singular=True).pdf(X)
                         for k in range(K)], axis=1)
        lls.append(np.log(dens.sum(1) + 1e-300).sum())
        r = dens / dens.sum(1, keepdims=True)        # E-step
        Nk = r.sum(0)                                 # M-step
        mu = (r.T @ X) / Nk[:, None]
        cov = np.array([(r[:, k:k+1] * (X - mu[k])).T @ (X - mu[k]) / Nk[k]
                        + 1e-6 * np.eye(d) for k in range(K)])
        pi = Nk / n
    lls = np.array(lls)
    fig, ax = plt.subplots(figsize=(8.4, 5.2))
    ax.plot(range(1, len(lls) + 1), lls, "o-", color=PURPLE, lw=2.4, ms=6)
    ax.fill_between(range(1, len(lls) + 1), lls.min() - 50, lls, color=PURPLE, alpha=0.06)
    # verify monotonic and annotate
    diffs = np.diff(lls)
    mono = bool((diffs >= -1e-6).all())
    ax.set_title(f"Observed-data log-likelihood increases every EM iteration  (monotonic = {mono})",
                 fontsize=11.5, fontweight="bold")
    ax.set_xlabel("EM iteration"); ax.set_ylabel("log-likelihood  log p(X | θ)")
    ax.annotate("each step never decreases it\n(the convergence guarantee)",
                xy=(len(lls), lls[-1]), xytext=(len(lls) * 0.42, lls.min() + (lls.max()-lls.min())*0.45),
                fontsize=10, color=GREEN,
                arrowprops=dict(arrowstyle="->", color=GREEN, lw=1.5))
    _despine(ax)
    fig.tight_layout(); fig.savefig(f"{OUT}/gmm_loglik.png", dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"wrote gmm_loglik.png  (monotonic increase = {mono}, "
          f"min step = {diffs.min():.4f})")


def gmm_bic():
    # data with a true 3 components; sweep k and show BIC dipping at the right model order
    rng = np.random.default_rng(0)
    X, _ = make_blobs(n_samples=600, centers=3, cluster_std=1.1, random_state=11)
    ks = range(1, 9)
    bics, aics = [], []
    for k in ks:
        gm = GaussianMixture(n_components=k, covariance_type="full", n_init=5, random_state=0).fit(X)
        bics.append(gm.bic(X)); aics.append(gm.aic(X))
    best_k = list(ks)[int(np.argmin(bics))]
    fig, ax = plt.subplots(figsize=(8.4, 5.2))
    ax.plot(list(ks), bics, "o-", color=BLUE, lw=2.4, ms=7, label="BIC")
    ax.plot(list(ks), aics, "s--", color=AMBER, lw=2.0, ms=6, label="AIC", alpha=0.9)
    ax.axvline(best_k, color=GREEN, lw=2.0, ls=":")
    ax.scatter([best_k], [min(bics)], s=180, facecolor="none", edgecolor=GREEN, lw=2.5, zorder=5)
    ax.annotate(f"BIC minimum at k = {best_k}\n(the true number of components)",
                xy=(best_k, min(bics)), xytext=(best_k + 0.6, min(bics) + (max(bics)-min(bics))*0.35),
                fontsize=10, color=GREEN, arrowprops=dict(arrowstyle="->", color=GREEN, lw=1.5))
    ax.set_title("Model selection: BIC dips at the correct number of components",
                 fontsize=11.5, fontweight="bold")
    ax.set_xlabel("number of components  k"); ax.set_ylabel("information criterion (lower = better)")
    ax.legend(frameon=True, fontsize=10); _despine(ax)
    fig.tight_layout(); fig.savefig(f"{OUT}/gmm_bic.png", dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"wrote gmm_bic.png  (BIC picks k={best_k})")


if __name__ == "__main__":
    gmm_em_iterations()
    gmm_vs_kmeans()
    gmm_loglik()
    gmm_bic()
    print("all GMM/EM diagrams written to", OUT)
