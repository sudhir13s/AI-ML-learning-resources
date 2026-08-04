"""Singular Value Decomposition on REAL data — the load-bearing module for the SVD chapter.

This is not a toy. Every number the chapter, the figures, and the notebook show is produced
here, from real matrices via real library calls (`numpy.linalg.svd`, `scipy.linalg.svd`,
`sklearn`). Two real demos anchor the whole topic:

  * **Demo 1 — image compression.** Load a real photograph (sklearn's bundled ``china.jpg``,
    427x640), convert to grayscale, and treat it as a real matrix ``A``. Compute its SVD once
    and reconstruct at ranks k = 1, 5, 20, 50, 100, full. Report the *measured* relative
    Frobenius error and the *measured* compression ratio at each k — the low-rank idea, seen.

  * **Demo 2 — SVD is PCA.** Load the real ``load_digits`` dataset (1797 handwritten 8x8
    digits), centre it, take the SVD of the centred data matrix, and show that the right
    singular vectors are the principal axes and that explained variance = sigma_i^2 / (n - 1).
    Reconstruct at low rank and read off the retained variance.

Around those, the module verifies the mathematics on real matrices: the reconstruction
identity ``A = U @ diag(s) @ Vt``; orthonormality ``UᵀU = VᵀV = I``; the eigen-connection
``sigma_i = sqrt(lambda_i(AᵀA))``; the **Eckart–Young–Mirsky** optimality of the truncation
(truncated SVD beats a random rank-k factor, every time); the Moore–Penrose pseudoinverse for
least squares; and the condition number ``sigma_max / sigma_min``.

Everything is seeded and CPU-only; runs standalone in a couple of seconds:

    python singular_value_decomposition.py
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from numpy.typing import NDArray
from scipy.linalg import svd as scipy_svd
from sklearn.datasets import load_digits, load_sample_images

# ---- Constants (hoisted; no magic numbers inline) ----------------------------------------------
RNG_SEED = 0  # one global seed so every "random" number in the module is reproducible
LUMA_RGB = (0.2126, 0.7152, 0.0722)  # Rec.709 luminance weights: grayscale = R*0.2126 + ...
COMPRESSION_RANKS = (1, 5, 20, 50, 100)  # ranks to reconstruct the image at (full added at runtime)
FLOAT_BYTES = 8  # a float64 component is 8 bytes (raw storage baseline for the ratio)
ORTHO_ATOL = 1e-10  # tolerance for "is this matrix orthonormal / is A = U S Vt" checks
DIGITS_RANK = 10  # low-rank reconstruction rank for the digits PCA demo (of 64 features)
LSTSQ_M, LSTSQ_N = 200, 5  # a real overdetermined system: 200 equations, 5 unknowns
LSTSQ_NOISE_STD = 0.5  # measurement noise added to the synthetic-but-realistic regression target


# ============================ 1. the decomposition itself =======================================
@dataclass
class SVDResult:
    """The three real factors of ``A = U @ diag(s) @ Vt`` (reduced / thin form).

    Shapes for ``A`` of shape (m, n) with r = min(m, n):
      U  : (m, r) — left singular vectors (orthonormal columns), an orthonormal basis of col-space
      s  : (r,)   — singular values, sorted DESCENDING, all >= 0
      Vt : (r, n) — right singular vectors as ROWS (V transposed); orthonormal rows
    """

    U: NDArray[np.float64]
    s: NDArray[np.float64]
    Vt: NDArray[np.float64]

    @property
    def rank_numerical(self) -> int:
        """Numerical rank: count of singular values above a relative tolerance (real-world rank).

        In exact arithmetic rank = count of nonzero sigmas; on real data noise makes the tail
        tiny-but-nonzero, so we threshold at ``max(m, n) * eps * sigma_max`` — numpy's own rule.
        """
        if self.s.size == 0:
            return 0
        tol = max(self.Vt.shape[1], self.U.shape[0]) * np.finfo(self.s.dtype).eps * self.s[0]
        return int((self.s > tol).sum())

    @property
    def condition_number(self) -> float:
        """kappa(A) = sigma_max / sigma_min — how much the map stretches worst vs best direction."""
        nonzero = self.s[self.s > 0]
        return float(nonzero[0] / nonzero[-1]) if nonzero.size else float("inf")


def compute_svd(a: NDArray[np.float64]) -> SVDResult:
    """Reduced SVD of a real matrix via LAPACK (``numpy.linalg.svd``, ``full_matrices=False``).

    Reduced (a.k.a. "thin") means U is (m, r) and Vt is (r, n) with r = min(m, n) — we drop the
    part of the full square U/V that would multiply the zero block of Sigma and contribute nothing.
    That is the form every application below actually uses.
    """
    u, s, vt = np.linalg.svd(a, full_matrices=False)
    return SVDResult(U=u, s=s, Vt=vt)


def reconstruct(svd: SVDResult, k: int | None = None) -> NDArray[np.float64]:
    """Rebuild the matrix from its top-``k`` singular triplets: sum_{i<k} sigma_i u_i v_iᵀ.

    ``k=None`` uses all triplets and returns A exactly (to float error). Truncating at k < r gives
    the best rank-k approximation of A (Eckart–Young–Mirsky, verified in ``eckart_young_check``).
    The einsum-free form ``(U[:, :k] * s[:k]) @ Vt[:k]`` scales each left vector by its sigma —
    cheaper and clearer than forming a diagonal matrix.
    """
    k = svd.s.size if k is None else k
    return (svd.U[:, :k] * svd.s[:k]) @ svd.Vt[:k]


def verify_identity(a: NDArray[np.float64], svd: SVDResult) -> float:
    """Return the max abs entrywise error of ``A == U diag(s) Vt`` — should be ~machine epsilon."""
    return float(np.abs(a - reconstruct(svd)).max())


def orthonormality_error(svd: SVDResult) -> tuple[float, float]:
    """Return (||UᵀU - I||, ||VᵀV - I||) — both ~0 because U, V have orthonormal columns.

    This is the property that makes SVD a *pair of rotations* around a pure scaling: an
    orthonormal matrix preserves lengths and angles, i.e. it is a rotation/reflection.
    """
    r = svd.s.size
    u_err = float(np.abs(svd.U.T @ svd.U - np.eye(r)).max())
    v_err = float(np.abs(svd.Vt @ svd.Vt.T - np.eye(r)).max())
    return u_err, v_err


def singular_values_from_eigen(a: NDArray[np.float64]) -> NDArray[np.float64]:
    """Recover sigma_i INDEPENDENTLY, as sqrt of the eigenvalues of AᵀA — the eigen-connection.

    AᵀA = V diag(sigma^2) Vᵀ is symmetric PSD, so its eigenvalues are sigma_i^2 (>= 0). Taking
    square roots and sorting descending must reproduce the singular values ``compute_svd`` returns —
    a real, independent cross-check that the two decompositions are the same object seen two ways.
    """
    ata = a.T @ a
    eigvals = np.linalg.eigvalsh(ata)  # ascending, real (symmetric matrix)
    eigvals = np.clip(eigvals, 0.0, None)  # kill tiny negative round-off before the sqrt
    return np.sqrt(eigvals)[::-1]  # descending, to match sigma ordering


# ============================ 2. Eckart–Young–Mirsky optimality ==================================
def truncation_error_frobenius(svd: SVDResult, k: int) -> float:
    """Predicted Frobenius error of the rank-k truncation: sqrt(sum_{i>k} sigma_i^2).

    Eckart–Young–Mirsky: NO rank-k matrix approximates A better (in Frobenius norm) than the
    truncated SVD, and its error is exactly the tail energy of the discarded singular values.
    """
    tail = svd.s[k:]
    return float(np.sqrt((tail**2).sum()))


def truncation_error_spectral(svd: SVDResult, k: int) -> float:
    """Predicted spectral-norm error of the rank-k truncation: exactly sigma_{k+1}.

    In the spectral (operator-2) norm the best rank-k error is the FIRST discarded singular value.
    """
    return float(svd.s[k]) if k < svd.s.size else 0.0


def random_rankk_error(a: NDArray[np.float64], k: int, seed: int = RNG_SEED) -> float:
    """Frobenius error of a *random* rank-k approximation — the foil for Eckart–Young.

    We project A onto a random k-dim column subspace: draw a random (n, k) matrix G, form the
    orthonormal basis Q of AG (a legitimate rank-k range), and take A_k = Q Qᵀ A. This is a
    perfectly valid rank-k matrix; Eckart–Young guarantees it can only do WORSE than truncated
    SVD, and this function measures by how much — so the notebook can *show* SVD is optimal.
    """
    rng = np.random.default_rng(seed)
    g = rng.standard_normal((a.shape[1], k))
    q, _ = np.linalg.qr(a @ g)  # orthonormal basis of a random rank-k column subspace
    a_k = q @ (q.T @ a)  # project A onto that subspace
    return float(np.linalg.norm(a - a_k, ord="fro"))


def eckart_young_check(a: NDArray[np.float64], svd: SVDResult, k: int) -> dict[str, float]:
    """Verify Eckart–Young on real data: truncated-SVD error == predicted, and <= random rank-k.

    Returns the measured truncation error, the closed-form prediction, and a random rank-k error;
    the invariant ``truncated <= random`` is the empirical proof of optimality.
    """
    measured = float(np.linalg.norm(a - reconstruct(svd, k), ord="fro"))
    predicted = truncation_error_frobenius(svd, k)
    random_err = random_rankk_error(a, k)
    return {"measured": measured, "predicted": predicted, "random_rankk": random_err}


# ============================ 3. Demo 1 — real image compression =================================
@dataclass
class CompressionPoint:
    """One rank-k reconstruction of the real image, with its measured error and storage."""

    k: int
    rel_frobenius_error: float  # ||A - A_k||_F / ||A||_F — measured, not assumed
    stored_values: int  # numbers you must keep for rank k: k*(m + n + 1)
    compression_ratio: float  # (m*n) / stored_values — how much smaller the rank-k form is


def load_grayscale_image() -> NDArray[np.float64]:
    """Load a REAL photograph (sklearn's bundled china.jpg) as a grayscale matrix, no download.

    ``load_sample_images`` ships inside scikit-learn, so this is fully reproducible and offline.
    We take the first image (a photo of the Forbidden City), collapse RGB to Rec.709 luminance,
    and return a real (427, 640) float matrix — a genuine, structured, real-world A.
    """
    images = load_sample_images()
    rgb = images.images[0].astype(np.float64)  # (H, W, 3) real photo
    return rgb @ np.array(LUMA_RGB)  # (H, W) luminance — a real matrix to decompose


def compression_curve(
    a: NDArray[np.float64], svd: SVDResult, ranks: tuple[int, ...] = COMPRESSION_RANKS
) -> list[CompressionPoint]:
    """Measure error + storage at each rank — the heart of the "SVD compresses" demo.

    Storage accounting: a rank-k reconstruction is kept as U[:, :k] (m*k), s[:k] (k), and Vt[:k]
    (k*n) — total ``k*(m + n + 1)`` numbers, versus ``m*n`` for the dense matrix. The ratio is the
    real compression factor; the error is the real quality cost.
    """
    m, n = a.shape
    a_norm = float(np.linalg.norm(a, ord="fro"))
    full_rank = svd.s.size
    points: list[CompressionPoint] = []
    for k in (*ranks, full_rank):
        err = float(np.linalg.norm(a - reconstruct(svd, k), ord="fro")) / a_norm
        stored = k * (m + n + 1)
        points.append(
            CompressionPoint(
                k=k,
                rel_frobenius_error=err,
                stored_values=stored,
                compression_ratio=(m * n) / stored,
            )
        )
    return points


def cumulative_energy(svd: SVDResult) -> NDArray[np.float64]:
    """Fraction of total 'energy' (sum of sigma^2) captured by the first k triplets, for all k.

    energy(k) = sum_{i<k} sigma_i^2 / sum_i sigma_i^2. This curve is why a handful of singular
    values usually suffice: real matrices have a fast-decaying spectrum, so energy saturates early.
    """
    power = svd.s**2
    return np.cumsum(power) / power.sum()


# ============================ 4. Demo 2 — SVD is PCA (real digits) ================================
@dataclass
class PCAResult:
    """PCA-via-SVD on a real dataset: principal axes, explained variance, and a low-rank rebuild."""

    components: NDArray[np.float64]  # (r, n_features) principal directions = right singular vectors
    explained_variance: NDArray[np.float64]  # sigma_i^2 / (n - 1) per component (real variances)
    explained_variance_ratio: NDArray[np.float64]  # each component's share of total variance
    singular_values: NDArray[np.float64]  # sigma_i of the CENTRED data matrix
    mean: NDArray[np.float64]  # feature means removed before the SVD (needed to reconstruct)


def pca_via_svd(x: NDArray[np.float64]) -> PCAResult:
    """PCA computed as the SVD of the centred data matrix — showing they are the same thing.

    Centre X (subtract the column means). Then X_c = U diag(sigma) Vᵀ, and:
      * the right singular vectors (rows of Vᵀ) ARE the principal component directions;
      * explained variance along component i is sigma_i^2 / (n - 1) — the sample variance of the
        i-th score, because the centred scores are U diag(sigma) and their columns are orthogonal.
    This is exactly what ``sklearn.decomposition.PCA`` does internally (a truncated SVD), verified
    against it in ``main``.
    """
    mean = x.mean(axis=0)
    x_c = x - mean
    n_samples = x.shape[0]
    _u, s, vt = np.linalg.svd(x_c, full_matrices=False)
    explained = s**2 / (n_samples - 1)  # variance captured by each principal component
    return PCAResult(
        components=vt,
        explained_variance=explained,
        explained_variance_ratio=explained / explained.sum(),
        singular_values=s,
        mean=mean,
    )


def pca_reconstruct(pca: PCAResult, x: NDArray[np.float64], k: int) -> NDArray[np.float64]:
    """Project real data onto the top-k principal axes and lift it back — lossy low-rank rebuild.

    scores = (X - mean) @ Vᵀ_topk ;  X_hat = scores @ V_topk + mean. Keeping k of n features, this
    is the dimensionality reduction PCA is used for, expressed purely through the SVD factors.
    """
    x_c = x - pca.mean
    comps = pca.components[:k]  # (k, n_features)
    scores = x_c @ comps.T  # (n_samples, k) — coordinates in the reduced basis
    return scores @ comps + pca.mean  # back to feature space


# ============================ 5. pseudoinverse & least squares ===================================
def pseudoinverse(svd: SVDResult, rcond: float = 1e-12) -> NDArray[np.float64]:
    """Moore–Penrose pseudoinverse via SVD: A⁺ = V diag(1/sigma) Uᵀ (reciprocate nonzero sigmas).

    For sigma_i above ``rcond * sigma_max`` we invert (1/sigma_i); tiny/zero sigmas are set to 0
    instead of exploding to infinity — the numerically stable definition that makes A⁺ solve the
    least-squares problem even when A is rank-deficient or ill-conditioned.
    """
    s = svd.s
    cutoff = rcond * s[0] if s.size else 0.0
    s_inv = np.where(s > cutoff, 1.0 / np.where(s > cutoff, s, 1.0), 0.0)
    return (svd.Vt.T * s_inv) @ svd.U.T


def build_overdetermined_system(
    m: int = LSTSQ_M, n: int = LSTSQ_N, noise_std: float = LSTSQ_NOISE_STD, seed: int = RNG_SEED
) -> tuple[NDArray[np.float64], NDArray[np.float64], NDArray[np.float64]]:
    """A real overdetermined least-squares problem: m equations, n unknowns, m >> n, with noise.

    Draw a random design matrix A (m, n) and a true coefficient vector x_true, then observe
    ``b = A x_true + noise``. Because m > n there is no exact solution — least squares finds the x
    minimising ``||A x - b||``, which the SVD pseudoinverse computes in closed form. Returns
    (A, b, x_true) so the caller can compare the recovered x to the truth.
    """
    rng = np.random.default_rng(seed)
    a = rng.standard_normal((m, n))
    x_true = rng.standard_normal(n)
    b = a @ x_true + noise_std * rng.standard_normal(m)
    return a, b, x_true


# ============================ 6. run it all: the printed proof ===================================
def main() -> None:
    """Run every real check and demo, printing the measured results the chapter cites."""
    print(f"numpy {np.__version__} | scipy {__import__('scipy').__version__}")
    import sklearn

    print(f"scikit-learn {sklearn.__version__}\n")
    rng = np.random.default_rng(RNG_SEED)

    # ---- core identities on a small real matrix ----
    a_small = rng.standard_normal((6, 4))
    svd_small = compute_svd(a_small)
    u_err, v_err = orthonormality_error(svd_small)
    print("=== Core SVD identities (6x4 real matrix) ===")
    print(f"  A = U diag(s) Vt   max entry error : {verify_identity(a_small, svd_small):.2e}")
    print(f"  ||UᵀU - I|| = {u_err:.2e}   ||VᵀV - I|| = {v_err:.2e}")
    sigma_svd = svd_small.s
    sigma_eig = singular_values_from_eigen(a_small)[: sigma_svd.size]
    print(f"  sigma from SVD      : {np.round(sigma_svd, 4)}")
    print(f"  sqrt(eig(AᵀA)) match: {np.allclose(sigma_svd, sigma_eig, atol=ORTHO_ATOL)}")
    print(f"  numerical rank = {svd_small.rank_numerical} | cond(A) = {svd_small.condition_number:.2f}")

    # cross-check numpy vs scipy give the same spectrum (independent LAPACK paths)
    s_scipy = scipy_svd(a_small, compute_uv=False)
    print(f"  numpy vs scipy singular values agree: {np.allclose(sigma_svd, s_scipy)}\n")

    # ---- Demo 1: real image compression ----
    img = load_grayscale_image()
    svd_img = compute_svd(img)
    print(f"=== Demo 1: image compression (real photo, {img.shape[0]}x{img.shape[1]}) ===")
    print(f"  {'rank k':>7} | {'rel Frob err':>12} | {'stored #':>10} | {'compression':>11}")
    print("  " + "-" * 50)
    for pt in compression_curve(img, svd_img):
        tag = "  (full)" if pt.k == svd_img.s.size else ""
        print(
            f"  {pt.k:>7} | {pt.rel_frobenius_error:>12.4f} | {pt.stored_values:>10,} | "
            f"{pt.compression_ratio:>10.1f}x{tag}"
        )
    energy = cumulative_energy(svd_img)
    for target in (0.90, 0.95, 0.99):
        k_needed = int(np.searchsorted(energy, target) + 1)
        print(f"  {target:.0%} of the image's energy is in the first {k_needed} singular values")
    print()

    # ---- Eckart–Young–Mirsky optimality (on the real image) ----
    print("=== Eckart–Young–Mirsky: truncated SVD is optimal (real image) ===")
    print(f"  {'k':>4} | {'truncated_F':>12} | {'predicted_F':>12} | {'random rank-k':>13}")
    print("  " + "-" * 50)
    for k in (5, 20, 50):
        r = eckart_young_check(img, svd_img, k)
        # explicit raises (not `assert`) so the checks survive `python -O`, which strips asserts
        if abs(r["measured"] - r["predicted"]) >= 1e-6 * r["predicted"]:
            raise AssertionError(f"Eckart–Young closed form must match measured error at k={k}")
        if r["measured"] > r["random_rankk"] + 1e-9:
            raise AssertionError(f"truncated SVD must be <= any rank-k approximation at k={k}")
        print(f"  {k:>4} | {r['measured']:>12.1f} | {r['predicted']:>12.1f} | {r['random_rankk']:>13.1f}")
    print("  -> truncated error == closed form, and always <= a random rank-k factor (optimal)\n")

    # ---- Demo 2: SVD is PCA on real digits ----
    digits = load_digits()
    x = digits.data.astype(np.float64)  # (1797, 64) real 8x8 handwritten-digit pixels
    pca = pca_via_svd(x)
    print(f"=== Demo 2: SVD == PCA (real digits, {x.shape[0]} samples x {x.shape[1]} features) ===")
    ratios = pca.explained_variance_ratio
    print(f"  explained-variance ratio, top 5 PCs: {np.round(ratios[:5], 4)}")
    for target in (0.80, 0.90, 0.95):
        k_needed = int(np.searchsorted(np.cumsum(ratios), target) + 1)
        print(f"  {target:.0%} of variance kept by the first {k_needed} of 64 components")
    # cross-check against sklearn's own PCA (which is a truncated SVD under the hood)
    from sklearn.decomposition import PCA

    sk = PCA(n_components=DIGITS_RANK).fit(x)
    match = np.allclose(pca.explained_variance[:DIGITS_RANK], sk.explained_variance_, atol=1e-6)
    print(f"  our explained-variance matches sklearn.PCA: {match}")
    x_hat = pca_reconstruct(pca, x, DIGITS_RANK)
    rel_err = np.linalg.norm(x - x_hat) / np.linalg.norm(x)
    kept = float(np.cumsum(ratios)[DIGITS_RANK - 1])
    print(f"  rank-{DIGITS_RANK} rebuild keeps {kept:.1%} of variance, rel error {rel_err:.4f}\n")

    # ---- pseudoinverse least squares on a real overdetermined system ----
    a_ls, b_ls, x_true = build_overdetermined_system()
    svd_ls = compute_svd(a_ls)
    x_hat = pseudoinverse(svd_ls) @ b_ls  # least-squares solution via SVD
    x_numpy, *_ = np.linalg.lstsq(a_ls, b_ls, rcond=None)  # reference solver
    residual = float(np.linalg.norm(a_ls @ x_hat - b_ls))
    print(f"=== Least squares via pseudoinverse ({LSTSQ_M} eqns, {LSTSQ_N} unknowns) ===")
    print(f"  ||x_svd - x_true||   = {np.linalg.norm(x_hat - x_true):.4f}  (noise makes this nonzero)")
    print(f"  matches numpy.lstsq  : {np.allclose(x_hat, x_numpy, atol=1e-8)}")
    print(f"  residual ||Ax - b||  = {residual:.4f} | cond(A) = {svd_ls.condition_number:.2f}")


if __name__ == "__main__":
    main()
