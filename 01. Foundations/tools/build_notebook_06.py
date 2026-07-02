"""Generate the step-by-step SVD teaching notebook (06-Singular-Value-Decomposition.ipynb).

The notebook mirrors ``singular_value_decomposition.py`` one operation at a time, so a reader can
open it, run every cell live, and *teach* SVD from it. Each numbered step has a short markdown
lead-in (the intuition) followed by ONE focused code cell with real output. This generator writes
the .ipynb; a separate nbconvert pass executes it headless so the outputs are embedded.

    python build_notebook_06.py         # writes the .ipynb (unexecuted) into the chapter's code/
    python -m nbconvert --to notebook --execute --inplace \
        "../06-Singular-Value-Decomposition/code/06-Singular-Value-Decomposition.ipynb"

This generator lives in the domain-level ``01. Foundations/tools/`` folder; the notebook it writes
(and the module it mirrors) stay in the chapter's ``code/`` folder. Kept as a generator (not a
hand-edited .ipynb) so the notebook and the module stay in lockstep: the same algorithm, typed once
in the module, demonstrated step-by-step here.
"""

from __future__ import annotations

import json
from pathlib import Path

# Written into the chapter's own code/ folder, one directory up from tools/ then into the chapter.
NB_PATH = (
    Path(__file__).resolve().parent.parent
    / "06-Singular-Value-Decomposition"
    / "code"
    / "06-Singular-Value-Decomposition.ipynb"
)


_CELL_ID = 0


def _next_id() -> str:
    """Stable, sequential cell id (silences nbformat's MissingIDFieldWarning)."""
    global _CELL_ID
    _CELL_ID += 1
    return f"cell-{_CELL_ID:02d}"


def md(source: str) -> dict:
    """A markdown cell."""
    return {
        "cell_type": "markdown",
        "id": _next_id(),
        "metadata": {},
        "source": source.splitlines(keepends=True),
    }


def code(source: str) -> dict:
    """A code cell (outputs filled in by the nbconvert execute pass)."""
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
    "# Singular Value Decomposition — a step-by-step, runnable teaching notebook\n"
    "\n"
    "This notebook builds SVD from the ground up on **real data**, one operation at a time. It is the "
    "executable companion to the chapter and to `singular_value_decomposition.py` — every function used "
    "here lives in that module, imported so the notebook and the module can never drift apart.\n"
    "\n"
    "By the end you will have **seen**, with real numbers and real images:\n"
    "\n"
    "1. the factorization $A = U\\Sigma V^\\top$ verified to machine precision;\n"
    "2. the geometry — a circle mapped to an ellipse whose semi-axes are the singular values;\n"
    "3. $\\sigma_i = \\sqrt{\\lambda_i(A^\\top A)}$, the eigen-connection, cross-checked;\n"
    "4. **image compression** — a real photo reconstructed at rank 1, 5, 20, 50, 100;\n"
    "5. **Eckart–Young–Mirsky** — truncated SVD is provably the best rank-$k$ approximation;\n"
    "6. **SVD = PCA** — on the real handwritten-digits dataset, matched against `sklearn.PCA`;\n"
    "7. the **pseudoinverse** solving a real least-squares problem;\n"
    "8. the **condition number**, numerical rank, and where SVD quietly saves the day.\n"
    "\n"
    "Everything runs on CPU in a few seconds, seeded for reproducibility."
)

# ---- Step 0: setup ----
add_md(
    "## Step 0 — Setup and version banner\n"
    "\n"
    "We import the real functions from the chapter module (so this notebook uses the *exact same code* "
    "the figures and the page use) and print the library versions the results were produced on."
)
add_code(
    "import numpy as np\n"
    "import scipy\n"
    "import sklearn\n"
    "import matplotlib\n"
    "import matplotlib.pyplot as plt\n"
    "\n"
    "from singular_value_decomposition import (\n"
    "    compute_svd, reconstruct, verify_identity, orthonormality_error,\n"
    "    singular_values_from_eigen, truncation_error_spectral,\n"
    "    eckart_young_check, load_grayscale_image, compression_curve,\n"
    "    cumulative_energy, pca_via_svd, pca_reconstruct, pseudoinverse,\n"
    "    build_overdetermined_system, RNG_SEED,\n"
    ")\n"
    "\n"
    "print(f'numpy {np.__version__} | scipy {scipy.__version__} | scikit-learn {sklearn.__version__} '\n"
    "      f'| matplotlib {matplotlib.__version__}')\n"
    "rng = np.random.default_rng(RNG_SEED)"
)

# ---- Step 1: the factorization ----
add_md(
    "## Step 1 — The factorization $A = U\\Sigma V^\\top$, made concrete\n"
    "\n"
    "SVD says **every** matrix — square or not, invertible or not — factors as\n"
    "\n"
    "$$A = U\\,\\Sigma\\,V^\\top,$$\n"
    "\n"
    "where $U$ and $V$ have **orthonormal columns** (they are rotations/reflections) and $\\Sigma$ is "
    "**diagonal** with non-negative, descending entries $\\sigma_1 \\ge \\sigma_2 \\ge \\dots \\ge 0$ "
    "(the *singular values*).\n"
    "\n"
    "Let's take a small, genuinely rectangular real matrix and read off the shapes. We use the **reduced** "
    "(thin) form: for $A \\in \\mathbb{R}^{m\\times n}$ with $r = \\min(m,n)$, we get $U\\in\\mathbb{R}^{m\\times r}$, "
    "$s\\in\\mathbb{R}^{r}$, $V^\\top\\in\\mathbb{R}^{r\\times n}$."
)
add_code(
    "A = rng.standard_normal((6, 4))   # a real 6x4 matrix (m=6 equations-worth, n=4 features-worth)\n"
    "svd = compute_svd(A)              # numpy.linalg.svd under the hood, reduced form\n"
    "\n"
    "print('A  shape :', A.shape)\n"
    "print('U  shape :', svd.U.shape,  '  (left singular vectors, orthonormal columns)')\n"
    "print('s  shape :', svd.s.shape,  '        (singular values, descending)')\n"
    "print('Vt shape :', svd.Vt.shape, '  (right singular vectors, as rows)')\n"
    "print('\\nsingular values:', np.round(svd.s, 4))\n"
    "print('descending?    :', bool(np.all(np.diff(svd.s) <= 0)))"
)

# ---- Step 2: verify the identity ----
add_md(
    "## Step 2 — Verify the decomposition is exact\n"
    "\n"
    "A factorization you can't reproduce is just a claim. Let's rebuild $A$ from its factors as "
    "$U\\,\\mathrm{diag}(s)\\,V^\\top$ and confirm we get the original back to floating-point precision "
    "(errors near $10^{-15}$ are pure round-off — the identity is exact)."
)
add_code(
    "A_rebuilt = reconstruct(svd)                    # (U[:, :r] * s) @ Vt  ==  U diag(s) Vt\n"
    "max_error = verify_identity(A, svd)\n"
    "print('max |A - U diag(s) Vt| =', f'{max_error:.2e}')\n"
    "print('exact to machine precision:', bool(np.allclose(A, A_rebuilt)))"
)

# ---- Step 3: orthonormality ----
add_md(
    "## Step 3 — $U$ and $V$ are rotations: orthonormality\n"
    "\n"
    "The reason SVD reads as *rotate → scale → rotate* is that $U$ and $V$ have **orthonormal columns**: "
    "$U^\\top U = I$ and $V^\\top V = I$. An orthonormal matrix preserves lengths and angles — it can only "
    "rotate or reflect, never stretch. All the stretching lives in $\\Sigma$."
)
add_code(
    "u_err, v_err = orthonormality_error(svd)\n"
    "print('||UᵀU - I|| =', f'{u_err:.2e}')\n"
    "print('||VᵀV - I|| =', f'{v_err:.2e}')\n"
    "print('\\nUᵀU (rounded):')\n"
    "print(np.round(svd.U.T @ svd.U, 6))   # should be the identity"
)

# ---- Step 4: the geometry ----
add_md(
    "## Step 4 — See it: the unit circle becomes an ellipse\n"
    "\n"
    "The cleanest way to *feel* SVD is in 2-D. Apply a real $2\\times 2$ matrix $A$ to every point on the "
    "unit circle and watch what happens in three stages:\n"
    "\n"
    "* $V^\\top$ **rotates** the circle (still a circle);\n"
    "* $\\Sigma$ **scales** the axes — the circle becomes an axis-aligned **ellipse** with semi-axis "
    "lengths exactly $\\sigma_1, \\sigma_2$;\n"
    "* $U$ **rotates** that ellipse into its final orientation.\n"
    "\n"
    "The ellipse's axis *lengths* are the singular values; its axis *directions* are the columns of $U$."
)
add_code(
    "A2 = np.array([[2.0, 1.2], [0.4, 1.6]])   # a real, non-symmetric 2x2 map\n"
    "svd2 = compute_svd(A2)\n"
    "theta = np.linspace(0, 2*np.pi, 400)\n"
    "circle = np.vstack([np.cos(theta), np.sin(theta)])\n"
    "\n"
    "after_Vt = svd2.Vt @ circle\n"
    "after_S  = svd2.s[:, None] * after_Vt\n"
    "after_U  = svd2.U @ after_S\n"
    "print('singular values of A2:', np.round(svd2.s, 4), ' <- these are the ellipse semi-axis lengths')\n"
    "print('U Σ Vᵀ (circle) == A2 (circle):', bool(np.allclose(after_U, A2 @ circle)))\n"
    "\n"
    "fig, axes = plt.subplots(1, 4, figsize=(13, 3.4))\n"
    "titles = ['unit circle', 'after Vᵀ (rotate)', 'after Σ (scale)', 'after U (rotate) = A x']\n"
    "for ax, pts, title in zip(axes, [circle, after_Vt, after_S, after_U], titles):\n"
    "    ax.plot(pts[0], pts[1], lw=2)\n"
    "    ax.set_aspect('equal')\n"
    "    ax.set_title(title)\n"
    "    ax.grid(alpha=0.3)\n"
    "    ax.set_xlim(-3, 3)\n"
    "    ax.set_ylim(-3, 3)\n"
    "plt.tight_layout()\n"
    "plt.show()"
)

# ---- Step 5: eigen connection ----
add_md(
    "## Step 5 — The eigen-connection: $\\sigma_i = \\sqrt{\\lambda_i(A^\\top A)}$\n"
    "\n"
    "SVD is not disconnected from eigenvalues — it *is* the eigendecomposition of the symmetric matrices "
    "$A^\\top A$ and $A A^\\top$. Since\n"
    "\n"
    "$$A^\\top A = V \\Sigma^\\top U^\\top U \\Sigma V^\\top = V\\,\\Sigma^2\\,V^\\top,$$\n"
    "\n"
    "the eigenvalues of $A^\\top A$ are $\\sigma_i^2$ and its eigenvectors are the right singular vectors "
    "$V$. So we can recover the singular values a **second, independent way** — as the square roots of the "
    "eigenvalues of $A^\\top A$ — and confirm they match."
)
add_code(
    "sigma_from_svd = svd.s\n"
    "sigma_from_eig = singular_values_from_eigen(A)[:sigma_from_svd.size]\n"
    "print('sigma  (SVD)          :', np.round(sigma_from_svd, 6))\n"
    "print('sqrt(eig(AᵀA))        :', np.round(sigma_from_eig, 6))\n"
    "print('they match            :', bool(np.allclose(sigma_from_svd, sigma_from_eig)))\n"
    "print('\\nWhy SVD always exists: AᵀA is symmetric PSD, so its eigenvalues are real and >= 0,')\n"
    "print('so their square roots (the singular values) are always real — even when A itself has')\n"
    "print('no eigendecomposition (rectangular, or defective).')"
)

# ---- Step 6: load the real image ----
add_md(
    "## Step 6 — A real matrix: load a photograph\n"
    "\n"
    "Now the payoff demo. An image is just a matrix of pixel intensities, so SVD applies directly. We load "
    "a **real photograph** bundled inside scikit-learn (no download), convert it to grayscale (Rec.709 "
    "luminance), and treat the result as a real $427\\times 640$ matrix $A$."
)
add_code(
    "img = load_grayscale_image()   # real china.jpg -> grayscale matrix, fully offline\n"
    "print('image matrix shape:', img.shape, '| dtype:', img.dtype)\n"
    "print('pixel range:', img.min(), '..', img.max())\n"
    "plt.figure(figsize=(6, 4))\n"
    "plt.imshow(img, cmap='gray')\n"
    "plt.title('the real image, as a matrix A')\n"
    "plt.axis('off')\n"
    "plt.show()"
)

# ---- Step 7: the spectrum ----
add_md(
    "## Step 7 — Its singular-value spectrum decays fast\n"
    "\n"
    "Compute the SVD of the image and look at the singular values. Real, structured data has a "
    "**fast-decaying spectrum**: a few large singular values carry the gross structure, and a long tail of "
    "tiny ones carries fine detail. That decay is *exactly* why low-rank compression works — most of the "
    "'energy' $\\sum_i \\sigma_i^2$ lives in the first handful of triplets."
)
add_code(
    "svd_img = compute_svd(img)\n"
    "s = svd_img.s\n"
    "print('number of singular values:', s.size)\n"
    "print('largest 5 :', np.round(s[:5], 1))\n"
    "print('smallest 5:', np.round(s[-5:], 3))\n"
    "print('ratio σ1/σ_min (condition number):', f'{svd_img.condition_number:.1f}')\n"
    "\n"
    "energy = cumulative_energy(svd_img)\n"
    "for tgt in (0.90, 0.95, 0.99):\n"
    "    k = int(np.searchsorted(energy, tgt) + 1)\n"
    "    print(f'{tgt:.0%} of energy in the first {k} singular values')\n"
    "\n"
    "fig, (a1, a2) = plt.subplots(1, 2, figsize=(12, 3.6))\n"
    "a1.semilogy(range(1, s.size + 1), s)\n"
    "a1.set_title('singular values (log)')\n"
    "a1.grid(alpha=0.3)\n"
    "a1.set_xlabel('index i')\n"
    "a1.set_ylabel('σ_i')\n"
    "a2.plot(range(1, energy.size + 1), energy * 100)\n"
    "a2.set_title('cumulative energy %')\n"
    "a2.grid(alpha=0.3)\n"
    "a2.set_xlabel('k')\n"
    "a2.set_ylabel('% energy')\n"
    "plt.tight_layout()\n"
    "plt.show()"
)

add_md(
    "> **A subtlety worth noticing.** The 90%-energy figure lands at a *tiny* $k$ because the single "
    "largest singular value captures the image's overall **brightness** (a near-constant background). "
    "Energy percentages are dominated by that one term. Visually, though, the *scene* only appears once "
    "the next several triplets are added — which is why the montage below still looks blank at rank 1 and "
    "recognizable only around rank 20. Energy and perceived quality are related but not identical."
)

# ---- Step 8: reconstruction at increasing rank ----
add_md(
    "## Step 8 — Low-rank reconstruction: watch the image appear\n"
    "\n"
    "A rank-$k$ approximation keeps only the top $k$ triplets: "
    "$A_k = \\sum_{i=1}^{k} \\sigma_i\\,u_i\\,v_i^\\top$. We reconstruct the photo at $k = 1, 5, 20, 50, 100$ "
    "and full rank, and print the **measured** relative error and compression ratio at each — the storage "
    "is $k(m+n+1)$ numbers versus $mn$ for the dense image."
)
add_code(
    "ranks = [1, 5, 20, 50, 100, s.size]\n"
    "fig, axes = plt.subplots(2, 3, figsize=(12, 8))\n"
    "for ax, k in zip(axes.ravel(), ranks):\n"
    "    rec = reconstruct(svd_img, k)                       # UNCLIPPED: error must match the table below\n"
    "    err = np.linalg.norm(img - rec) / np.linalg.norm(img)\n"
    "    ratio = (img.shape[0]*img.shape[1]) / (k*(img.shape[0]+img.shape[1]+1))\n"
    "    ax.imshow(np.clip(rec, 0, 255), cmap='gray', vmin=0, vmax=255)   # clip only the displayed pixels\n"
    "    label = 'full (exact)' if k == s.size else f'rank {k}'\n"
    "    ax.set_title(f'{label}  |  rel err {err:.3f}  |  {ratio:.1f}x smaller')\n"
    "    ax.axis('off')\n"
    "plt.tight_layout()\n"
    "plt.show()\n"
    "\n"
    "print(f\"{'rank k':>7} | {'rel err':>8} | {'stored #':>10} | {'compression':>11}\")\n"
    "for pt in compression_curve(img, svd_img):\n"
    "    print(f'{pt.k:>7} | {pt.rel_frobenius_error:>8.4f} | {pt.stored_values:>10,} | {pt.compression_ratio:>10.1f}x')"
)

# ---- Step 9: Eckart-Young ----
add_md(
    "## Step 9 — Eckart–Young–Mirsky: truncated SVD is *optimal*\n"
    "\n"
    "The headline theorem. Among **all** matrices of rank $\\le k$, the truncated SVD $A_k$ is the closest "
    "to $A$ — in both the Frobenius and spectral norms — and its error is exactly the tail of the spectrum:\n"
    "\n"
    "$$\\min_{\\mathrm{rank}(B)\\le k}\\|A - B\\|_F = \\|A - A_k\\|_F = \\sqrt{\\textstyle\\sum_{i>k}\\sigma_i^2},"
    "\\qquad \\|A - A_k\\|_2 = \\sigma_{k+1}.$$\n"
    "\n"
    "We verify two things on the real image: (a) the measured truncation error equals that closed form, and "
    "(b) a **random** rank-$k$ approximation is always *worse* — Eckart–Young made empirical."
)
add_code(
    "print(f\"{'k':>4} | {'measured_F':>11} | {'predicted_F':>11} | {'spectral σ_{k+1}':>15} | {'random rank-k':>13}\")\n"
    "for k in (5, 20, 50):\n"
    "    r = eckart_young_check(img, svd_img, k)\n"
    "    spec = truncation_error_spectral(svd_img, k)\n"
    "    print(f'{k:>4} | {r[\"measured\"]:>11.1f} | {r[\"predicted\"]:>11.1f} | {spec:>15.1f} | {r[\"random_rankk\"]:>13.1f}')\n"
    "    assert abs(r['measured'] - r['predicted']) < 1e-6 * r['predicted']\n"
    "    assert r['measured'] <= r['random_rankk'] + 1e-9   # truncated SVD is never beaten\n"
    "print('\\n✓ measured error == closed form, and always <= any random rank-k factor (optimality holds)')"
)

# ---- Step 10: SVD is PCA ----
add_md(
    "## Step 10 — SVD *is* PCA (on the real digits dataset)\n"
    "\n"
    "PCA and SVD are the same computation seen from two angles. Take a real data matrix $X$ "
    "($n$ samples $\\times$ $d$ features), **centre** it (subtract the column means), and take the SVD "
    "$X_c = U\\Sigma V^\\top$. Then:\n"
    "\n"
    "* the **right singular vectors** $V$ are the **principal component directions**;\n"
    "* the variance explained by component $i$ is $\\sigma_i^2 / (n-1)$.\n"
    "\n"
    "We run this on the real `load_digits` dataset (1797 handwritten $8\\times 8$ digits) and check our "
    "explained variances against scikit-learn's own `PCA` — which internally *is* a truncated SVD."
)
add_code(
    "from sklearn.datasets import load_digits\n"
    "from sklearn.decomposition import PCA\n"
    "\n"
    "X = load_digits().data.astype(np.float64)   # (1797, 64) real digit pixels\n"
    "pca = pca_via_svd(X)\n"
    "print('explained-variance ratio, top 5 PCs:', np.round(pca.explained_variance_ratio[:5], 4))\n"
    "for tgt in (0.80, 0.90, 0.95):\n"
    "    k = int(np.searchsorted(np.cumsum(pca.explained_variance_ratio), tgt) + 1)\n"
    "    print(f'{tgt:.0%} of variance kept by the first {k} of 64 components')\n"
    "\n"
    "sk = PCA(n_components=10).fit(X)\n"
    "print('\\nour explained variances match sklearn.PCA:',\n"
    "      bool(np.allclose(pca.explained_variance[:10], sk.explained_variance_, atol=1e-6)))"
)

# ---- Step 11: eigendigits + low-rank reconstruct ----
add_md(
    "## Step 11 — The principal directions are interpretable pixel patterns\n"
    "\n"
    "Because each principal direction is a vector in the 64-pixel feature space, we can reshape it back to "
    "$8\\times 8$ and *look at it*. These 'eigen-digits' are the dominant patterns the handwritten digits "
    "vary along. We then reconstruct the data from just the top 10 of 64 components and measure how much "
    "variance (and detail) survives — the dimensionality-reduction SVD is used for."
)
add_code(
    "fig, axes = plt.subplots(2, 5, figsize=(10, 4.4))\n"
    "for i, ax in enumerate(axes.ravel()):\n"
    "    comp = pca.components[i].reshape(8, 8)\n"
    "    ax.imshow(comp, cmap='RdBu_r')\n"
    "    ax.axis('off')\n"
    "    ax.set_title(f'PC {i + 1}: {pca.explained_variance_ratio[i]:.1%}')\n"
    "plt.suptitle('principal directions (right singular vectors) as 8x8 images')\n"
    "plt.tight_layout()\n"
    "plt.show()\n"
    "\n"
    "X_hat = pca_reconstruct(pca, X, k=10)\n"
    "rel = np.linalg.norm(X - X_hat) / np.linalg.norm(X)\n"
    "kept = float(np.cumsum(pca.explained_variance_ratio)[9])\n"
    "print(f'rank-10 reconstruction keeps {kept:.1%} of variance, relative error {rel:.4f}')"
)

# ---- Step 12: pseudoinverse least squares ----
add_md(
    "## Step 12 — The pseudoinverse solves least squares\n"
    "\n"
    "When $A$ is tall ($m > n$), $Ax = b$ usually has **no** exact solution — but SVD gives the best one. "
    "The Moore–Penrose pseudoinverse is\n"
    "\n"
    "$$A^+ = V\\,\\Sigma^+\\,U^\\top, \\qquad \\Sigma^+ = \\mathrm{diag}(1/\\sigma_i)\\ \\text{for } \\sigma_i>0,$$\n"
    "\n"
    "and $x = A^+ b$ minimises $\\|Ax - b\\|_2$. We build a real overdetermined system (200 noisy equations, "
    "5 unknowns), recover the coefficients, and confirm the SVD solution matches `numpy.linalg.lstsq`."
)
add_code(
    "A_ls, b_ls, x_true = build_overdetermined_system()   # 200 eqns, 5 unknowns, with noise\n"
    "svd_ls = compute_svd(A_ls)\n"
    "x_svd = pseudoinverse(svd_ls) @ b_ls\n"
    "x_np, *_ = np.linalg.lstsq(A_ls, b_ls, rcond=None)\n"
    "\n"
    "print('recovered coefficients (SVD)   :', np.round(x_svd, 3))\n"
    "print('true coefficients              :', np.round(x_true, 3))\n"
    "print('||x_svd - x_true||             :', f'{np.linalg.norm(x_svd - x_true):.4f}  (nonzero: noise)')\n"
    "print('matches numpy.lstsq            :', bool(np.allclose(x_svd, x_np, atol=1e-8)))\n"
    "print('residual ||A x - b||           :', f'{np.linalg.norm(A_ls @ x_svd - b_ls):.4f}')"
)

# ---- Step 13: condition number & numerical rank ----
add_md(
    "## Step 13 — Condition number and numerical rank\n"
    "\n"
    "Two diagnostics fall straight out of the singular values:\n"
    "\n"
    "* **Condition number** $\\kappa(A) = \\sigma_{\\max}/\\sigma_{\\min}$ measures how much the map stretches "
    "its most- vs least-amplified direction. A large $\\kappa$ means solving $Ax=b$ amplifies noise — the "
    "problem is *ill-conditioned*.\n"
    "* **Numerical rank** counts singular values above a tolerance. On real, noisy data the 'zero' singular "
    "values are tiny-but-nonzero, so the *effective* rank is what matters, not the algebraic one.\n"
    "\n"
    "We build a deliberately ill-conditioned matrix (a near-duplicate column) and watch $\\kappa$ explode "
    "while the numerical rank correctly reports the true dimensionality."
)
add_code(
    "well = rng.standard_normal((50, 5))\n"
    "ill = well.copy()\n"
    "ill[:, 4] = ill[:, 0] + 1e-6 * rng.standard_normal(50)   # column 4 ≈ column 0 -> near rank-deficient\n"
    "\n"
    "for name, M in (('well-conditioned', well), ('near-duplicate column', ill)):\n"
    "    d = compute_svd(M)\n"
    "    print(f'{name:>24}: cond = {d.condition_number:>12.1f} | numerical rank = {d.rank_numerical} of 5')\n"
    "\n"
    "print('\\nThe ill-conditioned matrix has a huge condition number: a tiny change in b can cause a')\n"
    "print('large change in the least-squares x. The pseudoinverse (Step 12) with its rcond cutoff is')\n"
    "print('what keeps this stable — it zeroes the reciprocal of the near-zero singular value instead')\n"
    "print('of dividing by it and exploding.')"
)

# ---- Recap ----
add_md(
    "## Recap\n"
    "\n"
    "In one runnable notebook, on real data, we established:\n"
    "\n"
    "| Step | What we saw | The takeaway |\n"
    "|---|---|---|\n"
    "| 1–3 | $A = U\\Sigma V^\\top$, exact; $U,V$ orthonormal | SVD is *rotate → scale → rotate* |\n"
    "| 4 | circle → ellipse | semi-axes = singular values, directions = columns of $U$ |\n"
    "| 5 | $\\sigma_i = \\sqrt{\\lambda_i(A^\\top A)}$ | why SVD exists for **any** matrix |\n"
    "| 6–8 | real photo, spectrum, rank-$k$ montage | low-rank compression, *measured* |\n"
    "| 9 | truncated vs random rank-$k$ | Eckart–Young: truncation is optimal |\n"
    "| 10–11 | digits, eigen-digits, `sklearn.PCA` match | **SVD is PCA** |\n"
    "| 12 | overdetermined solve | pseudoinverse = least squares |\n"
    "| 13 | $\\kappa$, numerical rank | SVD as a numerical-stability diagnostic |\n"
    "\n"
    "Every number here came from a real matrix and a real library call. That is the whole point: SVD is not "
    "an abstraction to memorize — it is a concrete, inspectable operation you can run, see, and trust."
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
