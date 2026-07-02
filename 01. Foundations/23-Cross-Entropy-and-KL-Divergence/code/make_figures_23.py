"""Figure generator for 23-Cross-Entropy-and-KL-Divergence — every number from the REAL pipeline.

All measured figures come from the same real corpora and datasets the chapter and notebook use
(``cross_entropy_kl.py``): the real 20-newsgroups text, the real ``load_digits`` classifier trained
by real gradient descent, real n-gram language models. Nothing is hand-typed. The one analytic
curve (the binary-entropy function) is computed from its formula, not drawn by hand.

Writes muted-palette PNGs to the shared chapter image dir (../../images/) with prefix ``found23_``:

  found23_entropy_surprise.png  -- REAL letter-frequency distribution + per-symbol surprise
                                   (-log2 p) of a real corpus, and the binary-entropy curve.
  found23_cross_entropy_loss.png-- REAL cross-entropy (log-loss) of a softmax classifier FALLING
                                   over gradient-descent steps on real digits, vs the ln(K) floor.
  found23_kl_identity.png       -- REAL H(p), cross-entropy H(p,q), and the KL gap stacked, plus
                                   the asymmetry D_KL(p||q) != D_KL(q||p) on real distributions.
  found23_forward_reverse_kl.png-- fit ONE Gaussian to a REAL bimodal distribution by forward KL
                                   (mode-covering) vs reverse KL (mode-seeking) — both fits shown.
  found23_perplexity.png        -- REAL perplexity of a unigram vs an interpolated bigram LM on
                                   real held-out text: context lowers it.
  found23_gaussian_kl.png       -- two real Gaussians and their KL: closed form vs numeric, and
                                   how KL grows as the model Gaussian drifts from the true one.

    python make_figures_23.py

Verified on Python 3.12 / matplotlib 3.10 / numpy 2.4 / scikit-learn 1.9.
"""

from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")  # headless: render straight to PNG, never open a window
import matplotlib.pyplot as plt
import numpy as np
from numpy.typing import NDArray
from sklearn.datasets import fetch_20newsgroups

from cross_entropy_kl import (
    BIGRAM_LAMBDA,
    binary_entropy_bits,
    bigram_perplexity,
    build_lm_corpus,
    compare_distributions,
    fit_gaussian_forward_kl,
    fit_gaussian_reverse_kl,
    kl_divergence_bits,
    kl_two_gaussians,
    kl_two_gaussians_numeric,
    letter_distribution,
    load_digits_split,
    mode_centers,
    real_bimodal_scores,
    shannon_entropy_bits,
    standardize,
    train_softmax_gd,
    unigram_perplexity,
)

# ---- Palette (matches the chapter's muted Mermaid classDefs) ------------------------------------
BLUE = "#3A6B96"  # data / input / true distribution p
PURPLE = "#5D4A8A"  # process / model distribution q
GREEN = "#2E7A5A"  # good / entropy floor / lower cost
RED = "#8B3B4A"  # cost / cross-entropy / the extra KL bits
SLATE = "#4A5B6E"  # neutral
AMBER = "#7A6528"  # highlight / surprise
NAVY = "#2A5B80"  # secondary data
INK = "#1C2530"  # labels
GRID = "#D4D9DF"  # gridlines

OUT_DIR = Path(__file__).resolve().parent.parent.parent / "images"
DPI = 120
IMG_PREFIX = "found23_"

_CATEGORY_A = "sci.space"
_CATEGORY_B = "rec.sport.baseball"
_REMOVE = ("headers", "footers", "quotes")


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


def _real_letter_pair() -> tuple[
    list[str], NDArray[np.float64], list[str], NDArray[np.float64]
]:
    """Real letter distributions of two real corpora (sci.space and baseball)."""
    a = fetch_20newsgroups(subset="train", categories=[_CATEGORY_A], remove=_REMOVE)
    b = fetch_20newsgroups(subset="train", categories=[_CATEGORY_B], remove=_REMOVE)
    letters_a, p = letter_distribution(" ".join(a.data))
    letters_b, q = letter_distribution(" ".join(b.data))
    return letters_a, p, letters_b, q


# ============================ Fig 1: entropy & surprise =========================================
def fig_entropy_surprise(letters: list[str], p: NDArray[np.float64]) -> None:
    """REAL letter frequencies + per-symbol surprise, and the binary-entropy curve.

    Left: the real letter-frequency distribution of a real corpus, coloured by how *surprising*
    each letter is (-log2 p). Rare letters (q, z) are tall in surprise, common ones (e, t) short.
    Entropy is the frequency-weighted average of that surprise. Right: the binary-entropy function,
    the shape of every 2-class cross-entropy, peaking at 1 bit for a fair coin.
    """
    surprise = -np.log2(p)
    h = shannon_entropy_bits(p)
    order = np.argsort(p)[::-1]  # most frequent first
    letters_sorted = [letters[i] for i in order]
    p_sorted = p[order]
    surprise_sorted = surprise[order]

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(13, 4.4))

    # left: frequency bars, annotated with surprise
    bars = ax1.bar(range(len(p_sorted)), p_sorted * 100, color=BLUE)
    # colour the three rarest (most surprising) letters amber to make surprise visible
    for idx in np.argsort(surprise_sorted)[-3:]:
        bars[idx].set_color(AMBER)
    ax1.set_xticks(range(len(letters_sorted)))
    ax1.set_xticklabels(letters_sorted, fontsize=8)
    ax1.set_xlabel("letter (sorted by frequency)")
    ax1.set_ylabel("frequency  (%)")
    ax1.set_title(
        f"Real letter distribution of a real corpus\nH(p) = {h:.3f} bits/letter "
        "(amber = rarest = most surprising)",
        fontsize=11,
    )
    _style_axis(ax1)

    # right: binary entropy curve
    grid = np.linspace(1e-4, 1 - 1e-4, 500)
    hb = binary_entropy_bits(grid)
    ax2.plot(grid, hb, color=GREEN, linewidth=2.2)
    ax2.axvline(0.5, color=RED, linestyle="--", linewidth=1)
    ax2.axhline(1.0, color=RED, linestyle=":", linewidth=1)
    ax2.annotate(
        "max = 1 bit at p=0.5\n(a fair coin: maximal uncertainty)",
        xy=(0.5, 1.0),
        xytext=(0.14, 0.55),
        fontsize=9,
        color=INK,
        arrowprops={"arrowstyle": "->", "color": INK},
    )
    ax2.set_xlabel("p  (probability of outcome 1)")
    ax2.set_ylabel("binary entropy  H(p)  (bits)")
    ax2.set_title("Binary-entropy function\n(the shape of every 2-class cross-entropy)", fontsize=11)
    ax2.set_ylim(0, 1.08)
    _style_axis(ax2)

    fig.tight_layout()
    _save(fig, f"{IMG_PREFIX}entropy_surprise.png")


# ============================ Fig 2: cross-entropy loss curve ====================================
def fig_cross_entropy_loss() -> None:
    """REAL cross-entropy of a softmax classifier FALLING over GD steps on real digits.

    The single most important picture in supervised learning: training a real classifier *is*
    minimising cross-entropy. We plot the measured loss at every gradient-descent step, mark the
    ln(K) 'uniform-guess' starting floor, and annotate the converged value — which equals
    sklearn's log_loss.
    """
    x_tr, x_te, y_tr, y_te = load_digits_split()
    x_tr_s, _ = standardize(x_tr, x_te)
    trained = train_softmax_gd(x_tr_s, y_tr)
    curve = trained.loss_curve
    ln_k = np.log(10)

    fig, ax = plt.subplots(figsize=(9, 5))
    ax.plot(range(1, curve.size + 1), curve, color=RED, linewidth=2.2, label="cross-entropy (NLL)")
    ax.axhline(ln_k, color=SLATE, linestyle="--", linewidth=1.2, label=f"uniform guess: ln(10) = {ln_k:.3f}")
    ax.axhline(0, color=GREEN, linestyle=":", linewidth=1, label="perfect fit: 0")
    ax.annotate(
        f"converged: {curve[-1]:.3f}",
        xy=(curve.size, curve[-1]),
        xytext=(curve.size * 0.55, curve[-1] + 0.35),
        fontsize=10,
        color=INK,
        arrowprops={"arrowstyle": "->", "color": INK},
    )
    ax.set_xlabel("gradient-descent step")
    ax.set_ylabel("cross-entropy loss  (nats)")
    ax.set_title(
        "Training a real classifier = minimising cross-entropy\n"
        "(softmax on real digits; the loss is the average -log p(true class))",
        fontsize=11,
    )
    ax.legend(frameon=False, fontsize=9)
    _style_axis(ax)
    fig.tight_layout()
    _save(fig, f"{IMG_PREFIX}cross_entropy_loss.png")


# ============================ Fig 3: KL identity + asymmetry =====================================
def fig_kl_identity(p: NDArray[np.float64], q: NDArray[np.float64]) -> None:
    """REAL H(p) + KL gap = cross-entropy, and the asymmetry of KL, on real distributions.

    Left: a stacked bar making H(p,q) = H(p) + D_KL(p||q) literal — the cross-entropy is the
    entropy floor plus the KL 'waste'. We use the real-letters-vs-uniform-model pair (a genuinely
    bad model) so the KL band is large and visible, rather than the near-identical two corpora.
    Right: forward vs reverse KL against that uniform model, showing they differ (a divergence,
    not a distance).
    """
    _ = compare_distributions(p, q)  # the two-corpus comparison is reported in the notebook/module
    uniform = np.full_like(p, 1.0 / p.size)
    h_p = shannon_entropy_bits(p)
    kl_pu = kl_divergence_bits(p, uniform)  # forward KL: extra bits from coding real p with uniform
    kl_up = kl_divergence_bits(uniform, p)  # reverse KL
    h_cross = h_p + kl_pu  # H(p, uniform) = H(p) + D_KL(p||uniform)

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4.4))

    # left: the decomposition H(p, uniform) = H(p) + KL(p||uniform), with a visible KL band
    labels = ["H(p)\n(entropy floor)", "H(p, uniform)\n(cross-entropy)"]
    ax1.bar(labels, [h_p, h_p], color=GREEN, label="H(p): irreducible")
    ax1.bar(labels, [0, kl_pu], bottom=[h_p, h_p], color=RED, label="D_KL(p‖uniform): the extra bits")
    ax1.text(1, h_p + kl_pu / 2, f"+{kl_pu:.3f}", ha="center", va="center", fontsize=10, color="white")
    ax1.text(0, h_p / 2, f"{h_p:.3f}", ha="center", va="center", fontsize=10, color="white")
    ax1.set_ylabel("bits / symbol")
    ax1.set_title(
        "H(p, q) = H(p) + D_KL(p‖q)\n(cross-entropy = floor + waste; q = a uniform model)",
        fontsize=11,
    )
    ax1.set_ylim(0, h_cross * 1.15)
    ax1.legend(frameon=False, fontsize=8, loc="upper left")
    _style_axis(ax1)

    # right: asymmetry
    labels = ["D_KL(real ‖ uniform)", "D_KL(uniform ‖ real)"]
    ax2.bar(labels, [kl_pu, kl_up], color=[BLUE, PURPLE])
    ax2.text(0, kl_pu + 0.02, f"{kl_pu:.3f}", ha="center", fontsize=10, color=INK)
    ax2.text(1, kl_up + 0.02, f"{kl_up:.3f}", ha="center", fontsize=10, color=INK)
    ax2.set_ylabel("KL divergence  (bits)")
    ax2.set_title(
        f"KL is asymmetric: {kl_up / kl_pu:.2f}× different\n(a divergence, not a distance)",
        fontsize=11,
    )
    ax2.set_ylim(0, max(kl_pu, kl_up) * 1.25)
    _style_axis(ax2)

    fig.tight_layout()
    _save(fig, f"{IMG_PREFIX}kl_identity.png")


# ============================ Fig 4: forward vs reverse KL =======================================
def fig_forward_reverse_kl() -> None:
    """Fit ONE Gaussian to a REAL bimodal distribution: forward (cover) vs reverse (seek) KL."""
    scores = real_bimodal_scores()
    grid = np.linspace(scores.min() - 1.5, scores.max() + 1.5, 500)
    centers = 0.5 * (grid[:-1] + grid[1:])
    fwd = fit_gaussian_forward_kl(scores, grid, centers)
    rev = fit_gaussian_reverse_kl(scores, grid, centers, mode_inits=mode_centers())

    xs = np.linspace(scores.min() - 1.5, scores.max() + 1.5, 400)

    def gauss(mu: float, sigma: float) -> NDArray[np.float64]:
        return np.exp(-0.5 * ((xs - mu) / sigma) ** 2) / (sigma * np.sqrt(2 * np.pi))

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(13, 4.6), sharey=True)
    for ax, fit, title, color in (
        (ax1, fwd, "Forward KL: argmin D_KL(p‖q)\nmode-COVERING (MLE)", GREEN),
        (ax2, rev, "Reverse KL: argmin D_KL(q‖p)\nmode-SEEKING (variational)", RED),
    ):
        ax.hist(scores, bins=40, density=True, color=BLUE, alpha=0.45, label="real bimodal data  p")
        ax.plot(xs, gauss(fit.mu, fit.sigma), color=color, linewidth=2.4,
                label=f"fitted Gaussian  q\nμ={fit.mu:+.2f}, σ={fit.sigma:.2f}")
        ax.set_xlabel("PC1 score (standardised)")
        ax.set_title(title, fontsize=11)
        ax.legend(frameon=False, fontsize=9)
        _style_axis(ax)
    ax1.set_ylabel("density")
    fig.suptitle(
        "One Gaussian, two objectives, on the SAME real bimodal data: forward KL spreads to cover "
        "both modes; reverse KL collapses onto one.",
        fontsize=12,
        color=INK,
    )
    fig.tight_layout(rect=(0, 0, 1, 0.94))
    _save(fig, f"{IMG_PREFIX}forward_reverse_kl.png")


# ============================ Fig 5: perplexity =================================================
def fig_perplexity() -> None:
    """REAL perplexity of a unigram vs interpolated bigram LM on real held-out text."""
    train, test, vocab_size = build_lm_corpus()
    uni = unigram_perplexity(train, test, vocab_size)
    bi = bigram_perplexity(train, test, vocab_size, lam=BIGRAM_LAMBDA)

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4.4))
    names = [uni.name, bi.name]

    ax1.bar(names, [uni.bits_per_word, bi.bits_per_word], color=[SLATE, GREEN])
    for i, r in enumerate((uni, bi)):
        ax1.text(i, r.bits_per_word + 0.05, f"{r.bits_per_word:.2f}", ha="center", fontsize=10, color=INK)
    ax1.set_ylabel("cross-entropy  (bits / word)")
    ax1.set_title("Held-out cross-entropy\n(lower = better model)", fontsize=11)
    _style_axis(ax1)

    ax2.bar(names, [uni.perplexity, bi.perplexity], color=[SLATE, GREEN])
    for i, r in enumerate((uni, bi)):
        ax2.text(i, r.perplexity + 8, f"{r.perplexity:.0f}", ha="center", fontsize=10, color=INK)
    ax2.set_ylabel("perplexity  = 2^(bits/word)")
    ax2.set_title(
        f"Perplexity on real text\nbigram is {uni.perplexity / bi.perplexity:.2f}× less perplexed",
        fontsize=11,
    )
    _style_axis(ax2)

    fig.suptitle(
        "Perplexity = 2^cross-entropy: the model's effective branching factor. Context (bigram) "
        "lowers cross-entropy, hence perplexity.",
        fontsize=11.5,
        color=INK,
    )
    fig.tight_layout(rect=(0, 0, 1, 0.93))
    _save(fig, f"{IMG_PREFIX}perplexity.png")


# ============================ Fig 6: Gaussian KL ================================================
def fig_gaussian_kl() -> None:
    """Two real Gaussians + KL closed-form vs numeric, and KL growing as q drifts from p."""
    mu0, sigma0 = 0.0, 1.0
    xs = np.linspace(-6, 8, 600)

    def gauss(mu: float, sigma: float) -> NDArray[np.float64]:
        return np.exp(-0.5 * ((xs - mu) / sigma) ** 2) / (sigma * np.sqrt(2 * np.pi))

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4.4))

    # left: two example Gaussians, KL closed == numeric
    mu1, sigma1 = 1.5, 2.0
    closed = kl_two_gaussians(mu0, sigma0, mu1, sigma1)
    numeric = kl_two_gaussians_numeric(mu0, sigma0, mu1, sigma1)
    ax1.plot(xs, gauss(mu0, sigma0), color=BLUE, linewidth=2.2, label="p = N(0, 1)")
    ax1.plot(xs, gauss(mu1, sigma1), color=PURPLE, linewidth=2.2, label="q = N(1.5, 2)")
    ax1.fill_between(xs, gauss(mu0, sigma0), color=BLUE, alpha=0.12)
    ax1.set_xlabel("x")
    ax1.set_ylabel("density")
    ax1.set_title(
        f"Two Gaussians and their KL\nclosed form {closed:.4f} = numeric {numeric:.4f} nats",
        fontsize=11,
    )
    ax1.legend(frameon=False, fontsize=9)
    _style_axis(ax1)

    # right: KL as q's mean drifts from p's mean (a smooth bowl, min at 0)
    shifts = np.linspace(-4, 4, 200)
    kls = [kl_two_gaussians(mu0, sigma0, mu0 + s, sigma0) for s in shifts]
    ax2.plot(shifts, kls, color=RED, linewidth=2.2)
    ax2.axvline(0, color=GREEN, linestyle="--", linewidth=1)
    ax2.annotate(
        "KL = 0 only when q = p",
        xy=(0, 0),
        xytext=(0.6, max(kls) * 0.5),
        fontsize=9,
        color=INK,
        arrowprops={"arrowstyle": "->", "color": INK},
    )
    ax2.set_xlabel("mean shift of q from p  (σ same)")
    ax2.set_ylabel("D_KL(p ‖ q)  (nats)")
    ax2.set_title("KL grows as the model drifts\n(≥ 0 always; the '½ (Δμ/σ)²' bowl)", fontsize=11)
    _style_axis(ax2)

    fig.tight_layout()
    _save(fig, f"{IMG_PREFIX}gaussian_kl.png")


def main() -> None:
    print(f"numpy {np.__version__} | matplotlib {matplotlib.__version__}")
    letters_a, p, _letters_b, q = _real_letter_pair()
    fig_entropy_surprise(letters_a, p)
    fig_cross_entropy_loss()
    fig_kl_identity(p, q)
    fig_forward_reverse_kl()
    fig_perplexity()
    fig_gaussian_kl()
    print(f"\nall figures written to {OUT_DIR} with prefix '{IMG_PREFIX}'")


if __name__ == "__main__":
    main()
