"""Cross-Entropy & KL Divergence on REAL data — the load-bearing module for the chapter.

This is not a toy. Every number the chapter, the figures, and the notebook show is produced
here, from real corpora and real datasets via real library calls (``numpy``, ``scipy``,
``scikit-learn``). The information-theory quantities are measured, not asserted:

  * **Entropy & surprise.** Shannon entropy ``H(p) = -sum p log2 p`` in *bits* on the real
    letter-frequency distribution of a real text corpus (sci.space newsgroup). English letters
    come out at ~4.19 bits/letter — the classic figure — and the binary-entropy curve peaks at
    exactly 1 bit at p = 0.5.

  * **Cross-entropy IS classification loss.** Train a real softmax (multinomial logistic)
    classifier on the real ``load_digits`` dataset by gradient descent, watch the cross-entropy
    fall from ln(10) to ~0.06, and show three facts that trip people up: NLL == cross-entropy ==
    -log p(true class); our loss equals ``sklearn.metrics.log_loss`` to 6 decimals; and the
    softmax+CE gradient collapses to the famous ``(p - y)``.

  * **KL divergence between REAL distributions.** ``D_KL(p||q) = sum p log2(p/q)`` between real
    empirical distributions, with the **asymmetry** ``D_KL(p||q) != D_KL(q||p)`` measured, ``KL
    >= 0`` verified (Gibbs), and the decomposition ``H(p,q) = H(p) + D_KL(p||q)`` confirmed
    numerically. Forward vs reverse KL is shown by fitting a *single* Gaussian to a *real
    bimodal* distribution: forward KL covers both modes (moment matching), reverse KL locks onto
    one (mode seeking).

  * **Perplexity.** ``perplexity = 2^cross-entropy`` of real n-gram language models (unigram vs
    an interpolated bigram) on real held-out newsgroup text — the bigram's context lowers it.

  * **Gaussian KL closed form** cross-checked against a numeric integral.

Everything is seeded and CPU-only; runs standalone in a few seconds (the first run downloads the
20-newsgroups corpus once into scikit-learn's cache, then is fully offline)::

    python cross_entropy_kl.py
"""

from __future__ import annotations

import re
from collections import Counter
from dataclasses import dataclass

import numpy as np
from numpy.typing import NDArray
from sklearn.datasets import fetch_20newsgroups, load_digits
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import log_loss
from sklearn.model_selection import train_test_split

# ---- Constants (hoisted; no magic numbers inline) ----------------------------------------------
RNG_SEED = 0  # one global seed so every "random" number in the module is reproducible
LOG2_EPS = 1e-12  # floor added before a log to avoid log(0) on empirically-zero bins
LAPLACE_ALPHA = 1.0  # add-one smoothing so KL is finite (q > 0 wherever p > 0)
N_CLASSES = 10  # digits 0..9
GD_STEPS = 300  # gradient-descent steps for the softmax classifier demo
GD_LR = 0.5  # learning rate for that classifier
TEST_FRACTION = 0.3  # held-out split for the digit classifier
LM_TOP_VOCAB = 5000  # keep the top-K words for the language model; rest -> <unk>
LM_HELDOUT = 0.10  # last 10% of tokens are the held-out test set for perplexity
BIGRAM_LAMBDA = 0.7  # interpolation weight on the bigram vs the unigram back-off
LM_SMOOTH_ALPHA = 0.1  # add-alpha smoothing inside the n-gram probabilities
# Two real newsgroup categories whose text distributions genuinely differ (space vs baseball):
CORPUS_A = "sci.space"
CORPUS_B = "rec.sport.baseball"
# Four categories give a larger corpus for a stable language-model perplexity estimate:
LM_CATEGORIES = ("sci.space", "rec.sport.baseball", "comp.graphics", "sci.med")
BIMODAL_DIGITS = (0, 6)  # two visually distinct digit classes -> a clearly bimodal PC1 score


# ============================ 1. entropy & surprise =============================================
def shannon_entropy_bits(p: NDArray[np.float64]) -> float:
    """Shannon entropy ``H(p) = -sum_i p_i log2 p_i`` in **bits** (expected surprise per symbol).

    Each symbol i carries surprise ``-log2 p_i`` bits: rare symbols surprise more. Entropy is the
    *average* surprise, and — by Shannon's source-coding theorem — the minimum average number of
    bits per symbol any lossless code can achieve. Empirically-zero probabilities contribute 0
    (the limit ``p log p -> 0`` as ``p -> 0``), enforced with a tiny floor before the log.
    """
    p = p[p > 0]  # 0 * log 0 := 0; drop the zeros so they contribute nothing
    return float(-np.sum(p * np.log2(p)))


def binary_entropy_bits(p: NDArray[np.float64]) -> NDArray[np.float64]:
    """The binary-entropy function ``H(p) = -p log2 p - (1-p) log2(1-p)`` over p in (0, 1).

    A coin's uncertainty as a function of its bias. Peaks at exactly 1 bit at p = 0.5 (a fair
    coin — maximally uncertain) and falls to 0 at p = 0 or 1 (a certain outcome — no surprise).
    This single curve is the shape of every cross-entropy loss for a binary classifier.
    """
    p = np.clip(p, LOG2_EPS, 1 - LOG2_EPS)  # keep the logs finite at the endpoints
    return -p * np.log2(p) - (1 - p) * np.log2(1 - p)


def letter_distribution(text: str) -> tuple[list[str], NDArray[np.float64]]:
    """Real letter-frequency distribution of a text: (sorted letters a..z, their probabilities).

    Lower-cases, keeps only ``a``-``z``, counts, and normalises to a probability vector. This is a
    genuine empirical distribution over a real corpus — the object we take the entropy of.
    """
    counts = Counter(ch for ch in text.lower() if ch.isalpha() and ch.isascii())
    letters = sorted(counts)
    freqs = np.array([counts[ch] for ch in letters], dtype=np.float64)
    return letters, freqs / freqs.sum()


# ============================ 2. cross-entropy & KL, defined =====================================
def cross_entropy_bits(p: NDArray[np.float64], q: NDArray[np.float64]) -> float:
    """Cross-entropy ``H(p,q) = -sum_i p_i log2 q_i`` in bits: the cost of coding ``p`` with ``q``.

    If the true symbol frequencies are ``p`` but you build your code (your model) assuming ``q``,
    each symbol costs ``-log2 q_i`` bits and you pay ``p_i`` of them on average. It is minimised,
    over all ``q``, exactly when ``q = p`` — at which point it equals the entropy ``H(p)``.
    """
    q = np.clip(q, LOG2_EPS, None)
    return float(-np.sum(p * np.log2(q)))


def kl_divergence_bits(p: NDArray[np.float64], q: NDArray[np.float64]) -> float:
    """KL divergence ``D_KL(p||q) = sum_i p_i log2(p_i / q_i)`` in bits: the *extra* coding cost.

    The number of **wasted** bits per symbol from using model ``q`` when the truth is ``p`` — i.e.
    ``H(p,q) - H(p)``. Always ``>= 0`` (Gibbs' inequality), zero iff ``p == q``, and **asymmetric**
    (``D_KL(p||q) != D_KL(q||p)`` in general) — which is why it is a *divergence*, not a distance.
    """
    mask = p > 0  # terms with p_i = 0 contribute 0 (0 * log 0 := 0)
    p_m, q_m = p[mask], np.clip(q[mask], LOG2_EPS, None)
    return float(np.sum(p_m * np.log2(p_m / q_m)))


def smoothed_distribution(
    counts: Counter[str], vocab: list[str], alpha: float = LAPLACE_ALPHA
) -> NDArray[np.float64]:
    """Turn raw counts into a probability vector over a shared ``vocab``, with add-alpha smoothing.

    KL and cross-entropy blow up if ``q`` puts zero probability on an event ``p`` allows, so we add
    ``alpha`` to every count before normalising — the minimal fix that keeps every probability
    strictly positive and the divergences finite on real, sparse data.
    """
    raw = np.array([counts[w] for w in vocab], dtype=np.float64) + alpha
    return raw / raw.sum()


@dataclass
class KLComparison:
    """A full two-distribution comparison on real data: entropies, cross-entropies, both KLs."""

    h_p: float  # H(p)   — entropy of the first distribution (bits)
    h_q: float  # H(q)   — entropy of the second distribution (bits)
    h_pq: float  # H(p,q) — cross-entropy coding p with q (bits)
    kl_pq: float  # D_KL(p||q) — forward KL (bits)
    kl_qp: float  # D_KL(q||p) — reverse KL (bits)

    @property
    def identity_holds(self) -> bool:
        """Verify H(p,q) == H(p) + D_KL(p||q) numerically — the decomposition that defines KL."""
        return bool(np.isclose(self.h_pq, self.h_p + self.kl_pq))

    @property
    def asymmetry_ratio(self) -> float:
        """D_KL(q||p) / D_KL(p||q): how far KL is from symmetric (1.0 would be symmetric)."""
        return self.kl_qp / self.kl_pq if self.kl_pq > 0 else float("inf")


def compare_distributions(p: NDArray[np.float64], q: NDArray[np.float64]) -> KLComparison:
    """Compute every entropy/cross-entropy/KL quantity for a real pair (p, q) and bundle them."""
    return KLComparison(
        h_p=shannon_entropy_bits(p),
        h_q=shannon_entropy_bits(q),
        h_pq=cross_entropy_bits(p, q),
        kl_pq=kl_divergence_bits(p, q),
        kl_qp=kl_divergence_bits(q, p),
    )


# ============================ 3. Demo — cross-entropy is classification loss =====================
def softmax(z: NDArray[np.float64]) -> NDArray[np.float64]:
    """Numerically-stable softmax over the last axis: subtract the row max before exponentiating.

    ``exp`` of a large logit overflows float64 at ~709; subtracting the row max makes the largest
    exponent 0 (``exp(0)=1``) without changing the ratios, so the probabilities are identical but
    finite. This is the log-sum-exp trick, and it is why real softmax code never calls bare ``exp``.
    """
    z = z - z.max(axis=1, keepdims=True)
    e = np.exp(z)
    return e / e.sum(axis=1, keepdims=True)


@dataclass
class SoftmaxTrainResult:
    """The trajectory and final state of a real softmax classifier trained by gradient descent."""

    loss_curve: NDArray[np.float64]  # cross-entropy (NLL) at each step — the decreasing loss
    weights: NDArray[np.float64]  # (n_features, n_classes) learned weight matrix
    bias: NDArray[np.float64]  # (n_classes,) learned bias
    probs_train: NDArray[np.float64]  # (n_train, n_classes) final predicted probabilities
    y_train: NDArray[np.int64]  # the true labels (to index p(true class))


def train_softmax_gd(
    x: NDArray[np.float64],
    y: NDArray[np.int64],
    *,
    steps: int = GD_STEPS,
    lr: float = GD_LR,
) -> SoftmaxTrainResult:
    """Train a softmax classifier by full-batch gradient descent, recording the cross-entropy loss.

    The loss is the mean negative log-likelihood ``-1/n sum_i log p(y_i | x_i)`` = cross-entropy
    between the one-hot labels and the predictions. The gradient of that loss w.r.t. the logits is
    exactly ``(P - Y)`` (predicted minus one-hot true) — the clean cancellation derived in the
    chapter — so the weight update is ``X^T (P - Y) / n``. No autodiff, no toy: the real algorithm.
    """
    n, d = x.shape
    y_onehot = np.eye(N_CLASSES)[y]  # (n, K) one-hot targets
    w = np.zeros((d, N_CLASSES))
    b = np.zeros(N_CLASSES)
    losses = np.empty(steps)
    for step in range(steps):
        logits = x @ w + b
        probs = softmax(logits)
        # cross-entropy = NLL = -mean log p(true class); the +eps guards log(0) on early steps
        losses[step] = -np.mean(np.log(probs[np.arange(n), y] + LOG2_EPS))
        grad_logits = probs - y_onehot  # THE (p - y) gradient — softmax+CE derivative
        w -= lr * (x.T @ grad_logits) / n
        b -= lr * grad_logits.mean(axis=0)
    return SoftmaxTrainResult(
        loss_curve=losses,
        weights=w,
        bias=b,
        probs_train=softmax(x @ w + b),
        y_train=y,
    )


def standardize(
    x_train: NDArray[np.float64], x_test: NDArray[np.float64]
) -> tuple[NDArray[np.float64], NDArray[np.float64]]:
    """Zero-mean, unit-variance each feature using TRAIN statistics only (no test leakage)."""
    mu = x_train.mean(axis=0)
    sd = x_train.std(axis=0) + LOG2_EPS
    return (x_train - mu) / sd, (x_test - mu) / sd


def load_digits_split() -> tuple[
    NDArray[np.float64], NDArray[np.float64], NDArray[np.int64], NDArray[np.int64]
]:
    """Load the real 8x8 handwritten-digits dataset and make a stratified train/test split."""
    data = load_digits()
    x = data.data.astype(np.float64)
    y = data.target.astype(np.int64)
    return train_test_split(x, y, test_size=TEST_FRACTION, random_state=RNG_SEED, stratify=y)


# ============================ 4. forward vs reverse KL (Gaussian fit) ============================
@dataclass
class GaussianFit:
    """A single-Gaussian fit to a real bimodal target, under forward or reverse KL."""

    mu: float
    sigma: float
    kl: float  # the divergence achieved at this fit


def _grid_pmf_from_samples(
    samples: NDArray[np.float64], grid: NDArray[np.float64]
) -> NDArray[np.float64]:
    """Discretise real samples into a normalised histogram (a pmf) on a shared grid."""
    hist, _ = np.histogram(samples, bins=grid, density=True)
    pmf = hist + LOG2_EPS
    return pmf / pmf.sum()


def _gaussian_pmf(mu: float, sigma: float, centers: NDArray[np.float64]) -> NDArray[np.float64]:
    """A Gaussian evaluated on the grid centers and normalised to a pmf (same support as p)."""
    q = np.exp(-0.5 * ((centers - mu) / sigma) ** 2) / (sigma * np.sqrt(2 * np.pi))
    q = q + LOG2_EPS
    return q / q.sum()


def _kl_nats(a: NDArray[np.float64], b: NDArray[np.float64]) -> float:
    """KL in nats (natural log) for the fitting objective — units don't affect the argmin."""
    return float(np.sum(a * np.log(a / b)))


def fit_gaussian_forward_kl(
    samples: NDArray[np.float64], grid: NDArray[np.float64], centers: NDArray[np.float64]
) -> GaussianFit:
    """Forward KL fit ``argmin_q D_KL(p||q)``: the *moment-matching* / mode-COVERING solution.

    For a Gaussian family, minimising the forward KL to the data has a closed form — set the mean
    and variance of q to the mean and variance of the data. Because forward KL penalises ``q`` for
    being small wherever ``p`` is large, the fit stretches to *cover every mode* (it cannot afford
    to ignore either bump), landing a wide Gaussian spanning both modes. This is the MLE solution.
    """
    p = _grid_pmf_from_samples(samples, grid)
    mu = float(samples.mean())
    sigma = float(samples.std())
    return GaussianFit(mu=mu, sigma=sigma, kl=_kl_nats(p, _gaussian_pmf(mu, sigma, centers)))


def fit_gaussian_reverse_kl(
    samples: NDArray[np.float64],
    grid: NDArray[np.float64],
    centers: NDArray[np.float64],
    mode_inits: tuple[float, ...],
) -> GaussianFit:
    """Reverse KL fit ``argmin_q D_KL(q||p)``: the mode-SEEKING solution (used by VI / ELBO).

    Reverse KL penalises ``q`` for putting mass where ``p`` is small, so the best single Gaussian
    *collapses onto one mode* and ignores the other — a narrow spike, not a wide cover. There are
    multiple local minima (one per mode); we try an init near each real mode and keep the best.
    Minimised with a small coordinate search over (mu, log sigma) — no external optimiser needed.
    """
    p = _grid_pmf_from_samples(samples, grid)

    def objective(mu: float, log_sigma: float) -> float:
        return _kl_nats(_gaussian_pmf(mu, float(np.exp(log_sigma)), centers), p)

    best: GaussianFit | None = None
    for init_mu in mode_inits:
        mu, log_sigma = init_mu, float(np.log(0.5))
        step_mu, step_ls = 0.5, 0.3
        # Nelder-Mead-free local search: shrink steps, greedily accept any improving move.
        for _ in range(60):
            current = objective(mu, log_sigma)
            improved = False
            for dmu, dls in ((step_mu, 0), (-step_mu, 0), (0, step_ls), (0, -step_ls)):
                trial = objective(mu + dmu, log_sigma + dls)
                if trial < current:
                    mu, log_sigma, current, improved = mu + dmu, log_sigma + dls, trial, True
                    break
            if not improved:
                step_mu, step_ls = step_mu * 0.6, step_ls * 0.6
        fit = GaussianFit(mu=mu, sigma=float(np.exp(log_sigma)), kl=objective(mu, log_sigma))
        if best is None or fit.kl < best.kl:
            best = fit
    if best is None:  # mode_inits is non-empty by construction; guard the invariant explicitly
        raise AssertionError("no Gaussian fit found — mode_inits was unexpectedly empty")
    return best


def real_bimodal_scores(digits: tuple[int, int] = BIMODAL_DIGITS) -> NDArray[np.float64]:
    """A real, clearly-bimodal 1-D distribution: PC1 scores of two distinct digit classes.

    Project the pixels of two visually different digit classes onto their joint first principal
    component and standardise. The two classes separate along PC1 into two bumps — a genuine
    bimodal empirical distribution to fit a single Gaussian to (no synthetic mixture needed).
    """
    from sklearn.decomposition import PCA

    data = load_digits()
    mask = np.isin(data.target, digits)
    scores = PCA(n_components=1, random_state=RNG_SEED).fit_transform(data.data[mask])[:, 0]
    return (scores - scores.mean()) / scores.std()


def mode_centers(digits: tuple[int, int] = BIMODAL_DIGITS) -> tuple[float, float]:
    """The two real mode locations (per-class PC1 means) — used to initialise the reverse-KL search."""
    from sklearn.decomposition import PCA

    data = load_digits()
    mask = np.isin(data.target, digits)
    sub = data.data[mask]
    labels = data.target[mask]
    scores = PCA(n_components=1, random_state=RNG_SEED).fit_transform(sub)[:, 0]
    scores = (scores - scores.mean()) / scores.std()
    return float(scores[labels == digits[0]].mean()), float(scores[labels == digits[1]].mean())


# ============================ 5. perplexity of a real n-gram LM ==================================
@dataclass
class LMResult:
    """A language-model evaluation on real held-out text: bits/word and perplexity."""

    name: str
    bits_per_word: float  # cross-entropy of the model on held-out text (bits)
    perplexity: float  # 2 ** bits_per_word — the effective branching factor


def _tokenize(text: str) -> list[str]:
    """Lower-case word tokens (runs of a-z) — the unit our language models predict."""
    return re.findall(r"[a-z]+", text.lower())


def build_lm_corpus(
    categories: tuple[str, ...] = LM_CATEGORIES, top_vocab: int = LM_TOP_VOCAB
) -> tuple[list[str], list[str], int]:
    """Real newsgroup text -> (train tokens, held-out test tokens, vocab size), rare words -> <unk>.

    Restricting to the top-``top_vocab`` words (mapping the rest to a single ``<unk>``) is the
    standard n-gram recipe: it bounds the vocabulary so smoothing is well-behaved and out-of-
    vocabulary words at test time are handled honestly rather than crashing the probability.
    """
    raw = fetch_20newsgroups(
        subset="train", categories=list(categories), remove=("headers", "footers", "quotes")
    )
    tokens = _tokenize(" ".join(raw.data))
    keep = {w for w, _ in Counter(tokens).most_common(top_vocab)}
    tokens = [w if w in keep else "<unk>" for w in tokens]
    cut = int((1 - LM_HELDOUT) * len(tokens))
    return tokens[:cut], tokens[cut:], len(keep) + 1  # +1 for <unk>


def unigram_perplexity(train: list[str], test: list[str], vocab_size: int) -> LMResult:
    """Perplexity of an add-alpha **unigram** model on held-out text: the context-free baseline.

    A unigram model ignores context entirely — ``P(w)`` from raw frequencies (add-alpha smoothed).
    Its cross-entropy on held-out text is the average ``-log2 P(w)``, and perplexity is 2 to that
    power: the effective number of equally-likely words the model is choosing among each step.
    """
    counts = Counter(train)
    total = sum(counts.values())
    denom = total + LM_SMOOTH_ALPHA * vocab_size
    bits = -np.mean([np.log2((counts[w] + LM_SMOOTH_ALPHA) / denom) for w in test])
    return LMResult("unigram", float(bits), float(2**bits))


def bigram_perplexity(
    train: list[str], test: list[str], vocab_size: int, lam: float = BIGRAM_LAMBDA
) -> LMResult:
    """Perplexity of an interpolated **bigram** model: context lowers the cross-entropy.

    ``P(w | prev) = lam * P_bigram(w | prev) + (1 - lam) * P_unigram(w)`` (Jelinek-Mercer
    interpolation), each term add-alpha smoothed. Conditioning on the previous word gives the model
    real information, so its bits/word — and therefore its perplexity — drop below the unigram's.
    """
    uni = Counter(train)
    bi = Counter(zip(train, train[1:]))
    total = sum(uni.values())
    uni_denom = total + LM_SMOOTH_ALPHA * vocab_size

    def logp2(prev: str, word: str) -> float:
        p_uni = (uni[word] + LM_SMOOTH_ALPHA) / uni_denom
        p_bi = (bi[(prev, word)] + LM_SMOOTH_ALPHA) / (uni[prev] + LM_SMOOTH_ALPHA * vocab_size)
        return float(np.log2(lam * p_bi + (1 - lam) * p_uni))

    bits = -np.mean([logp2(a, b) for a, b in zip(test, test[1:])])
    return LMResult("bigram (interpolated)", float(bits), float(2**bits))


# ============================ 6. Gaussian KL closed form vs numeric ==============================
def kl_two_gaussians(mu0: float, sigma0: float, mu1: float, sigma1: float) -> float:
    """Closed-form ``D_KL(N0 || N1)`` (nats) for two 1-D Gaussians — a formula worth knowing cold.

    ``D_KL = log(sigma1/sigma0) + (sigma0^2 + (mu0-mu1)^2)/(2 sigma1^2) - 1/2``. It appears verbatim
    as the VAE regulariser (KL of the encoder to a unit Gaussian) — one of the few KLs with a clean
    analytic form, which is exactly why the Gaussian family is so convenient in variational methods.
    """
    return float(
        np.log(sigma1 / sigma0)
        + (sigma0**2 + (mu0 - mu1) ** 2) / (2 * sigma1**2)
        - 0.5
    )


def kl_two_gaussians_numeric(
    mu0: float, sigma0: float, mu1: float, sigma1: float, *, span: float = 20.0, n: int = 20000
) -> float:
    """The same KL by brute-force numeric integration — the cross-check that the formula is right."""
    grid = np.linspace(mu0 - span, mu1 + span, n)
    dx = grid[1] - grid[0]

    def pdf(mu: float, sigma: float) -> NDArray[np.float64]:
        return np.exp(-0.5 * ((grid - mu) / sigma) ** 2) / (sigma * np.sqrt(2 * np.pi))

    p, q = pdf(mu0, sigma0), pdf(mu1, sigma1)
    return float(np.sum(p * np.log(p / q)) * dx)


# ============================ 7. run it all: the printed proof ===================================
def main() -> None:
    """Run every real check and demo, printing the measured results the chapter cites."""
    import scipy

    print(f"numpy {np.__version__} | scipy {scipy.__version__}")
    import sklearn

    print(f"scikit-learn {sklearn.__version__}\n")

    # ---- 1. entropy of a real corpus, in bits ----
    corpus_a = fetch_20newsgroups(
        subset="train", categories=[CORPUS_A], remove=("headers", "footers", "quotes")
    )
    corpus_b = fetch_20newsgroups(
        subset="train", categories=[CORPUS_B], remove=("headers", "footers", "quotes")
    )
    letters_a, p_letters = letter_distribution(" ".join(corpus_a.data))
    letters_b, q_letters = letter_distribution(" ".join(corpus_b.data))
    print("=== 1. Entropy & surprise (real text: letter frequencies) ===")
    print(f"  H(sci.space letters)        = {shannon_entropy_bits(p_letters):.4f} bits/letter")
    print(f"  H(baseball  letters)        = {shannon_entropy_bits(q_letters):.4f} bits/letter")
    print(f"  max possible (log2 26)      = {np.log2(26):.4f} bits (a uniform alphabet)")
    bh = binary_entropy_bits(np.array([0.5]))[0]
    print(f"  binary entropy at p=0.5     = {bh:.4f} bit (a fair coin — maximal)\n")

    # ---- 2. cross-entropy = H(p) + KL, KL >= 0, asymmetry, all on real distributions ----
    cmp = compare_distributions(p_letters, q_letters)
    print("=== 2. Cross-entropy, KL, and their identities (real letter distributions) ===")
    print(f"  H(p)                        = {cmp.h_p:.4f} bits")
    print(f"  H(p, q) cross-entropy       = {cmp.h_pq:.4f} bits")
    print(f"  D_KL(p||q) forward          = {cmp.kl_pq:.4f} bits")
    print(f"  D_KL(q||p) reverse          = {cmp.kl_qp:.4f} bits")
    print(f"  H(p) + D_KL(p||q)           = {cmp.h_p + cmp.kl_pq:.4f} bits")
    if not cmp.identity_holds:
        raise AssertionError("H(p,q) must equal H(p) + D_KL(p||q)")
    if cmp.kl_pq < 0 or cmp.kl_qp < 0:
        raise AssertionError("KL divergence must be >= 0 (Gibbs' inequality)")
    print(f"  -> H(p,q) == H(p)+KL(p||q): {cmp.identity_holds}; both KL >= 0; asymmetric "
          f"(ratio {cmp.asymmetry_ratio:.3f})")

    # a clean, large-asymmetry example: real letters vs a uniform model
    uniform = np.full_like(p_letters, 1.0 / p_letters.size)
    kl_pu = kl_divergence_bits(p_letters, uniform)
    kl_up = kl_divergence_bits(uniform, p_letters)
    print(f"  D_KL(real||uniform)={kl_pu:.4f}  D_KL(uniform||real)={kl_up:.4f}  "
          f"(asymmetry {kl_up / kl_pu:.2f}x)")
    if not np.isclose(kl_pu, np.log2(p_letters.size) - cmp.h_p):
        raise AssertionError("D_KL(p||uniform) must equal log2(K) - H(p)")
    print(f"  and D_KL(real||uniform) == log2(26) - H(p) = {np.log2(26) - cmp.h_p:.4f}  (exact)\n")
    _ = (letters_a, letters_b)  # names surfaced for the notebook/figures; kept explicit

    # ---- 3. cross-entropy IS the classification loss (real digits, real GD) ----
    x_tr, x_te, y_tr, y_te = load_digits_split()
    x_tr_s, x_te_s = standardize(x_tr, x_te)
    trained = train_softmax_gd(x_tr_s, y_tr)
    n_tr = x_tr_s.shape[0]
    our_ce = -np.mean(np.log(trained.probs_train[np.arange(n_tr), y_tr]))
    sk_ce = log_loss(y_tr, trained.probs_train)
    print("=== 3. Cross-entropy = classification loss (real digits, softmax + GD) ===")
    print(f"  CE at step 0                = {trained.loss_curve[0]:.4f}  (= ln 10 = "
          f"{np.log(N_CLASSES):.4f}, uniform guess)")
    print(f"  CE at step {GD_STEPS}             = {trained.loss_curve[-1]:.4f}  (training converged)")
    print(f"  our CE                      = {our_ce:.6f}")
    print(f"  sklearn.metrics.log_loss    = {sk_ce:.6f}")
    if not np.isclose(our_ce, sk_ce, atol=1e-6):
        raise AssertionError("our cross-entropy must equal sklearn.log_loss")
    i = 0
    nll_i = -np.log(trained.probs_train[i, y_tr[i]])
    print(f"  NLL of one example          = -log p(true={y_tr[i]}) = {nll_i:.4f}  "
          "(NLL == CE == -log p(true))")
    clf = LogisticRegression(max_iter=2000, C=10.0).fit(x_tr_s, y_tr)
    sk_test_ce = log_loss(y_te, clf.predict_proba(x_te_s))
    print(f"  sklearn LogisticRegression  : test log-loss {sk_test_ce:.4f}, "
          f"accuracy {clf.score(x_te_s, y_te):.4f}\n")

    # ---- 4. forward vs reverse KL: fitting one Gaussian to a real bimodal distribution ----
    scores = real_bimodal_scores()
    grid = np.linspace(scores.min() - 1.5, scores.max() + 1.5, 500)
    centers = 0.5 * (grid[:-1] + grid[1:])
    fwd = fit_gaussian_forward_kl(scores, grid, centers)
    rev = fit_gaussian_reverse_kl(scores, grid, centers, mode_inits=mode_centers())
    m0, m1 = mode_centers()
    print("=== 4. Forward vs reverse KL (fit ONE Gaussian to a real bimodal distribution) ===")
    print(f"  real modes (digit PC1)      : {m0:.3f} and {m1:.3f}  (two bumps)")
    print(f"  forward  KL fit  N(mu,sig)  : mu={fwd.mu:+.3f} sigma={fwd.sigma:.3f}  "
          "-> mode-COVERING (spans both)")
    print(f"  reverse  KL fit  N(mu,sig)  : mu={rev.mu:+.3f} sigma={rev.sigma:.3f}  "
          "-> mode-SEEKING (locks one)")
    if rev.sigma >= fwd.sigma:
        raise AssertionError("reverse-KL (mode-seeking) fit must be narrower than forward-KL")
    print("  -> forward KL = MLE / mode-covering; reverse KL = VI / mode-seeking\n")

    # ---- 5. perplexity of real n-gram language models ----
    train, test, vocab_size = build_lm_corpus()
    uni = unigram_perplexity(train, test, vocab_size)
    bi = bigram_perplexity(train, test, vocab_size)
    print(f"=== 5. Perplexity = 2^cross-entropy (real n-gram LM, {len(train):,} train tokens) ===")
    print(f"  {uni.name:<22}: {uni.bits_per_word:.4f} bits/word -> perplexity {uni.perplexity:.1f}")
    print(f"  {bi.name:<22}: {bi.bits_per_word:.4f} bits/word -> perplexity {bi.perplexity:.1f}")
    if bi.perplexity >= uni.perplexity:
        raise AssertionError("the bigram's context must lower perplexity below the unigram's")
    print("  -> context lowers cross-entropy, hence perplexity (the model is 'less perplexed')\n")

    # ---- 6. Gaussian KL closed form == numeric ----
    m0g, s0g, m1g, s1g = 0.0, 1.0, 1.5, 2.0
    closed = kl_two_gaussians(m0g, s0g, m1g, s1g)
    numeric = kl_two_gaussians_numeric(m0g, s0g, m1g, s1g)
    print("=== 6. Gaussian KL: closed form vs numeric integration ===")
    print(f"  D_KL(N(0,1) || N(1.5,2)) closed  = {closed:.5f} nats")
    print(f"  D_KL(...)                numeric = {numeric:.5f} nats")
    if not np.isclose(closed, numeric, atol=1e-3):
        raise AssertionError("closed-form Gaussian KL must match the numeric integral")
    print(f"  -> match: {np.isclose(closed, numeric, atol=1e-3)}")


if __name__ == "__main__":
    main()
