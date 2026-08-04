"""A Variational Autoencoder on REAL image data, from scratch, with the two claims that matter PROVEN.

This is not a toy. It trains a real VAE **from scratch** — the encoder ``q_phi(z|x) = N(mu, sigma^2 I)``, the
**reparameterization trick** ``z = mu + sigma * eps``, and a Bernoulli decoder ``p_theta(x|z)`` — on **real
handwritten digits** (torchvision **MNIST**; a real ``sklearn`` ``load_digits`` fallback if MNIST can't be
fetched), by maximizing the **Evidence Lower BOund (ELBO)**. It then makes the two claims the chapter rests on
*checkable*, with hard ``assert`` statements rather than hand-waving:

  1. **The closed-form Gaussian KL is correct.** The KL regulariser in the ELBO,
     ``KL(N(mu, sigma^2 I) || N(0, I)) = 1/2 * sum(mu^2 + sigma^2 - 1 - log sigma^2)``, is a *closed form*. We
     verify it equals a **Monte-Carlo estimate** ``E_q[log q(z|x) - log p(z)]`` to Monte-Carlo tolerance. The
     term the loss actually uses is the term the math claims.

  2. **The reparameterization trick is what lets gradients flow.** Writing ``z = mu + sigma * eps`` (with
     ``eps ~ N(0, I)`` a parameter-free noise source) keeps ``z`` differentiable in ``mu`` and ``sigma``, so
     ``d(loss)/d(mu)`` and ``d(loss)/d(logvar)`` exist. Sampling ``z`` *directly* from the distribution (a
     ``.sample()`` call) severs that path — the sampled ``z`` carries **no gradient** to the encoder. We show,
     with an ``assert``, that the reparameterized path has real gradients through ``mu``/``logvar`` while the
     direct-sample path does not. This is the "real thing" proof: the trick is not cosmetic; without it the
     encoder cannot be trained by backprop.

  3. **The VAE actually generates.** After training we (a) **reconstruct** real held-out digits, (b) **sample**
     ``z ~ N(0, I)`` and decode — novel digits the model dreamed up (the payoff a plain autoencoder cannot give),
     and (c) with a 2-D latent space, sweep a grid of ``z`` to reveal the **smooth, organized latent manifold**.
     We also measure the ``beta``-VAE trade-off: a larger KL weight buys a smaller (tighter-to-prior) KL at the
     cost of worse reconstruction.

Everything is **seeded** (``torch.manual_seed`` + NumPy) and pinned to **CPU** so the numbers are reproducible on
any machine. Run::

    python vae.py

If **MNIST cannot be downloaded** (offline), the module *detects* that and falls back to the real
``sklearn.datasets.load_digits`` 8x8 digit set — still real image data, never mocked — and the banner says which
path it took. The Gaussian-KL proof and the reparameterization-gradient proof are dataset-independent and always
run.

Verified on Python 3.12 / numpy 2.4 / torch 2.12 / torchvision 0.27 (CPU).
"""

from __future__ import annotations

import math
import os
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from torch import nn

SEED = 0

# Training hyperparameters for the headline MNIST run. Chosen (see the chapter) so a small MLP VAE trains from
# scratch to recognizable reconstructions and a smooth 2-D latent manifold on CPU in a couple of minutes. Every
# value is a plain dial you can turn in the notebook.
LATENT_DIM = 2  # 2-D so we can *draw* the latent manifold; the payoff figure of the chapter
HIDDEN_DIM = 400
N_EPOCHS = 15
BATCH_SIZE = 128
LR = 1e-3
N_TRAIN = 12_000  # a real MNIST subset — enough to learn, small enough to train fast on CPU
N_TEST = 2_000

# Where torchvision may cache MNIST. Kept OUT of the repo (a /tmp path) so weights/data are never committed.
_DATA_ROOT = Path(os.environ.get("VAE_DATA_ROOT", "/private/tmp/vae-data"))


def get_device() -> torch.device:
    """Detect the best available device for *reporting*; training itself pins CPU for a reproducible trace.

    A small MLP VAE on 28x28 digits trains fast enough on CPU, and CPU float math keeps the learning curve
    bit-reproducible across machines — which is what a teaching artifact needs. We still surface what hardware is
    present so the banner is honest.
    """
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


# ================================================================================================
# The data: real MNIST digits (torchvision), with a real sklearn load_digits fallback (never a mock)
# ================================================================================================


@dataclass
class DataSpec:
    """A uniform handle over the image data used for training: flattened pixels in [0, 1] plus the facts we need."""

    x_train: torch.Tensor  # [N_train, D] pixel intensities in [0, 1], flattened
    x_test: torch.Tensor  # [N_test,  D]
    y_test: np.ndarray  # test labels (used only to pick example digits for figures)
    img_shape: tuple[int, int]  # (H, W) so figures can un-flatten
    n_pixels: int  # D = H * W
    label: str
    source: str  # "mnist" or "sklearn-digits-fallback"


def load_data(*, n_train: int = N_TRAIN, n_test: int = N_TEST, seed: int = SEED) -> DataSpec:
    """Load real digit images as flattened pixel tensors in [0, 1]; prefer MNIST, else sklearn load_digits.

    MNIST pixels are grayscale in [0, 1] and we treat each pixel as the mean of a Bernoulli — the standard VAE
    likelihood for binarized images (the decoder outputs a per-pixel probability). The sklearn fallback (8x8,
    real handwritten digits) is scaled from its native 0..16 range into [0, 1] the same way. Both are real; we
    never fabricate an image.
    """
    try:
        from torchvision import datasets, transforms

        tfm = transforms.ToTensor()  # -> float tensor in [0, 1], shape [1, 28, 28]
        train_full = datasets.MNIST(str(_DATA_ROOT), train=True, download=True, transform=tfm)
        test_full = datasets.MNIST(str(_DATA_ROOT), train=False, download=True, transform=tfm)

        rng = np.random.default_rng(seed)
        tr_idx = rng.choice(len(train_full), size=min(n_train, len(train_full)), replace=False)
        te_idx = rng.choice(len(test_full), size=min(n_test, len(test_full)), replace=False)

        x_train = torch.stack([train_full[i][0].view(-1) for i in tr_idx])
        x_test = torch.stack([test_full[i][0].view(-1) for i in te_idx])
        y_test = np.array([int(test_full[i][1]) for i in te_idx])
        return DataSpec(x_train, x_test, y_test, (28, 28), 28 * 28, "MNIST (28x28)", "mnist")
    except Exception:  # noqa: BLE001 — any failure (offline, missing torchvision) => real sklearn fallback
        from sklearn.datasets import load_digits

        digits = load_digits()
        x = torch.tensor(digits.images.reshape(len(digits.images), -1), dtype=torch.float32) / 16.0
        y = np.asarray(digits.target)
        rng = np.random.default_rng(seed)
        perm = rng.permutation(len(x))
        x, y = x[perm], y[perm]
        n_tr = min(n_train, int(0.8 * len(x)))
        return DataSpec(
            x[:n_tr], x[n_tr:], y[n_tr:], (8, 8), 64,
            "sklearn load_digits (8x8)", "sklearn-digits-fallback",
        )


# ================================================================================================
# The model: encoder q_phi(z|x)=N(mu, sigma^2 I), reparameterization z=mu+sigma*eps, Bernoulli decoder
# ================================================================================================


class VAE(nn.Module):
    """A from-scratch Variational Autoencoder: a Gaussian encoder, the reparameterization trick, a Bernoulli decoder.

    The **encoder** is an MLP that maps an image ``x`` to the parameters of an approximate posterior
    ``q_phi(z|x) = N(mu(x), diag(sigma^2(x)))`` — it outputs a mean vector ``mu`` and a *log-variance* ``logvar``
    (we predict ``log sigma^2`` rather than ``sigma`` so it is unconstrained in sign and numerically stable). The
    **reparameterization trick** draws ``z = mu + sigma * eps`` with ``eps ~ N(0, I)`` fixed — this keeps ``z``
    differentiable in ``mu`` and ``logvar`` (see ``reparameterize``). The **decoder** is an MLP that maps ``z``
    back to per-pixel Bernoulli *logits*; a sigmoid turns them into pixel probabilities ``p_theta(x|z)``.
    """

    def __init__(self, n_pixels: int, latent_dim: int = LATENT_DIM, hidden: int = HIDDEN_DIM) -> None:
        super().__init__()
        self.n_pixels = n_pixels
        self.latent_dim = latent_dim
        # Encoder: x -> hidden -> (mu, logvar)
        self.enc_fc1 = nn.Linear(n_pixels, hidden)
        self.enc_mu = nn.Linear(hidden, latent_dim)
        self.enc_logvar = nn.Linear(hidden, latent_dim)
        # Decoder: z -> hidden -> pixel logits
        self.dec_fc1 = nn.Linear(latent_dim, hidden)
        self.dec_out = nn.Linear(hidden, n_pixels)

    def encode(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """Map images to the posterior parameters ``mu`` and ``logvar`` (i.e. ``log sigma^2``)."""
        h = F.relu(self.enc_fc1(x))
        return self.enc_mu(h), self.enc_logvar(h)

    def reparameterize(self, mu: torch.Tensor, logvar: torch.Tensor) -> torch.Tensor:
        """Sample ``z ~ N(mu, sigma^2 I)`` *differentiably*: ``z = mu + sigma * eps``, ``eps ~ N(0, I)``.

        This is the trick that makes the whole thing trainable. We cannot backprop through a raw draw from
        ``N(mu, sigma^2)`` — sampling is not a differentiable function of ``mu``/``sigma``. So we move the
        randomness into a *parameter-free* ``eps`` and reconstruct ``z`` by a differentiable affine map of it:
        ``z = mu + exp(0.5 * logvar) * eps``. Now ``dz/dmu = 1`` and ``dz/dsigma = eps`` are well defined, and the
        gradient of the loss flows back into the encoder. ``eps`` is resampled every forward pass (training) so
        each step sees fresh noise.
        """
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + std * eps

    def decode(self, z: torch.Tensor) -> torch.Tensor:
        """Map a latent ``z`` to per-pixel Bernoulli **logits** (apply sigmoid to get pixel probabilities)."""
        h = F.relu(self.dec_fc1(z))
        return self.dec_out(h)

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Full pass: encode -> reparameterize -> decode. Returns (pixel logits, mu, logvar)."""
        mu, logvar = self.encode(x)
        z = self.reparameterize(mu, logvar)
        return self.decode(z), mu, logvar


# ================================================================================================
# The ELBO: reconstruction (Bernoulli log-likelihood) + beta * KL-to-prior (closed-form Gaussian KL)
# ================================================================================================


def gaussian_kl_to_standard_normal(mu: torch.Tensor, logvar: torch.Tensor) -> torch.Tensor:
    """Closed-form ``KL(N(mu, sigma^2 I) || N(0, I))`` per example, summed over latent dimensions.

    For a diagonal Gaussian posterior against a unit-Gaussian prior the KL has an exact closed form,
    ``KL = 1/2 * sum_j (mu_j^2 + sigma_j^2 - 1 - log sigma_j^2)`` with ``sigma_j^2 = exp(logvar_j)``. This is the
    single place most practitioners meet a closed-form KL in the wild — it is the same two-Gaussian KL derived in
    the Cross-Entropy & KL-Divergence chapter, specialized to a standard-normal prior. Returns a per-example
    tensor of shape ``[batch]``.
    """
    return 0.5 * torch.sum(mu.pow(2) + logvar.exp() - 1.0 - logvar, dim=1)


def elbo_loss(
    logits: torch.Tensor, x: torch.Tensor, mu: torch.Tensor, logvar: torch.Tensor, *, beta: float = 1.0,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """The negative ELBO the VAE minimizes: reconstruction + ``beta`` * KL, averaged over the batch.

    Maximizing the ELBO ``log p(x) >= E_q[log p(x|z)] - KL(q(z|x) || p(z))`` is the same as *minimizing* its
    negation. The two terms:

      - **reconstruction** ``-E_q[log p_theta(x|z)]``: with a Bernoulli decoder this is the per-pixel binary
        cross-entropy between the decoder's pixel probabilities and the input, summed over pixels. It pulls the
        decoder to reconstruct the input from ``z``.
      - **KL** ``KL(q_phi(z|x) || N(0, I))``: the closed-form Gaussian KL above. It pulls every encoded posterior
        toward the prior, so the aggregate of all posteriors fills the prior — which is what makes sampling
        ``z ~ N(0, I)`` decode to something real.

    ``beta`` weights the KL (``beta = 1`` is the standard VAE; ``beta > 1`` is a beta-VAE, trading reconstruction
    for a tighter, more disentangled latent). We return the total plus the two terms separately (all per-example
    means) so the chapter can report the trade-off.
    """
    # Bernoulli reconstruction: sum BCE over pixels, then average over the batch.
    recon = F.binary_cross_entropy_with_logits(logits, x, reduction="none").sum(dim=1).mean()
    kl = gaussian_kl_to_standard_normal(mu, logvar).mean()
    return recon + beta * kl, recon, kl


# ================================================================================================
# Training: from-scratch minibatch gradient descent on the negative ELBO
# ================================================================================================


@dataclass
class TrainResult:
    elbo: np.ndarray  # per-epoch negative-ELBO (total loss) on the training set
    recon: np.ndarray  # per-epoch reconstruction term
    kl: np.ndarray  # per-epoch KL term
    beta: float
    latent_dim: int
    model: VAE = field(repr=False)
    final_elbo: float = 0.0
    final_recon: float = 0.0
    final_kl: float = 0.0


def train_vae(
    data: DataSpec,
    *,
    latent_dim: int = LATENT_DIM,
    hidden: int = HIDDEN_DIM,
    beta: float = 1.0,
    n_epochs: int = N_EPOCHS,
    batch_size: int = BATCH_SIZE,
    lr: float = LR,
    seed: int = SEED,
) -> TrainResult:
    """Train a VAE from scratch and return the measured per-epoch ELBO/recon/KL curves plus the trained model.

    The loop is the whole algorithm: for each minibatch, (1) encode to ``mu``/``logvar``; (2) reparameterize to a
    differentiable ``z``; (3) decode to pixel logits; (4) compute the negative ELBO (reconstruction + ``beta`` *
    KL); (5) one Adam step. Everything is seeded and runs on CPU for a reproducible trace.
    """
    torch.manual_seed(seed)
    np.random.seed(seed)
    model = VAE(data.n_pixels, latent_dim=latent_dim, hidden=hidden)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)

    x = data.x_train
    n = x.shape[0]
    elbo_hist, recon_hist, kl_hist = np.zeros(n_epochs), np.zeros(n_epochs), np.zeros(n_epochs)

    for epoch in range(n_epochs):
        perm = torch.randperm(n)
        tot_elbo = tot_recon = tot_kl = 0.0
        n_batches = 0
        for start in range(0, n, batch_size):
            xb = x[perm[start : start + batch_size]]
            logits, mu, logvar = model(xb)
            loss, recon, kl = elbo_loss(logits, xb, mu, logvar, beta=beta)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            tot_elbo += loss.item()
            tot_recon += recon.item()
            tot_kl += kl.item()
            n_batches += 1
        elbo_hist[epoch] = tot_elbo / n_batches
        recon_hist[epoch] = tot_recon / n_batches
        kl_hist[epoch] = tot_kl / n_batches

    return TrainResult(
        elbo_hist, recon_hist, kl_hist, beta, latent_dim, model,
        final_elbo=float(elbo_hist[-1]), final_recon=float(recon_hist[-1]), final_kl=float(kl_hist[-1]),
    )


# ================================================================================================
# Proof 1 — the closed-form Gaussian KL equals a Monte-Carlo estimate of KL(q || p)
# ================================================================================================


@dataclass
class KLProof:
    closed_form: float  # 1/2 sum(mu^2 + sigma^2 - 1 - logvar), the term the loss uses
    monte_carlo: float  # E_q[log q(z|x) - log p(z)] estimated by sampling z ~ q
    abs_error: float
    n_samples: int


def _diag_gaussian_logprob(z: torch.Tensor, mu: torch.Tensor, logvar: torch.Tensor) -> torch.Tensor:
    """Log density of a diagonal Gaussian ``N(mu, exp(logvar))`` at ``z``, summed over dimensions. Shape [n]."""
    var = logvar.exp()
    return -0.5 * torch.sum(math.log(2 * math.pi) + logvar + (z - mu).pow(2) / var, dim=1)


def prove_gaussian_kl(*, latent_dim: int = 8, n_samples: int = 400_000, seed: int = SEED) -> KLProof:
    """Verify the closed-form Gaussian KL against a Monte-Carlo estimate ``E_q[log q(z|x) - log p(z)]``.

    The closed form ``KL = 1/2 sum(mu^2 + sigma^2 - 1 - log sigma^2)`` is exact, but the *definition* of KL is an
    expectation, ``KL(q||p) = E_{z~q}[log q(z) - log p(z)]``. If the closed form is right, sampling ``z ~ q`` and
    averaging ``log q(z|x) - log p(z)`` must converge to it. We fix an arbitrary posterior ``(mu, logvar)`` and
    check both numbers agree to Monte-Carlo tolerance. This certifies the exact term the ELBO loss uses.
    """
    torch.manual_seed(seed)
    mu = torch.randn(1, latent_dim) * 0.8  # an arbitrary but fixed posterior mean
    logvar = torch.randn(1, latent_dim) * 0.5  # ...and log-variance

    closed = float(gaussian_kl_to_standard_normal(mu, logvar).item())

    std = torch.exp(0.5 * logvar)
    z = mu + std * torch.randn(n_samples, latent_dim)  # z ~ q(z|x)
    log_q = _diag_gaussian_logprob(z, mu.expand_as(z), logvar.expand_as(z))
    log_p = _diag_gaussian_logprob(z, torch.zeros_like(z), torch.zeros_like(z))  # prior N(0, I): logvar = 0
    mc = float((log_q - log_p).mean().item())

    return KLProof(closed_form=closed, monte_carlo=mc, abs_error=abs(closed - mc), n_samples=n_samples)


# ================================================================================================
# Proof 2 — the reparameterization trick is what lets gradients reach the encoder
# ================================================================================================


@dataclass
class ReparamProof:
    grad_mu_reparam: float  # ||d loss / d mu|| with z = mu + sigma * eps  (reparameterized: should be > 0)
    grad_logvar_reparam: float  # ||d loss / d logvar|| reparameterized     (should be > 0)
    direct_sample_has_grad: bool  # does z from a raw .sample() carry a grad back to mu? (should be False)


def prove_reparameterization(*, latent_dim: int = 4, seed: int = SEED) -> ReparamProof:
    """Show the reparameterized ``z`` gives gradients through ``mu``/``logvar``; a raw ``.sample()`` does not.

    We build a tiny scalar "loss" ``||z||^2`` from a sampled latent, two ways: (a) the **reparameterized** draw
    ``z = mu + sigma * eps`` used in training, and (b) a **direct sample** ``z ~ N(mu, sigma^2)`` via
    ``torch.distributions.Normal(mu, sigma).sample()``. In (a), ``z`` is a differentiable function of ``mu`` and
    ``logvar``, so ``loss.backward()`` populates ``mu.grad`` and ``logvar.grad``. In (b), ``.sample()`` returns a
    tensor detached from ``mu``/``sigma`` — no gradient reaches the encoder parameters. This is *why* the trick
    exists: without it, backprop cannot train the encoder.
    """
    torch.manual_seed(seed)

    # (a) reparameterized: z = mu + sigma * eps, eps a parameter-free constant
    mu_a = torch.zeros(1, latent_dim, requires_grad=True)
    logvar_a = torch.zeros(1, latent_dim, requires_grad=True)
    eps = torch.randn(1, latent_dim)
    z_a = mu_a + torch.exp(0.5 * logvar_a) * eps
    (z_a.pow(2).sum()).backward()
    grad_mu = float(mu_a.grad.norm().item())
    grad_logvar = float(logvar_a.grad.norm().item())

    # (b) direct sample: z ~ Normal(mu, sigma).sample() severs the graph
    mu_b = torch.zeros(1, latent_dim, requires_grad=True)
    logvar_b = torch.zeros(1, latent_dim, requires_grad=True)
    std_b = torch.exp(0.5 * logvar_b)
    z_b = torch.distributions.Normal(mu_b, std_b).sample()  # NOT rsample: detached from mu_b/std_b
    direct_has_grad = z_b.requires_grad  # False — the sample is a leaf constant w.r.t. the encoder

    return ReparamProof(
        grad_mu_reparam=grad_mu, grad_logvar_reparam=grad_logvar, direct_sample_has_grad=bool(direct_has_grad),
    )


# ================================================================================================
# Generation: reconstructions, prior samples, the 2-D latent manifold, and an interpolation
# ================================================================================================


def reconstruct(model: VAE, x: torch.Tensor) -> np.ndarray:
    """Encode real images to their posterior mean and decode — the VAE's reconstruction of each input."""
    model.eval()
    with torch.no_grad():
        mu, _ = model.encode(x)
        probs = torch.sigmoid(model.decode(mu))  # decode the mean (no noise) for a clean reconstruction
    return probs.numpy()


def sample_prior(model: VAE, *, n: int = 15, seed: int = SEED) -> np.ndarray:
    """Sample ``z ~ N(0, I)`` and decode — novel digits the model generates, the payoff a plain AE cannot give."""
    torch.manual_seed(seed)
    model.eval()
    with torch.no_grad():
        z = torch.randn(n, model.latent_dim)
        probs = torch.sigmoid(model.decode(z))
    return probs.numpy()


def latent_manifold(model: VAE, *, grid: int = 15, span: float = 2.5) -> np.ndarray:
    """Decode a grid of ``z`` across a 2-D latent square — visualizes the smooth, organized latent space.

    Only meaningful when ``latent_dim == 2``. We sweep each coordinate across ``[-span, span]`` (roughly the bulk
    of the standard-normal prior) and decode every grid point, so the returned array is the manifold of digits the
    latent space encodes — neighbours in ``z`` decode to visually similar digits, the hallmark of a VAE.
    """
    if model.latent_dim != 2:
        raise ValueError("latent_manifold requires latent_dim == 2")
    model.eval()
    coords = np.linspace(-span, span, grid, dtype=np.float32)
    zs = torch.tensor([[a, b] for b in coords for a in coords])  # row-major grid
    with torch.no_grad():
        probs = torch.sigmoid(model.decode(zs))
    return probs.numpy()


def interpolate(model: VAE, x0: torch.Tensor, x1: torch.Tensor, *, steps: int = 10) -> np.ndarray:
    """Walk a straight line in latent space between two encoded digits and decode each step — a smooth morph.

    Encode ``x0`` and ``x1`` to their posterior means, linearly interpolate between the two latent points, and
    decode each. Because the VAE's latent space is smooth and continuous (that is what the KL term buys), the
    decoded frames morph gradually from one digit into the other rather than cutting abruptly.
    """
    model.eval()
    with torch.no_grad():
        z0, _ = model.encode(x0.unsqueeze(0))
        z1, _ = model.encode(x1.unsqueeze(0))
        alphas = torch.linspace(0, 1, steps).unsqueeze(1)
        zs = (1 - alphas) * z0 + alphas * z1
        probs = torch.sigmoid(model.decode(zs))
    return probs.numpy()


# ================================================================================================
# The full experiment, bundled (figures and the notebook reuse this one measured run)
# ================================================================================================


@dataclass
class Experiment:
    torch_version: str
    numpy_version: str
    torchvision_version: str
    device_available: str
    data_label: str
    data_source: str
    img_shape: tuple[int, int]
    n_pixels: int
    seed: int
    main: TrainResult = field(repr=False, default=None)  # type: ignore[assignment]
    beta_runs: dict[float, TrainResult] = field(repr=False, default_factory=dict)
    kl_proof: KLProof | None = None
    reparam_proof: ReparamProof | None = None
    data: DataSpec = field(repr=False, default=None)  # type: ignore[assignment]


def run_experiment(
    *, betas: tuple[float, ...] = (1.0, 4.0), n_epochs: int = N_EPOCHS, seed: int = SEED,
) -> Experiment:
    """Run the whole measured pipeline once and return every quantity the chapter, figures, and notebook cite."""
    try:
        import torchvision

        tv_version = torchvision.__version__
    except Exception:  # noqa: BLE001
        tv_version = "not installed"

    data = load_data(seed=seed)
    # main model: beta = 1, latent_dim = 2 (so we can draw the manifold) — the headline run
    beta_runs = {
        b: train_vae(data, beta=b, latent_dim=LATENT_DIM, n_epochs=n_epochs, seed=seed) for b in betas
    }
    main = beta_runs[1.0]

    kl_proof = prove_gaussian_kl()
    reparam_proof = prove_reparameterization()

    return Experiment(
        torch_version=torch.__version__, numpy_version=np.__version__, torchvision_version=tv_version,
        device_available=str(get_device()), data_label=data.label, data_source=data.source,
        img_shape=data.img_shape, n_pixels=data.n_pixels, seed=seed,
        main=main, beta_runs=beta_runs, kl_proof=kl_proof, reparam_proof=reparam_proof, data=data,
    )


# ================================================================================================
# Report — every number the chapter quotes, each headline relationship guarded by a hard assert
# ================================================================================================


def main() -> None:
    exp = run_experiment()
    assert exp.kl_proof and exp.reparam_proof and exp.main

    print(f"torch {exp.torch_version} | numpy {exp.numpy_version} | torchvision {exp.torchvision_version}")
    print(f"(training on CPU for a reproducible trace; best available device = {exp.device_available}; "
          f"seed={exp.seed}; data: {exp.data_source})")
    print(f"dataset: {exp.data_label} — {exp.n_pixels} pixels/image, {exp.data.x_train.shape[0]} train / "
          f"{exp.data.x_test.shape[0]} test\n")

    print("=== 1. The closed-form Gaussian KL EQUALS a Monte-Carlo estimate of KL(q||p) ===")
    kp = exp.kl_proof
    print(f"  closed form  1/2 sum(mu^2 + sigma^2 - 1 - logvar) = {kp.closed_form:.5f}")
    print(f"  Monte-Carlo  E_q[log q(z|x) - log p(z)]  (N={kp.n_samples:,}) = {kp.monte_carlo:.5f}")
    print(f"  |closed - MC| = {kp.abs_error:.2e}  (=> the ELBO's KL term is the exact KL)\n")

    print("=== 2. The REPARAMETERIZATION trick is what lets gradients reach the encoder ===")
    rp = exp.reparam_proof
    print(f"  reparameterized z = mu + sigma*eps :  ||d loss/d mu|| = {rp.grad_mu_reparam:.3f},  "
          f"||d loss/d logvar|| = {rp.grad_logvar_reparam:.3f}  (gradients flow)")
    print(f"  direct sample  z ~ Normal(mu,sigma).sample() :  z.requires_grad = {rp.direct_sample_has_grad}  "
          f"(NO gradient to the encoder)\n")

    print(f"=== 3. A from-scratch VAE learns to reconstruct AND generate ({exp.data_label}) ===")
    m = exp.main
    print(f"  [beta=1] final negative-ELBO = {m.final_elbo:.2f}  (recon = {m.final_recon:.2f}  +  "
          f"KL = {m.final_kl:.2f})  after {N_EPOCHS} epochs, latent_dim={m.latent_dim}")

    print("\n=== 4. beta-VAE trade-off: a larger KL weight buys a smaller KL at the cost of reconstruction ===")
    for b in sorted(exp.beta_runs):
        r = exp.beta_runs[b]
        print(f"  beta={b:>3.0f}: recon = {r.final_recon:6.2f}   KL = {r.final_kl:5.2f}   "
              f"total = {r.final_elbo:6.2f}")
    r1, r4 = exp.beta_runs[1.0], exp.beta_runs[4.0]

    # ---- hard asserts on the headline relationships (raise, not print, if a lesson breaks) ----
    assert kp.abs_error < 1e-2, "the closed-form Gaussian KL must match the Monte-Carlo KL estimate"
    assert rp.grad_mu_reparam > 1e-6 and rp.grad_logvar_reparam > 1e-6, \
        "reparameterization must give real gradients through mu and logvar"
    assert rp.direct_sample_has_grad is False, \
        "a direct .sample() must NOT carry a gradient to the encoder (that is why we reparameterize)"
    assert m.final_recon < m.recon[0], "training must reduce the reconstruction loss (the VAE must learn)"
    assert r4.final_kl < r1.final_kl, "a larger beta must yield a smaller (tighter-to-prior) KL"
    assert r4.final_recon > r1.final_recon, "a larger beta must cost reconstruction quality (the trade-off)"
    print("\nAll checks passed: the closed-form Gaussian KL matches the Monte-Carlo KL; the reparameterization "
          "trick gives gradients where a raw sample gives none; a from-scratch VAE learns to reconstruct and "
          "generate real digits; and the beta trade-off (smaller KL, worse reconstruction) is measured.")


if __name__ == "__main__":
    main()
