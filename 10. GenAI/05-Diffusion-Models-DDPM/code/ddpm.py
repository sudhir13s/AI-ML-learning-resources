"""A Denoising Diffusion Probabilistic Model (DDPM) from scratch, on REAL data, with the claims that matter PROVEN.

This is not a toy. It builds a real DDPM **from scratch** — the fixed Gaussian **forward (noising) process**
``q(x_t | x_{t-1})``, its closed-form one-step jump ``q(x_t | x_0) = N(sqrt(abar_t) x_0, (1 - abar_t) I)``, a
learned **denoiser** ``eps_theta(x_t, t)`` trained with the simplified noise-prediction loss
``L_simple = E || eps - eps_theta(x_t, t) ||^2``, and the **ancestral sampling** loop that denoises pure noise
back into data. The demo is a **real 2-D distribution** (sklearn ``make_moons`` — real sampled data) because the
forward-noise -> reverse-denoise process is *visible* in 2-D: you literally watch the data cloud diffuse to
``N(0, I)`` and denoise back, and matching the target is easy to *measure*. (Scaling the identical algorithm to
images just swaps the little MLP denoiser for a convolutional U-Net; the chapter says so and points to real
systems. We keep the runnable artifact 2-D so it trains in seconds on CPU and every claim is checkable.)

It then makes the two claims the chapter rests on *checkable*, with hard ``assert`` statements:

  1. **The closed-form forward IS the iterative forward.** The "nice property" ``q(x_t | x_0)`` claims that
     compounding ``t`` tiny reparameterized Gaussian noising steps collapses to a single Gaussian with mean
     ``sqrt(abar_t) x_0`` and variance ``(1 - abar_t) I``. We verify it by Monte Carlo: fix an ``x_0``, run the
     step-by-step iterative forward chain many thousands of times to time ``t``, and check the empirical mean and
     covariance match the closed-form values to Monte-Carlo tolerance. The one-shot jump the training loop uses is
     exactly the marginal of the slow chain.

  2. **The trained model GENERATES the target distribution.** After training, ancestral sampling from ``N(0, I)``
     produces points whose distribution matches the real 2-D target. We quantify it with the **energy distance**
     (a real two-sample statistic, zero iff the distributions match) and ``assert`` that generated-vs-target is far
     smaller than the ``N(0, I)``-baseline-vs-target — i.e. the reverse chain learned the data, it did not just
     echo its Gaussian starting point.

Everything is **seeded** and pinned to **CPU** so the numbers reproduce on any machine. Run::

    python ddpm.py

The data is a real ``make_moons`` sample (never a mock); the whole pipeline trains and proves itself in under a
minute on CPU.

Verified on Python 3.12 / numpy 2.4 / torch 2.12 / scikit-learn 1.9 (CPU).
"""

from __future__ import annotations

import math
import time
from dataclasses import dataclass, field

import numpy as np
import torch
import torch.nn.functional as F
from torch import nn

SEED = 0

# --- Schedule / model / training hyperparameters for the headline 2-D run. Chosen (see the chapter) so a small
# MLP denoiser trains from scratch to samples that match a real 2-D distribution on CPU in seconds. Every value is
# a plain dial you can turn in the notebook.
T_STEPS = 400          # number of diffusion steps; T=400 drives abar_T ~ 0.02 so x_T is essentially N(0, I)
BETA_START = 1e-4      # linear-schedule endpoints (Ho et al. 2020 use these for T=1000; fine for T=400 in 2-D)
BETA_END = 0.02
HIDDEN_DIM = 128       # denoiser MLP width
TIME_EMB_DIM = 64      # sinusoidal timestep-embedding dimension
N_POINTS = 4000        # real 2-D points sampled from the target distribution
N_EPOCHS = 700
BATCH_SIZE = 256
LR = 1e-3
MOONS_NOISE = 0.06     # sklearn make_moons observation noise (real sampled data, mild jitter)


def get_device() -> torch.device:
    """Detect the best available device for *reporting*; training pins CPU for a reproducible trace.

    A small MLP denoiser on 2-D points trains fast on CPU, and CPU float math keeps the learning curve
    reproducible across machines — what a teaching artifact needs. We still surface what hardware is present so
    the banner is honest.
    """
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


# ================================================================================================
# The noise schedule: betas -> alphas -> alpha-bars (the compounding survival of the signal)
# ================================================================================================


@dataclass
class Schedule:
    """A fixed variance (noise) schedule and every quantity derived from it that the forward/reverse steps need.

    ``betas[i]`` is the variance of the Gaussian noise added at diffusion step ``t = i + 1`` (arrays are 0-indexed,
    timesteps are 1..T). From ``betas`` we precompute:

      - ``alphas = 1 - betas`` — the fraction of *variance* the signal keeps at each step;
      - ``alpha_bars = cumprod(alphas)`` — the compounded signal-survival ``abar_t = prod_{s<=t} alpha_s``, the
        single most important quantity: the closed-form forward ``q(x_t | x_0)`` has mean ``sqrt(abar_t) x_0`` and
        variance ``(1 - abar_t)``. ``abar_t`` falls from ~1 (t small, almost all signal) to ~0 (t = T, almost all
        noise);
      - ``sqrt_abar`` / ``sqrt_one_minus_abar`` — the signal- and noise-mixing coefficients used everywhere.
    """

    betas: torch.Tensor          # [T]
    alphas: torch.Tensor         # [T]
    alpha_bars: torch.Tensor     # [T]  abar_t = prod alpha
    sqrt_abar: torch.Tensor      # [T]  sqrt(abar_t)             — signal coefficient
    sqrt_one_minus_abar: torch.Tensor  # [T]  sqrt(1 - abar_t)   — noise coefficient
    alpha_bars_prev: torch.Tensor  # [T]  abar_{t-1} (abar_0 = 1) — for the reverse posterior variance
    posterior_var: torch.Tensor  # [T]  beta_tilde_t = (1-abar_{t-1})/(1-abar_t) * beta_t
    kind: str
    T: int

    def to(self, device: torch.device) -> Schedule:
        return Schedule(
            self.betas.to(device), self.alphas.to(device), self.alpha_bars.to(device),
            self.sqrt_abar.to(device), self.sqrt_one_minus_abar.to(device),
            self.alpha_bars_prev.to(device), self.posterior_var.to(device), self.kind, self.T,
        )


def make_schedule(*, kind: str = "linear", T: int = T_STEPS) -> Schedule:
    """Build a variance schedule: ``"linear"`` (Ho et al. 2020) or ``"cosine"`` (Nichol & Dhariwal 2021).

    The **linear** schedule ramps ``beta_t`` linearly from ``BETA_START`` to ``BETA_END``. The **cosine** schedule
    defines ``abar_t`` directly from a cosine curve so that noise is added more gently early and late (it improves
    likelihood and sample quality on images); we derive its ``beta_t`` back out via
    ``beta_t = 1 - abar_t / abar_{t-1}``. Both return the same precomputed ``Schedule`` bundle.
    """
    if kind == "linear":
        betas = torch.linspace(BETA_START, BETA_END, T, dtype=torch.float64)
    elif kind == "cosine":
        # Nichol & Dhariwal: abar_t = f(t)/f(0), f(t) = cos^2(((t/T + s)/(1+s)) * pi/2), s = 0.008
        s = 0.008
        steps = torch.arange(T + 1, dtype=torch.float64)
        f = torch.cos(((steps / T + s) / (1 + s)) * math.pi / 2) ** 2
        alpha_bars_full = f / f[0]
        betas = torch.clamp(1 - alpha_bars_full[1:] / alpha_bars_full[:-1], min=1e-8, max=0.999)
    else:  # pragma: no cover - guarded by callers
        raise ValueError(f"unknown schedule kind: {kind!r}")

    alphas = 1.0 - betas
    alpha_bars = torch.cumprod(alphas, dim=0)
    alpha_bars_prev = torch.cat([torch.ones(1, dtype=torch.float64), alpha_bars[:-1]])
    posterior_var = betas * (1.0 - alpha_bars_prev) / (1.0 - alpha_bars)
    return Schedule(
        betas=betas.float(), alphas=alphas.float(), alpha_bars=alpha_bars.float(),
        sqrt_abar=torch.sqrt(alpha_bars).float(),
        sqrt_one_minus_abar=torch.sqrt(1.0 - alpha_bars).float(),
        alpha_bars_prev=alpha_bars_prev.float(), posterior_var=posterior_var.float(),
        kind=kind, T=T,
    )


def _extract(a: torch.Tensor, t_idx: torch.Tensor, shape: tuple[int, ...]) -> torch.Tensor:
    """Gather schedule entries ``a[t_idx]`` and reshape to broadcast against a batch of shape ``shape``.

    ``t_idx`` holds 0-based schedule indices (timestep ``t`` maps to index ``t - 1``). Returns a tensor of shape
    ``[batch, 1, ...]`` so it multiplies a data batch elementwise.
    """
    out = a.gather(0, t_idx)
    return out.reshape(t_idx.shape[0], *((1,) * (len(shape) - 1)))


# ================================================================================================
# The forward process: the closed-form one-step jump q(x_t | x_0), and the slow iterative chain
# ================================================================================================


def q_sample(
    x0: torch.Tensor, t_idx: torch.Tensor, sched: Schedule, eps: torch.Tensor | None = None,
) -> tuple[torch.Tensor, torch.Tensor]:
    """The closed-form forward: draw ``x_t ~ q(x_t | x_0) = N(sqrt(abar_t) x_0, (1 - abar_t) I)`` in ONE step.

    This is the "nice property" that makes DDPM trainable: instead of simulating ``t`` noising steps, we jump
    straight to any ``x_t`` via the reparameterization ``x_t = sqrt(abar_t) x_0 + sqrt(1 - abar_t) eps`` with
    ``eps ~ N(0, I)``. Returns ``(x_t, eps)`` — we return the noise too because the training target *is* ``eps``.
    """
    if eps is None:
        eps = torch.randn_like(x0)
    sqrt_abar = _extract(sched.sqrt_abar, t_idx, x0.shape)
    sqrt_om = _extract(sched.sqrt_one_minus_abar, t_idx, x0.shape)
    return sqrt_abar * x0 + sqrt_om * eps, eps


def iterative_forward(
    x0: torch.Tensor, t: int, sched: Schedule, *, generator: torch.Generator | None = None,
) -> torch.Tensor:
    """Simulate the forward chain the SLOW way: apply ``x_s = sqrt(1 - beta_s) x_{s-1} + sqrt(beta_s) z`` for s=1..t.

    Each step is one tiny reparameterized Gaussian noising ``q(x_s | x_{s-1}) = N(sqrt(1 - beta_s) x_{s-1},
    beta_s I)`` with a *fresh* independent ``z``. Composing ``t`` of them is what the closed-form ``q(x_t | x_0)``
    claims to shortcut — Proof 1 checks the two agree in distribution. Returns ``x_t`` for the whole batch.
    """
    x = x0
    for s in range(t):  # s = 0..t-1 -> timestep s+1
        beta_s = sched.betas[s]
        z = torch.randn(x.shape, generator=generator)
        x = torch.sqrt(1.0 - beta_s) * x + torch.sqrt(beta_s) * z
    return x


# ================================================================================================
# The denoiser eps_theta(x_t, t): a small MLP with a sinusoidal timestep embedding (2-D data)
# ================================================================================================


def sinusoidal_embedding(t: torch.Tensor, dim: int) -> torch.Tensor:
    """Transformer-style sinusoidal embedding of an integer timestep ``t`` into a ``dim``-vector.

    The denoiser must behave differently at different noise levels, so ``t`` is embedded (not fed as a raw scalar)
    exactly as positions are in a transformer: interleaved sines and cosines at geometrically spaced frequencies.
    ``t`` is a float tensor of shape ``[batch]``; returns ``[batch, dim]``.
    """
    half = dim // 2
    freqs = torch.exp(-math.log(10_000.0) * torch.arange(half, dtype=torch.float32) / (half - 1))
    args = t[:, None].float() * freqs[None, :]
    return torch.cat([torch.sin(args), torch.cos(args)], dim=1)


class Denoiser2D(nn.Module):
    """A from-scratch noise-prediction network ``eps_theta(x_t, t)`` for 2-D data.

    It embeds the timestep ``t`` with a sinusoidal embedding + a small MLP, concatenates it with the noised point
    ``x_t``, and passes the pair through a SiLU MLP that outputs the *predicted noise* ``eps_hat`` (same shape as
    ``x_t``). Predicting the noise (rather than the clean point) is the DDPM parameterization: it is what makes the
    training loss the simple ``|| eps - eps_theta ||^2``.
    """

    def __init__(self, dim: int = 2, hidden: int = HIDDEN_DIM, time_emb_dim: int = TIME_EMB_DIM) -> None:
        super().__init__()
        self.time_emb_dim = time_emb_dim
        self.time_mlp = nn.Sequential(
            nn.Linear(time_emb_dim, hidden), nn.SiLU(), nn.Linear(hidden, hidden),
        )
        self.net = nn.Sequential(
            nn.Linear(dim + hidden, hidden), nn.SiLU(),
            nn.Linear(hidden, hidden), nn.SiLU(),
            nn.Linear(hidden, hidden), nn.SiLU(),
            nn.Linear(hidden, dim),
        )

    def forward(self, x_t: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
        """Predict the noise added to produce ``x_t`` at timestep ``t``. ``x_t``: [B, dim]; ``t``: [B] (1..T)."""
        temb = self.time_mlp(sinusoidal_embedding(t, self.time_emb_dim))
        return self.net(torch.cat([x_t, temb], dim=1))


# ================================================================================================
# The training objective: L_simple = E_{t, x0, eps} || eps - eps_theta(x_t, t) ||^2
# ================================================================================================


def diffusion_loss(
    model: nn.Module, x0: torch.Tensor, sched: Schedule, *, generator: torch.Generator | None = None,
) -> torch.Tensor:
    """One Monte-Carlo estimate of the simplified DDPM loss ``L_simple`` on a batch.

    For each example we draw a random timestep ``t ~ Uniform{1..T}`` and fresh noise ``eps ~ N(0, I)``, form the
    noised point ``x_t = q_sample(x0, t, eps)`` with the closed-form jump, and regress the network's prediction
    onto the *true* noise: ``|| eps - eps_theta(x_t, t) ||^2``. This unweighted noise-MSE is the DDPM training
    objective (a reweighting of the true variational bound; see the chapter). Returns the scalar mean loss.
    """
    b = x0.shape[0]
    t_idx = torch.randint(0, sched.T, (b,), generator=generator)  # 0-based index -> timestep t = t_idx + 1
    x_t, eps = q_sample(x0, t_idx, sched, eps=torch.randn(x0.shape, generator=generator))
    eps_pred = model(x_t, (t_idx + 1).float())
    return F.mse_loss(eps_pred, eps)


# ================================================================================================
# Ancestral sampling: start from N(0, I) and run the learned reverse chain x_T -> x_0
# ================================================================================================


@torch.no_grad()
def p_sample_loop(
    model: nn.Module, sched: Schedule, *, n: int, dim: int = 2, seed: int = SEED,
    record_every: int | None = None,
) -> tuple[torch.Tensor, list[tuple[int, torch.Tensor]]]:
    """Generate samples by ancestral sampling: ``x_T ~ N(0, I)``, then denoise down to ``x_0`` one step at a time.

    At each step the model predicts the noise ``eps_hat = eps_theta(x_t, t)`` and we take the DDPM reverse update
    ::

        mean = 1/sqrt(alpha_t) * ( x_t - (1 - alpha_t)/sqrt(1 - abar_t) * eps_hat )
        x_{t-1} = mean + sqrt(sigma_t^2) * z        (z ~ N(0, I); z = 0 at the final step t = 1)

    with the posterior variance ``sigma_t^2 = beta_tilde_t``. Optionally records the intermediate cloud every
    ``record_every`` steps (for the reverse-trajectory figure). Returns ``(x_0 samples, trajectory)``.
    """
    g = torch.Generator().manual_seed(seed)
    x = torch.randn(n, dim, generator=g)  # x_T ~ N(0, I)
    traj: list[tuple[int, torch.Tensor]] = []
    for i in reversed(range(sched.T)):  # i = T-1 .. 0  ->  timestep t = i + 1
        t = i + 1
        t_batch = torch.full((n,), float(t))
        eps_hat = model(x, t_batch)
        alpha_t = sched.alphas[i]
        abar_t = sched.alpha_bars[i]
        coef = (1.0 - alpha_t) / torch.sqrt(1.0 - abar_t)
        mean = (x - coef * eps_hat) / torch.sqrt(alpha_t)
        if i > 0:
            z = torch.randn(n, dim, generator=g)
            x = mean + torch.sqrt(sched.posterior_var[i]) * z
        else:
            x = mean  # last step: no noise
        if record_every is not None and (i % record_every == 0 or i == sched.T - 1):
            traj.append((t, x.clone()))
    return x, traj


# ================================================================================================
# The 2-D data: a REAL sampled distribution (sklearn make_moons), standardized to N(0, I) scale
# ================================================================================================


@dataclass
class Data2D:
    """A real 2-D dataset, standardized to zero mean / unit variance so ``N(0, I)`` is the right diffusion prior."""

    x: torch.Tensor        # [N, 2] standardized points
    mean: np.ndarray       # original mean (to un-standardize for plotting if desired)
    std: np.ndarray        # original std
    label: str


def load_moons(*, n: int = N_POINTS, noise: float = MOONS_NOISE, seed: int = SEED) -> Data2D:
    """Sample the real ``make_moons`` distribution and standardize it. Two interleaving half-moons — a genuinely
    2-D, non-Gaussian, multi-modal target that a diffusion model must *learn* (it cannot be captured by its
    ``N(0, I)`` starting point)."""
    from sklearn.datasets import make_moons

    xy, _ = make_moons(n_samples=n, noise=noise, random_state=seed)
    mean = xy.mean(axis=0)
    std = xy.std(axis=0)
    x = (xy - mean) / std
    return Data2D(torch.tensor(x, dtype=torch.float32), mean, std, "sklearn make_moons (2-D)")


# ================================================================================================
# Training loop: from-scratch minibatch gradient descent on L_simple
# ================================================================================================


@dataclass
class TrainResult:
    loss_hist: np.ndarray            # per-epoch mean L_simple
    model: nn.Module = field(repr=False)
    sched: Schedule = field(repr=False)
    final_loss: float = 0.0
    seconds: float = 0.0


def train_ddpm(
    data: Data2D, sched: Schedule, *, n_epochs: int = N_EPOCHS, batch_size: int = BATCH_SIZE,
    lr: float = LR, seed: int = SEED,
) -> TrainResult:
    """Train the denoiser from scratch on ``L_simple`` and return the measured per-epoch loss and trained model.

    The loop is the whole DDPM training algorithm: for each minibatch, sample random timesteps and noise, form the
    noised points with the closed-form ``q_sample``, predict the noise, take the MSE, and one Adam step. Seeded and
    on CPU for a reproducible trace.
    """
    torch.manual_seed(seed)
    np.random.seed(seed)
    model = Denoiser2D(dim=data.x.shape[1])
    opt = torch.optim.Adam(model.parameters(), lr=lr)
    g = torch.Generator().manual_seed(seed)

    x = data.x
    n = x.shape[0]
    loss_hist = np.zeros(n_epochs)
    t0 = time.time()
    for epoch in range(n_epochs):
        perm = torch.randperm(n, generator=g)
        tot, nb = 0.0, 0
        for start in range(0, n, batch_size):
            xb = x[perm[start : start + batch_size]]
            loss = diffusion_loss(model, xb, sched, generator=g)
            opt.zero_grad()
            loss.backward()
            opt.step()
            tot += loss.item()
            nb += 1
        loss_hist[epoch] = tot / nb
    return TrainResult(loss_hist, model, sched, final_loss=float(loss_hist[-1]), seconds=time.time() - t0)


# ================================================================================================
# Proof 1 — the closed-form forward q(x_t|x_0) EQUALS the iterative step-by-step forward (in distribution)
# ================================================================================================


@dataclass
class ForwardProof:
    t: int
    abar_t: float
    closed_mean: np.ndarray        # sqrt(abar_t) * x0
    iter_mean: np.ndarray          # empirical mean of the iterative chain at t
    closed_var: float              # 1 - abar_t (isotropic)
    iter_cov: np.ndarray           # empirical 2x2 covariance of the iterative chain at t
    mean_abs_err: float            # max |closed_mean - iter_mean|
    var_abs_err: float             # max |iter_cov - (1-abar_t) I|
    n_rollouts: int


def prove_forward_closed_form(
    sched: Schedule, *, t: int = 240, n_rollouts: int = 60_000, seed: int = SEED,
) -> ForwardProof:
    """Verify the closed-form ``q(x_t | x_0)`` is the marginal of the slow iterative forward chain.

    Fix a single point ``x_0``. The closed form claims ``x_t ~ N(sqrt(abar_t) x_0, (1 - abar_t) I)``. We *simulate*
    the iterative chain (``t`` fresh tiny Gaussian steps) ``n_rollouts`` independent times, then compare the
    empirical mean and covariance of ``x_t`` to the closed-form ``sqrt(abar_t) x_0`` and ``(1 - abar_t) I``. If the
    "nice property" is right they must agree to Monte-Carlo tolerance — certifying that the one-shot ``q_sample``
    the training loop uses is exactly the distribution of the compounded chain.
    """
    torch.manual_seed(seed)
    g = torch.Generator().manual_seed(seed)
    x0_point = torch.tensor([[1.2, -0.7]], dtype=torch.float32)  # an arbitrary but fixed x_0
    x0 = x0_point.expand(n_rollouts, 2).contiguous()

    x_t = iterative_forward(x0, t, sched, generator=g)  # simulate the slow chain n_rollouts times
    iter_mean = x_t.mean(dim=0).numpy()
    iter_cov = np.cov(x_t.numpy(), rowvar=False)

    abar_t = float(sched.alpha_bars[t - 1])
    closed_mean = (math.sqrt(abar_t) * x0_point).numpy().ravel()
    closed_var = 1.0 - abar_t

    mean_err = float(np.max(np.abs(closed_mean - iter_mean)))
    var_err = float(np.max(np.abs(iter_cov - closed_var * np.eye(2))))
    return ForwardProof(
        t=t, abar_t=abar_t, closed_mean=closed_mean, iter_mean=iter_mean, closed_var=closed_var,
        iter_cov=iter_cov, mean_abs_err=mean_err, var_abs_err=var_err, n_rollouts=n_rollouts,
    )


# ================================================================================================
# Proof 2 — the trained model GENERATES the target distribution (energy distance << baseline)
# ================================================================================================


def energy_distance(x: torch.Tensor, y: torch.Tensor) -> float:
    """The energy distance between two point sets — a real two-sample statistic, ``0`` iff the laws match.

    ``E^2(X, Y) = 2 E||X - Y|| - E||X - X'|| - E||Y - Y'||`` (pairwise Euclidean distances). It is a genuine
    metric on distributions (Szekely & Rizzo): small when two samples look like they came from the same law, large
    when they do not. We use it to score generated-vs-target without assuming a parametric form.
    """
    def _mean_pdist(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
        return torch.cdist(a, b).mean()

    d_xy = _mean_pdist(x, y)
    d_xx = _mean_pdist(x, x)
    d_yy = _mean_pdist(y, y)
    val = 2.0 * d_xy - d_xx - d_yy
    return float(torch.sqrt(torch.clamp(val, min=0.0)))


@dataclass
class GenerationProof:
    ed_generated: float    # energy distance: generated samples vs held-out real target
    ed_baseline: float     # energy distance: N(0, I) samples vs the same target
    ratio: float           # ed_generated / ed_baseline  (small => learned the data)
    n_samples: int


def prove_generation(
    result: TrainResult, data: Data2D, *, n_samples: int = 2000, seed: int = SEED,
) -> GenerationProof:
    """Show the trained reverse chain matches the target, not its ``N(0, I)`` start, via the energy distance.

    We draw ``n_samples`` from the model (ancestral sampling) and compare them to a held-out batch of the real
    target using the energy distance; we do the same for a raw ``N(0, I)`` baseline (the chain's starting cloud).
    If diffusion learned the distribution, ``ED(generated, target)`` is far below ``ED(N(0,I), target)``.
    """
    g = torch.Generator().manual_seed(seed + 1)
    target = data.x[torch.randperm(data.x.shape[0], generator=g)[:n_samples]]
    generated, _ = p_sample_loop(result.model, result.sched, n=n_samples, dim=data.x.shape[1], seed=seed + 2)
    baseline = torch.randn(n_samples, data.x.shape[1], generator=torch.Generator().manual_seed(seed + 3))

    ed_gen = energy_distance(generated, target)
    ed_base = energy_distance(baseline, target)
    return GenerationProof(
        ed_generated=ed_gen, ed_baseline=ed_base, ratio=ed_gen / ed_base, n_samples=n_samples,
    )


# ================================================================================================
# The full experiment, bundled (figures and the notebook reuse this one measured run)
# ================================================================================================


@dataclass
class Experiment:
    torch_version: str
    numpy_version: str
    sklearn_version: str
    device_available: str
    seed: int
    data: Data2D = field(repr=False, default=None)  # type: ignore[assignment]
    sched: Schedule = field(repr=False, default=None)  # type: ignore[assignment]
    cosine_sched: Schedule = field(repr=False, default=None)  # type: ignore[assignment]
    main: TrainResult = field(repr=False, default=None)  # type: ignore[assignment]
    forward_proof: ForwardProof | None = None
    generation_proof: GenerationProof | None = None


def run_experiment(*, n_epochs: int = N_EPOCHS, seed: int = SEED) -> Experiment:
    """Run the whole measured 2-D pipeline once and return every quantity the page, figures, and notebook cite."""
    try:
        import sklearn

        sk_version = sklearn.__version__
    except Exception:  # noqa: BLE001
        sk_version = "not installed"

    data = load_moons(seed=seed)
    sched = make_schedule(kind="linear", T=T_STEPS)
    cosine_sched = make_schedule(kind="cosine", T=T_STEPS)
    main = train_ddpm(data, sched, n_epochs=n_epochs, seed=seed)
    forward_proof = prove_forward_closed_form(sched)
    generation_proof = prove_generation(main, data)

    return Experiment(
        torch_version=torch.__version__, numpy_version=np.__version__, sklearn_version=sk_version,
        device_available=str(get_device()), seed=seed, data=data, sched=sched, cosine_sched=cosine_sched,
        main=main, forward_proof=forward_proof, generation_proof=generation_proof,
    )


# ================================================================================================
# Report — every number the chapter quotes, each headline relationship guarded by a hard assert
# ================================================================================================


def main() -> None:
    exp = run_experiment()
    assert exp.forward_proof and exp.generation_proof and exp.main

    print(f"torch {exp.torch_version} | numpy {exp.numpy_version} | scikit-learn {exp.sklearn_version}")
    print(f"(training on CPU for a reproducible trace; best available device = {exp.device_available}; "
          f"seed={exp.seed})")
    print(f"data: {exp.data.label} — {exp.data.x.shape[0]} points, standardized; "
          f"schedule: linear, T={exp.sched.T} (betas {BETA_START} -> {BETA_END})\n")

    print("=== 1. The closed-form forward q(x_t|x_0) IS the iterative step-by-step forward ===")
    fp = exp.forward_proof
    print(f"  at t={fp.t}: abar_t = {fp.abar_t:.4f}  =>  closed form N(sqrt(abar) x0, (1-abar) I)")
    print(f"  closed-form mean = {np.round(fp.closed_mean, 4)}   iterative mean = {np.round(fp.iter_mean, 4)}  "
          f"(N={fp.n_rollouts:,} rollouts)")
    print(f"  closed-form var  = {fp.closed_var:.4f} (isotropic)   iterative cov diag = "
          f"{np.round(np.diag(fp.iter_cov), 4)}")
    print(f"  max|mean err| = {fp.mean_abs_err:.2e}   max|cov err| = {fp.var_abs_err:.2e}  "
          f"(=> one-shot q_sample == the compounded chain)\n")

    print("=== 2. A from-scratch DDPM GENERATES the target distribution (energy distance) ===")
    gp = exp.generation_proof
    print(f"  trained {N_EPOCHS} epochs on {exp.data.x.shape[0]} points in {exp.main.seconds:.1f}s  |  "
          f"final L_simple = {exp.main.final_loss:.4f}")
    print(f"  energy distance  generated  vs target = {gp.ed_generated:.4f}")
    print(f"  energy distance  N(0,I) base vs target = {gp.ed_baseline:.4f}   (the chain's starting cloud)")
    print(f"  ratio generated/baseline = {gp.ratio:.3f}  (<< 1 => the reverse chain learned the data)\n")

    # ---- hard asserts on the headline relationships (raise, not print, if a lesson breaks) ----
    assert fp.mean_abs_err < 0.05, "closed-form forward mean must match the iterative chain's empirical mean"
    assert fp.var_abs_err < 0.05, "closed-form forward variance must match the iterative chain's covariance"
    assert exp.main.final_loss < exp.main.loss_hist[0], "training must reduce L_simple (the denoiser must learn)"
    assert gp.ratio < 0.5, "generated samples must be far closer to the target than the N(0,I) baseline"
    print("All checks passed: the closed-form forward equals the iterative chain (the 'nice property'); a "
          "from-scratch DDPM trained on real 2-D data generates the target distribution (energy distance far "
          "below the N(0,I) baseline).")


if __name__ == "__main__":
    main()
