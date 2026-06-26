"""Scaling laws, made visible from scratch: fit a power law, find the compute-optimal split, do the 6ND math.

Four self-contained demos, each the small-scale version of what a frontier lab does before a
nine-figure run:

  1. FIT THE POWER LAW. Generate synthetic loss-vs-N points from a KNOWN law L = E + A/N^alpha,
     then recover alpha by a log-log linear fit -- teaching "a power law is a straight line on
     log-log axes." We assert the recovered exponent matches the ground truth.
  2. THE COMPUTE-OPTIMAL FRONTIER. For a fixed compute budget C, sweep model size N, set the data
     D = C/(6N) so 6ND = C holds, evaluate the parametric loss L(N, D), and find the N* that
     minimizes it -- the U-shaped iso-compute curve. We print the implied tokens/param.
  3. THE 6ND CALCULATOR. Given N and D, training compute C = 6ND. Worked for GPT-3 and for
     Chinchilla (70B x 1.4T), with the asserts that pin the arithmetic.
  4. EXTRAPOLATE THE LOSS. Fit the compute power law L(C) = E + (Cc/C)^alpha_C to small runs,
     then predict the loss at 10x and 100x the budget -- the payoff that lets a lab commit to a
     run it hasn't done. We assert each 10x of compute multiplies the reducible term by 10^-alpha_C.

These are the SAME numbers embedded in the concept page and the teaching notebook: the page's
"Output" block is this file's stdout, pasted verbatim, so page == script == notebook.

Verified on Python 3.12 / torch 2.x. Device-agnostic (CUDA / MPS / CPU). The whole computation is
tiny and exact arithmetic, so the numbers are identical on any device; the trace pins to CPU
(TRACE_DEVICE) only so the printed run is byte-for-byte reproducible regardless of the machine.

Run:
    python scaling_laws.py
"""

from __future__ import annotations

import numpy as np
import torch

# --- C = 6ND : the FLOP-counting constants (see the page's "deriving C = 6ND") ---------------
FLOPS_PER_PARAM_FWD = 2  # one multiply + one add per weight per token (a multiply-accumulate)
FLOPS_PER_PARAM_BWD = 4  # backward computes input-grads AND weight-grads ~= 2x the forward
FLOPS_PER_PARAM = FLOPS_PER_PARAM_FWD + FLOPS_PER_PARAM_BWD  # = 6: the factor in C = 6ND
FLOPS_PER_SECOND_1PF = 1e15  # 1 petaFLOP/s, the unit for "petaFLOP-days"
SECONDS_PER_DAY = 86_400

# --- Chinchilla parametric fit L(N,D) = E + A/N^alpha + B/D^beta (Hoffmann et al. 2022, Table A3)
CHINCHILLA_E = 1.69  # irreducible loss: the entropy floor of the data, in nats/token
CHINCHILLA_A = 406.4  # coefficient on the model-size penalty term A/N^alpha
CHINCHILLA_B = 410.7  # coefficient on the data penalty term B/D^beta
CHINCHILLA_ALPHA = 0.34  # model-size exponent
CHINCHILLA_BETA = 0.28  # data exponent
TOKENS_PER_PARAM_RULE = 20  # the canonical Chinchilla rule: D* ~= 20 * N*

# --- compute-optimal sweep grid: span 10^7..10^13 params, fine enough to read the minimum -----
SWEEP_LOG10_N_MIN = 7
SWEEP_LOG10_N_MAX = 13
SWEEP_POINTS = 200_000

# --- power-law fit demo: a synthetic "measured" ladder with a KNOWN exponent to recover --------
LADDER_NS = (1e4, 3e4, 1e5, 3e5, 1e6, 3e6)  # model sizes, spanning >2 orders of magnitude
TRUE_E = 1.55  # ground-truth irreducible floor used to GENERATE the synthetic losses
TRUE_A = 18.0  # ground-truth coefficient
TRUE_ALPHA = 0.30  # ground-truth exponent -- the number the fit must recover
MEASUREMENT_NOISE_STD = 0.01  # small Gaussian noise so the points look "measured", not perfect
FIT_SEED = 0  # fixed seed -> the synthetic ladder (and the recovered exponent) is deterministic
ALPHA_FIT_TOL = 0.02  # recovered exponent must land within this of TRUE_ALPHA

# --- reference models, for the 6ND calculator and the undertraining story ---------------------
GPT3_N, GPT3_D = 175e9, 300e9  # GPT-3: 175B params, 300B tokens -> ~1.7 tokens/param (undertrained)
CHINCHILLA_N, CHINCHILLA_D = 70e9, 1.4e12  # Chinchilla: 70B params, 1.4T tokens -> ~20 tokens/param
GPT3_EXPECTED_C = 3.15e23  # the by-hand answer; asserted exact below
CHINCHILLA_EXPECTED_C = 5.88e23  # 6 * 70e9 * 1.4e12 = 5.88e23; asserted exact below

# --- loss-extrapolation demo: the compute power law L(C) = E + (Cc/C)^alpha_C ------------------
EXTRAP_E = 1.69  # irreducible floor (same entropy floor as the parametric fit)
EXTRAP_CC = 3.0e8  # fitted scale constant Cc (arbitrary units matching the budget axis)
EXTRAP_ALPHA_C = 0.057  # Kaplan's directly-fitted compute exponent (the small one that bites)
EXTRAP_C0 = 1e21  # the base budget; we predict the loss at 1x, 10x, 100x this
EXTRAP_FACTORS = (1, 10, 100)

# Run on the best available accelerator; CPU is the universal fallback.
DEVICE = (
    "cuda"
    if torch.cuda.is_available()
    else "mps"
    if torch.backends.mps.is_available()
    else "cpu"
)
# Pin the trace to CPU so the printed numbers are identical on every machine (the math is exact;
# this only removes any device-dependent float ordering from the reported run).
TRACE_DEVICE = "cpu"


def training_flops(n_params: float, n_tokens: float) -> float:
    """Training compute in FLOPs via C = 6ND.

    n_params: model parameter count N. n_tokens: training token count D.
    Returns C: forward (2N) + backward (4N) FLOPs per token, summed over D tokens.
    """
    return FLOPS_PER_PARAM * n_params * n_tokens  # 6 * N * D


def parametric_loss(
    n_params: torch.Tensor, n_tokens: torch.Tensor
) -> torch.Tensor:
    """Chinchilla parametric loss L(N,D) = E + A/N^alpha + B/D^beta, in nats/token.

    n_params, n_tokens: tensors of N and D (broadcastable). Returns L of the same shape:
    the irreducible floor E plus a model-size penalty and a data penalty.
    """
    return (
        CHINCHILLA_E
        + CHINCHILLA_A / n_params**CHINCHILLA_ALPHA
        + CHINCHILLA_B / n_tokens**CHINCHILLA_BETA
    )


def compute_optimal_split(
    budget_flops: float, device: str = TRACE_DEVICE
) -> tuple[float, float, float]:
    """Find the (N*, D*) on the iso-compute curve 6ND = C that minimizes the parametric loss.

    budget_flops: the fixed compute budget C. Returns (N*, D*, L*): sweep N across the grid,
    set D = C/(6N) to enforce the constraint, and pick the argmin of L(N, D) -- the bottom of
    the U-shaped iso-compute curve.
    """
    n_grid = torch.logspace(
        SWEEP_LOG10_N_MIN, SWEEP_LOG10_N_MAX, SWEEP_POINTS, device=device, dtype=torch.float64
    )
    d_grid = budget_flops / (FLOPS_PER_PARAM * n_grid)  # D = C/(6N): slide along the iso-compute line
    losses = parametric_loss(n_grid, d_grid)
    best = int(torch.argmin(losses))  # the valley floor: too-small N underfits, too-big N starves on data
    return float(n_grid[best]), float(d_grid[best]), float(losses[best])


def iso_compute_curve(
    budget_flops: float, n_samples: int = 7, device: str = TRACE_DEVICE
) -> list[tuple[float, float, float]]:
    """Sample the U-shaped loss-vs-N curve at one fixed budget, so the shape is readable in text.

    Returns a list of (N, D, L) rows spanning the sweep range -- print them and the U is visible:
    loss falls, hits a minimum near N*, then rises again.
    """
    n_grid = torch.logspace(
        SWEEP_LOG10_N_MIN, SWEEP_LOG10_N_MAX, n_samples, device=device, dtype=torch.float64
    )
    d_grid = budget_flops / (FLOPS_PER_PARAM * n_grid)
    losses = parametric_loss(n_grid, d_grid)
    return [(float(n), float(d), float(loss)) for n, d, loss in zip(n_grid, d_grid, losses)]


def make_synthetic_ladder(
    device: str = TRACE_DEVICE,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Generate a synthetic 'measured' loss ladder from the KNOWN law L = E + A/N^alpha + noise.

    Returns (Ns, losses): the model sizes and the noisy losses we will fit, with ground-truth
    exponent TRUE_ALPHA baked in for the fit to recover. Uses numpy's default_rng(FIT_SEED) for
    the noise so the ladder is byte-for-byte identical to the concept page's printed table.
    """
    rng = np.random.default_rng(FIT_SEED)  # deterministic noise -> reproducible recovered exponent
    n_array = np.array(LADDER_NS)
    clean = TRUE_E + TRUE_A / n_array**TRUE_ALPHA  # the true power law
    losses = clean + rng.normal(0, MEASUREMENT_NOISE_STD, n_array.shape)  # "measured" = true + noise
    n_sizes = torch.tensor(n_array, device=device, dtype=torch.float64)
    return n_sizes, torch.tensor(losses, device=device, dtype=torch.float64)


def fit_power_law_exponent(
    n_sizes: torch.Tensor, losses: torch.Tensor, irreducible: float
) -> float:
    """Recover the exponent alpha by a log-log linear fit of the REDUCIBLE loss vs N.

    Subtract the known floor to isolate the reducible part (L - E = A/N^alpha), take logs of
    both axes -> log(L - E) = log(A) - alpha * log(N): a straight line whose slope is -alpha.
    A degree-1 polyfit gives that slope; we return -slope = the recovered exponent.
    """
    log_n = torch.log(n_sizes)
    log_reducible = torch.log(losses - irreducible)  # isolate A/N^alpha before taking the log
    # least-squares slope of a line y = m*x + c, in closed form (no SciPy needed):
    x_centered = log_n - log_n.mean()
    y_centered = log_reducible - log_reducible.mean()
    slope = float((x_centered * y_centered).sum() / (x_centered**2).sum())
    return -slope  # log-log slope is -alpha, so alpha = -slope


def demo_fit_power_law() -> None:
    """Demo 1: recover a known power-law exponent from synthetic 'measured' losses."""
    print("[1] Fit a power law: recover the exponent from a measured ladder")
    n_sizes, losses = make_synthetic_ladder()
    recovered = fit_power_law_exponent(n_sizes, losses, TRUE_E)
    for n, loss in zip(n_sizes.tolist(), losses.tolist()):
        print(f"    N={n:>10.0f}  loss={loss:.4f}")
    print(f"    recovered alpha_N = {recovered:.3f}   (true {TRUE_ALPHA})")
    assert abs(recovered - TRUE_ALPHA) < ALPHA_FIT_TOL, (
        f"recovered exponent {recovered:.3f} not within {ALPHA_FIT_TOL} of {TRUE_ALPHA}"
    )
    print()


def demo_compute_optimal() -> None:
    """Demo 2: the U-shaped iso-compute curve and the compute-optimal split (~20 tokens/param)."""
    print("[2] Compute-optimal frontier: sweep N along 6ND = C, find the U-curve minimum")
    budget = 1e21  # a fixed compute budget, in FLOPs
    print(f"    budget C = {budget:.0e} FLOPs   (the iso-compute curve below; loss is U-shaped)")
    print(f"    {'N (params)':>12} | {'D (tokens)':>12} | {'L(N,D)':>8}")
    print("    " + "-" * 38)
    n_star, _, _ = compute_optimal_split(budget)
    for n, d, loss in iso_compute_curve(budget):
        mark = "  <- near N*" if abs(n - n_star) / n_star < 1.0 else ""
        print(f"    {n:>12.2e} | {d:>12.2e} | {loss:>8.4f}{mark}")
    n_star, d_star, loss_star = compute_optimal_split(budget)
    print(
        f"    optimum: N*={n_star:.2e}, D*={d_star:.2e}, "
        f"L*={loss_star:.4f}, tokens/param={d_star / n_star:.0f}"
    )
    # the parametric (0.34/0.28) optimum runs a bit above 20; the equal-exponent RULE gives ~20:
    n_rule = (budget / (FLOPS_PER_PARAM * TOKENS_PER_PARAM_RULE)) ** 0.5  # C = 6N(20N) = 120 N^2
    print(
        f"    20:1 rule: N*={n_rule:.2e} ({n_rule / 1e9:.2f}B), "
        f"D*={TOKENS_PER_PARAM_RULE * n_rule:.2e} ({TOKENS_PER_PARAM_RULE * n_rule / 1e9:.0f}B), "
        f"tokens/param={TOKENS_PER_PARAM_RULE}"
    )
    # the optimum is interior: strictly between the grid ends (a real minimum, not a boundary).
    assert 10**SWEEP_LOG10_N_MIN < n_star < 10**SWEEP_LOG10_N_MAX, "optimum hit a grid boundary"
    # and the rule's tokens/param is the canonical 20 by construction.
    assert abs((TOKENS_PER_PARAM_RULE * n_rule) / n_rule - TOKENS_PER_PARAM_RULE) < 1e-9
    print()


def demo_6nd_calculator() -> None:
    """Demo 3: the C = 6ND calculator, worked for GPT-3 and Chinchilla, with exact asserts."""
    print("[3] The 6ND calculator: training compute from N and D")
    for name, n, d in (
        ("GPT-3", GPT3_N, GPT3_D),
        ("Chinchilla", CHINCHILLA_N, CHINCHILLA_D),
    ):
        c = training_flops(n, d)
        pf_days = c / (FLOPS_PER_SECOND_1PF * SECONDS_PER_DAY)
        print(
            f"    {name:>11}: N={n:.0e}, D={d:.1e} -> C=6ND={c:.3e} FLOPs "
            f"({pf_days:.0f} PF-days), tokens/param={d / n:.1f}"
        )
    # exact arithmetic checks: the hand-computed answers must match to the quoted precision.
    assert abs(training_flops(GPT3_N, GPT3_D) - GPT3_EXPECTED_C) < 1e18, "GPT-3 6ND mismatch"
    assert abs(training_flops(CHINCHILLA_N, CHINCHILLA_D) - CHINCHILLA_EXPECTED_C) < 1e18, (
        "Chinchilla 6ND mismatch"
    )
    # the undertraining story, asserted: GPT-3 is far below 20:1, Chinchilla sits at ~20:1.
    assert GPT3_D / GPT3_N < 2, "GPT-3 should be ~1.7 tokens/param (undertrained)"
    assert abs(CHINCHILLA_D / CHINCHILLA_N - TOKENS_PER_PARAM_RULE) < 1, "Chinchilla should be ~20:1"
    print()


def predicted_loss(budget_flops: float) -> float:
    """Compute power law L(C) = E + (Cc/C)^alpha_C: the loss along the compute-optimal frontier.

    budget_flops: the compute C. Returns the predicted loss -- the irreducible floor plus a
    reducible term that decays as C^(-alpha_C).
    """
    return EXTRAP_E + (EXTRAP_CC / budget_flops) ** EXTRAP_ALPHA_C


def demo_extrapolate() -> None:
    """Demo 4: predict the loss at 10x and 100x compute from the fitted compute power law."""
    print("[4] Extrapolate the loss: predict 10x and 100x compute from the power law")
    base = predicted_loss(EXTRAP_C0)
    for factor in EXTRAP_FACTORS:
        loss = predicted_loss(factor * EXTRAP_C0)
        print(f"    {factor:4d}x compute -> L = {loss:.4f}")
    # each 10x of compute multiplies the REDUCIBLE term by exactly 10^-alpha_C (a fixed small slice).
    reducible_base = base - EXTRAP_E
    reducible_10x = predicted_loss(10 * EXTRAP_C0) - EXTRAP_E
    ratio = reducible_10x / reducible_base
    print(f"    10x shrinks the reducible term by 10^-alpha_C = {10 ** -EXTRAP_ALPHA_C:.4f}")
    assert abs(ratio - 10 ** -EXTRAP_ALPHA_C) < 1e-9, "10x should scale the reducible term by 10^-alpha_C"
    print()


def main() -> None:
    print(f"device: {DEVICE} (trace pinned to {TRACE_DEVICE})")
    print("torch:", torch.__version__)
    print()
    demo_fit_power_law()
    demo_compute_optimal()
    demo_6nd_calculator()
    demo_extrapolate()
    print(
        "all asserts passed: recovered exponent, interior optimum, 6ND arithmetic, "
        "and the loss extrapolation all hold."
    )


if __name__ == "__main__":
    main()
