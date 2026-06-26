"""From-scratch LLM evaluation: perplexity, pass@k, ECE, Elo, and an LLM-judge bias probe.

This is the ONE seeded source of truth for the chapter. The concept page, the teaching
notebook, and the figure generator all import the functions and constants below, so every
quoted number (perplexity, pass@k curve, ECE, final Elo ratings, bias flip-rate) is computed
in exactly one place and cannot silently drift between surfaces.

What it demonstrates, all from tiny hand-built inputs -- no model download, no GPU:
  * perplexity = exp(mean cross-entropy): a lower-CE model is a lower-perplexity model, and
    PPL == exp(CE) is proven by identity, not asserted by faith;
  * pass@k = 1 - C(n-c, k)/C(n, k) (the unbiased Chen-et-al. estimator): the chance at least
    one of k sampled completions passes, shown to rise monotonically with k -- AND why the
    naive "run k draws, average the indicator" estimator is BIASED low for the same (n, c);
  * ECE (expected calibration error): bin predictions by confidence, compare confidence to
    accuracy per bin -> a well-calibrated model has ECE ~ 0, an over-confident one does not
    (the gap measured, not asserted), the number behind every reliability diagram;
  * Elo from pairwise wins: ratings updated match-by-match from simulated games between agents
    of KNOWN true skill, shown to converge to the correct ORDERING;
  * LLM-judge position bias: a deterministic judge that mildly favours whichever answer it sees
    first -> swapping the order flips a measurable fraction of verdicts (bias measured, not
    asserted).

Mostly plain numpy + Python `math` -- the math here is device-independent. Where torch is used
(the perplexity cross-entropy) the reproducible trace in main() is pinned to CPU and prints the
device HONESTLY, matching the chapter's kv_cache.py / decoding_sampling.py exemplars, so the
printed device always matches where the numbers were actually produced.

Run:
    python llm_evaluation.py
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np
import torch
import torch.nn.functional as F

# ---- Perplexity demo: two toy models scored on the same held-out tokens ------------------
# A 6-word toy vocab. Two "models" assign a probability to the SAME held-out sequence; the
# better model puts more mass on the tokens that actually occur -> lower cross-entropy ->
# lower perplexity. Probabilities are over the vocab and sum to ~1 per row.
PPL_VOCAB: tuple[str, ...] = ("the", "cat", "sat", "on", "mat", "<eos>")

# The held-out sequence we score (token ids into PPL_VOCAB): "the cat sat on mat <eos>".
HELD_OUT_IDS: tuple[int, ...] = (0, 1, 2, 3, 4, 5)

# GOOD model: confidently right -- high probability on each token that actually occurs.
# Each row is the model's next-token distribution at that position (already conditioned).
GOOD_MODEL_PROBS: tuple[tuple[float, ...], ...] = (
    (0.70, 0.10, 0.05, 0.05, 0.05, 0.05),  # predicts "the"
    (0.05, 0.70, 0.10, 0.05, 0.05, 0.05),  # predicts "cat"
    (0.05, 0.10, 0.65, 0.10, 0.05, 0.05),  # predicts "sat"
    (0.05, 0.05, 0.10, 0.65, 0.10, 0.05),  # predicts "on"
    (0.05, 0.05, 0.05, 0.10, 0.70, 0.05),  # predicts "mat"
    (0.05, 0.05, 0.05, 0.05, 0.10, 0.70),  # predicts "<eos>"
)

# BAD model: nearly uniform -- it barely commits, so it wastes probability on wrong tokens
# and assigns less mass to the ones that actually occur. Higher CE, higher perplexity.
BAD_MODEL_PROBS: tuple[tuple[float, ...], ...] = (
    (0.25, 0.15, 0.15, 0.15, 0.15, 0.15),
    (0.15, 0.25, 0.15, 0.15, 0.15, 0.15),
    (0.15, 0.15, 0.25, 0.15, 0.15, 0.15),
    (0.15, 0.15, 0.15, 0.25, 0.15, 0.15),
    (0.15, 0.15, 0.15, 0.15, 0.25, 0.15),
    (0.15, 0.15, 0.15, 0.15, 0.15, 0.25),
)

# ---- pass@k demo: a code task where each sampled completion passes w.p. P_PASS -----------
PASS_N = 20          # number of completions sampled per problem (the budget n in pass@k)
PASS_C = 5           # number of those n that are correct (c correct out of n)
PASS_K_VALUES = (1, 2, 3, 5, 8, 10, 15, 20)  # the k's we sweep to draw the pass@k curve
PASS_NAIVE_TRIALS = 20000  # Monte-Carlo trials for the naive (biased) estimator comparison
PASS_NAIVE_SEED = 0        # reproducible naive-estimator sampling

# ---- ECE demo: synthetic (confidence, correct) data for two models -----------------------
ECE_N_BINS = 10        # number of equal-width confidence bins in [0, 1] (the standard 10-bin ECE)
ECE_N_SAMPLES = 6000   # synthetic predictions per model
ECE_SEED = 0           # reproducible confidences/outcomes
# An OVER-CONFIDENT model: it reports confidence p but is right only with probability
# (p - overconf_gap) -- so its accuracy lags its confidence in every bin. A CALIBRATED model
# is right with probability exactly p. The gap is what ECE puts a single number on.
ECE_OVERCONF_GAP = 0.15

# ---- Elo demo: four agents of KNOWN true skill, ranked from simulated pairwise games -----
ELO_K = 32           # Elo K-factor: how far a rating moves on a single surprising result
ELO_BASE = 1000.0    # every agent starts here -> ratings must SEPARATE from a flat start
ELO_SCALE = 400.0    # the 400-point convention: +400 rating = 10x expected odds of winning
ELO_MATCHES = 4000   # number of simulated pairwise games
ELO_SEED = 0         # one seed -> reproducible match schedule and outcomes

# True latent skills (NOT given to the Elo algorithm -- it only sees win/loss). The whole
# test is whether Elo RECOVERS this ordering A > B > C > D from outcomes alone.
TRUE_SKILL: dict[str, float] = {"A": 1600.0, "B": 1400.0, "C": 1200.0, "D": 1000.0}

# ---- LLM-judge position-bias demo --------------------------------------------------------
JUDGE_N_PAIRS = 500          # number of (answer_x, answer_y) pairs the judge rates
JUDGE_BIAS = 0.15            # the judge's thumb-on-the-scale for the FIRST answer it sees
JUDGE_SEED = 0               # reproducible pair qualities and verdicts

EPS = 1e-12                  # guards log(0) in the cross-entropy sum

# Run on the best available accelerator; CPU is the universal fallback (matches kv_cache.py).
DEVICE = (
    "cuda"
    if torch.cuda.is_available()
    else "mps"
    if torch.backends.mps.is_available()
    else "cpu"
)


# ==========================================================================================
# 1. Perplexity = exp(mean cross-entropy)
# ==========================================================================================
def cross_entropy_nats(model_probs: torch.Tensor, target_ids: tuple[int, ...]) -> float:
    """Mean per-token cross-entropy (in nats) of the held-out targets under the model.

    CE = -(1/N) * sum_t log p_model(target_t).  This is exactly the language-modeling loss:
    the average number of nats of 'surprise' the model assigns to each token that truly
    occurred. Lower = the model predicted the real text better.
    """
    chosen = model_probs[range(len(target_ids)), list(target_ids)].clamp_min(EPS)
    return float(-(torch.log(chosen).mean()).item())


def perplexity(model_probs: torch.Tensor, target_ids: tuple[int, ...]) -> float:
    """Perplexity = exp(cross-entropy in nats).

    Intuition: the effective number of equally-likely choices the model is hesitating
    between at each step. PPL = V means 'as confused as guessing uniformly among V words';
    PPL = 1 means perfect prediction. It is a monotone re-scaling of CE, so a lower-CE model
    is always a lower-perplexity model.
    """
    return math.exp(cross_entropy_nats(model_probs, target_ids))


# ==========================================================================================
# 2. pass@k -- the unbiased estimator from Chen et al. (2021)
# ==========================================================================================
def pass_at_k(n: int, c: int, k: int) -> float:
    """Unbiased pass@k: probability that at least one of k completions (drawn without
    replacement from n samples, c of them correct) passes the unit tests.

        pass@k = 1 - C(n-c, k) / C(n, k)

    The complement C(n-c, k)/C(n, k) is the chance ALL k drawn completions come from the
    (n-c) wrong ones -- i.e. you got unlucky and saw zero correct. One minus that is the
    chance you saw at least one correct. Naively averaging an indicator over k draws is a
    BIASED estimate of this; the combinatorial form is the unbiased fix (Chen et al. 2021).
    """
    assert 0 <= c <= n, "correct count c must be in [0, n]"
    assert 1 <= k <= n, "k must be in [1, n]"
    if n - c < k:
        # Fewer wrong answers than k -> any k draws MUST include a correct one -> pass@k = 1.
        return 1.0
    # 1 - prod_{i=0}^{k-1} (n-c-i)/(n-i)  is C(n-c,k)/C(n,k) computed without big factorials.
    fail = 1.0
    for i in range(k):
        fail *= (n - c - i) / (n - i)
    return 1.0 - fail


def pass_at_k_naive_mc(
    n: int, c: int, k: int, trials: int, seed: int
) -> float:
    """The BIASED naive estimator, by Monte-Carlo, so its bias is *measured* not asserted.

    A tempting-but-wrong recipe: draw k completions WITH replacement from the same n samples
    and report the fraction of trials in which at least one was correct. Sampling with
    replacement makes a repeated wrong answer 'count' as fresh failures, so it systematically
    UNDER-counts the chance of hitting a correct one -> it is biased LOW relative to the
    unbiased without-replacement estimator pass_at_k(n, c, k). We compute it by simulation to
    show the gap is real, not a rounding artefact.
    """
    assert 0 <= c <= n and 1 <= k <= n
    rng = np.random.default_rng(seed)
    correct_mask = np.zeros(n, dtype=bool)
    correct_mask[:c] = True  # first c of the n samples are the correct ones
    hits = 0
    for _ in range(trials):
        draw = rng.integers(0, n, size=k)  # WITH replacement -- the bias-inducing mistake
        if correct_mask[draw].any():
            hits += 1
    return hits / trials


# ==========================================================================================
# 3. Expected Calibration Error (ECE)
# ==========================================================================================
def expected_calibration_error(
    confidences: np.ndarray, correct: np.ndarray, n_bins: int = ECE_N_BINS
) -> tuple[float, list[dict[str, float]]]:
    """ECE: the gap between confidence and accuracy, averaged over confidence bins.

        ECE = sum_b (|B_b| / N) * | acc(B_b) - conf(B_b) |

    Partition the predictions into n_bins equal-width confidence bins. In each bin compare the
    mean confidence to the empirical accuracy; weight each bin's gap by how many predictions
    fall in it. A perfectly calibrated model has accuracy == confidence in every bin -> ECE 0.
    An over-confident model's accuracy sits below its confidence -> a positive ECE. Returns the
    scalar ECE plus per-bin stats (for the reliability diagram).
    """
    assert confidences.shape == correct.shape
    n_samples = len(confidences)
    edges = np.linspace(0.0, 1.0, n_bins + 1)
    # np.digitize assigns each confidence to a bin index in [1, n_bins]; clip the right edge
    # (confidence == 1.0) into the last bin so no prediction is dropped.
    bin_idx = np.clip(np.digitize(confidences, edges[1:-1], right=False), 0, n_bins - 1)
    ece = 0.0
    bins: list[dict[str, float]] = []
    for b, (lo, hi) in enumerate(zip(edges[:-1], edges[1:])):
        in_bin = bin_idx == b
        count = int(in_bin.sum())
        if count == 0:
            bins.append({"lo": lo, "hi": hi, "count": 0, "conf": 0.0, "acc": 0.0})
            continue
        conf = float(confidences[in_bin].mean())
        acc = float(correct[in_bin].mean())
        ece += (count / n_samples) * abs(acc - conf)
        bins.append({"lo": lo, "hi": hi, "count": count, "conf": conf, "acc": acc})
    return ece, bins


def make_calibration_data(
    n_samples: int, overconf_gap: float, seed: int
) -> tuple[np.ndarray, np.ndarray]:
    """Synthesize (confidence, correct) pairs whose accuracy lags confidence by overconf_gap.

    Each prediction gets a confidence drawn uniformly in [0.5, 1.0] (the model's stated
    probability of being right). It is then marked correct with probability max(conf - gap, 0):
    gap=0 gives a perfectly calibrated stream (accuracy == confidence), gap>0 gives a
    systematically over-confident one whose accuracy trails confidence -- the exact pattern ECE
    is designed to catch.
    """
    rng = np.random.default_rng(seed)
    confidences = rng.uniform(0.5, 1.0, size=n_samples)
    true_acc = np.clip(confidences - overconf_gap, 0.0, 1.0)
    correct = (rng.random(n_samples) < true_acc).astype(float)
    return confidences, correct


# ==========================================================================================
# 4. Elo from pairwise wins
# ==========================================================================================
def expected_score(rating_a: float, rating_b: float, scale: float = ELO_SCALE) -> float:
    """Elo's expected score for A vs B: a logistic in the rating gap.

        E_A = 1 / (1 + 10^((R_B - R_A)/scale))

    A 0-point gap -> 0.5 (a coin flip). A +400 gap (one 'scale' unit) -> ~0.91, i.e. ~10x
    odds. This logistic is the same Bradley-Terry model used for RLHF reward models -- Elo
    is online Bradley-Terry with a fixed step size.
    """
    return 1.0 / (1.0 + 10.0 ** ((rating_b - rating_a) / scale))


def elo_update(
    rating_a: float, rating_b: float, score_a: float, k: float = ELO_K
) -> tuple[float, float]:
    """One Elo update after a game. score_a is 1.0 if A won, 0.0 if A lost, 0.5 for a draw.

        R_A <- R_A + K (S_A - E_A);   R_B <- R_B + K (S_B - E_B)

    You gain points for winning more than expected and lose them for the reverse, so beating
    a strong opponent moves you far and beating a weak one barely moves you. The total rating
    in the pool is conserved (whatever A gains, B loses).
    """
    exp_a = expected_score(rating_a, rating_b)
    new_a = rating_a + k * (score_a - exp_a)
    new_b = rating_b + k * ((1.0 - score_a) - (1.0 - exp_a))
    return new_a, new_b


def simulate_elo(
    true_skill: dict[str, float],
    n_matches: int,
    seed: int,
    k: float = ELO_K,
    base: float = ELO_BASE,
) -> tuple[dict[str, float], list[dict[str, float]]]:
    """Simulate n_matches pairwise games between agents of KNOWN true skill and rate them
    with Elo from a flat start.

    The game outcome is sampled from the TRUE Bradley-Terry probability (so the data is
    honestly generated by the latent skills), but the Elo algorithm never sees those skills
    -- only the win/loss. Returns the final ratings and the per-step rating history (for the
    convergence figure). The test is whether the recovered RANKING matches the true one.
    """
    rng = np.random.default_rng(seed)
    names = list(true_skill)
    ratings = {name: base for name in names}  # flat start: every agent equal
    history: list[dict[str, float]] = [dict(ratings)]
    for _ in range(n_matches):
        a, b = rng.choice(names, size=2, replace=False)  # pick two distinct agents
        # TRUE probability A beats B, from the latent skills (the data-generating process).
        p_a_wins = expected_score(true_skill[a], true_skill[b])
        score_a = 1.0 if rng.random() < p_a_wins else 0.0
        ratings[a], ratings[b] = elo_update(ratings[a], ratings[b], score_a, k)
        history.append(dict(ratings))
    return ratings, history


def ranking(ratings: dict[str, float]) -> list[str]:
    """Agents sorted best-first by rating -- the ORDERING we compare against the true one."""
    return sorted(ratings, key=lambda name: ratings[name], reverse=True)


# ==========================================================================================
# 5. LLM-judge position bias
# ==========================================================================================
@dataclass(frozen=True)
class JudgeResult:
    """Outcome of the position-bias probe: how often the verdict flips when we swap order."""

    flip_rate: float          # fraction of pairs whose winner changed when order swapped
    first_pos_winrate: float  # win-rate of WHICHEVER answer was shown first (>0.5 = bias)


def position_biased_judge(
    quality_first: float, quality_second: float, bias: float
) -> int:
    """A toy 'LLM judge': returns 0 if it prefers the FIRST answer, 1 if the second.

    It compares true qualities but adds a fixed `bias` to whichever answer it sees first --
    exactly the position bias measured in real LLM judges (Zheng et al. 2023): the same two
    answers can swap winner purely by swapping their order.
    """
    return 0 if (quality_first + bias) >= quality_second else 1


def measure_position_bias(
    n_pairs: int, bias: float, seed: int
) -> JudgeResult:
    """Rate n_pairs of answers twice -- original order and swapped -- and measure how often
    the winner flips, plus the win-rate of the first-shown answer.

    With bias=0 the judge is order-invariant (flip_rate=0, first-position win-rate=0.5). A
    positive bias makes 'shown first' an advantage, so some near-tie pairs flip and the
    first-position win-rate rises above 0.5 -- the bias made visible as a number.
    """
    rng = np.random.default_rng(seed)
    flips = 0
    first_pos_wins = 0
    for _ in range(n_pairs):
        qx, qy = rng.random(), rng.random()  # two answers of independent random quality
        # Order 1: X first, Y second. Winner is the ANSWER (x or y), not the slot.
        winner_orig = "x" if position_biased_judge(qx, qy, bias) == 0 else "y"
        # Order 2: Y first, X second. Same two answers, order swapped.
        winner_swap = "y" if position_biased_judge(qy, qx, bias) == 0 else "x"
        if winner_orig != winner_swap:
            flips += 1
        # In each ordering, did the first-shown answer win? (slot-0 winner in both runs)
        first_pos_wins += int(position_biased_judge(qx, qy, bias) == 0)
        first_pos_wins += int(position_biased_judge(qy, qx, bias) == 0)
    return JudgeResult(
        flip_rate=flips / n_pairs,
        first_pos_winrate=first_pos_wins / (2 * n_pairs),
    )


def main() -> None:
    # Pin the reproducible trace to CPU and print the device HONESTLY: the printed device must
    # match where the math actually ran (the recurring device-honesty requirement). Most of
    # this demo is numpy/pure-Python; only the perplexity CE uses torch.
    trace_device = "cpu"
    print(f"device: {trace_device} (detected {DEVICE}; pinned to CPU for reproducibility)")
    print("torch:", torch.__version__, "| numpy:", np.__version__)
    print()

    # --- 1. Perplexity = exp(CE), and the better model has the lower PPL ------------------
    good = torch.tensor(GOOD_MODEL_PROBS, device=trace_device)
    bad = torch.tensor(BAD_MODEL_PROBS, device=trace_device)
    ce_good = cross_entropy_nats(good, HELD_OUT_IDS)
    ce_bad = cross_entropy_nats(bad, HELD_OUT_IDS)
    ppl_good = perplexity(good, HELD_OUT_IDS)
    ppl_bad = perplexity(bad, HELD_OUT_IDS)
    print("[perplexity] scoring the held-out sequence "
          f"'{' '.join(PPL_VOCAB[i] for i in HELD_OUT_IDS)}':")
    print(f"   GOOD model: CE={ce_good:.4f} nats   PPL={ppl_good:.3f}")
    print(f"   BAD  model: CE={ce_bad:.4f} nats   PPL={ppl_bad:.3f}")
    # Identity check: perplexity is EXACTLY exp(cross-entropy) -- proven, not asserted.
    assert abs(ppl_good - math.exp(ce_good)) < 1e-9, "PPL must equal exp(CE) by definition"
    assert abs(ppl_bad - math.exp(ce_bad)) < 1e-9, "PPL must equal exp(CE) by definition"
    # The better (lower-CE) model must have the lower perplexity.
    assert ppl_good < ppl_bad, "the confident-correct model must have lower perplexity"
    print(f"   => PPL == exp(CE) exactly; better model lower PPL "
          f"({ppl_good:.3f} < {ppl_bad:.3f}).\n")

    # --- 2. pass@k rises monotonically with k --------------------------------------------
    print(f"[pass@k]  n={PASS_N} samples, c={PASS_C} correct -> chance >=1 of k passes:")
    curve = [pass_at_k(PASS_N, PASS_C, k) for k in PASS_K_VALUES]
    for k, val in zip(PASS_K_VALUES, curve):
        bar = "#" * int(round(val * 40))
        print(f"   pass@{k:>2} = {val:.3f}  {bar}")
    # The whole point of reporting pass@k for several k: more attempts can only help.
    assert all(b >= a - 1e-12 for a, b in zip(curve, curve[1:])), (
        "pass@k must be monotonically non-decreasing in k"
    )
    assert abs(curve[0] - PASS_C / PASS_N) < 1e-9, "pass@1 must equal c/n (one draw)"
    assert abs(curve[-1] - 1.0) < 1e-9, "pass@n must be 1.0 (drawing all samples)"
    print(f"   => pass@1 = c/n = {curve[0]:.3f}; rises to pass@{PASS_N} = {curve[-1]:.3f}; "
          f"monotone in k.")
    # The naive with-replacement estimator is biased LOW -- show the gap at a mid k.
    k_demo = 5
    unbiased_k = pass_at_k(PASS_N, PASS_C, k_demo)
    naive_k = pass_at_k_naive_mc(PASS_N, PASS_C, k_demo, PASS_NAIVE_TRIALS, PASS_NAIVE_SEED)
    assert naive_k < unbiased_k - 1e-3, "naive with-replacement estimator must be biased low"
    print(f"   naive (with-replacement) pass@{k_demo} = {naive_k:.3f}  <  "
          f"unbiased pass@{k_demo} = {unbiased_k:.3f}  (the bias, measured).\n")

    # --- 3. ECE: a calibrated model has ECE ~ 0; an over-confident one does not ------------
    conf_cal, corr_cal = make_calibration_data(ECE_N_SAMPLES, 0.0, ECE_SEED)
    conf_over, corr_over = make_calibration_data(ECE_N_SAMPLES, ECE_OVERCONF_GAP, ECE_SEED)
    ece_cal, _ = expected_calibration_error(conf_cal, corr_cal)
    ece_over, _ = expected_calibration_error(conf_over, corr_over)
    print(f"[ece]  {ECE_N_SAMPLES} synthetic predictions, {ECE_N_BINS} confidence bins:")
    print(f"   calibrated model   (gap 0.00): ECE = {ece_cal:.4f}   "
          f"acc {corr_cal.mean():.3f} vs mean-conf {conf_cal.mean():.3f}")
    print(f"   over-confident     (gap {ECE_OVERCONF_GAP:.2f}): ECE = {ece_over:.4f}   "
          f"acc {corr_over.mean():.3f} vs mean-conf {conf_over.mean():.3f}")
    # A calibrated stream has tiny ECE (only sampling noise); the over-confident one is large.
    assert ece_cal < 0.03, "calibrated data must have near-zero ECE"
    assert ece_over > 0.10, "over-confident data must have a clearly positive ECE"
    assert ece_over > ece_cal, "over-confident ECE must exceed calibrated ECE"
    print(f"   => over-confidence lifts ECE {ece_cal:.4f} -> {ece_over:.4f}; "
          f"its accuracy trails its confidence by ~{ECE_OVERCONF_GAP:.2f}.\n")

    # --- 4. Elo recovers the TRUE skill ordering from win/loss alone ----------------------
    final_ratings, _history = simulate_elo(TRUE_SKILL, ELO_MATCHES, ELO_SEED)
    recovered = ranking(final_ratings)
    true_order = ranking(TRUE_SKILL)
    print(f"[elo]  {ELO_MATCHES} simulated games, flat {ELO_BASE:.0f} start, K={ELO_K:.0f}:")
    for name in recovered:
        print(f"   {name}: Elo {final_ratings[name]:7.1f}   (true skill {TRUE_SKILL[name]:.0f})")
    print(f"   recovered ranking: {' > '.join(recovered)}")
    print(f"   true      ranking: {' > '.join(true_order)}")
    # THE key result: Elo, seeing only outcomes, recovers the correct ORDERING of skills.
    assert recovered == true_order, "Elo must recover the true skill ordering from outcomes"
    print("   => Elo recovered the exact true ordering from win/loss alone.\n")

    # --- 5. LLM-judge position bias: swapping order flips verdicts ------------------------
    unbiased = measure_position_bias(JUDGE_N_PAIRS, bias=0.0, seed=JUDGE_SEED)
    biased = measure_position_bias(JUDGE_N_PAIRS, bias=JUDGE_BIAS, seed=JUDGE_SEED)
    print(f"[llm-judge] {JUDGE_N_PAIRS} answer pairs, rated in both orders:")
    print(f"   unbiased judge (bias=0.0):   flip-rate {unbiased.flip_rate:.3f}   "
          f"first-position win-rate {unbiased.first_pos_winrate:.3f}")
    print(f"   biased   judge (bias={JUDGE_BIAS}):  flip-rate {biased.flip_rate:.3f}   "
          f"first-position win-rate {biased.first_pos_winrate:.3f}")
    # A fair judge is order-invariant: no flips, first position wins exactly half the time.
    assert unbiased.flip_rate == 0.0, "an unbiased judge must never flip on order swap"
    assert abs(unbiased.first_pos_winrate - 0.5) < 1e-9, "fair judge: 50% first-position wins"
    # A biased judge flips some near-ties and favours the first slot -- bias measured, not asserted.
    assert biased.flip_rate > 0.0, "position bias must produce order-dependent flips"
    assert biased.first_pos_winrate > 0.5, "position bias must favour the first-shown answer"
    print(f"   => position bias flips {biased.flip_rate:.1%} of verdicts and lifts the "
          f"first-position win-rate to {biased.first_pos_winrate:.1%}.")


if __name__ == "__main__":
    main()
