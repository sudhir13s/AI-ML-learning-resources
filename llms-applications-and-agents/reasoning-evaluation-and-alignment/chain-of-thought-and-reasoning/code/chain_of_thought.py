"""Chain-of-Thought, measured from scratch: why steps help, and why voting helps more.

Two fully self-contained, deterministic demonstrations -- no LLM, no GPU, just numpy
modelling the *structure* of multi-step reasoning so the mechanism is visible:

  1. self_consistency_sweep()  -- model a reasoner as a noisy K-way classifier whose single
     best guess is only moderately reliable, but whose ERRORS are scattered while its
     CORRECT answer is the single most common one. Sample K independent chains and take the
     MAJORITY vote. Accuracy rises with K and beats a single chain -- the Wang et al. (2022)
     self-consistency effect, reproduced from first principles.

  2. direct_vs_cot()  -- a small symbolic multi-step task (chained modular arithmetic). A
     "direct" solver collapses the whole computation in one noisy shot; a "CoT" solver emits
     one intermediate result per step, carrying state forward. Emitting the intermediate
     state turns one hard joint guess into several easy local ones, so CoT accuracy beats
     direct accuracy -- the decomposition mechanism behind chain-of-thought.

DEVICE NOTE: this demo is pure numpy (CPU). There are no tensors and no accelerator to
detect; the device line printed by main() reports the torch-detected accelerator honestly
and states that the numerics are pinned to CPU numpy for bit-for-bit reproducibility. Every
number printed here is the SAME number quoted on the page, in the notebook, and in the
figures (make_figures_17.py imports these functions -- one seeded source of truth).

Run:
    python chain_of_thought.py
"""

from __future__ import annotations

import numpy as np

# ---- Self-consistency model parameters ----------------------------------------------
# A K-way reasoning problem. ONE chain lands on the correct answer with prob P_CORRECT;
# the remaining mass is spread over the (N_ANSWERS - 1) wrong answers. Crucially the
# correct answer is the single most likely *individual* outcome even when P_CORRECT < 0.5,
# so as we sample more chains the majority vote concentrates on it (Condorcet-style).
N_ANSWERS = 5  # distinct candidate final answers a chain can produce
P_CORRECT = 0.45  # prob a single sampled chain reaches the correct answer (< 0.5 on purpose, but
# still the single most likely individual outcome -- so the plurality vote concentrates on it)
SC_K_VALUES = (1, 3, 5, 7, 11, 21, 41)  # odd K avoids vote ties; sweep shows the rising curve
SC_TRIALS = 60_000  # Monte-Carlo trials per K -- large enough for ~0.002 standard error (clean curve)
SC_SEED = 0

# ---- Direct-vs-CoT compositional task parameters ------------------------------------
# A chain of N_STEPS modular updates: state_0 = x0; state_{i+1} = (state_i * a_i + b_i) mod M.
# The final answer is state_{N_STEPS}. Both solvers must, in effect, perform every internal
# operation correctly to land the final residue -- but they differ in HOW RELIABLY a single
# operation is performed, and that single difference is the whole point of CoT:
#
#   * DIRECT (blurt the answer): the model juggles all N_STEPS operations at once in its head
#     with no scratch space. Each internal op survives with the LOWER prob P_STEP_DIRECT, and
#     these survivals are a joint event (one slip with nothing written down derails the rest,
#     leaving a near-random residue). Reliability compounds: ~P_STEP_DIRECT ** N_STEPS.
#   * COT (show your work): writing each intermediate down turns each step into an ISOLATED,
#     easier sub-problem the model solves with the HIGHER prob P_STEP_COT. The final answer is
#     right iff every emitted step is right, so reliability is ~P_STEP_COT ** N_STEPS -- but
#     from a much higher base, so the product stays far larger.
#
# The asymmetry P_STEP_COT > P_STEP_DIRECT is the documented decomposition effect: a step done
# in isolation, with its inputs written down, is simply easier than the same step done as part
# of a single all-at-once guess. A slip with no anchor leaves a near-random guess over M residues.
N_STEPS = 5  # number of chained operations -- multi-step is where CoT earns its keep
MODULUS = 7  # keep the state space small so "guessing" has a real chance, sharpening the gap
P_STEP_DIRECT = 0.55  # per-op reliability with NO scratch space (juggling all steps at once)
P_STEP_COT = 0.90  # per-op reliability when the step is isolated and its inputs are written down
COT_TRIALS = 20_000
COT_SEED = 1


def majority_vote(samples: np.ndarray) -> np.ndarray:
    """Return the modal value along axis=1 (ties broken by lowest index via argmax on counts).

    samples: (trials, K) integer answers. Returns (trials,) the majority answer per trial.
    """
    trials = samples.shape[0]
    out = np.empty(trials, dtype=samples.dtype)
    for i in range(trials):
        counts = np.bincount(samples[i], minlength=N_ANSWERS)  # tally votes across the K chains
        out[i] = int(counts.argmax())  # the most-voted answer; argmax breaks ties toward lowest index
    return out


def sample_chains(rng: np.random.Generator, k: int, trials: int) -> np.ndarray:
    """Draw `trials` problems, each with `k` independent chains.

    Answer 0 is defined as the CORRECT answer. A chain returns 0 with prob P_CORRECT,
    else a uniformly-random WRONG answer in 1..N_ANSWERS-1. Returns (trials, k) ints.
    """
    correct = rng.random((trials, k)) < P_CORRECT  # True where the chain got it right
    wrong = rng.integers(1, N_ANSWERS, size=(trials, k))  # a wrong answer in 1..N_ANSWERS-1
    return np.where(correct, 0, wrong)  # correct chains -> answer 0, else a random wrong answer


def self_consistency_sweep(seed: int = SC_SEED) -> dict[int, float]:
    """Majority-vote accuracy as a function of the number of sampled chains K.

    Returns {K: accuracy}. accuracy[1] is single-chain accuracy (== P_CORRECT in expectation);
    larger K rises above it as the vote concentrates on the correct answer. Each K uses a
    fresh seeded rng (seed + K) so the curve reflects K alone, not a shared random walk --
    this makes the rise clean and monotonic up to Monte-Carlo noise.
    """
    acc: dict[int, float] = {}
    for k in SC_K_VALUES:
        rng = np.random.default_rng(seed + k)  # fresh stream per K -> curve isolates the effect of K
        samples = sample_chains(rng, k, SC_TRIALS)  # (trials, k) sampled chain answers
        voted = majority_vote(samples)  # (trials,) the per-problem majority answer
        acc[k] = float((voted == 0).mean())  # fraction of problems where the vote is correct (answer 0)
    return acc


def run_chain_direct(rng: np.random.Generator, params: np.ndarray, x0: int) -> int:
    """DIRECT solve: blurt the answer with no scratch space.

    Each internal op survives with the LOW prob P_STEP_DIRECT (the model is juggling all
    N_STEPS at once). If every op survives, the final residue is correct; the first slip has
    nothing written down to anchor the rest, so the answer collapses to a near-random residue
    over the M outcomes. Net reliability compounds as ~P_STEP_DIRECT ** N_STEPS.
    """
    state = x0
    all_ok = True
    for a, b in params:  # walk the true computation; each op may silently fail
        state = (state * int(a) + int(b)) % MODULUS  # the correct local update
        if rng.random() >= P_STEP_DIRECT:  # this operation slipped (low reliability, no scratch space)
            all_ok = False
    if all_ok:
        return state  # every op survived -> correct final residue
    return int(rng.integers(0, MODULUS))  # a slip with nothing to anchor on -> near-random residue


def run_chain_cot(rng: np.random.Generator, params: np.ndarray, x0: int) -> int:
    """CoT solve: show your work -- each isolated step is easier, so per-step reliability rises.

    Writing each intermediate down turns each step into a separate, easier sub-problem solved
    with the HIGHER prob P_STEP_COT. The emitted intermediate anchors the next step, so the
    model keeps computing from whatever it wrote; a slip emits a wrong residue but does not
    blow up the whole chain. The final answer is right iff every emitted step is right --
    reliability ~P_STEP_COT ** N_STEPS, from a much higher base than the direct solver.
    """
    state = x0
    for a, b in params:
        true_next = (state * int(a) + int(b)) % MODULUS  # what this isolated step SHOULD emit
        if rng.random() < P_STEP_COT:  # higher reliability: the step is isolated, inputs written down
            state = true_next  # step performed correctly -> emit the true intermediate
        else:
            state = int(rng.integers(0, MODULUS))  # local slip: emit a wrong residue, continue from it
    return state


def direct_vs_cot(seed: int = COT_SEED) -> tuple[float, float]:
    """Return (direct_accuracy, cot_accuracy) over COT_TRIALS random multi-step problems."""
    rng = np.random.default_rng(seed)
    direct_hits = 0
    cot_hits = 0
    for _ in range(COT_TRIALS):
        params = rng.integers(1, MODULUS, size=(N_STEPS, 2))  # (a_i, b_i) per step, in 1..M-1
        x0 = int(rng.integers(0, MODULUS))  # starting state
        truth = x0
        for a, b in params:  # the ground-truth answer, computed exactly
            truth = (truth * int(a) + int(b)) % MODULUS
        direct_hits += int(run_chain_direct(rng, params, x0) == truth)
        cot_hits += int(run_chain_cot(rng, params, x0) == truth)
    return direct_hits / COT_TRIALS, cot_hits / COT_TRIALS


def _detect_device_line() -> str:
    """Honest device line: report the detected accelerator but state we pin to CPU numpy."""
    try:
        import torch  # torch is optional here -- the demo itself is pure numpy

        detected = (
            "cuda"
            if torch.cuda.is_available()
            else "mps"
            if torch.backends.mps.is_available()
            else "cpu"
        )
        return (
            f"device: cpu (detected {detected}; pinned to CPU numpy for reproducibility)"
            f"  |  torch: {torch.__version__}"
        )
    except ModuleNotFoundError:
        return "device: cpu (numpy only; torch not installed)"


def main() -> None:
    print(_detect_device_line())
    print(f"numpy: {np.__version__}\n")

    # ---- Demo 1: self-consistency ----------------------------------------------------
    acc = self_consistency_sweep()
    single = acc[1]
    best_k = max(acc, key=lambda k: acc[k])
    best = acc[best_k]
    print("Self-consistency: majority-vote accuracy vs number of sampled chains K")
    print(f"  (single chain reaches the right answer with prob P_CORRECT={P_CORRECT})")
    print(f"  {'K':>4} | {'accuracy':>9}")
    print("  " + "-" * 18)
    for k in SC_K_VALUES:
        print(f"  {k:>4} | {acc[k]:>8.3f}")
    # Assert the qualitative results BEFORE any further reporting: voting beats a single chain,
    # and the curve rises with K (allowing a tiny Monte-Carlo slack between adjacent points).
    assert best > single, "self-consistency must beat a single chain"
    assert acc[SC_K_VALUES[-1]] > acc[1], "accuracy must rise with K"
    rising = all(
        acc[b] >= acc[a] - 0.003  # 0.003 slack ~ Monte-Carlo std error; the trend is monotone
        for a, b in zip(SC_K_VALUES, SC_K_VALUES[1:])
    )
    assert rising, "majority-vote accuracy must rise monotonically with K"
    print(
        f"\n  single-chain acc = {single:.3f}  ->  K={best_k} majority-vote acc = {best:.3f}"
        f"   (gain +{best - single:.3f})   assert PASSED\n"
    )

    # ---- Demo 2: direct vs CoT -------------------------------------------------------
    direct_acc, cot_acc = direct_vs_cot()
    print(f"Direct-vs-CoT on a {N_STEPS}-step modular-arithmetic chain (mod {MODULUS}):")
    print(f"  direct (one-shot)     accuracy = {direct_acc:.3f}")
    print(f"  CoT (emit each step)  accuracy = {cot_acc:.3f}")
    assert cot_acc > direct_acc, "CoT must beat direct on the compositional task"
    print(f"  CoT beats direct by +{cot_acc - direct_acc:.3f}   assert PASSED")


if __name__ == "__main__":
    main()
