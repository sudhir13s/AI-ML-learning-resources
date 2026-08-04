"""From-scratch hallucination & alignment demos: the ONE seeded source of truth.

This is the single place every number on the concept page, in the teaching notebook, and in
every figure is computed. The page, the notebook (`20-Hallucination-and-Alignment-Basics.ipynb`),
and the figure generator (`make_figures_20.py`) all import the functions and constants below,
so nothing can silently drift across surfaces.

Four deterministic demos, each asserting its qualitative point BEFORE printing it:

  (a) temperature -> hallucination
      A toy "knowledge" next-token sampler. The model "knows" a few facts (high-logit correct
      tokens) but the softmax always assigns SOME mass to plausible-but-wrong tokens. Raising
      the decoding temperature flattens the distribution, so the wrong tokens get sampled more
      often -> the unsupported-claim rate climbs monotonically with T. (Asserted.)

  (b) grounding (retrieval) -> fewer hallucinations
      Same sampler, but a retrieved context sharpens the logits of the SUPPORTED token (adds a
      grounding boost). The grounded distribution puts far more mass on the supported answer, so
      the unsupported-claim rate drops sharply vs the ungrounded one at the SAME temperature.

  (c) confidence vs correctness -> calibration & abstention
      A toy "model" emits a confidence (max softmax prob) and a correctness label. Sweeping an
      abstention threshold tau: answer only when confidence >= tau, abstain otherwise. As tau
      rises, ANSWERED accuracy rises (we keep only confident, mostly-correct cases) while
      COVERAGE falls -- the coverage/accuracy trade that abstention buys. (Asserted.)

  (d) helpful vs harmless -> the alignment Pareto frontier
      A one-knob refusal policy: a refusal threshold r. Low r -> the model answers everything
      (maximally helpful, but it also answers harmful requests -> unsafe). High r -> it refuses
      aggressively (safe, but it also refuses benign requests -> over-refusal, less helpful).
      Sweeping r traces a helpful-vs-harmless frontier: you cannot max both with one scalar knob.

Pure NumPy + a tiny bit of torch ONLY for the softmax (kept device-agnostic to match the
chapter's exemplars). The reproducible trace in main() is pinned to CPU and prints the device
honestly, so the printed device always matches where the numbers were actually produced.

Run:
    python hallucination_alignment.py
"""

from __future__ import annotations

import numpy as np
import torch
import torch.nn.functional as F

# ---- Global reproducibility ---------------------------------------------------------------
SEED = 0  # one seed -> identical numbers on page, notebook, and figures
EPS = 1e-12  # guards log(0) in entropy / calibration sums

# Run on the best available accelerator; CPU is the universal fallback (matches kv_cache.py).
DEVICE = (
    "cuda"
    if torch.cuda.is_available()
    else "mps"
    if torch.backends.mps.is_available()
    else "cpu"
)

# ===========================================================================================
# Shared softmax helper
# ===========================================================================================
def softmax_with_temperature(logits: torch.Tensor, temperature: float) -> torch.Tensor:
    """Temperature-scaled softmax p_i = softmax(z_i / T).

    T<1 divides logits up -> gaps widen -> sharper (more peaked) distribution.
    T>1 divides logits down -> gaps shrink -> flatter (more uniform) distribution.
    T->0 limit is a one-hot at the argmax (greedy); T=1 is the plain softmax.

    This is the SAME mechanism used in the Decoding & Sampling chapter (ch. 18) -- here we read
    it through the hallucination lens: flatter distribution => more mass on wrong tokens.
    """
    assert temperature > 0, "temperature must be > 0 (T->0 is the greedy/argmax limit)"
    return F.softmax(logits / temperature, dim=-1)


# ===========================================================================================
# (a) A toy "knowledge" next-token sampler: temperature -> hallucination
# ===========================================================================================
# A 6-way toy "answer vocabulary" for a single factual question, e.g. "Who wrote <book>?".
# Exactly ONE answer is SUPPORTED (the true, grounded answer). The others are plausible-but-
# unsupported distractors -- the kind of confident-sounding wrong answer a model invents.
ANSWER_VOCAB: tuple[str, ...] = (
    "Austen",   # the supported / correct author (index 0)
    "Bronte",   # plausible distractor (same era/genre)
    "Dickens",  # plausible distractor
    "Eliot",    # plausible distractor
    "Hardy",    # plausible distractor
    "Gaskell",  # plausible distractor
)
SUPPORTED_IDX = 0  # "Austen" is the only supported answer

# The model "knows" the answer: the supported token has the largest logit. But the softmax
# ALWAYS assigns nonzero mass to every distractor -- there is no "0 probability" option. That
# residual mass on wrong tokens is the seed of hallucination: at high temperature it grows.
BASE_LOGITS: tuple[float, ...] = (4.0, 2.2, 2.0, 1.8, 1.6, 1.4)

# Retrieval/grounding adds this much to the supported token's logit (a retrieved passage that
# names the true author makes that token far more likely). Applied ONLY to the supported index.
GROUNDING_BOOST = 4.0

# Temperatures swept in demo (a)/(b). Greedy (T->0) never hallucinates on this toy because the
# argmax is the supported token; the interesting regime is T >= ~0.7 where wrong tokens appear.
TEMPS_HALLUCINATION: tuple[float, ...] = (0.3, 0.5, 0.7, 1.0, 1.5, 2.0)
N_SAMPLES = 20000  # draws per temperature for a stable Monte-Carlo unsupported-claim rate


def base_logits_tensor(device: str = DEVICE) -> torch.Tensor:
    """The ungrounded next-token logits for the toy factual question."""
    return torch.tensor(BASE_LOGITS, dtype=torch.float32, device=device)


def grounded_logits_tensor(boost: float = GROUNDING_BOOST, device: str = DEVICE) -> torch.Tensor:
    """The grounded logits: the SAME base logits with the supported token boosted by retrieval.

    This models RAG at the logit level: a retrieved passage that names the true author makes
    the supported token decisively more probable. Only the supported index changes.
    """
    logits = base_logits_tensor(device=device).clone()
    logits[SUPPORTED_IDX] += boost
    return logits


def unsupported_claim_rate(logits: torch.Tensor, temperature: float, n_samples: int,
                           generator: torch.Generator) -> float:
    """Monte-Carlo fraction of sampled answers that are NOT the supported token.

    Draw n_samples answers from the temperature-scaled softmax; the unsupported-claim rate is
    the fraction landing on any distractor. This is a direct, sampled measure of how often the
    decoder emits an unsupported claim -- the toy analogue of a hallucination rate.
    """
    probs = softmax_with_temperature(logits, temperature)
    draws = torch.multinomial(probs, num_samples=n_samples, replacement=True, generator=generator)
    return float((draws != SUPPORTED_IDX).float().mean().item())


def hallucination_rate_curve(logits: torch.Tensor, temps: tuple[float, ...],
                             n_samples: int = N_SAMPLES, seed: int = SEED,
                             device: str = "cpu") -> list[float]:
    """Unsupported-claim rate at each temperature (one seeded generator, reproducible)."""
    gen = torch.Generator(device=device).manual_seed(seed)
    return [unsupported_claim_rate(logits, t, n_samples, gen) for t in temps]


# ===========================================================================================
# (c) Confidence vs correctness: calibration & abstention
# ===========================================================================================
# A synthetic but realistic calibration setup. We generate N (confidence, correct) pairs where
# higher confidence => higher probability of being correct, with NOISE -- exactly the regime
# where abstaining below a confidence threshold helps. confidence = the model's max-softmax
# probability for its chosen answer; correct = whether that answer was right.
N_CALIB = 4000


def make_calibration_data(n: int = N_CALIB, seed: int = SEED) -> tuple[np.ndarray, np.ndarray]:
    """Return (confidence, correct) arrays for n synthetic predictions.

    confidence ~ Uniform(0.5, 1.0) (a softmax max-prob over >=2 classes is always >= 0.5 for a
    binary head; we keep it in [0.5, 1] to mirror that). The probability that a prediction is
    CORRECT rises with its confidence -- a *roughly* calibrated model -- but with Bernoulli
    noise, so some high-confidence answers are still wrong (the hallucination case) and some
    low-confidence ones are right. Abstention exploits the *trend*, not perfection.
    """
    rng = np.random.default_rng(seed)
    confidence = rng.uniform(0.5, 1.0, size=n)
    # P(correct | confidence): a calibrated-ish ramp, linear in confidence then a Bernoulli draw.
    # Maps conf 0.5 -> 0.05 and conf 1.0 -> 0.95, so high-confidence answers are usually (not
    # always) right and low-confidence ones usually wrong -- the trend abstention exploits.
    p_correct = 0.05 + 0.9 * (confidence - 0.5) / 0.5
    p_correct = np.clip(p_correct, 0.0, 1.0)
    correct = (rng.uniform(0.0, 1.0, size=n) < p_correct).astype(np.int64)
    return confidence, correct


def coverage_accuracy_curve(confidence: np.ndarray, correct: np.ndarray,
                            thresholds: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """For each abstention threshold tau, return (coverage, answered_accuracy).

    Policy: ANSWER when confidence >= tau, ABSTAIN otherwise.
      coverage(tau)         = fraction of questions answered  (answered / total)
      answered_accuracy(tau) = accuracy ON THE ANSWERED ones   (correct & answered / answered)

    As tau rises we keep only the most confident -- and (because the model is roughly
    calibrated) mostly-correct -- predictions, so answered_accuracy rises while coverage falls.
    That trade is the entire value of abstention: trade some coverage for higher reliability.
    """
    coverages = []
    accuracies = []
    for tau in thresholds:
        answered = confidence >= tau
        n_answered = int(answered.sum())
        coverages.append(n_answered / len(confidence))
        if n_answered == 0:
            accuracies.append(float("nan"))  # nothing answered -> accuracy undefined
        else:
            accuracies.append(float(correct[answered].mean()))
    return np.array(coverages), np.array(accuracies)


def expected_calibration_error(confidence: np.ndarray, correct: np.ndarray,
                               n_bins: int = 10) -> float:
    """Expected Calibration Error (ECE): mean |confidence - accuracy| over confidence bins.

    Bin predictions by confidence; in each bin compare the mean confidence to the actual
    accuracy. ECE is the sample-weighted average gap. 0 = perfectly calibrated (confidence ==
    accuracy in every bin); large = the model's confidence does not match how often it is right
    -- the formal handle on "a model that is confidently wrong".
    """
    bins = np.linspace(0.0, 1.0, n_bins + 1)
    ece = 0.0
    n = len(confidence)
    for lo, hi in zip(bins[:-1], bins[1:]):
        in_bin = (confidence >= lo) & (confidence < hi)
        if hi == 1.0:  # include the right edge in the last bin
            in_bin = (confidence >= lo) & (confidence <= hi)
        count = int(in_bin.sum())
        if count == 0:
            continue
        bin_conf = float(confidence[in_bin].mean())
        bin_acc = float(correct[in_bin].mean())
        ece += (count / n) * abs(bin_conf - bin_acc)
    return ece


def reliability_bins(confidence: np.ndarray, correct: np.ndarray,
                     n_bins: int = 10) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Return (bin_centers, bin_accuracy, bin_count) for a reliability diagram."""
    bins = np.linspace(0.0, 1.0, n_bins + 1)
    centers, accs, counts = [], [], []
    for lo, hi in zip(bins[:-1], bins[1:]):
        in_bin = (confidence >= lo) & (confidence < hi)
        if hi == 1.0:
            in_bin = (confidence >= lo) & (confidence <= hi)
        count = int(in_bin.sum())
        centers.append((lo + hi) / 2.0)
        counts.append(count)
        accs.append(float(correct[in_bin].mean()) if count > 0 else float("nan"))
    return np.array(centers), np.array(accs), np.array(counts)


# ===========================================================================================
# (d) Helpful vs harmless: the alignment Pareto frontier
# ===========================================================================================
# A population of requests, each with a "harm score" in [0,1] (0 = benign, 1 = clearly harmful).
# A single permissiveness knob r decides the policy: ANSWER any request whose perceived harm is
# BELOW r, refuse the rest. r is "how willing the model is to answer":
#   - Low r  -> answers almost nothing -> very safe, but also refuses benign asks (over-refusal).
#   - High r -> answers almost everything -> very helpful, but also answers harmful asks (unsafe).
# Sweeping r traces a helpful-vs-harmless frontier: one scalar cannot maximize both.
N_REQUESTS = 4000
PERMISSIVENESS_THRESHOLDS = np.linspace(0.0, 1.0, 41)


def make_request_population(n: int = N_REQUESTS, seed: int = SEED) -> tuple[np.ndarray, np.ndarray]:
    """Return (harm_score, is_harmful) for n requests.

    harm_score ~ Uniform(0,1). A request is *truly* harmful if its harm score exceeds 0.7 (the
    minority of genuinely harmful asks). The model only sees a NOISY estimate of harm (its
    classifier), so its refusal decisions on the true harm boundary are imperfect -- which is
    exactly why one threshold can't separate helpful from harmless cleanly.
    """
    rng = np.random.default_rng(seed)
    true_harm = rng.uniform(0.0, 1.0, size=n)
    is_harmful = (true_harm > 0.7).astype(np.int64)  # ~30% truly harmful
    # The model's perceived harm = true harm + noise (an imperfect internal harm classifier).
    perceived_harm = np.clip(true_harm + rng.normal(0.0, 0.15, size=n), 0.0, 1.0)
    return perceived_harm, is_harmful


def helpful_harmless_curve(perceived_harm: np.ndarray, is_harmful: np.ndarray,
                           thresholds: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """For each permissiveness threshold r, return (helpfulness, harmlessness).

    Policy: ANSWER when perceived_harm < r, REFUSE otherwise.
      helpfulness(r)  = fraction of BENIGN requests that were answered   (helpful = answer good asks)
      harmlessness(r) = fraction of HARMFUL requests that were refused    (harmless = refuse bad asks)

    Low r answers almost nothing -> harmlessness ~1.0 but helpfulness ~0 (it refuses benign asks).
    High r answers almost everything -> helpfulness ~1.0 but harmlessness ~0 (it answers harmful asks).
    The frontier between them is the helpful/harmless trade RLHF must navigate (ch. 15).
    """
    benign = is_harmful == 0
    harmful = is_harmful == 1
    helpfulness = []
    harmlessness = []
    for r in thresholds:
        answered = perceived_harm < r
        refused = ~answered
        # helpfulness: of the benign requests, what fraction did we answer?
        helpfulness.append(float(answered[benign].mean()) if benign.any() else float("nan"))
        # harmlessness: of the harmful requests, what fraction did we refuse?
        harmlessness.append(float(refused[harmful].mean()) if harmful.any() else float("nan"))
    return np.array(helpfulness), np.array(harmlessness)


# ===========================================================================================
# Reproducible trace
# ===========================================================================================
def main() -> None:
    # Pin the trace to CPU and print the device HONESTLY: the printed device must match where
    # the math actually ran (the recurring device-honesty requirement across the chapter set).
    trace_device = "cpu"
    print(f"device: {trace_device} (detected {DEVICE}; pinned to CPU for reproducibility)")
    print("torch:", torch.__version__, " | numpy:", np.__version__)
    print()

    # --- (a) temperature -> hallucination ------------------------------------------------
    print("[a] temperature -> unsupported-claim rate (ungrounded toy 'knowledge' sampler):")
    base = base_logits_tensor(device=trace_device)
    rates_ungrounded = hallucination_rate_curve(base, TEMPS_HALLUCINATION, device=trace_device)
    for t, r in zip(TEMPS_HALLUCINATION, rates_ungrounded):
        print(f"   T={t:>3}:  unsupported-claim rate = {r:.3f}")
    # Contract: hotter decoding flattens the softmax -> more mass on wrong tokens -> the
    # unsupported-claim rate rises monotonically. (Allow tiny MC noise via a <= with margin.)
    for a, b in zip(rates_ungrounded, rates_ungrounded[1:]):
        assert b >= a - 0.01, "unsupported-claim rate must rise (not fall) with temperature"
    assert rates_ungrounded[-1] > rates_ungrounded[0] + 0.2, (
        "high temperature must hallucinate substantially more than low temperature"
    )
    print(
        f"   => rate climbs {rates_ungrounded[0]:.3f} (T=0.3) -> {rates_ungrounded[-1]:.3f} "
        f"(T=2.0): hotter decoding hallucinates more.\n"
    )

    # --- (b) grounding (retrieval) -> fewer hallucinations -------------------------------
    print("[b] grounding (retrieval boost) -> unsupported-claim rate at the SAME temperatures:")
    grounded = grounded_logits_tensor(device=trace_device)
    rates_grounded = hallucination_rate_curve(grounded, TEMPS_HALLUCINATION, device=trace_device)
    for t, ung, grd in zip(TEMPS_HALLUCINATION, rates_ungrounded, rates_grounded):
        print(f"   T={t:>3}:  ungrounded {ung:.3f}  ->  grounded {grd:.3f}")
    # Contract: at EVERY temperature, grounding lowers the unsupported-claim rate.
    for ung, grd in zip(rates_ungrounded, rates_grounded):
        assert grd <= ung, "grounding must not increase the unsupported-claim rate"
    # And the reduction at T=1.0 should be substantial (retrieval is a strong lever).
    i_t1 = TEMPS_HALLUCINATION.index(1.0)
    assert rates_grounded[i_t1] < rates_ungrounded[i_t1] - 0.2, (
        "grounding must substantially cut hallucination at T=1.0"
    )
    print(
        f"   => at T=1.0 grounding cuts unsupported claims "
        f"{rates_ungrounded[i_t1]:.3f} -> {rates_grounded[i_t1]:.3f}.\n"
    )

    # --- (c) confidence vs correctness: calibration & abstention -------------------------
    print("[c] abstention: answer only when confidence >= tau (coverage vs answered-accuracy):")
    confidence, correct = make_calibration_data()
    base_acc = float(correct.mean())  # accuracy if we answer EVERYTHING (tau = 0)
    ece = expected_calibration_error(confidence, correct)
    taus = np.array([0.5, 0.6, 0.7, 0.8, 0.9])
    cov, acc = coverage_accuracy_curve(confidence, correct, taus)
    print(f"   answer-everything accuracy = {base_acc:.3f}   ECE = {ece:.3f}")
    for tau, c, a in zip(taus, cov, acc):
        print(f"   tau={tau:.1f}:  coverage = {c:.3f}   answered-accuracy = {a:.3f}")
    # Contract 1: raising tau raises answered-accuracy (keep confident, mostly-correct cases).
    valid = ~np.isnan(acc)
    accs_valid = acc[valid]
    for a0, a1 in zip(accs_valid, accs_valid[1:]):
        assert a1 >= a0 - 0.02, "answered-accuracy must rise (not fall) as tau rises"
    # Contract 2: raising tau lowers coverage (we answer fewer questions).
    for c0, c1 in zip(cov, cov[1:]):
        assert c1 <= c0 + 1e-9, "coverage must fall as tau rises"
    # Contract 3: abstaining beats answering everything on the answered set.
    assert acc[-1] > base_acc, "the high-confidence answered set must beat the answer-all accuracy"
    print(
        f"   => abstaining at tau=0.9 lifts accuracy {base_acc:.3f} -> {acc[-1]:.3f} "
        f"while coverage falls 1.000 -> {cov[-1]:.3f}.\n"
    )

    # --- (d) helpful vs harmless: the alignment Pareto frontier --------------------------
    print("[d] helpful vs harmless: one permissiveness knob cannot maximize both:")
    perceived_harm, is_harmful = make_request_population()
    helpful, harmless = helpful_harmless_curve(
        perceived_harm, is_harmful, PERMISSIVENESS_THRESHOLDS
    )
    # Sample a few thresholds for the printed trace.
    for r in (0.0, 0.25, 0.5, 0.75, 1.0):
        i = int(np.argmin(np.abs(PERMISSIVENESS_THRESHOLDS - r)))
        print(f"   r={PERMISSIVENESS_THRESHOLDS[i]:.2f}:  helpfulness = {helpful[i]:.3f}   "
              f"harmlessness = {harmless[i]:.3f}")
    # Contract: helpfulness RISES and harmlessness FALLS as the permissiveness knob rises -- the
    # opposing pull that makes the trade a frontier, not a free lunch.
    assert helpful[-1] > helpful[0] + 0.5, "high permissiveness must be far more helpful than low"
    assert harmless[0] > harmless[-1] + 0.5, "low permissiveness must be far more harmless than high"
    # No single threshold scores >= 0.9 on BOTH at once -> the frontier is a genuine trade.
    both_high = (helpful >= 0.9) & (harmless >= 0.9)
    assert not both_high.any(), "no single scalar threshold should ace both helpful AND harmless"
    print(
        "   => as r rises, helpfulness rises and harmlessness falls; "
        "no single r aces both (a real Pareto trade).\n"
    )

    print("all four demos asserted their qualitative point before printing. done.")


if __name__ == "__main__":
    main()
