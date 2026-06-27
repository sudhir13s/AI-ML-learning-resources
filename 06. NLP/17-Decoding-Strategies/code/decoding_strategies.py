"""From-scratch decoding strategies: greedy, beam search, temperature, top-k, top-p (nucleus).

This is the ONE seeded source of truth for the chapter. The concept page, the teaching
notebook (17-Decoding-Strategies.ipynb), and the figure generator (make_figures_17.py) all
import the functions and constants below, so every quoted number -- joint probabilities,
beam scores, entropies, nucleus sizes, repetition rates -- is computed in exactly one place
and cannot silently drift between the prose, the notebook, and the figures.

It is the GENERAL sequence-generation view of decoding (beam search for closed-ended tasks
like machine translation / summarization + sampling for open-ended generation). The
LLM-systems angle (KV-cache interaction, speculative decoding throughput) lives in the
sibling chapter "09. LLMs / 18-Decoding-and-Sampling"; this file deliberately does not repeat
it.

What it demonstrates, all from tiny hand-built next-token distributions (no model download):
  * greedy = argmax per step (deterministic, locally optimal, globally MYOPIC);
  * beam search keeps the B best partial sequences and recovers a higher-JOINT-probability
    sequence that greedy misses (the AX-vs-BP trap), scored in numerically stable log-space;
  * length normalization removes beam's bias toward short sequences;
  * temperature reshapes the softmax (T<1 sharpens, T>1 flattens) -- measured via entropy;
  * top-k keeps a FIXED number of tokens; top-p (nucleus) keeps the smallest set whose
    cumulative prob >= p -- an ADAPTIVE cutoff that widens exactly when the model is uncertain;
  * neural text degeneration: a greedy/low-temperature loop on a self-reinforcing toy model
    collapses to a repeating token (low distinct-rate) while sampling stays diverse.

Every qualitative claim is asserted BEFORE it is printed, so running this file is a
self-check: if an assertion fails, a number on the page is wrong.

Pure NumPy so the numeric trace is fully deterministic and fast (well under a second on CPU);
no GPU and no model weights required. The reproducible trace in main() prints the host's
Python / NumPy versions honestly so the printed environment always matches where the numbers
were produced.

Run:
    python decoding_strategies.py
"""

from __future__ import annotations

import platform
from collections.abc import Sequence

import numpy as np

# ---- Tiny vocabularies and hand-built next-token distributions ----------------------------
# Small enough to SEE every strategy reshape the distribution by eye, large enough to expose
# the structural differences (greedy myopia, top-k vs top-p adaptivity).

# (1) The two-step tree behind "greedy is myopic". After the prompt the model splits its mass
#     between first tokens A and B; the CONTINUATIONS differ in shape. Greedy takes the better
#     first token (A) but ends at AX; the genuinely most-probable sequence is BP, reachable
#     only by keeping the lower-probability first token B. This is the canonical beam example.
FIRST_STEP: dict[str, float] = {"A": 0.55, "B": 0.45}
SECOND_STEP: dict[str, dict[str, float]] = {
    "A": {"X": 0.40, "Y": 0.35, "Z": 0.25},  # spread out -> best path only 0.55*0.40 = 0.22
    "B": {"P": 0.95, "Q": 0.05},             # piled on P  -> best path 0.45*0.95 = 0.4275
}

# (1b) The page's three-step LOG-SPACE beam trace over a tiny vocab {a,b,c}. Step-1 log-probs
#      from the prompt, then step-2 log-prob increments for each survivor's continuations. Beam
#      B=2 keeps {a, b} after step 1, then {ab: -0.9, ba: -1.0} after step 2 -- note ba beats aa
#      (-1.7) though 'a' was the better first token, because pruning is GLOBAL across expansions.
#      Kept here so the beam_trace figure is driven by this single source of truth (no drift).
BEAM_TRACE_STEP1: dict[str, float] = {"a": -0.5, "b": -0.7, "c": -2.0}
BEAM_TRACE_STEP2_INC: dict[str, dict[str, float]] = {
    "a": {"a": -1.2, "b": -0.4, "c": -1.6},
    "b": {"a": -0.3, "b": -1.5, "c": -2.4},
}
BEAM_TRACE_WIDTH = 2  # beam width used in the trace

# (2) A four-token logit vector for the temperature worked example (tokens A,B,C,D).
TEMP_LOGITS: tuple[float, ...] = (3.0, 2.0, 1.0, 0.0)
TEMPERATURES: tuple[float, ...] = (0.5, 1.0, 2.0)  # sharpen / unchanged / flatten

# (3) A realistic 8-token distribution (already sorted) for the top-k vs top-p comparison.
EIGHT_TOKEN_DIST: tuple[float, ...] = (0.40, 0.25, 0.13, 0.08, 0.06, 0.04, 0.025, 0.015)
TOP_K = 2     # fixed-size truncation for the top-k demo
TOP_P = 0.9   # nucleus mass threshold for the top-p demo

# (4) Two contrasting distributions (peaked vs flat) to show top-p ADAPTS while top-k does not.
#     Logits all-distinct so the k-th-largest cutoff is unambiguous (top-k tie-breaking is a
#     separate pitfall covered on the page, not the point here).
PEAKED_LOGITS: tuple[float, ...] = (5.0, 8.0, 4.0, 3.5, 3.0, 2.5, 2.0, 1.5, 1.0, 0.5)
FLAT_LOGITS: tuple[float, ...] = (2.20, 2.05, 1.95, 2.10, 1.80, 2.00, 1.70, 1.90, 1.85, 1.75)

# (5) Degeneration demo: a toy "language model" whose next-token distribution depends only on
#     the PREVIOUS token, with a self-reinforcing loop baked in (after the loop token, the loop
#     token is by far the most probable -- exactly the pattern-continuation pressure Holtzman et
#     al. identify). Greedy walks straight into the loop; sampling escapes it.
DEGEN_VOCAB: tuple[str, ...] = ("<s>", "I", "love", "pizza", "and", "cake", "too", ".")
# rows index the PREVIOUS token, columns the NEXT token; each row is a probability vector.
# The cycle I->love->pizza->.->I ... is given high (but < 1) probability so greedy loops while
# sampling can break out via the smaller alternative-mass on each row.
DEGEN_TRANSITIONS: tuple[tuple[float, ...], ...] = (
    (0.0, 0.90, 0.0, 0.0, 0.05, 0.03, 0.0, 0.02),   # <s> -> I (start)
    (0.0, 0.0, 0.85, 0.0, 0.05, 0.05, 0.03, 0.02),  # I    -> love
    (0.0, 0.0, 0.0, 0.85, 0.07, 0.05, 0.0, 0.03),   # love -> pizza
    (0.0, 0.0, 0.0, 0.0, 0.20, 0.10, 0.05, 0.65),   # pizza-> . (mostly) / and / cake
    (0.0, 0.10, 0.10, 0.55, 0.0, 0.20, 0.05, 0.0),  # and  -> pizza / cake
    (0.0, 0.10, 0.05, 0.10, 0.10, 0.0, 0.60, 0.05), # cake -> too
    (0.0, 0.05, 0.05, 0.05, 0.05, 0.05, 0.0, 0.75), # too  -> .
    (0.0, 0.80, 0.05, 0.05, 0.05, 0.05, 0.0, 0.0),  # .    -> I  (closes the I-love-pizza loop)
)
DEGEN_STEPS = 40          # generation length for the degeneration demo
DEGEN_SEED = 0            # one seed -> reproducible sampled sequences everywhere
EPS = 1e-12               # guards log(0) in entropy / log-prob sums


def softmax(logits: np.ndarray) -> np.ndarray:
    """Numerically stable softmax over the last axis (subtract the max before exponentiating)."""
    shifted = logits - logits.max(axis=-1, keepdims=True)  # max-subtraction avoids overflow in exp
    exp = np.exp(shifted)
    return exp / exp.sum(axis=-1, keepdims=True)


def softmax_with_temperature(logits: np.ndarray, temperature: float) -> np.ndarray:
    """Temperature-scaled softmax: p_i = softmax(z_i / T).

    T<1 divides logits up -> gaps widen -> sharper (more peaked) distribution.
    T>1 divides logits down -> gaps shrink -> flatter (more uniform) distribution.
    T=1 recovers the plain softmax; the T->0 limit is a one-hot at the argmax (= greedy).
    """
    assert temperature > 0, "temperature must be > 0 (T->0 is the greedy limit)"
    return softmax(logits / temperature)


def entropy_bits(probs: np.ndarray) -> float:
    """Shannon entropy H = -sum p log2 p, in bits -- a scalar 'peakiness' measure.

    Lower entropy = more peaked (mass concentrated); higher = flatter (mass spread).
    Max for an n-way distribution is log2(n) (perfectly uniform).
    """
    p = np.clip(probs, EPS, 1.0)  # clip so log2(0) never produces -inf * 0 = nan
    return float(-(p * np.log2(p)).sum())


# ---- Greedy & beam search -----------------------------------------------------------------

def greedy_two_step() -> tuple[str, float]:
    """Greedy on the two-step tree: take the best FIRST token, then the best continuation.

    Returns the (sequence, joint_probability). Greedy is locally optimal at each step but
    cannot see that a weaker first token (B) leads to a far stronger continuation (P).
    """
    first = max(FIRST_STEP, key=lambda t: FIRST_STEP[t])           # A (0.55 > 0.45)
    second = max(SECOND_STEP[first], key=lambda t: SECOND_STEP[first][t])
    return first + second, FIRST_STEP[first] * SECOND_STEP[first][second]


def best_two_step() -> tuple[str, float]:
    """Brute-force the globally most-probable two-token sequence over the whole tree."""
    candidates = [
        (t1 + t2, p1 * p2)
        for t1, p1 in FIRST_STEP.items()
        for t2, p2 in SECOND_STEP[t1].items()
    ]
    return max(candidates, key=lambda s: s[1])


def beam_two_step(beam_width: int) -> tuple[str, float]:
    """Beam search on the two-step tree, scored in LOG-space (the numerically stable form).

    Keeps the `beam_width` best partial sequences after step 1, expands each by every possible
    continuation, then returns the highest-scoring complete sequence. With width 2 it keeps both
    A and B alive and so recovers BP, the sequence greedy's tunnel vision skips.
    """
    assert beam_width >= 1
    # Step 1: score each first token by log p, keep the top `beam_width`.
    beams = sorted(
        ((t, float(np.log(p))) for t, p in FIRST_STEP.items()),
        key=lambda b: b[1],
        reverse=True,
    )[:beam_width]
    # Step 2: expand every surviving beam by every continuation; score = sum of log-probs.
    finished: list[tuple[str, float]] = []
    for seq, logp in beams:
        for tok, p in SECOND_STEP[seq].items():
            finished.append((seq + tok, logp + float(np.log(p))))
    seq, logp = max(finished, key=lambda c: c[1])
    return seq, float(np.exp(logp))  # convert the winning log-score back to a probability


def length_penalty(length: int, alpha: float) -> float:
    """Google-NMT length penalty lp(y) = ((5 + |y|)^alpha) / ((5 + 1)^alpha) (Wu et al. 2016).

    score(y) = (1 / lp(y)) * sum log p. alpha=0 -> no normalization; alpha=1 -> mean log-prob.
    The denominator grows with length, offsetting the accumulating negative log-probs so beam
    search stops preferring short sequences.
    """
    return ((5 + length) ** alpha) / ((5 + 1) ** alpha)


def normalized_beam_score(log_probs: Sequence[float], alpha: float) -> float:
    """Google-NMT length-normalized beam score: (sum of log-probs) / lp(length).

    Note this is the PRODUCTION heuristic with the +5 smoothing constant, NOT a pure mean: the
    constant deliberately damps the length correction for short sequences, which is why alpha is
    tuned to ~0.6-0.7 rather than the 1.0 you'd use for an exact average.
    """
    total = float(np.sum(log_probs))
    return total / length_penalty(len(log_probs), alpha)


def mean_log_prob(log_probs: Sequence[float]) -> float:
    """Per-token mean log-probability -- the conceptually simplest, fully length-FAIR score.

    Dividing the summed log-prob by the raw token count makes long and short hypotheses compete
    purely on per-token quality, with no length bias at all. The Google lp() form above is a
    softened version of this idea (its +5 constant keeps it from over-rewarding length).
    """
    return float(np.mean(log_probs))


def beam_trace_survivors() -> list[tuple[str, float]]:
    """Run the three-step log-space beam trace over the {a,b,c} constants; return step-2 survivors.

    Step 1 keeps the top BEAM_TRACE_WIDTH first tokens by log p; step 2 expands each survivor by
    every continuation, scores by adding the increment, and keeps the top BEAM_TRACE_WIDTH GLOBALLY.
    Returns the surviving (sequence, score) pairs, sorted best-first -- the single source of truth
    for the beam_trace figure and the page's log-space worked example.
    """
    step1 = sorted(BEAM_TRACE_STEP1.items(), key=lambda kv: kv[1], reverse=True)[:BEAM_TRACE_WIDTH]
    candidates: list[tuple[str, float]] = []
    for parent, parent_score in step1:
        for tok, inc in BEAM_TRACE_STEP2_INC[parent].items():
            candidates.append((parent + tok, parent_score + inc))
    candidates.sort(key=lambda c: c[1], reverse=True)
    return candidates[:BEAM_TRACE_WIDTH]


# ---- Truncation: top-k and top-p (nucleus) ------------------------------------------------

def top_k_keep(probs: np.ndarray, k: int) -> np.ndarray:
    """Boolean mask of the k highest-probability tokens. FIXED size: exactly k survive."""
    assert 1 <= k <= probs.shape[-1], "k must be in [1, vocab_size]"
    order = np.argsort(-probs)            # indices sorted by descending probability
    keep = np.zeros_like(probs, dtype=bool)
    keep[order[:k]] = True
    return keep


def top_p_keep(probs: np.ndarray, p: float) -> np.ndarray:
    """Boolean mask of the nucleus: the smallest set of top tokens whose cumulative prob >= p.

    ADAPTIVE size: on a peaked distribution a few tokens already cover p (small nucleus); on a
    flat distribution many tokens are needed (large nucleus). The crossing token itself is kept,
    so the kept mass is always >= p (never just under it).
    """
    assert 0 < p <= 1, "p must be in (0, 1]"
    order = np.argsort(-probs)
    cumulative = np.cumsum(probs[order])
    # Number kept = position of the first cumulative sum to reach p, INCLUSIVE (+1).
    n_keep = int(np.searchsorted(cumulative, p) + 1)
    n_keep = min(n_keep, probs.shape[-1])
    keep = np.zeros_like(probs, dtype=bool)
    keep[order[:n_keep]] = True
    return keep


def renormalize(probs: np.ndarray, keep: np.ndarray) -> np.ndarray:
    """Zero the masked tokens and rescale the survivors to sum to 1 (a valid distribution)."""
    kept = probs * keep
    total = kept.sum()
    assert total > 0, "the kept set must hold positive mass"
    return kept / total


def nucleus_size(probs: np.ndarray, p: float) -> int:
    """How many tokens land inside the top-p nucleus -- the adaptive count itself."""
    return int(top_p_keep(probs, p).sum())


# ---- Neural text degeneration (greedy loops, sampling escapes) ----------------------------

def distinct_rate(tokens: Sequence[int], n: int = 2) -> float:
    """distinct-n: fraction of UNIQUE n-grams among all n-grams. Lower = more repetitive.

    distinct-2 = 1.0 means every bigram is unique; ~0.1 means the text loops on a few bigrams.
    The single clearest scalar for the degeneration phenomenon.
    """
    grams = [tuple(tokens[i : i + n]) for i in range(len(tokens) - n + 1)]
    if not grams:
        return 0.0
    return len(set(grams)) / len(grams)


def generate_greedy(n_steps: int = DEGEN_STEPS) -> list[int]:
    """Greedy walk on the toy transition model: always take the most-probable next token.

    Starts at <s>, then at each step picks argmax of the row for the current token. Because the
    cycle tokens dominate their rows, greedy falls into the I-love-pizza-. loop and stays there.
    """
    transitions = np.array(DEGEN_TRANSITIONS)
    seq = [0]  # <s>
    current = 0
    for _ in range(n_steps):
        current = int(np.argmax(transitions[current]))  # deterministic: the row's top token
        seq.append(current)
    return seq


def generate_sampled(
    n_steps: int = DEGEN_STEPS, temperature: float = 1.0, seed: int = DEGEN_SEED
) -> list[int]:
    """Sampled walk on the toy transition model at a given temperature (seeded, reproducible).

    Each step samples the next token from the (temperature-reshaped) row of the current token.
    The occasional non-top choice breaks the self-reinforcing loop greedy gets stuck in.
    """
    transitions = np.array(DEGEN_TRANSITIONS)
    rng = np.random.default_rng(seed)
    seq = [0]  # <s>
    current = 0
    for _ in range(n_steps):
        row = transitions[current]
        # Reshape the row by temperature in LOGIT space (log p / T), guarding the structural
        # zeros so an impossible transition stays impossible at any temperature.
        logits = np.where(row > 0, np.log(np.clip(row, EPS, 1.0)), -np.inf)
        probs = softmax_with_temperature(logits, temperature)
        current = int(rng.choice(len(DEGEN_VOCAB), p=probs))
        seq.append(current)
    return seq


def _fmt_seq(ids: Sequence[int]) -> str:
    """Render a token-id sequence as readable text for the degeneration prints."""
    return " ".join(DEGEN_VOCAB[i] for i in ids if DEGEN_VOCAB[i] != "<s>")


def main() -> None:
    print(f"python: {platform.python_version()}   numpy: {np.__version__}   device: cpu (pure-numpy, deterministic)")
    print()

    # --- 1. Greedy is myopic; beam recovers the globally best sequence -------------------
    g_seq, g_p = greedy_two_step()
    best_seq, best_p = best_two_step()
    beam_seq, beam_p = beam_two_step(beam_width=2)
    print("[greedy vs beam] two-step tree (joint probability of the chosen 2-token sequence):")
    print(f"   greedy           -> {g_seq}  p = {g_p:.4f}")
    print(f"   beam (width 2)   -> {beam_seq}  p = {beam_p:.4f}")
    print(f"   global best      -> {best_seq}  p = {best_p:.4f}")
    # The whole point: greedy's locally-best first token (A) loses to beam's BP.
    assert g_seq == "AX" and abs(g_p - 0.22) < 1e-9, "greedy must end at AX with joint prob 0.22"
    assert beam_seq == "BP" and abs(beam_p - 0.4275) < 1e-9, "beam width 2 must recover BP at 0.4275"
    assert beam_p > g_p, "beam must find a higher-joint-probability sequence than greedy"
    assert (beam_seq, round(beam_p, 4)) == (best_seq, round(best_p, 4)), "beam must match the global best here"
    print(f"   => beam's BP ({beam_p:.4f}) nearly DOUBLES greedy's AX ({g_p:.4f}); greedy never considered B.\n")

    # --- 1b. The log-space beam trace (drives the beam_trace figure) ----------------------
    survivors = beam_trace_survivors()
    surv_str = "  ".join(f"{seq}:{score:.1f}" for seq, score in survivors)
    print(f"[beam trace] log-space {{a,b,c}} tree, B={BEAM_TRACE_WIDTH} step-2 survivors: {surv_str}")
    # The page's worked example: survivors are ab (-0.9) and ba (-1.0); ba beats aa (-1.7) though
    # 'a' was the better first token, because pruning is GLOBAL across all expansions.
    assert [seq for seq, _ in survivors] == ["ab", "ba"], "step-2 survivors must be ab then ba"
    assert abs(survivors[0][1] - (-0.9)) < 1e-9 and abs(survivors[1][1] - (-1.0)) < 1e-9, "scores ab=-0.9, ba=-1.0"
    print("   => ba (-1.0) survives over aa (-1.7) — a weaker first token wins on a strong continuation.\n")

    # --- 2. Length normalization removes beam's short-sequence bias ----------------------
    # The bias is structural: every token's log-prob is negative, so a longer hypothesis ALWAYS
    # has a smaller RAW SUM -- even when its per-token probabilities are HIGHER. We pit a short
    # hypothesis (2 mediocre tokens, p=0.55 each) against a longer one whose every token is
    # genuinely BETTER (p=0.70 each). Two scorers:
    #   raw sum-of-log-probs -> crowns the SHORT one purely for being short (the bug).
    #   mean log-prob        -> a fully length-FAIR per-token average, which correctly crowns the
    #                           higher-quality LONG one. That flip IS length normalization.
    # (The production Google-NMT lp() form is a softened version of the mean -- shown after.)
    short_logps = [np.log(0.55), np.log(0.55)]                      # 2 mediocre tokens
    long_logps = [np.log(0.70)] * 5                                 # 5 genuinely-better tokens
    raw_short, raw_long = float(np.sum(short_logps)), float(np.sum(long_logps))
    mean_short, mean_long = mean_log_prob(short_logps), mean_log_prob(long_logps)
    print("[length norm] short (2 tok, p=0.55) vs long (5 tok, p=0.70) -- higher score preferred:")
    print(f"   raw sum-of-log-probs : short {raw_short:.3f}   long {raw_long:.3f}   -> winner: "
          f"{'short' if raw_short > raw_long else 'long'}")
    print(f"   mean log-prob (fair) : short {mean_short:.3f}   long {mean_long:.3f}   -> winner: "
          f"{'short' if mean_short > mean_long else 'long'}")
    assert raw_short > raw_long, "raw scoring is biased to the shorter sequence (every token lowers the sum)"
    assert mean_long > mean_short, "the per-token mean (fully length-fair) must prefer the higher-quality long one"
    # The Google-NMT lp() is a *partial* correction: alpha tunes how much length is divided out.
    print("   Google-NMT lp() partial correction, sum / lp(len):")
    for alpha in (0.0, 0.7, 1.0):
        s = normalized_beam_score(short_logps, alpha)
        long_score = normalized_beam_score(long_logps, alpha)
        print(f"      alpha={alpha:>3}: short {s:.3f}   long {long_score:.3f}   "
              f"(the +5 smoothing keeps even alpha=1 from fully averaging)")
    print("   => raw sum prefers SHORT; the length-fair per-token mean flips it to the higher-quality LONG one.\n")

    # --- 3. Temperature reshapes the softmax: measure entropy at each T -------------------
    logits = np.array(TEMP_LOGITS)
    print("[temperature] logits [3,2,1,0] reshaped by T (entropy in bits, lower = peakier):")
    entropies: dict[float, float] = {}
    for temp in TEMPERATURES:
        probs = softmax_with_temperature(logits, temp)
        h = entropy_bits(probs)
        entropies[temp] = h
        shown = "  ".join(f"{x:.3f}" for x in probs)
        print(f"   T={temp:>3}:  [{shown}]   H={h:.3f} bits")
    assert entropies[0.5] < entropies[1.0] < entropies[2.0], "entropy must rise with temperature"
    # Pin the exact probabilities the page quotes (top token 0.644 at T=1, 0.865 at T=0.5).
    assert abs(softmax_with_temperature(logits, 1.0)[0] - 0.644) < 1e-3
    assert abs(softmax_with_temperature(logits, 0.5)[0] - 0.865) < 1e-3
    print(f"   => entropy rises {entropies[0.5]:.2f} -> {entropies[1.0]:.2f} -> {entropies[2.0]:.2f} bits as T goes 0.5 -> 1 -> 2.\n")

    # --- 4. top-k keeps a FIXED count; top-p keeps an ADAPTIVE mass -----------------------
    dist = np.array(EIGHT_TOKEN_DIST)
    k_keep = top_k_keep(dist, TOP_K)
    p_keep = top_p_keep(dist, TOP_P)
    print(f"[truncation] same 8-token dist {tuple(EIGHT_TOKEN_DIST)}:")
    print(f"   top-k (k={TOP_K}): keeps {int(k_keep.sum())} tokens, mass {dist[k_keep].sum():.3f}")
    print(f"   top-p (p={TOP_P}): keeps {int(p_keep.sum())} tokens, mass {dist[p_keep].sum():.3f}")
    assert int(k_keep.sum()) == 2 and abs(dist[k_keep].sum() - 0.65) < 1e-9, "top-k=2 keeps 2 tokens, mass 0.65"
    assert int(p_keep.sum()) == 5 and abs(dist[p_keep].sum() - 0.92) < 1e-9, "top-p=0.9 keeps 5 tokens, mass 0.92"
    # Renormalization preserves relative odds and sums to 1.
    renorm = renormalize(dist, p_keep)
    assert abs(renorm.sum() - 1.0) < 1e-9, "renormalized survivors must sum to 1"
    assert abs(renorm[0] / renorm[1] - dist[0] / dist[1]) < 1e-9, "relative odds among survivors are preserved"
    print(f"   => top-p adapts to the SHAPE (5 here); top-k is stuck at {TOP_K} no matter the shape.\n")

    # --- 5. top-p adapts across peaked vs flat; top-k does not ----------------------------
    peaked = softmax(np.array(PEAKED_LOGITS))
    flat = softmax(np.array(FLAT_LOGITS))
    n_peaked, n_flat = nucleus_size(peaked, TOP_P), nucleus_size(flat, TOP_P)
    print(f"[adaptivity] top-p={TOP_P} nucleus size: {n_peaked} on PEAKED dist, {n_flat} on FLAT dist.")
    assert n_peaked < n_flat, "nucleus must be smaller when the distribution is peaked (top-p adapts)"
    assert int(top_k_keep(peaked, 5).sum()) == int(top_k_keep(flat, 5).sum()) == 5, "top-k is fixed across shapes"
    print(f"   => top-p widens from {n_peaked} to {n_flat} as the model gets less certain; top-k stays fixed.\n")

    # --- 6. Neural text degeneration: greedy loops, sampling escapes ----------------------
    greedy_ids = generate_greedy()
    sampled_ids = generate_sampled(temperature=1.0)
    lowt_ids = generate_sampled(temperature=0.3)
    d_greedy = distinct_rate(greedy_ids)
    d_sampled = distinct_rate(sampled_ids)
    d_lowt = distinct_rate(lowt_ids)
    print("[degeneration] toy self-reinforcing model, 40 tokens (distinct-2: lower = more looping):")
    print(f"   greedy            distinct-2 = {d_greedy:.3f}  ->  {_fmt_seq(greedy_ids)[:70]}...")
    print(f"   sample (T=0.3)    distinct-2 = {d_lowt:.3f}  ->  {_fmt_seq(lowt_ids)[:70]}...")
    print(f"   sample (T=1.0)    distinct-2 = {d_sampled:.3f}  ->  {_fmt_seq(sampled_ids)[:70]}...")
    # The headline: greedy collapses into a loop (low distinct-2); full-temperature sampling
    # stays varied (high distinct-2); low-temperature sits in between, nearer greedy.
    assert d_greedy < d_sampled, "greedy must be MORE repetitive (lower distinct-2) than T=1 sampling"
    assert d_lowt < d_sampled, "low temperature must be more repetitive than full-temperature sampling"
    print(f"   => decoding ALONE moves distinct-2 from {d_greedy:.2f} (greedy loop) to {d_sampled:.2f} (sampled, varied).")


if __name__ == "__main__":
    main()
