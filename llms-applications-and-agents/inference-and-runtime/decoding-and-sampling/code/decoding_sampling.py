"""From-scratch decoding & sampling: greedy, temperature, top-k, top-p (nucleus).

This is the ONE seeded source of truth for the chapter. The concept page, the teaching
notebook, and the figure generator all import the functions and constants below, so every
quoted number (entropies, nucleus sizes, sampled tokens) is computed in exactly one place
and cannot silently drift between surfaces.

What it demonstrates, all from a tiny hand-built next-token distribution:
  * greedy = argmax (deterministic, myopic);
  * temperature reshapes the softmax (T<1 sharpens, T>1 flattens) -- measured via entropy;
  * top-k keeps a FIXED number of tokens (k), renormalises, samples;
  * top-p (nucleus) keeps the smallest set whose cumulative prob >= p -- an ADAPTIVE cutoff;
  * the key result: on a PEAKED distribution nucleus is small, on a FLAT distribution it is
    large -- the adaptivity that fixed-size top-k cannot match (asserted, not asserted-by-faith).

Device-agnostic (CUDA / MPS / CPU) to match the chapter's kv_cache.py exemplar. The logic is
device-independent; the reproducible trace in main() is pinned to CPU and prints the device
honestly so the printed device always matches where the numbers were actually produced.

Run:
    python decoding_sampling.py
"""

from __future__ import annotations

import torch
import torch.nn.functional as F

# ---- Toy vocabulary and two next-token logit vectors -------------------------------------
# A 10-word toy vocab is enough to SEE every strategy reshape the distribution. The two logit
# vectors are hand-chosen to be a "peaked" case (one word dominates) and a "flat" case (mass
# spread widely) -- the contrast that makes top-p's adaptivity visible.
VOCAB: tuple[str, ...] = (
    "the", "cat", "sat", "on", "mat", "dog", "ran", "fast", "blue", "sky",
)
VOCAB_SIZE = len(VOCAB)

# Peaked: "cat" dominates; a steep distribution where a single token owns most of the mass.
PEAKED_LOGITS = (5.0, 8.0, 4.0, 3.5, 3.0, 2.5, 2.0, 1.5, 1.0, 0.5)
# Flat: logits close together; mass is spread, no clear winner -- high uncertainty. Values are
# deliberately all-distinct (no exact ties) so the k-th-largest cutoff is unambiguous -- top-k's
# tie-breaking at the boundary is a separate pitfall covered on the page, not the point here.
FLAT_LOGITS = (2.20, 2.05, 1.95, 2.10, 1.80, 2.00, 1.70, 1.90, 1.85, 1.75)

# ---- Hyperparameters used across the demo (hoisted so page/notebook/figures agree) --------
TEMPERATURES = (0.5, 1.0, 2.0)  # T<1 sharpen, T=1 unchanged, T>1 flatten
TOP_K = 3                       # fixed-size truncation for the top-k demo
TOP_P = 0.9                     # nucleus mass threshold for the top-p demo
SAMPLE_SEED = 0                 # one seed -> reproducible sampled tokens everywhere
N_SAMPLES = 2000                # draws for the empirical-frequency demo
EPS = 1e-12                     # guards log(0) in the entropy sum

# Run on the best available accelerator; CPU is the universal fallback (matches kv_cache.py).
DEVICE = (
    "cuda"
    if torch.cuda.is_available()
    else "mps"
    if torch.backends.mps.is_available()
    else "cpu"
)


def softmax_with_temperature(logits: torch.Tensor, temperature: float) -> torch.Tensor:
    """Temperature-scaled softmax: p_i = softmax(z_i / T).

    T<1 divides logits up -> gaps widen -> sharper (more peaked) distribution.
    T>1 divides logits down -> gaps shrink -> flatter (more uniform) distribution.
    T=1 recovers the plain softmax. T->0 limit is a one-hot at the argmax (= greedy).
    """
    assert temperature > 0, "temperature must be > 0 (T->0 is the greedy limit)"
    return F.softmax(logits / temperature, dim=-1)


def entropy_bits(probs: torch.Tensor) -> float:
    """Shannon entropy H = -sum p log2 p, in bits -- a scalar 'peakiness' measure.

    Lower entropy = more peaked (mass concentrated); higher = flatter (mass spread).
    Max for a 10-way distribution is log2(10) ~= 3.32 bits (perfectly uniform).
    """
    p = probs.clamp_min(EPS)  # clamp so log2(0) never produces -inf * 0 = nan
    return float(-(p * torch.log2(p)).sum().item())


def greedy_token(logits: torch.Tensor) -> int:
    """Greedy decoding: pick the single highest-logit token (argmax). Deterministic."""
    return int(torch.argmax(logits).item())


def top_k_filter(logits: torch.Tensor, k: int) -> torch.Tensor:
    """Keep the k highest-logit tokens, mask the rest to -inf (so softmax zeros them).

    FIXED size: exactly k tokens survive regardless of the distribution's shape -- the
    limitation top-p fixes. Returns filtered LOGITS (renormalise via softmax afterwards).
    """
    assert 1 <= k <= logits.shape[-1], "k must be in [1, vocab_size]"
    kth_value = torch.topk(logits, k).values[..., -1]  # the k-th largest logit = cutoff
    mask = logits < kth_value  # everything below the k-th largest is removed
    return logits.masked_fill(mask, float("-inf"))


def top_p_filter(logits: torch.Tensor, p: float) -> torch.Tensor:
    """Nucleus (top-p) filtering: keep the smallest set of tokens whose cumulative
    probability >= p, mask the rest to -inf.

    ADAPTIVE size: on a peaked distribution a few tokens already cover p (small nucleus);
    on a flat distribution many tokens are needed (large nucleus). Returns filtered LOGITS.
    """
    assert 0 < p <= 1, "p must be in (0, 1]"
    probs = F.softmax(logits, dim=-1)
    sorted_probs, sorted_idx = torch.sort(probs, descending=True)
    cumulative = torch.cumsum(sorted_probs, dim=-1)
    # Keep tokens up to and INCLUDING the first one that crosses p: shifting the cumulative
    # mask right by one means the crossing token itself stays in the nucleus (so the kept
    # mass is always >= p, never just under it).
    remove_sorted = cumulative > p
    remove_sorted[..., 1:] = remove_sorted[..., :-1].clone()
    remove_sorted[..., 0] = False  # the top-1 token is always kept
    remove = torch.zeros_like(remove_sorted).scatter(-1, sorted_idx, remove_sorted)
    return logits.masked_fill(remove, float("-inf"))


def nucleus_size(logits: torch.Tensor, p: float) -> int:
    """How many tokens land inside the top-p nucleus -- the adaptive count itself."""
    kept = torch.isfinite(top_p_filter(logits, p))
    return int(kept.sum().item())


def sample_from_logits(logits: torch.Tensor, generator: torch.Generator) -> int:
    """Sample one token id from logits via the softmax distribution (seeded generator)."""
    probs = F.softmax(logits, dim=-1)
    return int(torch.multinomial(probs, num_samples=1, generator=generator).item())


def empirical_frequencies(
    logits: torch.Tensor, n_samples: int, generator: torch.Generator
) -> torch.Tensor:
    """Draw n_samples tokens and return their observed frequency over the vocab.

    Used to show, empirically, that sampling RECOVERS the (filtered) distribution -- the
    sampled frequencies track the theoretical probabilities the filter produced.
    """
    probs = F.softmax(logits, dim=-1)
    draws = torch.multinomial(probs, num_samples=n_samples, replacement=True, generator=generator)
    return torch.bincount(draws, minlength=logits.shape[-1]).float() / n_samples


def _fmt_probs(probs: torch.Tensor, top: int = 4) -> str:
    """Compact 'word:prob' string for the highest-probability tokens, for readable prints."""
    order = torch.argsort(probs, descending=True)[:top]
    return "  ".join(f"{VOCAB[i]}:{probs[i].item():.3f}" for i in order)


def main() -> None:
    # Pin the reproducible trace to CPU and print the device HONESTLY: the printed device must
    # match where the math actually ran (the recurring device-honesty requirement).
    trace_device = "cpu"
    print(f"device: {trace_device} (detected {DEVICE}; pinned to CPU for reproducibility)")
    print("torch:", torch.__version__)
    print()

    peaked = torch.tensor(PEAKED_LOGITS, device=trace_device)
    flat = torch.tensor(FLAT_LOGITS, device=trace_device)

    # --- 1. Greedy is the argmax (deterministic) -----------------------------------------
    g = greedy_token(peaked)
    print(f"[greedy] peaked -> argmax = '{VOCAB[g]}' (id {g})")
    assert VOCAB[g] == "cat", "greedy must pick the single highest-logit token"
    print()

    # --- 2. Temperature reshapes the softmax: measure entropy (peakiness) at each T -------
    print("[temperature] peaked distribution reshaped by T (entropy in bits, lower = peakier):")
    entropies: dict[float, float] = {}
    for temp in TEMPERATURES:
        probs = softmax_with_temperature(peaked, temp)
        h = entropy_bits(probs)
        entropies[temp] = h
        print(f"   T={temp:>3}:  H={h:.3f} bits   top tokens  {_fmt_probs(probs)}")
    # The contract: hotter temperature => flatter => strictly higher entropy.
    assert entropies[0.5] < entropies[1.0] < entropies[2.0], (
        "entropy must increase with temperature (T sharpens at 0.5, flattens at 2.0)"
    )
    print(
        f"   => entropy rises {entropies[0.5]:.3f} -> {entropies[1.0]:.3f} -> "
        f"{entropies[2.0]:.3f} bits as T goes 0.5 -> 1.0 -> 2.0\n"
    )

    # --- 3. top-k keeps a FIXED count; top-p keeps an ADAPTIVE count ----------------------
    k_peaked = int(torch.isfinite(top_k_filter(peaked, TOP_K)).sum().item())
    k_flat = int(torch.isfinite(top_k_filter(flat, TOP_K)).sum().item())
    print(f"[top-k]  k={TOP_K}: keeps {k_peaked} tokens on PEAKED, {k_flat} on FLAT (always k).")
    assert k_peaked == TOP_K and k_flat == TOP_K, "top-k size is fixed regardless of shape"

    n_peaked = nucleus_size(peaked, TOP_P)
    n_flat = nucleus_size(flat, TOP_P)
    print(f"[top-p]  p={TOP_P}: nucleus = {n_peaked} tokens on PEAKED, {n_flat} on FLAT (adapts).")
    # THE key result: nucleus is SMALLER when peaked, LARGER when flat. This is the whole
    # reason top-p beats fixed top-k -- it widens exactly when the model is uncertain.
    assert n_peaked < n_flat, (
        "nucleus must be smaller on a peaked distribution than on a flat one (top-p adapts)"
    )
    print(
        f"   => top-p adapts: {n_peaked} tokens (peaked) < {n_flat} tokens (flat); "
        f"top-k is stuck at {TOP_K} for both.\n"
    )

    # --- 4. Sampling recovers the filtered distribution (empirical check) -----------------
    # Use the FLAT distribution's nucleus here: it keeps several tokens, so the empirical
    # frequencies actually spread out (the peaked nucleus is ~1 token -- a trivial check).
    gen = torch.Generator(device=trace_device).manual_seed(SAMPLE_SEED)
    filtered = top_p_filter(flat, TOP_P)
    theo = F.softmax(filtered, dim=-1)
    freqs = empirical_frequencies(filtered, N_SAMPLES, gen)
    max_gap = float((theo - freqs).abs().max().item())
    print(f"[sampling] {N_SAMPLES} draws from the top-p={TOP_P} nucleus of the FLAT dist:")
    print(f"   theoretical {_fmt_probs(theo)}")
    print(f"   empirical   {_fmt_probs(freqs)}")
    print(f"   max |theory - empirical| = {max_gap:.3f}  (sampling recovers the distribution)")
    assert max_gap < 0.05, "empirical frequencies must track the theoretical probabilities"
    # Tokens outside the nucleus must NEVER be sampled (their prob was masked to exactly 0).
    assert (freqs[~torch.isfinite(filtered)] == 0).all(), "masked tokens must never be sampled"
    print()

    # --- 5. Temperature widens diversity: distinct tokens over many seeded draws ----------
    # Greedy (T->0) would emit the SAME token every time -> the repetition trap. As T rises,
    # the number of DISTINCT tokens seen over N draws grows -> more diverse, reproducibly.
    print("[diversity] distinct tokens over draws from the peaked dist (seed fixed per T):")
    distinct_counts: dict[float, int] = {}
    for temp in TEMPERATURES:
        gen_t = torch.Generator(device=trace_device).manual_seed(SAMPLE_SEED)
        scaled = peaked / temp
        draws = [sample_from_logits(scaled, gen_t) for _ in range(N_SAMPLES)]
        distinct = len(set(draws))
        distinct_counts[temp] = distinct
        print(f"   T={temp:>3}: {distinct:>2} distinct tokens in {N_SAMPLES} draws")
    # Contract: hotter temperature => more distinct tokens seen (more diversity).
    assert distinct_counts[0.5] < distinct_counts[2.0], (
        "raising temperature must increase token diversity (distinct-token count)"
    )
    print(
        f"   => diversity rises {distinct_counts[0.5]} -> {distinct_counts[1.0]} -> "
        f"{distinct_counts[2.0]} distinct tokens as T goes 0.5 -> 1.0 -> 2.0"
    )


if __name__ == "__main__":
    main()
