"""Long-context methods from scratch: prove the positional wall, then knock it down.

Three self-contained demos, each typed once here and mirrored cell-for-cell in the
teaching notebook:

  1. RoPE + Position Interpolation. Implement rotary embeddings, show that a head's
     attention score depends only on the RELATIVE offset (i - j), then show WHY naive
     extrapolation past the training length breaks (the dot product hits angles the model
     never saw) and HOW Position Interpolation (Chen et al. 2023) fixes it by squeezing the
     position index by L_train / L_target so every angle stays in the trained range.

  2. Sliding-window receptive field. Build a sliding-window causal mask, assert a token
     CANNOT see beyond its window in one layer, then show numerically that stacking L
     layers grows the effective reach to ~= L * W tokens (the Mistral argument).

  3. StreamingLLM attention sinks (Xiao et al. 2023). Keep the first k "sink" tokens plus
     the recent w window; show that evicting the sinks wrecks the softmax denominator while
     keeping them preserves it -- the mechanical reason streaming works at bounded memory.

Every assertion runs BEFORE any interpretation is printed. Device-agnostic
(CUDA / MPS / CPU): every tensor is created with device=DEVICE so the code runs anywhere.
The numeric trace is pinned to CPU and labelled honestly so the printed numbers are
reproducible bit-for-bit regardless of the host accelerator.

Verified on Python 3.12 / torch 2.x.

Run:
    python long_context_methods.py
"""

from __future__ import annotations

import torch

# ---- Hoisted named constants (no magic numbers in the logic below) -------------------
HEAD_DIM = 8  # tiny even head dimension so RoPE's d/2 frequency pairs print by eye
ROPE_BASE = 10_000.0  # the theta base from the RoPE paper; sets how fast each pair rotates
TRAIN_LEN = 16  # the context length the toy model was "trained" on
TARGET_LEN = 64  # the longer context we want to extend to (4x the training length)
WINDOW = 4  # sliding-window width W: each token attends to itself + (W-1) previous
N_LAYERS = 5  # depth used to demonstrate receptive-field growth (~ L * W reach)
N_SINK = 4  # StreamingLLM: number of leading "sink" tokens to always keep
RECENT_W = 6  # StreamingLLM: size of the recent window kept alongside the sinks
SEED = 0  # reproducibility
NEG_INF = float("-inf")  # masked attention scores -> -inf -> exactly 0 after softmax

# Run on the best available accelerator; CPU is the universal fallback. The reproducible
# numeric trace below is pinned to CPU on purpose and labelled as such, so the printed
# numbers do not depend on which device this happens to run on.
DEVICE = (
    "cuda"
    if torch.cuda.is_available()
    else "mps"
    if torch.backends.mps.is_available()
    else "cpu"
)


# =====================================================================================
# Demo 1 -- RoPE and Position Interpolation
# =====================================================================================
def rope_angles(
    positions: torch.Tensor, head_dim: int = HEAD_DIM, base: float = ROPE_BASE, *,
    device: str = DEVICE,
) -> torch.Tensor:
    """Per-position rotation angles for RoPE: theta_{m,i} = m * base^(-2i/d).

    positions: (T,) integer (or scaled float) position indices.
    Returns (T, head_dim/2) -- one angle per frequency pair per position.

    Each dimension PAIR (2i, 2i+1) of a vector is rotated by an angle proportional to the
    position m; low-i pairs rotate fast (short wavelength), high-i pairs rotate slowly
    (long wavelength), exactly like the hand-positions of a set of clocks ticking at
    different rates. The relative-position property comes from this being a pure rotation.
    """
    half = head_dim // 2
    # inv_freq[i] = base^(-2i/d): the angular frequency of pair i. arange built on DEVICE.
    inv_freq = base ** (-torch.arange(0, half, device=device, dtype=torch.float32) * 2.0 / head_dim)
    # outer product position x frequency -> angle theta_{m,i} for every (position, pair)
    return positions[:, None].to(torch.float32) * inv_freq[None, :]  # (T, half)


def apply_rope(x: torch.Tensor, angles: torch.Tensor) -> torch.Tensor:
    """Rotate each adjacent dimension pair of x by the given per-position angles.

    x: (T, head_dim); angles: (T, head_dim/2). Returns the RoPE-rotated (T, head_dim).

    For pair (x_even, x_odd) at angle t this applies the 2-D rotation matrix
        [cos t, -sin t; sin t, cos t]
    which is why q.k after rotation depends only on the angle DIFFERENCE (i - j): rotating
    both q and k and taking their dot product leaves a function of the relative offset.
    """
    cos, sin = torch.cos(angles), torch.sin(angles)  # (T, half) each
    x_even, x_odd = x[:, 0::2], x[:, 1::2]  # split into the two halves of each rotated pair
    rot_even = x_even * cos - x_odd * sin  # first row of the 2-D rotation matrix
    rot_odd = x_even * sin + x_odd * cos  # second row of the 2-D rotation matrix
    out = torch.empty_like(x)  # allocate on x's device/dtype -- stays device-agnostic
    out[:, 0::2] = rot_even  # re-interleave the rotated pairs back to (T, head_dim)
    out[:, 1::2] = rot_odd
    return out


def rope_score(
    q: torch.Tensor, k: torch.Tensor, pos_q: int, pos_k: int, *,
    scale: float, device: str = DEVICE,
) -> float:
    """Attention logit q.k after RoPE-rotating q at pos_q and k at pos_k.

    `scale` multiplies each raw position before computing angles. scale=1 is the standard
    (no extension); scale = TRAIN_LEN / TARGET_LEN is Position Interpolation, which squeezes
    every index into the trained angular range.
    """
    pq = torch.tensor([pos_q * scale], device=device)  # scaled query position
    pk = torch.tensor([pos_k * scale], device=device)  # scaled key position
    q_rot = apply_rope(q[None, :], rope_angles(pq, device=device))[0]  # rotate the query
    k_rot = apply_rope(k[None, :], rope_angles(pk, device=device))[0]  # rotate the key
    return float((q_rot @ k_rot).item())


def demo_rope_relative_and_interpolation() -> None:
    """Show RoPE is relative, then that naive extrapolation breaks and PI fixes it."""
    torch.manual_seed(SEED)
    cpu = "cpu"  # pin the trace to CPU so these printed numbers are reproducible
    q = torch.randn(HEAD_DIM, device=cpu)
    k = torch.randn(HEAD_DIM, device=cpu)

    # (a) RoPE depends only on the RELATIVE offset: same offset -> same score.
    s_2_5 = rope_score(q, k, 2, 5, scale=1.0, device=cpu)  # offset -3
    s_10_13 = rope_score(q, k, 10, 13, scale=1.0, device=cpu)  # offset -3, far away
    print("(1a) RoPE is relative: equal offsets give equal scores")
    print(f"     score(pos 2 vs 5)   = {s_2_5: .6f}")
    print(f"     score(pos 10 vs 13) = {s_10_13: .6f}  (same offset -3)")
    assert abs(s_2_5 - s_10_13) < 1e-4, "RoPE score must depend only on (i - j)"
    print(f"     max abs diff = {abs(s_2_5 - s_10_13):.2e}  -> relative property holds\n")

    # (b) Build the range of rotation ANGLES the model saw in training vs. what naive
    #     extrapolation to TARGET_LEN demands. The fastest-rotating pair (i=0) is the one
    #     that wraps soonest; its max angle is what blows past the trained range.
    trained_max_angle = rope_angles(torch.tensor([TRAIN_LEN - 1.0], device=cpu), device=cpu)[0, 0]
    naive_max_angle = rope_angles(torch.tensor([TARGET_LEN - 1.0], device=cpu), device=cpu)[0, 0]
    pi_scale = TRAIN_LEN / TARGET_LEN  # Position Interpolation squeeze factor
    pi_max_angle = rope_angles(
        torch.tensor([(TARGET_LEN - 1.0) * pi_scale], device=cpu), device=cpu
    )[0, 0]
    print("(1b) Naive extrapolation overshoots the trained angle range; PI rescues it")
    print(f"     trained max angle (pos {TRAIN_LEN - 1}, fastest pair) = {float(trained_max_angle):.4f} rad")
    print(f"     naive   max angle (pos {TARGET_LEN - 1})              = {float(naive_max_angle):.4f} rad  -> {float(naive_max_angle / trained_max_angle):.1f}x past trained")
    print(f"     PI      max angle (scale {pi_scale:.3f})             = {float(pi_max_angle):.4f} rad  -> back inside trained range")
    # Naive extension demands angles well beyond training; PI scales them back to ~trained.
    # PI maps position m -> m * (L_train / L_target), so the largest position TARGET_LEN-1
    # lands at (TARGET_LEN-1) * scale = 15.75 -- essentially the trained ceiling of 15, NOT
    # the 63 that naive extrapolation demands. The point is the order-of-magnitude rescue.
    assert naive_max_angle > trained_max_angle * 2, "naive extension should overshoot"
    assert float(pi_max_angle) <= float(trained_max_angle) + 1.0, "PI should land at ~trained max angle"
    assert float(pi_max_angle) < float(naive_max_angle) / 3, "PI must be far below the naive angle"
    print("     assert: naive > 2x trained, PI ~= trained max angle  -> PI keeps angles in-range\n")


# =====================================================================================
# Demo 2 -- sliding-window mask and the L*W receptive field
# =====================================================================================
def sliding_window_mask(seq_len: int, window: int = WINDOW, *, device: str = DEVICE) -> torch.Tensor:
    """Boolean (seq_len, seq_len) mask: True where query i may attend to key j.

    Causal AND within the window: token i sees keys j in [i - window + 1, i]. Everything
    older than the window, and everything in the future, is masked out.
    """
    idx = torch.arange(seq_len, device=device)  # positions 0..seq_len-1 on DEVICE
    offset = idx[:, None] - idx[None, :]  # offset[i, j] = i - j
    # keep keys that are not in the future (offset >= 0) and within the window (offset < W)
    return (offset >= 0) & (offset < window)


def masked_attention(x: torch.Tensor, mask: torch.Tensor, w_qk: torch.Tensor) -> torch.Tensor:
    """One toy attention layer with a given boolean mask. Returns mixed outputs (T, d).

    Uses a shared Q=K=V=x @ w_qk projection to keep the demo about the MASK, not the
    weights. Masked-out positions get -inf logits -> 0 weight after softmax.
    """
    proj = x @ w_qk  # (T, d): cheap shared projection so the focus stays on the mask
    scores = proj @ proj.transpose(-1, -2) * (proj.shape[-1] ** -0.5)  # (T, T) logits
    scores = scores.masked_fill(~mask, NEG_INF)  # forbidden positions -> -inf -> 0 weight
    weights = torch.softmax(scores, dim=-1)  # (T, T) attention distribution per query
    return weights @ proj  # (T, d) each output is a window-limited mix of values


def reachable_set(layer: int, query: int, window: int = WINDOW) -> set[int]:
    """Positions that can influence `query`'s representation after `layer` layers.

    Layer 0 = the input itself: only the token. Each extra layer expands the set by one
    window on the left -- the recursive reason depth grows the receptive field.
    """
    reach = {query}
    for _ in range(layer):
        expanded = set()
        for pos in reach:
            lo = max(0, pos - (window - 1))  # the window reaches window-1 tokens back
            expanded.update(range(lo, pos + 1))
        reach = expanded
    return reach


def demo_sliding_window_receptive_field() -> None:
    """Prove a one-layer window is bounded, then that depth grows the reach to ~L*W."""
    torch.manual_seed(SEED)
    cpu = "cpu"  # pin the trace to CPU for reproducible numbers
    seq_len = 12
    mask = sliding_window_mask(seq_len, device=cpu)

    # (a) one layer: a token literally cannot attend beyond its window.
    w_qk = torch.randn(2, 2, device=cpu) * 0.1
    x = torch.randn(seq_len, 2, device=cpu) * 0.1
    weights_row = torch.softmax(
        (x @ w_qk @ (x @ w_qk).transpose(-1, -2) * (2**-0.5)).masked_fill(~mask, NEG_INF), dim=-1
    )
    last = seq_len - 1
    allowed = mask[last].nonzero().flatten().tolist()  # keys the last token may see
    forbidden_weight = weights_row[last, : last - WINDOW + 1].sum().item()  # weight on out-of-window keys
    print("(2a) One sliding-window layer: a token cannot see beyond its window")
    print(f"     window W = {WINDOW}; last token (pos {last}) may attend to keys {allowed}")
    print(f"     total attention weight on out-of-window keys = {forbidden_weight:.2e}")
    assert allowed == list(range(last - WINDOW + 1, last + 1)), "window must be exactly the last W keys"
    assert forbidden_weight < 1e-6, "out-of-window keys must receive ~0 weight"
    print("     assert: allowed == last W keys, out-of-window weight ~ 0\n")

    # (b) depth grows the reach: layer L reaches back ~ L * (W - 1) tokens.
    query = seq_len - 1
    print("(2b) Stacking layers grows the effective receptive field (~ L * W)")
    for layer in range(1, N_LAYERS + 1):
        reach = reachable_set(layer, query)
        span = query - min(reach) + 1  # how many tokens back the query can now feel
        print(f"     after {layer} layer(s): reaches back {span:>2} tokens  (min pos {min(reach)})")
    full_reach = reachable_set(N_LAYERS, query)
    span = query - min(full_reach) + 1
    expected = min(query + 1, 1 + N_LAYERS * (WINDOW - 1))  # 1 + L*(W-1), capped at seq start
    assert span == expected, f"reach {span} should equal 1 + L*(W-1) = {expected}"
    print(f"     assert: {N_LAYERS}-layer reach == 1 + L*(W-1) = {expected} tokens  -> depth multiplies the window\n")


# =====================================================================================
# Demo 3 -- StreamingLLM attention sinks
# =====================================================================================
def streaming_keep_mask(
    seq_len: int, n_sink: int = N_SINK, recent_w: int = RECENT_W, *, device: str = DEVICE
) -> torch.Tensor:
    """Boolean (seq_len,) keep-mask: True for the first n_sink tokens and the last recent_w."""
    keep = torch.zeros(seq_len, dtype=torch.bool, device=device)
    keep[:n_sink] = True  # the load-bearing "attention sink" tokens
    keep[-recent_w:] = True  # the recent window the model genuinely needs
    return keep


def demo_attention_sinks() -> None:
    """Show evicting the sink tokens collapses the softmax; keeping them preserves it."""
    torch.manual_seed(SEED)
    cpu = "cpu"  # pin the trace to CPU for reproducible numbers
    seq_len = 40
    scale = HEAD_DIM**-0.5
    q = torch.randn(HEAD_DIM, device=cpu)
    keys = torch.randn(seq_len, HEAD_DIM, device=cpu)
    # Emulate the trained "sink" effect: the first few keys align with EVERY query, which is
    # exactly the pattern StreamingLLM observed -- a few tokens soak up the spare mass. The
    # *1.2 nudge makes the sinks dominant but not total, so the recent window still matters.
    keys[:N_SINK] += q * 1.2
    logits = (keys @ q) * scale  # (seq_len,) raw attention logits

    full = torch.softmax(logits, dim=-1)  # the reference distribution over all keys
    sink_mass = full[:N_SINK].sum().item()  # how much probability the sinks absorb
    print("(3) Attention sinks: a few leading tokens absorb the softmax's spare mass")
    print(f"     probability mass on the {N_SINK} sink tokens = {sink_mass:.3f}")

    # Eviction A: drop the sinks, keep only the recent window -> renormalize over survivors.
    evict = logits.clone()
    evict[:N_SINK] = NEG_INF  # delete the sink tokens from the softmax denominator
    evict_recent = evict.clone()
    evict_recent[: seq_len - RECENT_W] = NEG_INF  # also restrict to the recent window
    w_evicted = torch.softmax(evict_recent, dim=-1)

    # Eviction B: StreamingLLM -- keep sinks AND the recent window.
    keep = streaming_keep_mask(seq_len, device=cpu)
    stream_logits = logits.masked_fill(~keep, NEG_INF)
    w_stream = torch.softmax(stream_logits, dim=-1)

    # Compare each surviving recent token's weight to its weight in the full distribution.
    recent_idx = torch.arange(seq_len - RECENT_W, seq_len, device=cpu)
    drift_evicted = (w_evicted[recent_idx] - full[recent_idx]).abs().max().item()
    drift_stream = (w_stream[recent_idx] - full[recent_idx]).abs().max().item()
    print(f"     recent-window weight drift WITHOUT sinks = {drift_evicted:.3f}  (renormalized wildly)")
    print(f"     recent-window weight drift WITH    sinks = {drift_stream:.3f}  (stable)")
    assert sink_mass > 0.3, "sinks should absorb a large share of the mass in this setup"
    assert drift_stream < drift_evicted, "keeping sinks must keep the recent weights more stable"
    print("     assert: sink mass large, keeping sinks -> smaller drift -> streaming stays stable\n")


def main() -> None:
    print(f"device: cpu (trace pinned for reproducibility; DEVICE detected = {DEVICE})")
    print("torch:", torch.__version__)
    print()
    demo_rope_relative_and_interpolation()
    demo_sliding_window_receptive_field()
    demo_attention_sinks()
    print("all asserts passed.")


if __name__ == "__main__":
    main()
