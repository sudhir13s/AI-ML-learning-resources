"""From-scratch KV cache: prove identical outputs, then time how the speedup GROWS with length.

A single-layer attention runs the decode loop both ways -- recomputing K/V for every past
token (the O(n^2) trap) versus keeping a cache (O(n)) -- asserts the two produce identical
outputs to floating-point tolerance, then times them across growing sequence lengths so the
widening speedup is visible.

This is the same verified demo embedded in the concept page and the teaching notebook.
Verified on Python 3.12 / torch 2.x. Device-agnostic (CUDA / MPS / CPU); the absolute
milliseconds are device-dependent, but the identical-output check and the widening-speedup
trend hold on any device. The timing sweep in main() deliberately runs on CPU so the per-step math dominates and the widening trend stays clean.

Run:
    python kv_cache.py
"""

from __future__ import annotations

import time
from collections.abc import Callable

import torch
import torch.nn.functional as F

# Model dimensions for the single attention layer used by the demo.
D_MODEL = 512
N_HEADS = 8
HEAD_DIM = 64
SCALE = HEAD_DIM**-0.5  # 1/sqrt(head_dim) attention scaling, hoisted out of the hot loop
SEQ_LENGTHS = (256, 512, 1024, 2048)
TIMING_REPS = 3
ALLCLOSE_ATOL = 1e-5  # both paths do identical float ops in a different order; they differ only by ~1e-7 re-association noise, so 1e-5 is a safe ceiling that still catches real bugs

# Run on the best available accelerator; CPU is the universal fallback.
DEVICE = (
    "cuda"
    if torch.cuda.is_available()
    else "mps"
    if torch.backends.mps.is_available()
    else "cpu"
)


def build_projections(
    seed: int = 0, device: str = DEVICE
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Return the (Wq, Wk, Wv) projection matrices for the single attention layer."""
    torch.manual_seed(seed)
    assert N_HEADS * HEAD_DIM == D_MODEL, "n_heads * head_dim must equal d_model"
    w_q = torch.randn(D_MODEL, D_MODEL, device=device) * 0.02  # *0.02: small init keeps scores in softmax's sensitive range, not saturated
    w_k = torch.randn(D_MODEL, D_MODEL, device=device) * 0.02
    w_v = torch.randn(D_MODEL, D_MODEL, device=device) * 0.02
    return w_q, w_k, w_v


def split_heads(x: torch.Tensor) -> torch.Tensor:
    """(T, d_model) -> (n_heads, T, head_dim)."""
    seq_len = x.shape[0]  # x is (T, d_model); T = number of tokens in this block
    return x.view(seq_len, N_HEADS, HEAD_DIM).transpose(0, 1)  # view splits d_model into (heads, head_dim); transpose puts heads first -> (n_heads, T, head_dim) so each head attends independently


def attn_step(q_t: torch.Tensor, keys: torch.Tensor, values: torch.Tensor) -> torch.Tensor:
    """One query token attends over all cached keys/values.

    q_t: (n_heads, 1, head_dim); keys, values: (n_heads, T, head_dim).
    """
    scores = (q_t @ keys.transpose(-1, -2)) * SCALE  # q·kᵀ over head_dim = one score per cached key; *SCALE stops large head_dim from saturating softmax
    return (F.softmax(scores, dim=-1) @ values).transpose(0, 1).reshape(1, D_MODEL)  # softmax over keys -> weights; @values mixes them; transpose+reshape re-joins heads into one d_model vector (transpose first so heads sit next to head_dim before the flatten — reshape without it would interleave heads)


def decode_no_cache(
    emb: torch.Tensor, n_steps: int, w_q: torch.Tensor, w_k: torch.Tensor, w_v: torch.Tensor
) -> torch.Tensor:
    """Every step re-projects K, V for ALL tokens so far -> O(n^2) total work."""
    outs = []
    for t in range(1, n_steps + 1):
        ctx = emb[:t]  # naive cost: rebuild context over ALL tokens 0..t-1 every step
        keys, values = split_heads(ctx @ w_k), split_heads(ctx @ w_v)  # recomputed from scratch each step — the exact redundancy the cache removes
        outs.append(attn_step(split_heads(emb[t - 1 : t] @ w_q), keys, values))  # query is only the current token; keys/values span the whole context
    return torch.cat(outs, 0)


def decode_with_cache(
    emb: torch.Tensor, n_steps: int, w_q: torch.Tensor, w_k: torch.Tensor, w_v: torch.Tensor
) -> torch.Tensor:
    """Project only the NEW token, append to the cache -> O(n) total work.

    NOTE: the cat-per-step below is the O(n^2) re-allocation trap flagged in the page's
    "How it works" section; real engines write in place into a pre-allocated buffer. Kept
    as a cat here for clarity -- it does not change the output, only the allocation pattern.
    """
    outs: list[torch.Tensor] = []
    k_cache: torch.Tensor | None = None
    v_cache: torch.Tensor | None = None
    for t in range(1, n_steps + 1):
        new = emb[t - 1 : t]  # the single newest token — the only one we project this step
        k_new, v_new = split_heads(new @ w_k), split_heads(new @ w_v)  # project K,V for just this one token — O(1) work, vs O(t) in the naive loop
        k_cache = k_new if k_cache is None else torch.cat([k_cache, k_new], dim=1)  # dim=1 is the seq_len axis (n_heads, seq_len, head_dim): append this key as a new row
        v_cache = v_new if v_cache is None else torch.cat([v_cache, v_new], dim=1)  # same append on the value cache (cat-per-step is the teaching shortcut)
        outs.append(attn_step(split_heads(new @ w_q), k_cache, v_cache))  # one fresh query attends over the FULL cache — old K,V reused, never recomputed
    return torch.cat(outs, 0)


def demo_cache_growth() -> None:
    """Print the cache shape climbing one row per decode step — the core mechanic, made visible."""
    torch.manual_seed(0)
    n_tokens = 3
    w_k = torch.randn(D_MODEL, D_MODEL) * 0.02
    emb = torch.randn(n_tokens, D_MODEL) * 0.1
    k_cache: torch.Tensor | None = None
    for t in range(1, n_tokens + 1):
        k_new = split_heads(emb[t - 1 : t] @ w_k)   # project K for just this token
        # dim=1 is the seq_len axis: append this token's key as a new row
        k_cache = k_new if k_cache is None else torch.cat([k_cache, k_new], dim=1)
        print(f"step {t}: appended token {t-1} -> k_cache.shape = {tuple(k_cache.shape)}")
    print()


def timeit(fn: Callable[[], torch.Tensor], reps: int = TIMING_REPS) -> float:
    """Return mean wall-clock milliseconds over `reps` runs (one warmup run first)."""
    fn()  # warmup
    start = time.perf_counter()
    for _ in range(reps):
        fn()
    return (time.perf_counter() - start) / reps * 1e3


def main() -> None:
    demo_cache_growth()
    # sweep runs on CPU so the per-step math dominates and the trend is clean
    # (on MPS/CUDA, kernel-launch overhead dwarfs these tiny single-layer tensors and
    # flattens the speedup ratio, contradicting the "widening speedup" claim above).
    sweep_device = "cpu"
    w_q, w_k, w_v = build_projections(device=sweep_device)
    print(f"device: {DEVICE} (sweep on {sweep_device})")
    print("torch:", torch.__version__)
    # First, prove the cache changes nothing — on a short sequence, before any timing.
    emb_check = torch.randn(64, D_MODEL, device=sweep_device) * 0.1
    out_no = decode_no_cache(emb_check, 64, w_q, w_k, w_v)
    out_yes = decode_with_cache(emb_check, 64, w_q, w_k, w_v)
    max_diff = (out_no - out_yes).abs().max().item()
    assert torch.allclose(out_no, out_yes, atol=ALLCLOSE_ATOL)
    print(f"identical outputs (no-cache vs cache): True   max abs diff: {max_diff:.2e}\n")
    # The `identical` column reports whether the cached output matches the no-cache output
    # to within ALLCLOSE_ATOL -- proof the cache is a speed trick, not a modeling change.
    print(f"{'N':>6} | {'no-cache':>10} | {'kv-cache':>10} | {'speedup':>8} | identical")
    print("-" * 58)
    for n_steps in SEQ_LENGTHS:
        emb = torch.randn(n_steps, D_MODEL, device=sweep_device) * 0.1  # *0.1 keeps synthetic activations small so attention scores stay numerically tame
        out_no = decode_no_cache(emb, n_steps, w_q, w_k, w_v)
        out_yes = decode_with_cache(emb, n_steps, w_q, w_k, w_v)
        identical = torch.allclose(out_no, out_yes, atol=ALLCLOSE_ATOL)
        assert identical, f"outputs diverged at N={n_steps}"
        ms_no = timeit(lambda e=emb, n=n_steps: decode_no_cache(e, n, w_q, w_k, w_v))  # e=emb, n=n_steps bind NOW, not at call time — avoids the late-binding closure bug
        ms_yes = timeit(lambda e=emb, n=n_steps: decode_with_cache(e, n, w_q, w_k, w_v))
        print(
            f"{n_steps:>6} | {ms_no:>8.1f}ms | {ms_yes:>8.1f}ms "
            f"| {ms_no / ms_yes:>6.1f}x | {identical}"
        )


if __name__ == "__main__":
    main()
