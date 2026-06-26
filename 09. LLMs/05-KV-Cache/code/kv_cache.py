"""From-scratch KV cache: prove identical outputs, then time how the speedup GROWS with length.

A single-layer attention runs the decode loop both ways -- recomputing K/V for every past
token (the O(n^2) trap) versus keeping a cache (O(n)) -- asserts the two produce identical
outputs to floating-point tolerance, then times them across growing sequence lengths so the
widening speedup is visible. Runs on CPU in a few seconds; no GPU needed.

This is the same verified demo embedded in 05-KV-Cache.md and 05-KV-Cache.ipynb.
Verified on Python 3.12 / torch 2.x, CPU.

Run:
    python kv_cache.py
"""

from __future__ import annotations

import time

import torch
import torch.nn.functional as F

# Model dimensions for the single attention layer used by the demo.
D_MODEL = 512
N_HEADS = 8
HEAD_DIM = 64
SEQ_LENGTHS = (256, 512, 1024, 2048)
TIMING_REPS = 3
ALLCLOSE_ATOL = 1e-5


def build_projections(seed: int = 0) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Return the (Wq, Wk, Wv) projection matrices for the single attention layer."""
    torch.manual_seed(seed)
    assert N_HEADS * HEAD_DIM == D_MODEL, "n_heads * head_dim must equal d_model"
    w_q = torch.randn(D_MODEL, D_MODEL) * 0.02
    w_k = torch.randn(D_MODEL, D_MODEL) * 0.02
    w_v = torch.randn(D_MODEL, D_MODEL) * 0.02
    return w_q, w_k, w_v


def split_heads(x: torch.Tensor) -> torch.Tensor:
    """(T, d_model) -> (n_heads, T, head_dim)."""
    seq_len = x.shape[0]
    return x.view(seq_len, N_HEADS, HEAD_DIM).transpose(0, 1)


def attn_step(q_t: torch.Tensor, keys: torch.Tensor, values: torch.Tensor) -> torch.Tensor:
    """One query token attends over all cached keys/values.

    q_t: (n_heads, 1, head_dim); keys, values: (n_heads, T, head_dim).
    """
    scores = (q_t @ keys.transpose(-1, -2)) / HEAD_DIM**0.5
    return (F.softmax(scores, dim=-1) @ values).transpose(0, 1).reshape(1, D_MODEL)


def decode_no_cache(
    emb: torch.Tensor, n_steps: int, w_q: torch.Tensor, w_k: torch.Tensor, w_v: torch.Tensor
) -> torch.Tensor:
    """Every step re-projects K, V for ALL tokens so far -> O(n^2) total work."""
    outs = []
    for t in range(1, n_steps + 1):
        ctx = emb[:t]
        keys, values = split_heads(ctx @ w_k), split_heads(ctx @ w_v)  # recomputed from scratch
        outs.append(attn_step(split_heads(emb[t - 1 : t] @ w_q), keys, values))
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
        new = emb[t - 1 : t]
        k_new, v_new = split_heads(new @ w_k), split_heads(new @ w_v)
        k_cache = k_new if k_cache is None else torch.cat([k_cache, k_new], dim=1)
        v_cache = v_new if v_cache is None else torch.cat([v_cache, v_new], dim=1)
        outs.append(attn_step(split_heads(new @ w_q), k_cache, v_cache))
    return torch.cat(outs, 0)


def timeit(fn, reps: int = TIMING_REPS) -> float:
    """Return mean wall-clock milliseconds over `reps` runs (one warmup run first)."""
    fn()  # warmup
    start = time.perf_counter()
    for _ in range(reps):
        fn()
    return (time.perf_counter() - start) / reps * 1e3


def main() -> None:
    w_q, w_k, w_v = build_projections()
    print(f"{'N':>6} | {'no-cache':>10} | {'kv-cache':>10} | {'speedup':>8} | identical")
    print("-" * 58)
    for n_steps in SEQ_LENGTHS:
        emb = torch.randn(n_steps, D_MODEL) * 0.1
        out_no = decode_no_cache(emb, n_steps, w_q, w_k, w_v)
        out_yes = decode_with_cache(emb, n_steps, w_q, w_k, w_v)
        identical = torch.allclose(out_no, out_yes, atol=ALLCLOSE_ATOL)
        ms_no = timeit(lambda e=emb, n=n_steps: decode_no_cache(e, n, w_q, w_k, w_v))
        ms_yes = timeit(lambda e=emb, n=n_steps: decode_with_cache(e, n, w_q, w_k, w_v))
        print(
            f"{n_steps:>6} | {ms_no:>8.1f}ms | {ms_yes:>8.1f}ms "
            f"| {ms_no / ms_yes:>6.1f}x | {identical}"
        )


if __name__ == "__main__":
    main()
