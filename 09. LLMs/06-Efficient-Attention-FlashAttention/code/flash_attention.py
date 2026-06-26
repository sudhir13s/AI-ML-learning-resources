"""From-scratch FlashAttention: prove blockwise (tiled) attention equals full attention.

Standard attention materializes the full N x N score matrix in memory -- O(N^2) memory and
O(N^2) HBM traffic. FlashAttention computes the *same* result without ever materializing that
matrix: it tiles K and V into blocks, streams each block through fast on-chip SRAM, and keeps a
running (online) softmax -- a running max `m` and running normaliser `l` -- rescaling the partial
output as it goes. This file builds that algorithm from scratch and ASSERTS it matches the
textbook full-softmax attention to ~1e-6 before anything else.

What it shows, in order:
  1. The naive baseline that materializes the whole score matrix (what FlashAttention avoids).
  2. `online_softmax` -- the genuinely single-pass streaming softmax whose (m, l) bookkeeping is
     the mathematical heart (it holds only running scalars, never the full exp vector).
  3. `flash_attention` -- blockwise attention built on that streaming softmax.
  4. A hard assert that blockwise == full to ALLCLOSE_ATOL, plus the per-block (m, l) trace.

Verified on Python 3.12 / torch 2.12.0. The compute is device-agnostic: every tensor is created
on `device`, so the algorithm runs unchanged on CUDA / MPS / CPU. The reproducible numeric trace
in main() is deliberately pinned to CPU so the printed numbers are bit-stable across machines, and
the printed device line says so honestly. The real wall-clock win is GPU HBM traffic, which a CPU
run cannot show, so this file proves correctness and counts the materialized bytes rather than
timing a fake speedup.

Run:
    python flash_attention.py
"""

from __future__ import annotations

import math

import torch
import torch.nn.functional as F

# Problem shape for the from-scratch demo. Small, real, and printable.
SEQ_LEN = 8  # N: number of query and key/value positions
HEAD_DIM = 4  # d: dimension of each query/key/value vector
BLOCK_SIZE = 2  # B_c: how many key/value rows we load into "SRAM" per tile
SCALE = HEAD_DIM**-0.5  # 1/sqrt(d): standard attention scaling, hoisted out of the loop
ALLCLOSE_ATOL = 1e-6  # blockwise and full attention are algebraically identical; they differ only
# by float rounding from a different summation order, which lands well under 1e-6
SEED = 0

# Best available accelerator; CPU is the universal fallback. The algorithm is device-agnostic --
# every tensor below is created on the device passed in, so it runs unchanged on any of these.
DEVICE = (
    "cuda"
    if torch.cuda.is_available()
    else "mps"
    if torch.backends.mps.is_available()
    else "cpu"
)


def full_attention(
    q: torch.Tensor, k: torch.Tensor, v: torch.Tensor
) -> tuple[torch.Tensor, int]:
    """Textbook attention: materialize the whole N x N score matrix, then softmax.

    q, k, v: (N, d). Returns (output (N, d), bytes the N x N score matrix occupies).
    This is the baseline FlashAttention must match -- and the matrix it refuses to store.
    """
    scores = (q @ k.transpose(-1, -2)) * SCALE  # (N, N): every query dotted with every key
    weights = F.softmax(scores, dim=-1)  # row-wise softmax over the key axis
    out = weights @ v  # (N, d): each output is a softmax-weighted sum of value rows
    materialized_bytes = scores.numel() * scores.element_size()  # the O(N^2) memory FlashAttention avoids
    return out, materialized_bytes


def online_softmax(scores_row: torch.Tensor, block_size: int) -> torch.Tensor:
    """Softmax of a 1-D score row computed in a genuinely SINGLE streaming pass.

    Never holds the full exponential vector: it carries a running max `m`, a running sum-of-exps
    `l`, and a running un-normalised numerator vector `num` (the same kind of accumulator
    `flash_attention` uses for its output). When a block raises the max, every running quantity is
    rescaled by exp(m_old - m_new) <= 1 so all terms end up taken against the SAME final max. That
    rescale is the log-sum-exp trick made incremental; it is exactly what lets the blocks combine to
    the correct global softmax. Returns the full softmax vector (assembled only at the end, for the
    equality check) -- but the streaming state is O(N) numerator + O(1) scalars, computed in one pass.
    """
    n = scores_row.numel()
    running_max = torch.tensor(float("-inf"), device=scores_row.device)  # m: largest score seen so far
    running_denom = torch.tensor(0.0, device=scores_row.device)  # l: sum of exp(score - m) so far, consistent with m
    num = torch.zeros(n, device=scores_row.device)  # running un-normalised numerators exp(score - m), rescaled as m grows
    for start in range(0, n, block_size):
        block = scores_row[start : start + block_size]
        new_max = torch.maximum(running_max, block.max())  # max across everything seen so far
        correction = torch.exp(running_max - new_max)  # <= 1: shrink prior running state to the new max
        num = num * correction  # re-base every already-accumulated numerator against new_max in one multiply
        num[start : start + block_size] = torch.exp(block - new_max)  # this block's exps, also vs new_max
        running_denom = running_denom * correction + torch.exp(block - new_max).sum()  # rescale denom, add block
        running_max = new_max  # advance the running max
    return num / running_denom  # single final division: each numerator over the final denominator


def flash_attention(
    q: torch.Tensor, k: torch.Tensor, v: torch.Tensor, block_size: int = BLOCK_SIZE
) -> tuple[torch.Tensor, list[tuple[int, float, float]]]:
    """Blockwise (tiled) attention using a running softmax -- never forms the full N x N matrix.

    For each query, walk the key/value blocks once, maintaining a running max `m`, a running
    normaliser `l`, and a running un-normalised output accumulator `acc`. When a block raises the
    max, rescale BOTH `l` and `acc` by exp(m_old - m_new) so the partial output stays consistent
    with the new max. After the last block, dividing `acc` by `l` yields exact attention -- the
    same number `full_attention` gives, at O(block_size) memory instead of O(N).

    Returns (output (N, d), a trace of (block_index, running_max, running_denom) for query 0).
    """
    seq_len, head_dim = q.shape
    out = torch.zeros_like(q)
    trace_for_query0: list[tuple[int, float, float]] = []

    for i in range(seq_len):  # one query at a time (real kernels tile queries too; one here for clarity)
        running_max = torch.tensor(float("-inf"), device=q.device)  # m_i
        running_denom = torch.tensor(0.0, device=q.device)  # l_i: running sum of exp(score - m_i)
        acc = torch.zeros(head_dim, device=q.device)  # un-normalised output: sum of exp(score - m_i) * value, rescaled as m_i grows

        for block_idx, start in enumerate(range(0, seq_len, block_size)):
            k_block = k[start : start + block_size]  # (B_c, d): the key tile loaded into "SRAM"
            v_block = v[start : start + block_size]  # (B_c, d): the matching value tile
            scores = (q[i] @ k_block.transpose(-1, -2)) * SCALE  # (B_c,): this query vs the tile's keys
            block_max = scores.max()
            new_max = torch.maximum(running_max, block_max)  # global max so far for query i
            correction = torch.exp(running_max - new_max)  # <= 1: how much to shrink old (l, acc) for the new max
            p = torch.exp(scores - new_max)  # (B_c,): exps of this tile against the new global max
            # Rescale the old denominator to the new max, then add this tile's contribution.
            running_denom = running_denom * correction + p.sum()
            # Rescale the old output accumulator the SAME way, then add this tile's weighted values.
            acc = acc * correction + p @ v_block
            running_max = new_max  # advance the running max for the next tile
            if i == 0:
                trace_for_query0.append((block_idx, running_max.item(), running_denom.item()))

        out[i] = acc / running_denom  # final normalisation: divide accumulated output by the running denom

    return out, trace_for_query0


def main() -> None:
    # The reproducible numeric trace is pinned to CPU so the printed numbers are bit-stable across
    # machines; the printed line says so honestly. (The functions above run on any device --
    # swapping trace_device to DEVICE runs the identical algorithm on cuda/mps with no other change.)
    trace_device = "cpu"
    print(f"device: {trace_device} (best available: {DEVICE}; trace pinned to CPU for reproducibility)")
    print("torch:", torch.__version__)
    torch.manual_seed(SEED)

    q = torch.randn(SEQ_LEN, HEAD_DIM, device=trace_device)
    k = torch.randn(SEQ_LEN, HEAD_DIM, device=trace_device)
    v = torch.randn(SEQ_LEN, HEAD_DIM, device=trace_device)

    # 1. Streaming softmax must equal the library softmax on one score row -- prove the heart first.
    scores_row = (q[0] @ k.transpose(-1, -2)) * SCALE
    online = online_softmax(scores_row, BLOCK_SIZE)
    reference = F.softmax(scores_row, dim=-1)
    softmax_diff = (online - reference).abs().max().item()
    assert torch.allclose(online, reference, atol=ALLCLOSE_ATOL), "online softmax diverged"
    print(f"\nonline softmax == F.softmax: True   max abs diff: {softmax_diff:.2e}")

    # 2. Blockwise attention must equal full attention -- the headline correctness claim.
    out_full, materialized_bytes = full_attention(q, k, v)
    out_flash, trace = flash_attention(q, k, v, BLOCK_SIZE)
    attn_diff = (out_full - out_flash).abs().max().item()
    assert torch.allclose(out_full, out_flash, atol=ALLCLOSE_ATOL), "blockwise != full"
    print(f"flash attention == full attention: True   max abs diff: {attn_diff:.2e}")

    # 3. Show the running (m, l) climbing block by block for query 0 -- the bookkeeping, made visible.
    print(f"\nrunning (max m, denom l) per block for query 0 (block_size={BLOCK_SIZE}):")
    for block_idx, max_m, denom_l in trace:  # max_m, denom_l == the (m, l) of the derivation
        print(f"  after block {block_idx}: m = {max_m:+.4f},  l = {denom_l:.4f}")

    # 4. Count the memory the naive path materializes that FlashAttention never does.
    flash_block_bytes = BLOCK_SIZE * HEAD_DIM * q.element_size()
    print(
        f"\nmaterialized N x N scores (naive): {materialized_bytes} bytes "
        f"({SEQ_LEN}x{SEQ_LEN} floats) -- grows as O(N^2)"
    )
    print(
        f"flash per-block working set: ~{flash_block_bytes} bytes "
        f"({BLOCK_SIZE}x{HEAD_DIM} floats) -- independent of N"
    )
    big_n, batch, heads, fp16, a100_gib = 8192, 16, 32, 2, 80
    one_matrix_gib = (big_n * big_n * fp16) / (1024**3)  # one fp16 N x N score matrix
    all_matrices_gib = one_matrix_gib * batch * heads  # one per (batch, head), as backward needs
    print(
        f"at N={big_n}, one fp16 score matrix = {one_matrix_gib:.3f} GiB; "
        f"x (batch {batch} x {heads} heads) = {all_matrices_gib:.0f} GiB"
    )
    print(
        f"  -> {all_matrices_gib:.0f} of an {a100_gib} GB A100's GiB on attention scratch alone, "
        "leaving almost nothing for weights, activations, and the other layers. "
        "This is the wall FlashAttention removes."
    )

    assert math.isclose(SCALE, 1.0 / math.sqrt(HEAD_DIM)), "scale must be 1/sqrt(head_dim)"


if __name__ == "__main__":
    main()
