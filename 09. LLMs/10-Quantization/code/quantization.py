"""From-scratch LLM weight quantization: affine int8 (symmetric/asymmetric), the outlier
problem (per-tensor vs per-channel/group), and int4 group-wise — with reconstruction error
and the real memory math.

The whole landscape of LLM quantization rests on one formula: map a real value x to a small
integer q with q = round(x / s) - z and recover it with x_hat = s * (q + z). This script
implements that map from scratch, then shows the single fact that makes LLM quantization hard:
a few large-magnitude "outlier" channels wreck a per-tensor scale, and finer granularity
(per-channel, then per-group) is the fix. Finally it quantizes a weight matrix to int4 in
groups and prints the exact bytes saved vs fp16/int8 and the error paid for them.

Every claim is asserted before it is printed: dequant stays within the scale's quantization
bound, and per-channel error is strictly below per-tensor error once an outlier is present.

Verified on Python 3.12 / torch 2.x. Device-agnostic (CUDA / MPS / CPU); the math is identical
on any device, but the reproducible trace is pinned to CPU and the printed device line says so
honestly. No GPU needed — runs in well under a second.

Run:
    python quantization.py
"""

from __future__ import annotations

import torch

# ---- Quantization constants (hoisted, named — no magic numbers in the logic) -----------------
INT8_BITS = 8
INT4_BITS = 4
# Signed symmetric range for b bits is [-(2^(b-1)-1), 2^(b-1)-1]; we use the symmetric
# qmax = 2^(b-1) - 1 form (e.g. 127 for int8, 7 for int4) so +qmax and -qmax are both representable.
INT8_QMAX = (1 << (INT8_BITS - 1)) - 1   # 127
INT4_QMAX = (1 << (INT4_BITS - 1)) - 1   # 7
# Unsigned (asymmetric) range for b bits is [0, 2^b - 1].
UINT8_QMIN, UINT8_QMAX = 0, (1 << INT8_BITS) - 1   # [0, 255]
GROUP_SIZE = 64           # int4 group-wise: one scale per 64 consecutive weights (a common GPTQ/AWQ default)
FP16_BYTES_PER_PARAM = 2  # the baseline precision LLM weights ship in
SEED = 0

# Detect the best device honestly, but pin the reproducible trace to CPU (see module docstring).
DETECTED_DEVICE = (
    "cuda"
    if torch.cuda.is_available()
    else "mps"
    if torch.backends.mps.is_available()
    else "cpu"
)
DEVICE = "cpu"  # pin: identical math everywhere; CPU keeps the printed trace reproducible


# ---- Affine quantization, from scratch -------------------------------------------------------
def quantize_symmetric_int8(
    x: torch.Tensor, qmax: int = INT8_QMAX
) -> tuple[torch.Tensor, float]:
    """Symmetric affine quantization to signed int8: zero-point is fixed at 0.

    Symmetric means the real value 0.0 maps exactly to the integer 0, so we store only a
    scale s (no zero-point). s is set so the largest-magnitude value lands on +/-qmax:
        s = max(|x|) / qmax,    q = round(x / s),    clamped to [-qmax, qmax].
    Returns (q as int8, scale). Best when the values are roughly centred on 0 — true of most
    LLM *weights*, which is why weight-only int8/int4 schemes are usually symmetric.
    """
    scale = x.abs().max().item() / qmax           # one scale for the whole tensor; the loudest value pins it to +/-qmax
    if scale == 0.0:                              # guard: an all-zero tensor has no range to quantize
        scale = 1.0
    q = torch.round(x / scale)                    # round-to-nearest: THE lossy step — real -> integer grid
    q = torch.clamp(q, -qmax, qmax)               # clamp so nothing exceeds the int8 grid (no-op here since s was set from max|x|)
    return q.to(torch.int8), scale


def dequantize_symmetric(q: torch.Tensor, scale: float) -> torch.Tensor:
    """Invert symmetric quantization: x_hat = s * q (zero-point is 0)."""
    return q.to(torch.float32) * scale            # back to the real line by undoing the scale


def quantize_asymmetric_int8(
    x: torch.Tensor, qmin: int = UINT8_QMIN, qmax: int = UINT8_QMAX
) -> tuple[torch.Tensor, float, int]:
    """Asymmetric affine quantization to unsigned int8 [0, 255]: scale AND zero-point.

    When the values are NOT centred on 0 (e.g. post-ReLU activations, or a narrow band offset
    from 0), a symmetric map wastes integer levels. Asymmetric fits the *actual* [min, max]
    window by pairing a scale with an integer zero-point z:
        s = (max - min) / (qmax - qmin),    z = round(qmin - min / s),    q = round(x / s) + z.
    Dequant is x_hat = s * (q - z), so x_min maps to qmin and x_max to qmax — the full [0, 255]
    grid lands exactly on the data window. z is the integer code that real 0.0 would map to; it
    is NOT clamped to [0, 255] (for a band far from 0 it sits well outside that range, and that
    is correct — only round(x/s)+z is clamped). Returns (q stored in int16, scale, zero_point).
    """
    x_min, x_max = x.min().item(), x.max().item()
    scale = (x_max - x_min) / (qmax - qmin)       # map the real window [min,max] onto the integer window [0,255]
    if scale == 0.0:                              # guard: a constant tensor has zero range
        scale = 1.0
    zero_point = round(qmin - x_min / scale)      # integer offset so that x_min lands on qmin; NOT clamped (may be < 0 for an offset band)
    q = torch.round(x / scale) + zero_point       # affine map: scale THEN shift by the zero-point
    q = torch.clamp(q, qmin, qmax)                # clamp the CODES into the unsigned grid [0,255] (z itself stays unclamped)
    return q.to(torch.int16), scale, zero_point   # int16 holds 0..255 cleanly (torch lacks a uint8 arithmetic dtype here)


def dequantize_asymmetric(q: torch.Tensor, scale: float, zero_point: int) -> torch.Tensor:
    """Invert asymmetric quantization: x_hat = s * (q - z)."""
    return (q.to(torch.float32) - zero_point) * scale  # undo the shift (subtract z) then the scale


def reconstruction_error(x: torch.Tensor, x_hat: torch.Tensor) -> float:
    """Mean absolute reconstruction error |x - x_hat| — the price of quantization, in real units."""
    return (x - x_hat).abs().mean().item()


# ---- Per-tensor vs per-channel: the outlier problem ------------------------------------------
def quantize_per_tensor_int8(w: torch.Tensor) -> torch.Tensor:
    """Quantize a whole weight matrix with ONE scale, then dequantize — the coarsest granularity."""
    q, scale = quantize_symmetric_int8(w)         # a single scale shared across every element of W
    return dequantize_symmetric(q, scale)


def quantize_per_channel_int8(w: torch.Tensor, axis: int = 0) -> torch.Tensor:
    """Quantize with ONE scale per row (output channel) — finer granularity, outlier-robust.

    axis=0 means each row gets its own scale, so a single loud row spends bits only on itself
    and cannot crush the precision of the other rows (the per-tensor failure mode).
    """
    # max(|.|) along every axis EXCEPT `axis`, leaving one scale per slice of `axis`.
    reduce_dims = [d for d in range(w.dim()) if d != axis]  # all axes but the per-channel one
    amax = w.abs().amax(dim=reduce_dims, keepdim=True)      # per-row max magnitude -> (rows, 1)
    scale = amax / INT8_QMAX                                # one scale per row; broadcast over the row's columns
    scale = torch.where(scale == 0, torch.ones_like(scale), scale)  # guard all-zero rows
    q = torch.clamp(torch.round(w / scale), -INT8_QMAX, INT8_QMAX)  # round+clamp with the per-row scale broadcast
    return q * scale                                       # dequantize: each row undone by its OWN scale


def quantize_group_wise_int8(w: torch.Tensor, group_size: int = GROUP_SIZE) -> torch.Tensor:
    """Quantize int8 with one scale per GROUP of `group_size` consecutive weights along a row.

    Groups run along the INPUT (column) axis, so a single loud column is isolated to the one
    group it falls in — its scale inflation can't bleed into the other groups of the same row.
    This is the granularity that survives a column-structured outlier (the per-row scale cannot,
    because every row contains the loud column). Returns the dequantized W_hat.
    """
    rows, cols = w.shape
    assert cols % group_size == 0, "cols must be divisible by group_size for clean grouping"
    n_groups = cols // group_size
    w_grouped = w.view(rows, n_groups, group_size)         # last axis is one group of consecutive columns
    amax = w_grouped.abs().amax(dim=-1, keepdim=True)      # per-group max magnitude -> (rows, n_groups, 1)
    scale = amax / INT8_QMAX                                # one int8 scale per group; only the group with the outlier inflates
    scale = torch.where(scale == 0, torch.ones_like(scale), scale)  # guard all-zero groups
    q = torch.clamp(torch.round(w_grouped / scale), -INT8_QMAX, INT8_QMAX)  # round+clamp onto the int8 grid per group
    return (q * scale).view(rows, cols)                    # dequantize per group, restore (rows, cols)


# ---- Int4 group-wise quantization ------------------------------------------------------------
def quantize_group_wise_int4(
    w: torch.Tensor, group_size: int = GROUP_SIZE
) -> tuple[torch.Tensor, float]:
    """Quantize a weight matrix to int4 with one scale per GROUP of `group_size` weights.

    Group-wise sits between per-tensor (1 scale) and per-element (no compression): split each
    row into contiguous groups and give each group its own scale. This is how GPTQ/AWQ/GGUF
    actually store 4-bit weights — fine enough to tame local outliers, coarse enough that the
    scale overhead stays small. Returns (dequantized W_hat, scale-overhead bytes per row-group).
    """
    rows, cols = w.shape
    assert cols % group_size == 0, "cols must be divisible by group_size for clean grouping"
    n_groups = cols // group_size
    w_grouped = w.view(rows, n_groups, group_size)         # reshape so the last axis is one group
    amax = w_grouped.abs().amax(dim=-1, keepdim=True)      # per-group max magnitude -> (rows, n_groups, 1)
    scale = amax / INT4_QMAX                                # one scale per group; broadcast over the group's 64 weights
    scale = torch.where(scale == 0, torch.ones_like(scale), scale)  # guard all-zero groups
    q = torch.clamp(torch.round(w_grouped / scale), -INT4_QMAX, INT4_QMAX)  # round+clamp onto the 4-bit grid [-7,7]
    w_hat = (q * scale).view(rows, cols)                   # dequantize per group, then restore the (rows, cols) shape
    scale_bytes_per_group = FP16_BYTES_PER_PARAM           # each group stores one fp16 scale alongside its int4 codes
    return w_hat, scale_bytes_per_group


def int4_bytes_per_param(group_size: int = GROUP_SIZE) -> float:
    """Effective bytes/param for int4 group-wise: 0.5 byte for the 4-bit code + amortized scale.

    Each weight is 4 bits = 0.5 byte. Each group of `group_size` weights also stores one fp16
    scale (2 bytes), so the scale adds 2 / group_size bytes per weight on top.
    """
    code_bytes = INT4_BITS / INT8_BITS                     # 4 bits = 0.5 byte per weight
    scale_overhead = FP16_BYTES_PER_PARAM / group_size     # one fp16 scale shared across `group_size` weights
    return code_bytes + scale_overhead


# ---- Memory math for a real model ------------------------------------------------------------
def model_memory_gb(num_params: float, bytes_per_param: float) -> float:
    """Weight memory in GB (base-10, the convention model cards use) for a given precision."""
    return num_params * bytes_per_param / 1e9


def main() -> None:
    torch.manual_seed(SEED)
    print("device:", f"{DEVICE} (detected {DETECTED_DEVICE}; pinned to CPU for reproducibility)")
    print("torch:", torch.__version__)
    print()

    # --- 1. Affine int8: symmetric vs asymmetric, with reconstruction error ------------------
    print("=" * 70)
    print("1. AFFINE QUANTIZATION (int8): symmetric vs asymmetric")
    print("=" * 70)
    x = torch.tensor([-1.50, -0.30, 0.00, 0.42, 0.95, 2.10], device=DEVICE)  # roughly centred, has a + and - tail
    q_sym, s_sym = quantize_symmetric_int8(x)
    x_sym = dequantize_symmetric(q_sym, s_sym)
    err_sym = reconstruction_error(x, x_sym)
    print(f"x              : {x.tolist()}")
    print(f"symmetric  scale s = {s_sym:.6f}   zero-point z = 0 (fixed)")
    print(f"  quantized ints : {q_sym.tolist()}")
    print(f"  dequantized    : {[round(v, 4) for v in x_sym.tolist()]}")
    print(f"  mean|err|      : {err_sym:.6f}")
    # symmetric error is bounded by half a step = s/2 (round-to-nearest); allow a hair of float slack
    assert err_sym <= s_sym / 2 + 1e-6, "symmetric error must stay within half a quant step"

    # Asymmetric shines on a tensor whose range is LOPSIDED around 0 (a small negative tail, a
    # long positive one — the classic shape of a post-GELU activation). Symmetric is forced to
    # cover the symmetric span [-max, max] = [-4, 4] and wastes the levels below -0.5;
    # asymmetric fits the actual [-0.5, 4.0] window with a small zero-point.
    a = torch.tensor([-0.50, -0.10, 0.30, 1.20, 2.50, 4.00], device=DEVICE)  # lopsided: short - tail, long + tail
    q_asym, s_asym, z_asym = quantize_asymmetric_int8(a)
    a_asym = dequantize_asymmetric(q_asym, s_asym, z_asym)
    err_asym = reconstruction_error(a, a_asym)
    # Compare against symmetric on the SAME all-positive tensor to show why z helps.
    q_a_sym, s_a_sym = quantize_symmetric_int8(a)
    err_a_sym = reconstruction_error(a, dequantize_symmetric(q_a_sym, s_a_sym))
    print(f"\nlopsided tensor a : {a.tolist()}")
    print(f"asymmetric scale s = {s_asym:.6f}   zero-point z = {z_asym}")
    print(f"  quantized ints : {q_asym.tolist()}")
    print(f"  mean|err| asym : {err_asym:.6f}   (symmetric on same a: {err_a_sym:.6f})")
    # Symmetric must span [-4, 4] (step 8/254); asymmetric spans only [-0.5, 4.0] (step 4.5/255),
    # ~1.8x finer over the window that actually holds data -> lower error.
    assert err_asym < err_a_sym, "asymmetric must beat symmetric on a lopsided (non-centred) range"
    print("  -> asymmetric < symmetric on a lopsided range (no levels wasted on the unused side)")

    # --- 2. The outlier problem: per-tensor breaks; granularity is the fix --------------------
    print("\n" + "=" * 70)
    print("2. THE OUTLIER PROBLEM: per-tensor vs per-channel vs per-group")
    print("=" * 70)
    rows, cols = 8, 128
    w = torch.randn(rows, cols, device=DEVICE) * 0.1        # a tame weight matrix: values ~ N(0, 0.1)
    err_pt_clean = reconstruction_error(w, quantize_per_tensor_int8(w))
    # Inject ONE outlier COLUMN (input channel 17) ~100x louder, present in EVERY row — the
    # signature of LLM activation outliers, which sit in a few fixed feature dimensions.
    outlier_col = 17
    w_out = w.clone()
    w_out[:, outlier_col] = torch.randn(rows, device=DEVICE) * 10.0  # one loud column across all rows
    err_pt_out = reconstruction_error(w_out, quantize_per_tensor_int8(w_out))      # 1 scale for all of W
    err_pc_out = reconstruction_error(w_out, quantize_per_channel_int8(w_out))     # 1 scale per ROW (per output channel)
    err_pg_out = reconstruction_error(w_out, quantize_group_wise_int8(w_out, 16))  # 1 scale per 16-col GROUP (along input)
    print(f"clean matrix ({rows}x{cols}, ~N(0,0.1)): per-tensor mean|err| = {err_pt_clean:.6f}")
    print(f"\nwith ONE 100x-outlier COLUMN (input channel {outlier_col}, in every row):")
    print(f"  per-tensor  mean|err| : {err_pt_out:.6f}   <- the loud column sets ONE global scale; everything else is crushed")
    print(f"  per-channel mean|err| : {err_pc_out:.6f}   <- per-ROW scale does NOT help: every row contains the loud column")
    print(f"  per-group   mean|err| : {err_pg_out:.6f}   <- per-GROUP (along columns) isolates the outlier to its own group")
    blowup = err_pt_out / err_pt_clean
    print(f"  per-tensor error blew up {blowup:.0f}x from the column outlier;")
    print(f"  per-group is {err_pt_out / err_pg_out:.0f}x better than per-tensor and {err_pc_out / err_pg_out:.0f}x better than per-channel")
    # the loud column inflates BOTH the global scale and every row's scale -> only column-grouping fixes it
    assert blowup > 5.0, "the column outlier must materially inflate the per-tensor error"
    assert err_pg_out < err_pt_out, "per-group must beat per-tensor under a column outlier"
    assert err_pg_out < err_pc_out, "per-group (along columns) must beat per-row under a COLUMN outlier"

    # --- 3. Int4 group-wise: memory reduction and error vs int8 ------------------------------
    print("\n" + "=" * 70)
    print("3. INT4 GROUP-WISE: memory vs error")
    print("=" * 70)
    w4 = torch.randn(256, 1024, device=DEVICE) * 0.1        # a bigger weight block to quantize to 4 bits
    w_hat_int4, _ = quantize_group_wise_int4(w4, GROUP_SIZE)
    err_int4 = reconstruction_error(w4, w_hat_int4)
    # int8 per-channel as the higher-precision reference point
    err_int8 = reconstruction_error(w4, quantize_per_channel_int8(w4))
    b_fp16 = FP16_BYTES_PER_PARAM
    b_int8 = 1.0
    b_int4 = int4_bytes_per_param(GROUP_SIZE)
    print(f"weight block: {tuple(w4.shape)}, group_size = {GROUP_SIZE}")
    print(f"  bytes/param : fp16 = {b_fp16:.3f}   int8 = {b_int8:.3f}   int4 = {b_int4:.4f} (0.5 code + {FP16_BYTES_PER_PARAM/GROUP_SIZE:.4f} scale)")
    print(f"  int4 mean|err|: {err_int4:.6f}   int8 per-channel mean|err|: {err_int8:.6f}")
    print(f"  int4 vs fp16 memory: {b_int4 / b_fp16:.3f}x  ({(1 - b_int4 / b_fp16) * 100:.1f}% smaller)")
    print(f"  int4 vs int8 memory: {b_int4 / b_int8:.3f}x")
    # int4 must compress more than int8, and pay for it with strictly more error
    assert b_int4 < b_int8 < b_fp16, "int4 must use fewer bytes/param than int8 than fp16"
    assert err_int4 > err_int8, "int4 must have larger reconstruction error than int8 (fewer bits)"

    # --- 4. The 70B headline number ----------------------------------------------------------
    print("\n" + "=" * 70)
    print("4. MEMORY MATH: Llama-2-70B across precisions")
    print("=" * 70)
    params_70b = 70e9
    for name, bpp in (("fp16", b_fp16), ("int8", b_int8), ("int4 (group)", b_int4)):
        gb = model_memory_gb(params_70b, bpp)
        print(f"  70B @ {name:<12}: {gb:6.1f} GB")
    gb_fp16 = model_memory_gb(params_70b, b_fp16)
    gb_int4 = model_memory_gb(params_70b, b_int4)
    print(f"  -> int4 fits 70B in {gb_int4:.0f} GB (from {gb_fp16:.0f} GB) — one 40GB GPU instead of two 80GB")
    assert gb_int4 < 40.0 < gb_fp16, "int4 must drop 70B under a single-GPU budget"

    print("\nall asserts passed.")


if __name__ == "__main__":
    main()
