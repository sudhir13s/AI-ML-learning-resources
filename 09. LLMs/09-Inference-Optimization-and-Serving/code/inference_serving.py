"""Inference serving as a back-of-the-envelope model: roofline, batching, speculation.

Three from-scratch models that make the serving mechanism concrete and reproducible. Pure
NumPy -- there is no GPU here and we never pretend otherwise. Every hardware number (HBM
bandwidth, peak FLOP/s, weight bytes, KV bytes/token) is a NAMED constant from a real A100,
and every printed latency/throughput is a *modeled* value computed from those constants, not a
wall-clock measurement. The point is to expose the arithmetic the serving stack is built on:

  1. roofline_table  -- decode is memory-bound; batching B amortizes the weight read across B
     tokens, so throughput rises with B until the compute roofline, then saturates.
  2. batching_sim    -- static vs continuous (in-flight) batching on ragged requests; continuous
     batching fills the gaps static batching wastes on head-of-line blocking.
  3. speculative_speedup -- a small draft model proposes k tokens, the target verifies in one
     pass; expected speedup as a function of draft acceptance rate alpha.

Qualitative asserts (throughput rises with B; continuous >= static; speculation helps when
alpha is high) run BEFORE any number is printed, so the teaching claims are checked, not hoped.

This is the same verified demo embedded in the concept page and the teaching notebook.
Verified on Python 3.12 / NumPy. No torch, no device: every number below is modeled, labelled
as modeled, and reproducible on any machine.

Run:
    python inference_serving.py
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

# --- Hardware constants: one real NVIDIA A100-80GB SXM, FP16 ---------------------------------
HBM_BANDWIDTH_BYTES_PER_S = 2.0e12  # ~2.0 TB/s HBM2e bandwidth (the decode bottleneck)
PEAK_FLOPS = 312e12                 # ~312 TFLOP/s dense FP16 tensor-core throughput
GPU_FLOP_PER_BYTE = PEAK_FLOPS / HBM_BANDWIDTH_BYTES_PER_S  # ~156: the roofline ridge point

# --- Model constants: Llama-3-8B (GQA-8), the cross-linked example from the KV-Cache chapter -
MODEL_PARAMS = 8.0e9                 # 8B parameters
BYTES_PER_PARAM_FP16 = 2            # FP16 weights
WEIGHT_BYTES = MODEL_PARAMS * BYTES_PER_PARAM_FP16  # 16 GB streamed once per decode step
FLOPS_PER_PARAM_PER_TOKEN = 2       # one multiply-add per weight per token = 2 FLOPs
# KV bytes/token = 2(K,V) * n_layers * n_kv_heads * head_dim * bytes; Llama-3-8B: 32,8,128,FP16
KV_BYTES_PER_TOKEN = 2 * 32 * 8 * 128 * BYTES_PER_PARAM_FP16  # = 0.125 MiB/token

MIB = 1024 * 1024
BATCH_SIZES = (1, 8, 32, 64, 128, 156, 256, 512)
ROOFLINE_CONTEXT_TOKENS = 256  # short context: weights dominate, so the compute roofline is reachable


def decode_step_time_s(batch: int, context_tokens: int) -> tuple[float, float]:
    """Modeled time (seconds) for one decode step at batch size `batch`.

    Returns (memory_bound_time, compute_bound_time). The real step takes the MAX of the two:
    the GPU cannot finish until both the bytes are streamed AND the math is done, and one
    always dominates. Memory: read every weight once + each sequence's KV (B * context * kv).
    Compute: 2 * params FLOPs per token * B tokens. Batching amortizes the fixed weight read.
    """
    bytes_moved = WEIGHT_BYTES + batch * context_tokens * KV_BYTES_PER_TOKEN  # weights once + KV per seq
    memory_time = bytes_moved / HBM_BANDWIDTH_BYTES_PER_S  # bandwidth-bound time
    flops = FLOPS_PER_PARAM_PER_TOKEN * MODEL_PARAMS * batch  # compute scales with batch only
    compute_time = flops / PEAK_FLOPS  # compute-bound time
    return memory_time, compute_time


def roofline_crossover_batch(context_tokens: int) -> float | None:
    """The batch size where compute time overtakes memory time (decode becomes compute-bound).

    Solve memory_time(B) == compute_time(B) for B. If the KV term grows faster with B than the
    compute-equivalent term, the line never crosses at this context -- decode stays memory-bound
    forever, and we return None (a real, teachable outcome at long context).
    """
    # WEIGHT + B*ctx*kv = (2*params*B/PEAK_FLOPS) * HBM   (compute_time * HBM expressed in bytes)
    compute_equiv_bytes_per_batch = FLOPS_PER_PARAM_PER_TOKEN * MODEL_PARAMS / PEAK_FLOPS * HBM_BANDWIDTH_BYTES_PER_S
    kv_bytes_per_batch = context_tokens * KV_BYTES_PER_TOKEN
    net = compute_equiv_bytes_per_batch - kv_bytes_per_batch  # bytes of weight-read each extra B can "hide"
    if net <= 0:
        return None  # KV growth outpaces compute headroom -> never compute-bound at this context
    return WEIGHT_BYTES / net


def roofline_table(context_tokens: int = ROOFLINE_CONTEXT_TOKENS) -> list[tuple[int, float, float, str]]:
    """Build (batch, latency_ms_per_token, throughput_tok_per_s, bound) rows over BATCH_SIZES."""
    rows: list[tuple[int, float, float, str]] = []
    for batch in BATCH_SIZES:
        memory_time, compute_time = decode_step_time_s(batch, context_tokens)
        step_time = max(memory_time, compute_time)  # the slower resource sets the wall clock
        bound = "memory" if memory_time >= compute_time else "compute"
        throughput = batch / step_time  # B tokens produced per step / step time = tokens/s
        rows.append((batch, step_time * 1e3, throughput, bound))
    return rows


# --- Static vs continuous batching ----------------------------------------------------------

@dataclass(frozen=True)
class Request:
    """A serving request: a label and how many decode steps (output tokens) it needs."""

    name: str
    output_tokens: int


def simulate_static_batching(
    requests: list[Request], step_ms: float, slots: int
) -> tuple[float, float]:
    """Static batching: the engine runs requests in fixed WAVES of `slots`. A wave is locked --
    a finished request's slot sits IDLE until the wave's LONGEST request completes; only then
    does the next wave start (head-of-line blocking).

    This is the cost static batching pays on ragged sequences: short requests in a wave waste
    their slot for every step the wave's longest request is still running.
    Returns (makespan_ms, gpu_utilization). Utilization = useful token-steps / offered slot-steps.
    """
    total_steps = 0  # wall-clock steps across all waves
    useful_steps = 0  # steps that actually produced a token
    for wave_start in range(0, len(requests), slots):  # carve requests into back-to-back waves
        wave = requests[wave_start : wave_start + slots]
        wave_length = max(request.output_tokens for request in wave)  # wave ends at its slowest request
        total_steps += wave_length  # the whole wave is held this long, idle slots included
        useful_steps += sum(request.output_tokens for request in wave)  # real token-steps in the wave
    makespan_ms = total_steps * step_ms
    offered_slot_steps = total_steps * slots  # capacity the engine offered over the run
    utilization = useful_steps / offered_slot_steps
    return makespan_ms, utilization


def simulate_continuous_batching(
    requests: list[Request], step_ms: float, slots: int
) -> tuple[float, float]:
    """Continuous (in-flight) batching: the moment a request finishes, its slot is refilled with
    a queued request the very NEXT step -- no wave barrier, no idle slots while work remains
    (the Orca idea). Same `slots`-wide engine, same requests, no head-of-line blocking.

    Each step, every occupied slot advances one token; finished requests free their slot and a
    waiting request takes it immediately. Returns (makespan_ms, gpu_utilization).
    """
    remaining = [request.output_tokens for request in requests]  # tokens left per request
    queue = list(range(len(requests)))  # indices waiting to be admitted
    active: list[int] = []  # indices currently occupying a slot
    steps = 0
    busy_slot_steps = 0
    while queue or active:
        while len(active) < slots and queue:  # backfill every free slot before stepping
            active.append(queue.pop(0))
        busy_slot_steps += len(active)  # slots doing real work this step (no idle wait)
        for index in active:
            remaining[index] -= 1  # advance each active request one decode step
        active = [index for index in active if remaining[index] > 0]  # release finished slots now
        steps += 1
    makespan_ms = steps * step_ms
    offered_slot_steps = steps * slots  # capacity offered over the whole run
    utilization = busy_slot_steps / offered_slot_steps
    return makespan_ms, utilization


# --- Speculative decoding -------------------------------------------------------------------

def speculative_speedup(alpha: float, draft_k: int, draft_cost_ratio: float) -> float:
    """Modeled wall-clock speedup of speculative decoding vs plain autoregressive decode.

    A draft model proposes `draft_k` tokens; the target verifies all of them in ONE forward pass
    and accepts the longest correct prefix. With per-token acceptance probability `alpha`, the
    expected number of tokens accepted per target pass is the truncated-geometric mean
        E[accepted] = (1 - alpha**(k+1)) / (1 - alpha)
    (Leviathan et al. 2023, Eq. for the expected acceptance). Each target verification also pays
    for `draft_k` cheap draft steps at `draft_cost_ratio` of a target step. So:
        speedup = E[accepted] / (1 + draft_k * draft_cost_ratio)
    >1 means faster than vanilla decode; <1 means the draft overhead is not earning its keep.
    """
    if not 0.0 <= alpha < 1.0:
        raise ValueError("alpha (acceptance rate) must be in [0, 1)")
    expected_accepted = (1.0 - alpha ** (draft_k + 1)) / (1.0 - alpha)  # mean accepted draft prefix + 1 (the target's always-accepted bonus token after the last accepted draft)
    target_passes_cost = 1.0 + draft_k * draft_cost_ratio  # one target pass + k draft steps
    return expected_accepted / target_passes_cost


# --- Asserts: check the teaching claims BEFORE printing any number --------------------------

def _verify_claims() -> None:
    """Qualitative checks that must hold for the lessons to be true. Run before any output."""
    # 1. Throughput rises monotonically with batch in the memory-bound regime, then saturates.
    rows = roofline_table()
    throughputs = [row[2] for row in rows]
    assert throughputs[0] < throughputs[1] < throughputs[2], "throughput must rise with batch while memory-bound"
    assert throughputs[-1] >= throughputs[0], "large-batch throughput must not fall below batch-1"
    # The smallest batches are memory-bound (weight read dominates), the largest is compute-bound.
    assert rows[0][3] == "memory", "batch-1 decode must be memory-bound"
    assert rows[-1][3] == "compute", "very large batch must reach the compute roofline"

    # 2. On ragged requests with more work than slots, continuous batching strictly beats static.
    ragged = [Request("a", 4), Request("b", 6), Request("c", 20), Request("d", 60), Request("e", 5), Request("f", 8)]
    step_ms = 10.0
    static_makespan, static_util = simulate_static_batching(ragged, step_ms, slots=3)
    cont_makespan, cont_util = simulate_continuous_batching(ragged, step_ms, slots=3)
    assert cont_makespan < static_makespan, "continuous batching must finish strictly sooner here"
    assert cont_util > static_util, "continuous batching must use the GPU strictly better here"

    # 3. Speculation helps when acceptance is high; the draft overhead can make it hurt when low.
    high = speculative_speedup(alpha=0.8, draft_k=4, draft_cost_ratio=0.1)
    low = speculative_speedup(alpha=0.1, draft_k=4, draft_cost_ratio=0.4)
    assert high > 1.0, "high acceptance should give a real speedup"
    assert high > low, "speedup must increase with acceptance rate"


def main() -> None:
    _verify_claims()  # all teaching claims checked before a single number is shown
    print("All values below are MODELED from named A100/Llama-3-8B constants (no GPU, no wall-clock).")
    print(f"HBM bandwidth: {HBM_BANDWIDTH_BYTES_PER_S/1e12:.1f} TB/s | peak: {PEAK_FLOPS/1e12:.0f} TFLOP/s | "
          f"ridge: {GPU_FLOP_PER_BYTE:.0f} FLOP/byte")
    print(f"Weights: {WEIGHT_BYTES/1e9:.0f} GB (FP16) | KV/token: {KV_BYTES_PER_TOKEN/MIB:.3f} MiB (GQA-8)\n")

    # 1) Roofline: throughput vs batch -------------------------------------------------------
    print(f"[1] Decode roofline at {ROOFLINE_CONTEXT_TOKENS}-token context (modeled):")
    print(f"{'batch':>6} | {'latency/token':>14} | {'throughput':>16} | bound")
    print("-" * 56)
    for batch, latency_ms, throughput, bound in roofline_table():
        print(f"{batch:>6} | {latency_ms:>11.2f} ms | {throughput:>11.0f} tok/s | {bound}")
    crossover = roofline_crossover_batch(ROOFLINE_CONTEXT_TOKENS)
    crossover_text = f"B* = {crossover:.0f}" if crossover is not None else "never (memory-bound at all B)"
    print(f"compute-roofline crossover: {crossover_text}")
    long_crossover = roofline_crossover_batch(2048)
    long_text = f"B* = {long_crossover:.0f}" if long_crossover is not None else "never (KV growth outpaces compute headroom)"
    print(f"  ...at 2048-token context: {long_text}\n")

    # 2) Static vs continuous batching -------------------------------------------------------
    ragged = [Request("a", 4), Request("b", 6), Request("c", 20), Request("d", 60), Request("e", 5), Request("f", 8)]
    slots = 3  # engine width: 3 concurrent slots, 6 ragged requests (4/6/20/60/5/8 tokens)
    step_ms = 10.0  # one modeled decode step; constant here to isolate the batching effect
    static_makespan, static_util = simulate_static_batching(ragged, step_ms, slots)
    cont_makespan, cont_util = simulate_continuous_batching(ragged, step_ms, slots)
    print("[2] Static vs continuous batching: 6 ragged requests (4/6/20/60/5/8 tok), 3 slots, modeled:")
    print(f"  static     : makespan {static_makespan:6.0f} ms | GPU util {static_util*100:4.0f}%")
    print(f"  continuous : makespan {cont_makespan:6.0f} ms | GPU util {cont_util*100:4.0f}%")
    print(f"  -> continuous backfills freed slots immediately: {static_util:.0%} -> {cont_util:.0%} util, "
          f"makespan {static_makespan/cont_makespan:.2f}x shorter\n")

    # 3) Speculative decoding ----------------------------------------------------------------
    print("[3] Speculative decoding expected speedup (draft_k=4, draft_cost=0.1x target), modeled:")
    print(f"{'acceptance alpha':>18} | {'expected speedup':>16}")
    print("-" * 38)
    for alpha in (0.1, 0.3, 0.5, 0.7, 0.8, 0.9):
        speedup = speculative_speedup(alpha=alpha, draft_k=4, draft_cost_ratio=0.1)
        print(f"{alpha:>18.1f} | {speedup:>14.2f}x")
    print("  -> speedup climbs with acceptance; a cheap, well-aligned draft is what makes it pay.")


if __name__ == "__main__":
    main()
