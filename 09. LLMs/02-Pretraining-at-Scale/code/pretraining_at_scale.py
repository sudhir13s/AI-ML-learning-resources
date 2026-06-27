"""Pretraining-at-scale mechanics, from scratch: the LR schedule, grad-accum equivalence,
a tiny real-recipe training loop, and a toy MFU.

Four self-contained demos that make the *systems* of pretraining concrete -- because at scale
the objective (next-token cross-entropy) is the easy part and the engineering around it is the
work:
  1. The warmup + cosine learning-rate schedule, printed across steps so the ramp-then-decay
     shape is visible and the endpoints (peak at end of warmup, floor at the end) are asserted.
  2. Gradient-accumulation EQUIVALENCE: K micro-batches of size m produce (to float tolerance)
     the same gradient as one batch of size m*K -- the "fake a big batch" trick, proven.
  3. A tiny end-to-end pretraining loop with the REAL recipe (AdamW + warmup->cosine + grad clip)
     so the loss visibly drops on a toy corpus.
  4. A toy MFU (model FLOPs utilization) = achieved 6ND-rate / peak FLOPs.

This is the same verified demo embedded in the concept page and the teaching notebook.
Verified on Python 3.12 / torch 2.x. Device-agnostic (CUDA / MPS / CPU); the trace deliberately
runs on CPU (TRACE_DEVICE) so the printed loss curve and LR/grad numbers reproduce on every
machine -- MPS/CUDA reorder float ops and shift only the low-order digits.

Run:
    python pretraining_at_scale.py
"""

from __future__ import annotations

import math

import torch
import torch.nn as nn
import torch.nn.functional as F

# --- Learning-rate schedule hyperparameters (named, not magic) ---------------
LR_PEAK = 3e-4  # the maximum learning rate, reached at the end of warmup
LR_FLOOR = 3e-5  # the minimum LR the cosine decays down to (here 10% of peak, a common choice)
WARMUP_STEPS = 10  # linear ramp-in length for the standalone schedule demo
TOTAL_STEPS_SCHEDULE = 100  # total steps for the standalone schedule demo
LR_ENDPOINT_TOL = 1e-12  # endpoints must hit peak/floor exactly (pure arithmetic, no float drift)

# --- Gradient-accumulation demo dimensions -----------------------------------
MICRO_BATCH = 4  # m: examples per micro-batch
ACCUM_STEPS = 8  # K: number of micro-batches accumulated before one optimizer step
FEATURE_DIM = 16  # input width of the toy linear model
GRAD_EQUIV_ATOL = 1e-6  # the two paths feed identical data through identical math; they differ only by float rounding from one big matmul vs K small ones (~1e-9), so 1e-6 is a safe ceiling that still catches a real bug (e.g. a missing /K)

# --- Tiny pretraining-loop hyperparameters (the real recipe, scaled down) ----
VOCAB_SIZE = 16  # toy vocabulary
CONTEXT_LEN = 8  # sequence length per training example
EMBED_DIM = 32  # token embedding width
PATTERN_PERIOD = 4  # the toy corpus repeats a length-4 cycle, so next-token is LEARNABLE (a clean loss drop)
CORPUS_TOKENS = 256  # size of the toy token-id corpus
TRAIN_STEPS = 300  # optimizer steps for the tiny training loop
WARMUP_STEPS_TRAIN = 20  # warmup length for the training loop's schedule
ADAM_BETAS = (0.9, 0.95)  # beta2=0.95 (not the 0.999 default) is the LLM-stability choice
WEIGHT_DECAY = 0.1  # AdamW decoupled weight decay, a standard pretraining value
GRAD_CLIP_NORM = 1.0  # clip the global grad norm: the seatbelt against loss spikes
REPORT_EVERY = 60  # print one training row every this many steps
FINAL_LOSS_BOUND = 1.0  # the toy loss must end below this; the periodic corpus is learnable so the loss drops clearly, and the bound holds on CPU/MPS/CUDA alike

# --- Toy MFU inputs -----------------------------------------------------------
PRETEND_TOKENS_PER_SEC = 5e4  # a stand-in throughput measurement for the MFU demo
PRETEND_PEAK_FLOPS = 1e12  # a stand-in hardware peak (real GPUs ~3e14 bf16 for an A100, up to ~1e15 for H100)
FLOPS_PER_PARAM_PER_TOKEN = 6  # the 6 in 6ND: ~2N forward + ~4N backward FLOPs per token
NUM_GPUS = 1  # num_GPUs=1 here; a real run divides by the whole cluster's peak

SEED = 0

# Run on the best available accelerator; CPU is the universal fallback.
DEVICE = (
    "cuda"
    if torch.cuda.is_available()
    else "mps"
    if torch.backends.mps.is_available()
    else "cpu"
)

# Run the reproducible trace on CPU so the printed numbers match the page/notebook on every
# machine (MPS/CUDA reorder float ops and shift the low-order digits).
TRACE_DEVICE = "cpu"


def lr_at_step(
    step: int,
    peak: float = LR_PEAK,
    floor: float = LR_FLOOR,
    warmup: int = WARMUP_STEPS,
    total: int = TOTAL_STEPS_SCHEDULE,
) -> float:
    """Return the learning rate at `step` under a linear-warmup + cosine-decay schedule.

    Warmup (step < warmup): a straight line from 0 up to `peak` -- ramps the LR in gently so a
    full-size step doesn't wreck randomly-initialized weights before Adam's stats settle.
    Cosine decay (step >= warmup): a smooth half-cosine from `peak` down to `floor`.
    """
    if step < warmup:
        return peak * step / warmup  # linear warmup: 0 at step 0, exactly `peak` at step==warmup
    progress = (step - warmup) / (total - warmup)  # OFF-BY-ONE FIX: the cosine clock restarts AT warmup, not at step 0
    # cos goes 1 -> -1 as progress goes 0 -> 1, so the bracket goes peak -> floor smoothly
    return floor + 0.5 * (peak - floor) * (1 + math.cos(math.pi * progress))


def make_linear(seed: int, in_dim: int) -> nn.Linear:
    """Return a fresh single-output Linear with identical init given the same seed.

    Re-seeding before construction guarantees the 'big batch' and 'accumulated' models start
    from the *same* weights, so any gradient difference is the accumulation logic, not the init.
    """
    torch.manual_seed(seed)
    return nn.Linear(in_dim, 1)


def big_batch_grad(
    model: nn.Linear, features: torch.Tensor, targets: torch.Tensor
) -> torch.Tensor:
    """Gradient of the mean MSE loss over the FULL batch in one backward pass."""
    F.mse_loss(model(features), targets).backward()  # mean over all m*K examples -> one gradient
    assert model.weight.grad is not None  # backward populated .grad; assert for the type checker
    return model.weight.grad.clone()  # clone so a later zero_grad can't mutate what we return


def accumulated_grad(
    model: nn.Linear,
    features: torch.Tensor,
    targets: torch.Tensor,
    micro_batch: int,
    accum_steps: int,
) -> torch.Tensor:
    """Gradient assembled by accumulating over `accum_steps` micro-batches of size `micro_batch`.

    Each micro-batch's mean loss is divided by accum_steps BEFORE backward, so summing the
    accumulated gradients gives the grand mean -- the mean-of-means identity. Drop the /K and the
    accumulated gradient is accum_steps-times too large (the classic large-batch bug).
    """
    for k in range(accum_steps):
        start = k * micro_batch  # this micro-batch covers examples [start, start+micro_batch)
        chunk_x = features[start : start + micro_batch]
        chunk_y = targets[start : start + micro_batch]
        # /accum_steps: averaging K micro-batch means == the single grand mean (linearity of grad)
        (F.mse_loss(model(chunk_x), chunk_y) / accum_steps).backward()  # grads ACCUMULATE in .grad across iterations (no zero_grad inside the loop)
    assert model.weight.grad is not None
    return model.weight.grad.clone()


class TinyGPT(nn.Module):
    """A minimal next-token model: embed -> nonlinearity -> per-token vocab logits.

    Deliberately tiny (no real attention) -- enough to show that the REAL training recipe
    (AdamW + warmup->cosine + grad clip) drives the loss down, which is the point of the demo.
    """

    def __init__(self, vocab_size: int, embed_dim: int) -> None:
        super().__init__()
        self.embed = nn.Embedding(vocab_size, embed_dim)  # token id -> dense vector
        self.feed_forward = nn.Linear(embed_dim, embed_dim)  # one mixing layer
        self.head = nn.Linear(embed_dim, vocab_size)  # project to one logit per vocab token

    def forward(self, token_ids: torch.Tensor) -> torch.Tensor:
        """token_ids: [batch, seq] -> logits: [batch, seq, vocab]."""
        hidden = torch.tanh(self.feed_forward(self.embed(token_ids)))  # tanh: cheap nonlinearity
        return self.head(hidden)  # one logit vector over the vocabulary at every position


def demo_lr_schedule() -> list[float]:
    """Print the warmup+cosine LR at sample steps and return the full schedule."""
    schedule = [lr_at_step(t) for t in range(TOTAL_STEPS_SCHEDULE + 1)]
    print(
        f"LR schedule (warmup={WARMUP_STEPS}, total={TOTAL_STEPS_SCHEDULE}, "
        f"peak={LR_PEAK:.0e}, floor={LR_FLOOR:.0e}):"
    )
    for t in (0, 5, WARMUP_STEPS, 30, 55, TOTAL_STEPS_SCHEDULE):
        print(f"  step {t:>3}: lr = {schedule[t]:.3e}")
    return schedule


def demo_grad_accum_equivalence() -> float:
    """Prove K micro-batches == one big batch for the gradient; return the max abs difference."""
    torch.manual_seed(1)
    n_examples = MICRO_BATCH * ACCUM_STEPS  # the full effective batch
    features = torch.randn(n_examples, FEATURE_DIM)  # toy inputs
    targets = torch.randn(n_examples, 1)  # toy regression targets

    big_model = make_linear(seed=2, in_dim=FEATURE_DIM)  # same init as the accumulated model
    big_grad = big_batch_grad(big_model, features, targets)

    accum_model = make_linear(seed=2, in_dim=FEATURE_DIM)  # identical starting weights
    accum_grad = accumulated_grad(accum_model, features, targets, MICRO_BATCH, ACCUM_STEPS)

    max_diff = (big_grad - accum_grad).abs().max().item()
    print(f"grad-accum equivalence: max|big - accumulated| = {max_diff:.2e}")
    return max_diff


def demo_training_loop() -> float:
    """Train TinyGPT with the real recipe on a toy corpus; print loss dropping; return final loss."""
    torch.manual_seed(SEED)
    # Build/run on TRACE_DEVICE (CPU) so the printed curve is reproducible everywhere.
    # A PERIODIC corpus (repeat 0,1,2,3,0,1,2,3,...) makes next-token genuinely learnable, so the
    # loss drops cleanly -- a random corpus would have no pattern to learn and the loss would just
    # hover near log(VOCAB_SIZE).
    cycle = torch.arange(PATTERN_PERIOD, device=TRACE_DEVICE)  # [0, 1, 2, 3]
    corpus = cycle.repeat(CORPUS_TOKENS // PATTERN_PERIOD)  # [0,1,2,3, 0,1,2,3, ...]
    model = TinyGPT(VOCAB_SIZE, EMBED_DIM).to(TRACE_DEVICE)
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=LR_PEAK, betas=ADAM_BETAS, weight_decay=WEIGHT_DECAY
    )

    print(f"{'step':>4} | {'lr':>9} | {'loss':>7}")
    print("-" * 28)
    final_loss = float("nan")
    for step in range(TRAIN_STEPS):
        start = torch.randint(0, len(corpus) - CONTEXT_LEN - 1, (1,)).item()  # random window into the corpus
        inputs = corpus[start : start + CONTEXT_LEN].unsqueeze(0)  # [1, ctx] token ids
        targets = corpus[start + 1 : start + CONTEXT_LEN + 1].unsqueeze(0)  # next-token labels (shift by 1)

        lr = lr_at_step(step, LR_PEAK, LR_FLOOR, WARMUP_STEPS_TRAIN, TRAIN_STEPS)  # real schedule
        for group in optimizer.param_groups:
            group["lr"] = lr  # apply the scheduled LR to the optimizer this step

        logits = model(inputs)  # [1, ctx, vocab]
        assert logits.shape == (1, CONTEXT_LEN, VOCAB_SIZE)  # boundary shape check: [batch, seq, vocab]
        loss = F.cross_entropy(logits.reshape(-1, VOCAB_SIZE), targets.reshape(-1))  # next-token CE
        optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), GRAD_CLIP_NORM)  # cap the step: the seatbelt against loss spikes
        optimizer.step()

        final_loss = loss.item()
        if step % REPORT_EVERY == 0 or step == TRAIN_STEPS - 1:
            print(f"{step:>4} | {lr:.3e} | {final_loss:>7.4f}")
    return final_loss


def demo_mfu(model: nn.Module) -> float:
    """Print and return a toy MFU = achieved 6ND-rate / peak FLOPs."""
    n_params = sum(p.numel() for p in model.parameters())  # N: this model's parameter count
    flops_per_token = FLOPS_PER_PARAM_PER_TOKEN * n_params  # 6N FLOPs/token (forward + backward)
    achieved_flops = flops_per_token * PRETEND_TOKENS_PER_SEC  # 6N x tokens/sec = useful FLOP/s
    # num_GPUs=1 here; a real run divides by the whole cluster's peak
    mfu = achieved_flops / (PRETEND_PEAK_FLOPS * NUM_GPUS)  # fraction of peak hardware compute actually used
    print(f"params N = {n_params:,}   training FLOPs/token = 6N = {flops_per_token:,}")
    print(
        f"toy MFU = (6N x {PRETEND_TOKENS_PER_SEC:.0e} tok/s) / {PRETEND_PEAK_FLOPS:.0e} peak "
        f"= {mfu:.4%}"
    )
    return mfu


# =============================================================================
# Seeded trace + table helpers for the figure/animation generators.
#
# These are PURE, additive helpers: they reproduce the exact same seeded runs as the demos above
# (same SEED, same recipe, same construction order) but RETURN the full per-step traces instead of
# only printing every REPORT_EVERY-th row -- so make_figures_02.py / make_animation_02.py can plot
# the real numbers the page reports, never orphan values. They do not change any existing behavior.
# =============================================================================


def lr_schedule_trace(
    peak: float = LR_PEAK,
    floor: float = LR_FLOOR,
    warmup: int = WARMUP_STEPS,
    total: int = TOTAL_STEPS_SCHEDULE,
) -> tuple[list[int], list[float]]:
    """Return (steps, lr) for the standalone schedule demo, step 0 .. total inclusive."""
    steps = list(range(total + 1))
    return steps, [lr_at_step(t, peak, floor, warmup, total) for t in steps]


def training_loop_trace() -> tuple[list[int], list[float], list[float], list[float]]:
    """Re-run demo_training_loop's EXACT seeded recipe and return per-step traces.

    Returns (steps, losses, lrs, grad_norms) for every one of TRAIN_STEPS steps -- the same numbers
    demo_training_loop prints every REPORT_EVERY rows, but complete, so the figure shows the real
    descent (start ~2.84 -> end ~0.75) the page quotes. grad_norms is the TOTAL grad norm measured
    BEFORE clipping (what the stability dashboard watches).
    """
    torch.manual_seed(SEED)
    cycle = torch.arange(PATTERN_PERIOD, device=TRACE_DEVICE)
    corpus = cycle.repeat(CORPUS_TOKENS // PATTERN_PERIOD)
    model = TinyGPT(VOCAB_SIZE, EMBED_DIM).to(TRACE_DEVICE)
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=LR_PEAK, betas=ADAM_BETAS, weight_decay=WEIGHT_DECAY
    )
    steps: list[int] = []
    losses: list[float] = []
    lrs: list[float] = []
    grad_norms: list[float] = []
    for step in range(TRAIN_STEPS):
        start = torch.randint(0, len(corpus) - CONTEXT_LEN - 1, (1,)).item()
        inputs = corpus[start : start + CONTEXT_LEN].unsqueeze(0)
        targets = corpus[start + 1 : start + CONTEXT_LEN + 1].unsqueeze(0)
        lr = lr_at_step(step, LR_PEAK, LR_FLOOR, WARMUP_STEPS_TRAIN, TRAIN_STEPS)
        for group in optimizer.param_groups:
            group["lr"] = lr
        logits = model(inputs)
        loss = F.cross_entropy(logits.reshape(-1, VOCAB_SIZE), targets.reshape(-1))
        optimizer.zero_grad()
        loss.backward()
        # measure the total grad norm BEFORE clipping (clip_grad_norm_ returns exactly this)
        grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), GRAD_CLIP_NORM).item()
        optimizer.step()
        steps.append(step)
        losses.append(loss.item())
        lrs.append(lr)
        grad_norms.append(grad_norm)
    return steps, losses, lrs, grad_norms


def grad_accum_norms() -> tuple[float, float, float, float]:
    """Reproduce the grad-accum demo and return (big_norm, accum_norm, buggy_norm, max_diff).

    big_norm: |grad| of one batch of size m*K. accum_norm: |grad| of K correctly-averaged
    micro-batches (==big_norm to float noise). buggy_norm: |grad| when the /K is dropped (the
    classic bug -- accum_steps-times too large). max_diff: max|big - accumulated| (float noise).
    """
    torch.manual_seed(1)
    n_examples = MICRO_BATCH * ACCUM_STEPS
    features = torch.randn(n_examples, FEATURE_DIM)
    targets = torch.randn(n_examples, 1)

    big_model = make_linear(seed=2, in_dim=FEATURE_DIM)
    big_grad = big_batch_grad(big_model, features, targets)

    accum_model = make_linear(seed=2, in_dim=FEATURE_DIM)
    accum_grad = accumulated_grad(accum_model, features, targets, MICRO_BATCH, ACCUM_STEPS)

    buggy_model = make_linear(seed=2, in_dim=FEATURE_DIM)  # same init; drop the /K
    for k in range(ACCUM_STEPS):
        start = k * MICRO_BATCH
        chunk_x = features[start : start + MICRO_BATCH]
        chunk_y = targets[start : start + MICRO_BATCH]
        F.mse_loss(buggy_model(chunk_x), chunk_y).backward()  # BUG: no /ACCUM_STEPS -> grads SUM
    assert buggy_model.weight.grad is not None
    buggy_grad = buggy_model.weight.grad

    max_diff = (big_grad - accum_grad).abs().max().item()
    return (
        big_grad.norm().item(),
        accum_grad.norm().item(),
        buggy_grad.norm().item(),
        max_diff,
    )


# --- Production / budgeting numbers (the 6ND rule applied to real models) ------
# (label, params N, train tokens D) for the production table; compute is computed, not hardcoded.
PRODUCTION_MODELS: tuple[tuple[str, float, float], ...] = (
    ("GPT-3", 175e9, 300e9),
    ("Chinchilla", 70e9, 1.4e12),
    ("Llama-2-70B", 70e9, 2.0e12),
    ("Llama-3-8B", 8e9, 15e12),
    ("Llama-3-70B", 70e9, 15e12),
)
# Adam mixed-precision memory tax, per parameter, in bytes (the optimizer-state tax from the page).
# The canonical "~16 bytes/param" = 2 (bf16 weight) + 2 (bf16 gradient) + 4 (fp32 master weight)
# + 4 (fp32 momentum) + 4 (fp32 variance). The bf16 gradient is easy to forget but real: the
# backward pass materializes a 2-byte gradient per parameter before the fp32 optimizer update.
ADAM_BYTES_BREAKDOWN: tuple[tuple[str, int], ...] = (
    ("bf16 weight", 2),
    ("bf16 gradient", 2),
    ("fp32 master weight", 4),
    ("Adam momentum (fp32)", 4),
    ("Adam variance (fp32)", 4),
)


def compute_6nd(n_params: float, n_tokens: float) -> float:
    """Training FLOPs estimate C = 6ND."""
    return FLOPS_PER_PARAM_PER_TOKEN * n_params * n_tokens


def production_table() -> list[tuple[str, float, float, float, float]]:
    """Return (label, N, D, tokens_per_param, compute_6ND) for each production model."""
    return [
        (label, n, d, d / n, compute_6nd(n, d)) for (label, n, d) in PRODUCTION_MODELS
    ]


def adam_memory_tax(n_params: float) -> tuple[list[tuple[str, int]], float]:
    """Return (per-param byte breakdown, total GiB) for Adam mixed-precision state at n_params."""
    total_bytes_per_param = sum(b for _, b in ADAM_BYTES_BREAKDOWN)
    total_gib = n_params * total_bytes_per_param / (1024**3)
    return list(ADAM_BYTES_BREAKDOWN), total_gib


def main() -> None:
    print(
        f"compute device available: {DEVICE} "
        f"(the reproducible trace runs on CPU for stable numbers)"
    )
    print("torch:", torch.__version__, "\n")

    # 1. Warmup + cosine LR schedule -- endpoints must be exact.
    schedule = demo_lr_schedule()
    assert abs(schedule[WARMUP_STEPS] - LR_PEAK) < LR_ENDPOINT_TOL, "LR must hit peak exactly at end of warmup"
    assert abs(schedule[TOTAL_STEPS_SCHEDULE] - LR_FLOOR) < LR_ENDPOINT_TOL, "LR must hit floor exactly at the end"
    print()

    # 2. Gradient-accumulation equivalence -- K micro-batches == one big batch.
    max_diff = demo_grad_accum_equivalence()
    assert max_diff < GRAD_EQUIV_ATOL, "accumulated gradient must match the big-batch gradient (did you drop the /K?)"
    print()

    # 3. A tiny pretraining loop with the real recipe -- loss must drop.
    final_loss = demo_training_loop()
    assert final_loss < FINAL_LOSS_BOUND, "the toy loss should drop below the bound; if not, the recipe/loop is broken"
    print()

    # 4. Toy MFU on the trained model's parameter count.
    model = TinyGPT(VOCAB_SIZE, EMBED_DIM)  # same shape as the trained model -> same N
    demo_mfu(model)


if __name__ == "__main__":
    main()
