"""From-scratch LoRA: build a low-rank adapter, prove the win, prove zero-latency merge.

A LoRA-wrapped linear layer freezes its pretrained weight ``W`` and trains only a low-rank
update ``ΔW = (alpha/r) · B·A`` (``B`` zero-init, ``A`` random-init), so training starts exactly
at the pretrained model (``ΔW = 0``) and updates a tiny fraction of the parameters. This script:

  1. wraps a frozen ``Linear`` with trainable ``A``, ``B`` and prints the trainable-vs-frozen
     parameter counts and the reduction ratio (the headline LoRA win);
  2. verifies that only ``A`` and ``B`` receive gradients while ``W`` stays frozen;
  3. trains the adapter on a tiny synthetic task and shows the loss falling;
  4. **merges** ``ΔW`` into ``W`` and asserts the merged layer's output equals the unmerged
     LoRA forward to floating-point tolerance -- proving the famous *zero added inference
     latency*: at serving time LoRA is just a plain matmul;
  5. sweeps the rank ``r`` to show the params-vs-fit tradeoff.

This is the same verified demo embedded in the concept page and the teaching notebook.
Verified on Python 3.12 / torch 2.x. Device-agnostic (CUDA / MPS / CPU); the reproducible
trace is pinned to CPU so the printed numbers match the page and notebook on any machine.

Run:
    python lora_peft.py
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

# ---- Hyperparameters (hoisted, named -- the knobs the page discusses) -----------------
IN_FEATURES = 1024  # d: input width of the wrapped linear layer
OUT_FEATURES = 1024  # k: output width (square here so the d^2 vs 2rd contrast is clean)
LORA_RANK = 8  # r: the low-rank bottleneck, r << min(d, k) -- this is the whole point
LORA_ALPHA = 16  # alpha: scaling numerator; the update is multiplied by alpha/r
SEED = 0
TRAIN_STEPS = 300
LEARNING_RATE = 1e-2
N_SAMPLES = 256  # rows in the tiny synthetic dataset
TRUE_RANK = 4  # the synthetic target update is genuinely low-rank (rank 4) -- LoRA's premise
RANK_SWEEP = (1, 2, 4, 8, 16, 32)  # ranks compared in the params-vs-fit sweep
MERGE_ATOL = 1e-5  # merged vs unmerged outputs feed identical math; only float rounding differs

# Detect the best accelerator for honesty in the printout, but pin the reproducible trace to
# CPU so the numbers below are bit-stable across machines (the strict device-honesty rule).
DETECTED_DEVICE = (
    "cuda"
    if torch.cuda.is_available()
    else "mps"
    if torch.backends.mps.is_available()
    else "cpu"
)
DEVICE = "cpu"  # reproducible trace runs on CPU regardless of what is detected


class LoRALinear(nn.Module):
    """A frozen ``Linear`` (weight ``W``) plus a trainable low-rank update ``ΔW = (alpha/r)·B·A``.

    Forward: ``h = x·Wᵀ + (alpha/r)·(x·Aᵀ)·Bᵀ``. The first term is the untouched pretrained
    layer; the second is the adapter. At init ``B = 0`` so ``ΔW = 0`` and the layer is exactly
    the pretrained one -- training begins from the pretrained model, not from noise.
    """

    def __init__(
        self,
        in_features: int,
        out_features: int,
        rank: int,
        alpha: float,
        device: str = DEVICE,
    ) -> None:
        super().__init__()
        self.rank = rank
        self.scaling = alpha / rank  # the alpha/r scale: decouples update magnitude from rank r

        # Pretrained weight W (out, in). FROZEN: requires_grad=False so the optimizer never
        # touches it -- this is what makes LoRA parameter-efficient and lets one base be shared.
        self.weight = nn.Parameter(
            torch.randn(out_features, in_features, device=device) * 0.02,
            requires_grad=False,  # W is frozen -- only A and B below are trained
        )

        # Low-rank factors. A: (r, in) random-init (Kaiming) so B·A starts as a generic
        # projection; B: (out, r) ZERO-init so ΔW = B·A = 0 at step 0.
        self.lora_a = nn.Parameter(torch.empty(rank, in_features, device=device))
        self.lora_b = nn.Parameter(torch.zeros(out_features, rank, device=device))  # B=0 -> ΔW=0
        nn.init.kaiming_uniform_(self.lora_a, a=5**0.5)  # standard LoRA A init (He uniform)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        base = F.linear(x, self.weight)  # frozen pretrained path: x·Wᵀ
        # adapter path: project down to rank r (x·Aᵀ), back up (·Bᵀ), scale by alpha/r.
        update = F.linear(F.linear(x, self.lora_a), self.lora_b) * self.scaling
        return base + update

    def merged_weight(self) -> torch.Tensor:
        """Return ``W + (alpha/r)·B·A`` -- the single matrix used at inference (zero extra latency)."""
        delta_w = (self.lora_b @ self.lora_a) * self.scaling  # B·A is (out, in), same shape as W
        return self.weight + delta_w


def count_parameters(layer: LoRALinear) -> tuple[int, int]:
    """Return (trainable, frozen) parameter counts for a LoRA layer."""
    trainable = sum(p.numel() for p in layer.parameters() if p.requires_grad)
    frozen = sum(p.numel() for p in layer.parameters() if not p.requires_grad)
    return trainable, frozen


def make_synthetic_task(
    seed: int = SEED, device: str = DEVICE
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """A regression task whose target update is genuinely low-rank (rank ``TRUE_RANK``).

    Returns (x, y, base_weight). The targets are ``y = x·(W + ΔW*)ᵀ`` where ``ΔW*`` has rank
    ``TRUE_RANK`` -- so a LoRA of rank >= TRUE_RANK can fit it exactly, modelling the paper's
    finding that fine-tuning updates have low intrinsic rank.
    """
    torch.manual_seed(seed)
    base_weight = torch.randn(OUT_FEATURES, IN_FEATURES, device=device) * 0.02
    # A genuinely low-rank target update: product of two thin matrices -> rank TRUE_RANK.
    u = torch.randn(OUT_FEATURES, TRUE_RANK, device=device) * 0.1
    v = torch.randn(TRUE_RANK, IN_FEATURES, device=device) * 0.1
    delta_star = u @ v  # rank-TRUE_RANK update the adapter must learn to reproduce
    x = torch.randn(N_SAMPLES, IN_FEATURES, device=device)
    y = F.linear(x, base_weight + delta_star)  # ground-truth uses the adapted weight
    return x, y, base_weight


def train_adapter(
    layer: LoRALinear, x: torch.Tensor, y: torch.Tensor, steps: int = TRAIN_STEPS
) -> list[float]:
    """Train ONLY the LoRA factors (A, B) on the synthetic task; return the loss curve."""
    # The optimizer is handed only the parameters that require grad -- i.e. A and B, never W.
    trainable_params = [p for p in layer.parameters() if p.requires_grad]
    optimizer = torch.optim.Adam(trainable_params, lr=LEARNING_RATE)
    losses: list[float] = []
    for _ in range(steps):
        optimizer.zero_grad()
        loss = F.mse_loss(layer(x), y)
        loss.backward()
        optimizer.step()
        losses.append(loss.item())
    return losses


def check_grad_flow(layer: LoRALinear, x: torch.Tensor, y: torch.Tensor) -> dict[str, bool]:
    """One backward pass; report which parameters received a gradient.

    Confirms A and B get grads while the frozen W gets none -- the mechanical reason LoRA
    trains so few parameters.
    """
    layer.zero_grad(set_to_none=True)
    loss = F.mse_loss(layer(x), y)
    loss.backward()
    return {
        "lora_a_has_grad": layer.lora_a.grad is not None,
        "lora_b_has_grad": layer.lora_b.grad is not None,
        "frozen_w_has_grad": layer.weight.grad is not None,  # expected False -- W is frozen
    }


def verify_merge(layer: LoRALinear, x: torch.Tensor) -> tuple[bool, float]:
    """Assert the merged single-matrix forward equals the unmerged LoRA forward.

    This is the zero-added-latency proof: at inference you replace ``W`` with ``W + ΔW`` and run
    one plain matmul -- no extra layers, no extra FLOPs, identical output.
    """
    unmerged = layer(x)  # base path + adapter path (two matmuls)
    merged = F.linear(x, layer.merged_weight())  # single fused matmul with W + ΔW
    max_diff = (unmerged - merged).abs().max().item()
    return torch.allclose(unmerged, merged, atol=MERGE_ATOL), max_diff


def rank_sweep(x: torch.Tensor, y: torch.Tensor) -> list[tuple[int, int, float]]:
    """For each rank r: (r, trainable params, final loss) -- the params-vs-fit tradeoff."""
    rows: list[tuple[int, int, float]] = []
    for r in RANK_SWEEP:
        layer = LoRALinear(IN_FEATURES, OUT_FEATURES, rank=r, alpha=LORA_ALPHA).to(DEVICE)
        layer.weight.data.copy_(BASE_WEIGHT)  # same frozen base for every rank -> fair comparison
        trainable, _ = count_parameters(layer)
        final_loss = train_adapter(layer, x, y)[-1]
        rows.append((r, trainable, final_loss))
    return rows


# Shared frozen base so every experiment adapts the SAME pretrained weight (module-level
# constant set once from the synthetic task; never mutated after).
_X, _Y, BASE_WEIGHT = make_synthetic_task()


def main() -> None:
    torch.manual_seed(SEED)
    print(f"device: {DEVICE} (detected {DETECTED_DEVICE}; pinned to CPU for reproducibility)")
    print("torch:", torch.__version__)
    print()

    layer = LoRALinear(IN_FEATURES, OUT_FEATURES, rank=LORA_RANK, alpha=LORA_ALPHA).to(DEVICE)
    layer.weight.data.copy_(BASE_WEIGHT)

    # 1) The headline win: trainable vs frozen parameters and the reduction ratio.
    trainable, frozen = count_parameters(layer)
    full_ft = IN_FEATURES * OUT_FEATURES  # full fine-tuning would update all d*k params of W
    print("=== Parameter count (one d x d linear, d = {}) ===".format(IN_FEATURES))
    print(f"full fine-tuning (train all of W): {full_ft:>12,} params  (d^2)")
    print(f"LoRA trainable (A: r x d, B: d x r): {trainable:>10,} params  (2*r*d, r={LORA_RANK})")
    print(f"frozen (W, untouched):             {frozen:>12,} params")
    print(f"reduction: {full_ft / trainable:>6.1f}x fewer trainable params "
          f"({100 * trainable / full_ft:.2f}% of full FT)")
    print()

    # 2) Gradients flow to A and B only -- W is frozen.
    grad = check_grad_flow(layer, _X, _Y)
    print("=== Gradient flow (one backward pass) ===")
    print(f"A receives grad: {grad['lora_a_has_grad']}   "
          f"B receives grad: {grad['lora_b_has_grad']}   "
          f"frozen W receives grad: {grad['frozen_w_has_grad']}")
    assert grad["lora_a_has_grad"] and grad["lora_b_has_grad"]
    assert not grad["frozen_w_has_grad"], "W must stay frozen"
    print()

    # 3) ΔW = 0 at init (B = 0): the adapted layer starts identical to the pretrained one.
    fresh = LoRALinear(IN_FEATURES, OUT_FEATURES, rank=LORA_RANK, alpha=LORA_ALPHA).to(DEVICE)
    fresh.weight.data.copy_(BASE_WEIGHT)
    delta_at_init = (fresh.lora_b @ fresh.lora_a).abs().max().item()
    print("=== B = 0 init => ΔW = 0 at step 0 ===")
    print(f"max|ΔW| before any training: {delta_at_init:.1e}  (exactly zero -> starts at pretrained)")
    print()

    # 4) Train the adapter -- the loss falls as A, B learn the low-rank update.
    losses = train_adapter(layer, _X, _Y)
    print("=== Train the adapter (frozen W, trainable A, B) ===")
    print(f"loss: {losses[0]:.4e} (step 0) -> {losses[-1]:.4e} (step {len(losses)}) "
          f"[{losses[0] / losses[-1]:.0f}x lower]")
    print()

    # 5) Merge ΔW into W and prove zero-latency equivalence.
    merge_ok, max_diff = verify_merge(layer, _X)
    print("=== Merge B·A into W -> zero added inference latency ===")
    print(f"merged == unmerged forward: {merge_ok}   max abs diff: {max_diff:.2e} "
          f"({'bitwise-identical' if max_diff == 0 else 'within float-noise tolerance'})")
    assert merge_ok, "merged weight must reproduce the unmerged LoRA forward"
    print()

    # 6) Rank sweep: params grow linearly in r; fit saturates once r >= the true rank.
    print("=== Rank sweep: params (2*r*d) vs fit (final loss), true update rank = {} ===".format(TRUE_RANK))
    print(f"{'rank r':>7} | {'trainable':>10} | {'% of full FT':>12} | {'final loss':>12}")
    print("-" * 52)
    for r, params, final_loss in rank_sweep(_X, _Y):
        print(f"{r:>7} | {params:>10,} | {100 * params / full_ft:>11.3f}% | {final_loss:>12.4e}")


if __name__ == "__main__":
    main()
