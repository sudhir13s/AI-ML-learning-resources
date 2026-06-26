"""From-scratch preference optimization: the Bradley-Terry reward loss and the DPO loss.

This is the single verified source of truth for the RLHF & DPO chapter. Every number quoted
on the page and rendered into the figures is computed by the functions below -- the notebook
imports these same functions, and ``make_figures_15.py`` calls them too, so the page, the
notebook, the script, and the four PNGs can never silently drift apart.

What it demonstrates, all from scratch and deterministic:

  (a) The **Bradley-Terry reward-model loss** -log sigma(r_w - r_l) on hand-picked reward gaps,
      reproducing the page's worked-example numbers (gap +3 -> P=0.9526, loss=0.0486; gap 0 ->
      P=0.5, loss=0.6931 = log 2; gap -3 -> P=0.0474, loss=3.0486).

  (b) A **tiny reward model** (one linear value head over a pooled-embedding stand-in) trained
      with the Bradley-Terry loss on a few synthetic preference pairs. We ASSERT it scores the
      chosen answers above the rejected ones after training -- the reward model has learned the
      ranking from comparisons alone.

  (c) The **DPO loss and its gradient** at initialization (policy == reference): the loss is
      exactly log 2 and the gradient PUSHES the chosen log-prob up (-0.05) and the rejected
      log-prob down (+0.05) with beta=0.1.

  (d) A **measured toy DPO run**: a two-log-prob policy starting at the reference (-5.0) is
      optimized under the DPO loss for 120 steps. We ASSERT the chosen log-prob rises, the
      rejected one falls, and the implicit-reward MARGIN beta*Dlog(pi/pi_ref) grows step by
      step while the loss decays -- the derivation, watched happening. This is the run plotted
      in ``dpo_update.png``.

Device-agnostic (CUDA / MPS / CPU): the model in (b) is moved with ``.to(DEVICE)`` and every
tensor is created on ``DEVICE``. The numeric demos are pinned to CPU so the printed table is
bit-reproducible regardless of the accelerator present (float reductions on MPS/CUDA can differ
in the last digits); the honest device line reports both.

Verified on Python 3.12 / torch 2.x.

Run:
    python rlhf_dpo.py
"""

from __future__ import annotations

import math

import torch
import torch.nn as nn
import torch.nn.functional as F

# ---- Hoisted constants (identical in the notebook and the figure generator) ---------------
BETA = 0.1  # the KL-leash / preference temperature: same beta in PPO's penalty and the DPO loss
REF_LOGPROB = -5.0  # both policy log-probs start AT the frozen reference, so the margin starts at 0
SEED = 0

# Bradley-Terry worked-example reward pairs (r_chosen, r_rejected) -> the page's three rows.
BT_EXAMPLE_PAIRS: tuple[tuple[float, float], ...] = ((2.0, -1.0), (0.0, 0.0), (-1.0, 2.0))

# Tiny reward-model demo.
FEAT_DIM = 8  # width of the pooled-embedding stand-in fed to the scalar reward head
N_PAIRS = 6  # number of synthetic preference pairs
CLUSTER_MEAN = 0.6  # chosen features cluster around +mu, rejected around -mu (separable on purpose)
CLUSTER_STD = 0.3
RM_STEPS = 200
RM_LR = 0.05

# DPO margin-progression table (chosen delta, rejected delta) -> the page's worked-example table.
DPO_TABLE_DELTAS: tuple[tuple[float, float], ...] = (
    (0.0, 0.0),
    (1.0, -1.0),
    (2.0, -2.0),
    (3.0, -3.0),
    (5.0, -5.0),
)

# Measured toy-DPO run (plotted in dpo_update.png).
DPO_RUN_STEPS = 120
DPO_RUN_LR = 2.0  # chosen by matching the figure: ends chosen +2.35, rejected -12.35, margin 1.47

# Run on the best available accelerator; CPU is the universal fallback (matches kv_cache.py).
DEVICE = (
    "cuda"
    if torch.cuda.is_available()
    else "mps"
    if torch.backends.mps.is_available()
    else "cpu"
)


# ============================================================================================
# (a) The Bradley-Terry reward-model loss
# ============================================================================================
def bt_loss(r_chosen: torch.Tensor, r_rejected: torch.Tensor) -> torch.Tensor:
    """Bradley-Terry reward-model loss: -log sigma(r_chosen - r_rejected), averaged over pairs.

    This is binary cross-entropy on the reward GAP -- the reward model is a classifier of
    "is the chosen answer better than the rejected one?", and only the gap matters (Elo-style).
    ``F.logsigmoid`` is the numerically stable log(sigmoid(.)) -- never compute log(sigmoid)
    as two separate ops, which overflows for large-magnitude gaps.
    """
    return -F.logsigmoid(r_chosen - r_rejected).mean()


def bt_worked_examples() -> list[dict[str, float]]:
    """Reproduce the page's three Bradley-Terry rows: gap -> P(chosen>rejected) -> loss."""
    rows: list[dict[str, float]] = []
    for r_w, r_l in BT_EXAMPLE_PAIRS:
        gap = r_w - r_l
        prob = 1.0 / (1.0 + math.exp(-gap))  # sigma(gap) = P(chosen preferred)
        loss = bt_loss(torch.tensor(r_w), torch.tensor(r_l)).item()
        rows.append({"r_w": r_w, "r_l": r_l, "gap": gap, "P": prob, "loss": loss})
    return rows


# ============================================================================================
# (b) A tiny reward model trained with the Bradley-Terry loss
# ============================================================================================
def make_preference_features(device: str = "cpu") -> tuple[torch.Tensor, torch.Tensor]:
    """Synthetic (chosen, rejected) feature pairs that a linear reward head can separate.

    Each row stands in for a pooled response embedding. Chosen rows cluster around +CLUSTER_MEAN
    and rejected around -CLUSTER_MEAN so a scalar reward head CAN rank them -- the lesson is
    watching the Bradley-Terry loss *learn* that ranking from comparisons, not the difficulty.
    """
    torch.manual_seed(SEED)
    chosen = torch.randn(N_PAIRS, FEAT_DIM, device=device) * CLUSTER_STD + CLUSTER_MEAN
    rejected = torch.randn(N_PAIRS, FEAT_DIM, device=device) * CLUSTER_STD - CLUSTER_MEAN
    return chosen, rejected


def train_reward_model(
    chosen: torch.Tensor, rejected: torch.Tensor, device: str = "cpu"
) -> tuple[nn.Linear, dict[str, float]]:
    """Train a one-linear-layer reward model with the Bradley-Terry loss; return it + stats.

    The reward model is the SFT body with its token head swapped for a single scalar VALUE head;
    here we model just that head as a Linear(FEAT_DIM, 1) over the pooled feature, which is the
    only new, trainable piece in a real reward model.
    """
    torch.manual_seed(SEED)
    reward_head = nn.Linear(FEAT_DIM, 1).to(device)  # .to(device): honor the chosen accelerator
    optimizer = torch.optim.Adam(reward_head.parameters(), lr=RM_LR)

    def scores() -> tuple[torch.Tensor, torch.Tensor]:
        r_c = reward_head(chosen).squeeze(-1)  # (N_PAIRS,) scalar reward per chosen answer
        r_r = reward_head(rejected).squeeze(-1)
        return r_c, r_r

    with torch.no_grad():
        r_c0, r_r0 = scores()
        before = {
            "chosen": r_c0.mean().item(),
            "rejected": r_r0.mean().item(),
            "loss": bt_loss(r_c0, r_r0).item(),
        }

    for _ in range(RM_STEPS):
        r_c, r_r = scores()
        loss = bt_loss(r_c, r_r)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

    with torch.no_grad():
        r_c1, r_r1 = scores()
        after = {
            "chosen": r_c1.mean().item(),
            "rejected": r_r1.mean().item(),
            "loss": bt_loss(r_c1, r_r1).item(),
            "n_correct": int((r_c1 > r_r1).sum().item()),  # pairs scored chosen > rejected
        }
    return reward_head, {"before": before, "after": after}


# ============================================================================================
# (c) + (d) The DPO loss, its gradient, and a measured run
# ============================================================================================
def dpo_loss(
    pi_lp_w: torch.Tensor,
    pi_lp_l: torch.Tensor,
    ref_lp_w: torch.Tensor,
    ref_lp_l: torch.Tensor,
    beta: float = BETA,
) -> torch.Tensor:
    """DPO loss from sequence log-probs of (chosen=w, rejected=l) under policy and reference.

    -log sigma( beta * [ (logpi_w - logpi_l) - (logref_w - logref_l) ] ). The bracket is the
    implicit-reward MARGIN: beta*log(pi/pi_ref) for chosen minus the same for rejected. The
    reference appears only as a subtraction, so it acts as a frozen anchor (the absorbed KL leash).
    """
    pi_logratios = pi_lp_w - pi_lp_l  # does the policy prefer chosen over rejected?
    ref_logratios = ref_lp_w - ref_lp_l  # ... relative to the frozen reference's preference
    return -F.logsigmoid(beta * (pi_logratios - ref_logratios)).mean()


def implicit_reward(pi_logprob: torch.Tensor | float, ref_logprob: torch.Tensor | float, beta: float = BETA) -> float:
    """The implicit reward beta*log(pi/pi_ref) -- the reward DPO optimizes without ever naming it."""
    pi_val = pi_logprob.item() if isinstance(pi_logprob, torch.Tensor) else pi_logprob
    ref_val = ref_logprob.item() if isinstance(ref_logprob, torch.Tensor) else ref_logprob
    return beta * (pi_val - ref_val)  # log(pi/pi_ref) = logpi - logref for log-probs


def dpo_init_gradient() -> dict[str, float]:
    """DPO loss and gradient at init (policy == reference): loss = log 2, grads = -/+ beta/2."""
    ref_w = torch.tensor(REF_LOGPROB)
    ref_l = torch.tensor(REF_LOGPROB)
    pi_w = torch.tensor(REF_LOGPROB, requires_grad=True)  # policy logprob of chosen, starts == ref
    pi_l = torch.tensor(REF_LOGPROB, requires_grad=True)  # policy logprob of rejected, starts == ref
    loss = dpo_loss(pi_w, pi_l, ref_w, ref_l)
    loss.backward()
    return {
        "loss": loss.item(),  # -log sigma(0) = log 2 = 0.6931
        "grad_chosen": pi_w.grad.item(),  # negative -> the optimizer RAISES the chosen log-prob
        "grad_rejected": pi_l.grad.item(),  # positive -> the optimizer LOWERS the rejected log-prob
    }


def dpo_margin_table() -> list[dict[str, float]]:
    """Reproduce the page's DPO worked-example table: (chosen d, rejected d) -> margin, loss, P."""
    ref_w = torch.tensor(REF_LOGPROB)
    ref_l = torch.tensor(REF_LOGPROB)
    rows: list[dict[str, float]] = []
    for d_w, d_l in DPO_TABLE_DELTAS:
        # ref log-ratio is 0 at init, so margin = beta * (d_w - d_l)
        margin = BETA * (d_w - d_l)
        loss = dpo_loss(
            torch.tensor(REF_LOGPROB + d_w), torch.tensor(REF_LOGPROB + d_l), ref_w, ref_l
        ).item()
        prob = 1.0 / (1.0 + math.exp(-margin))  # sigma(margin) = P(chosen preferred)
        rows.append({"d_w": d_w, "d_l": d_l, "margin": margin, "loss": loss, "P": prob})
    return rows


def run_toy_dpo(
    steps: int = DPO_RUN_STEPS, lr: float = DPO_RUN_LR, beta: float = BETA
) -> dict[str, list[float]]:
    """Optimize a two-log-prob policy under the DPO loss; record the trajectory for plotting.

    The policy is just two scalars -- log pi(chosen) and log pi(rejected) -- both initialized AT
    the frozen reference. Minimizing the DPO loss must raise the first, lower the second, grow
    the implicit-reward margin, and shrink the loss. This is the run drawn in dpo_update.png.
    """
    torch.manual_seed(SEED)
    ref_w = torch.tensor(REF_LOGPROB)
    ref_l = torch.tensor(REF_LOGPROB)
    pi_w = torch.tensor(REF_LOGPROB, requires_grad=True)
    pi_l = torch.tensor(REF_LOGPROB, requires_grad=True)
    optimizer = torch.optim.SGD([pi_w, pi_l], lr=lr)

    history: dict[str, list[float]] = {"step": [], "chosen": [], "rejected": [], "margin": [], "loss": []}
    for step in range(steps + 1):
        margin = beta * ((pi_w - pi_l) - (ref_w - ref_l))
        loss = -F.logsigmoid(margin)
        history["step"].append(step)
        history["chosen"].append(pi_w.item())
        history["rejected"].append(pi_l.item())
        history["margin"].append(margin.item())
        history["loss"].append(loss.item())
        if step < steps:  # take steps after recording, so step 0 is the untouched init
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
    return history


# ============================================================================================
# Driver
# ============================================================================================
def main() -> None:
    # Numeric demos pinned to CPU so the printed tables are bit-reproducible on any machine;
    # the reward-model training in (b) also runs on the honest accelerator to prove device-agnosticism.
    demo_device = "cpu"
    print(f"device: {demo_device} (detected {DEVICE}; pinned to CPU for reproducibility)")
    print("torch:", torch.__version__)
    print()

    # ---- (a) Bradley-Terry worked examples ------------------------------------------------
    print("(a) Bradley-Terry RM loss  (gap -> P(chosen>rejected) -> loss):")
    for row in bt_worked_examples():
        print(
            f"    r_w={row['r_w']:+.1f} r_l={row['r_l']:+.1f}  gap={row['gap']:+.1f}  "
            f"P={row['P']:.4f}  loss={row['loss']:.4f}"
        )
    print()

    # ---- (b) Train a tiny reward model with the BT loss -----------------------------------
    print(f"(b) Tiny reward model trained with the Bradley-Terry loss ({RM_STEPS} steps):")
    chosen, rejected = make_preference_features(device=DEVICE)  # honor the real accelerator
    _, stats = train_reward_model(chosen, rejected, device=DEVICE)
    b = stats["before"]
    a = stats["after"]
    print(
        f"    before: mean r(chosen)={b['chosen']:+.3f}  mean r(rejected)={b['rejected']:+.3f}  "
        f"loss={b['loss']:.4f}"
    )
    print(
        f"    after : mean r(chosen)={a['chosen']:+.3f}  mean r(rejected)={a['rejected']:+.3f}  "
        f"loss={a['loss']:.4f}"
    )
    print(f"    pairs scored chosen > rejected: {a['n_correct']}/{N_PAIRS}")
    # Assert BEFORE we move on: the reward model learned the ranking from comparisons alone.
    assert a["chosen"] > a["rejected"], "reward model must score chosen above rejected after training"
    assert a["n_correct"] == N_PAIRS, "reward model must rank every pair correctly after training"
    assert a["loss"] < b["loss"], "Bradley-Terry loss must decrease during training"
    print("    OK: reward model ranks chosen above rejected (asserted)")
    print()

    # ---- (c) DPO loss and gradient at init ------------------------------------------------
    init = dpo_init_gradient()
    print("(c) DPO at init (policy == reference):")
    print(f"    loss = {init['loss']:.4f}  = log 2 (no preference yet)")
    print(
        f"    d loss / d logp(chosen)   = {init['grad_chosen']:+.4f}  -> training PUSHES chosen UP"
    )
    print(
        f"    d loss / d logp(rejected) = {init['grad_rejected']:+.4f}  -> training PUSHES rejected DOWN"
    )
    assert init["grad_chosen"] < 0, "gradient must raise the chosen log-prob"
    assert init["grad_rejected"] > 0, "gradient must lower the rejected log-prob"
    assert abs(init["loss"] - math.log(2)) < 1e-6, "init loss must equal log 2"
    print()

    # ---- DPO margin progression table -----------------------------------------------------
    print("    DPO margin progression (beta=0.1), matching the worked-example table:")
    for row in dpo_margin_table():
        print(
            f"      chosen{row['d_w']:+.0f}, rejected{row['d_l']:+.0f}: margin={row['margin']:+.2f}  "
            f"loss={row['loss']:.4f}  P(w>l)={row['P']:.4f}"
        )
    print()

    # ---- Implicit reward + adaptive gradient weight ---------------------------------------
    print("    Implicit reward beta*log(pi/pi_ref) for the chosen response:")
    for d_lp in (0.0, 0.8, 2.0):
        print(
            f"      chosen logprob {d_lp:+.1f} above reference -> implicit reward = "
            f"{implicit_reward(REF_LOGPROB + d_lp, REF_LOGPROB):+.3f}"
        )
    print("    Adaptive gradient weight beta*sigma(-margin)  (big when the model is wrong):")
    for m in (-2.0, 0.0, 2.0):
        weight = BETA * (1.0 / (1.0 + math.exp(m)))  # beta * sigma(-m)
        print(f"      margin={m:+.1f}: weight={weight:.4f}")
    print()

    # ---- (d) Measured toy DPO run ---------------------------------------------------------
    print(f"(d) Measured toy DPO run ({DPO_RUN_STEPS} steps, beta={BETA}):")
    history = run_toy_dpo()
    for s in (0, 40, 80, 120):
        i = history["step"].index(s)
        print(
            f"    step {s:3d}: chosen={history['chosen'][i]:+.3f}  rejected={history['rejected'][i]:+.3f}  "
            f"margin={history['margin'][i]:+.4f}  loss={history['loss'][i]:.4f}"
        )
    # Assert the derivation's promise: chosen rises, rejected falls, margin grows, loss falls.
    assert history["chosen"][-1] > history["chosen"][0], "chosen log-prob must rise over the run"
    assert history["rejected"][-1] < history["rejected"][0], "rejected log-prob must fall over the run"
    assert history["margin"][-1] > history["margin"][0], "implicit-reward margin must grow over the run"
    assert history["loss"][-1] < history["loss"][0], "DPO loss must fall over the run"
    final_implicit = implicit_reward(history["chosen"][-1], REF_LOGPROB)
    print(
        f"    final implicit reward for chosen = beta*log(pi/pi_ref) = {final_implicit:+.3f}  "
        f"(margin {history['margin'][-1]:+.3f}, loss {history['loss'][-1]:.4f})"
    )
    print("    OK: chosen up, rejected down, margin grows, loss falls (asserted)")


if __name__ == "__main__":
    main()
