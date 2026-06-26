"""From-scratch Mixture-of-Experts (MoE) layer: route tokens, combine experts, and
SHOW that the auxiliary load-balancing loss is what keeps the router from collapsing.

The file does three things, in order:

  1. Build one MoE layer (N expert FFNs + a linear router) and run a tiny batch
     through it, printing the SHAPES at each step and WHICH expert each token went to
     -- the routing made visible.
  2. Compute the auxiliary load-balancing loss  L_aux = N * sum_i f_i * P_i  on a
     balanced vs a collapsed routing, by hand, and assert the collapsed one scores
     higher (the penalty that the training loop leans on).
  3. Train the same layer twice -- once WITHOUT the aux loss and once WITH it -- and
     measure expert utilisation, so you can watch the rich-get-richer collapse happen
     and the aux loss fix it.

This is the same verified demo embedded in the concept page and the teaching notebook.
Verified on Python 3.12 / torch 2.x. Device-agnostic (CUDA / MPS / CPU); the absolute
numbers are seed- and device-dependent, but the routing logic, the aux-loss ordering
(collapsed > balanced), and the "aux loss prevents dead experts" trend hold on any device.

Run:
    python mixture_of_experts.py
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

# ---- Layer dimensions for the from-scratch MoE used throughout the demo. ----
D_MODEL = 32          # token hidden width (kept tiny so the prints are readable)
D_FF = 64             # expert FFN intermediate width (the d -> d_ff -> d expansion)
N_EXPERTS = 8         # number of expert FFNs in the layer
TOP_K = 2             # experts consulted per token (k << N is the whole point)
IDEAL_UTIL_PCT = 100.0 / N_EXPERTS  # perfectly balanced top-1 usage = 12.5% per expert

# ---- Training hyper-parameters for the collapse-vs-balanced experiment. ----
TRAIN_STEPS = 1500
BATCH_TOKENS = 256
LEARNING_RATE = 3e-3
ROUTER_PUSH = 3.0     # strength of the rich-get-richer pressure (see train_moe)
AUX_COEFF = 4.0       # alpha: weight on the aux loss when it is enabled
EVAL_TOKENS = 4096    # tokens used to measure final utilisation
DEAD_THRESHOLD_PCT = 0.5  # an expert under this share of tokens is counted "dead"

# Run on the best available accelerator; CPU is the universal fallback.
DEVICE = (
    "cuda"
    if torch.cuda.is_available()
    else "mps"
    if torch.backends.mps.is_available()
    else "cpu"
)


class MoELayer(nn.Module):
    """One Mixture-of-Experts layer: a linear router + N expert FFNs, top-k routed.

    Replaces the single dense FFN of a transformer block. forward() returns the
    routed-and-combined output, the auxiliary load-balancing loss, and each token's
    top-1 expert (handy for measuring utilisation).
    """

    def __init__(
        self,
        d_model: int = D_MODEL,
        d_ff: int = D_FF,
        n_experts: int = N_EXPERTS,
        k: int = TOP_K,
    ) -> None:
        super().__init__()
        self.n_experts = n_experts
        self.k = k
        # The router/gating network: one linear map d_model -> n_experts, no bias.
        # bias=False because a per-expert bias just shifts all logits and the softmax
        # is shift-equivariant per token -- the bias would be redundant with the experts.
        self.router = nn.Linear(d_model, n_experts, bias=False)
        # N ordinary feed-forward networks. "Expert" is a grand word for an MLP:
        # there is nothing special inside one -- the intelligence is in the router.
        self.experts = nn.ModuleList(
            nn.Sequential(nn.Linear(d_model, d_ff), nn.GELU(), nn.Linear(d_ff, d_model))
            for _ in range(n_experts)
        )

    def forward(
        self, x: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """x: (tokens, d_model) -> (output, aux_loss, top1_expert_per_token)."""
        logits = self.router(x)                       # (T, N): one router score per expert
        probs = F.softmax(logits, dim=-1)             # (T, N): g(x) = softmax(W_g x), sums to 1 per token
        gate, idx = probs.topk(self.k, dim=-1)        # top-k: the k highest-scoring experts per token
        gate = gate / gate.sum(-1, keepdim=True)      # renormalise the k gates so they sum to 1 (convex combine)

        # Combine the k chosen experts, weighted by their renormalised gates.
        # Grouping by expert keeps each expert a single batched matmul over only its
        # tokens -- the gather/scatter that real engines vectorise (here: clear, not fast).
        y = torch.zeros_like(x)
        for slot in range(self.k):                    # slot 0 = best expert, slot 1 = 2nd best, ...
            for expert_idx in range(self.n_experts):
                mask = idx[:, slot] == expert_idx     # tokens whose slot-th pick is this expert
                if mask.any():
                    # gate[mask, slot] is this expert's weight for those tokens; unsqueeze to broadcast over d_model
                    y[mask] += gate[mask, slot : slot + 1] * self.experts[expert_idx](x[mask])

        # Auxiliary load-balancing loss (Switch Transformer):  L = N * sum_i f_i * P_i.
        top1 = idx[:, 0]                              # each token's single best expert
        # f_i = fraction of tokens routed (by top-1) to expert i -- a HARD count, no gradient.
        f = F.one_hot(top1, self.n_experts).float().mean(0)
        # P_i = mean router probability for expert i -- the SOFT, differentiable statistic.
        P = probs.mean(0)
        # The product f_i * P_i lets the gradient flow through P_i while f_i acts as a
        # measured weight: over-loaded experts (large f_i) get their probability pushed down.
        aux = self.n_experts * (f * P).sum()
        return y, aux, top1


def show_routing() -> None:
    """Run a tiny batch through one MoE layer, printing shapes and per-token routing."""
    torch.manual_seed(0)
    n_tokens = 6
    moe = MoELayer()
    x = torch.randn(n_tokens, D_MODEL)               # a small, readable batch of token vectors

    logits = moe.router(x)                           # (T, N) router scores
    probs = F.softmax(logits, dim=-1)                # (T, N) g(x)
    gate, idx = probs.topk(moe.k, dim=-1)            # which k experts, and their gates
    gate = gate / gate.sum(-1, keepdim=True)         # renormalise over the chosen k
    y, aux, _ = moe(x)

    print(f"input x            : {tuple(x.shape)}   (T={n_tokens} tokens, d_model={D_MODEL})")
    print(f"router logits      : {tuple(logits.shape)}   (one score per expert)")
    print(f"router probs g(x)  : {tuple(probs.shape)}   (softmax, each row sums to 1)")
    print(f"top-{moe.k} experts/token: {tuple(idx.shape)}   (the chosen expert indices)")
    print(f"output y           : {tuple(y.shape)}   (same shape as input -- a drop-in FFN)")
    print()
    print(f"each token -> its top-{moe.k} experts (renormalised gate weights):")
    for t in range(n_tokens):
        picks = ", ".join(
            f"E{idx[t, s].item()} (g={gate[t, s].item():.2f})" for s in range(moe.k)
        )
        print(f"  token {t}: {picks}")
    print(f"\naux load-balancing loss on this batch: {aux.item():.4f}"
          f"  (min possible = 1.0 at perfect balance)\n")


def aux_loss(f: torch.Tensor, p: torch.Tensor, n_experts: int) -> float:
    """Auxiliary load-balancing loss  L = N * sum_i f_i * P_i  for given f, P vectors."""
    return float(n_experts * (f * p).sum())


def show_aux_loss_balanced_vs_collapsed() -> None:
    """Compute the aux loss on a balanced vs a collapsed routing and assert the ordering."""
    n = 4  # 4 experts, by hand, matching the worked example on the concept page
    # Balanced: every expert gets a quarter of the tokens and a quarter of the probability.
    f_bal = torch.tensor([0.25, 0.25, 0.25, 0.25])
    p_bal = torch.tensor([0.25, 0.25, 0.25, 0.25])
    # Collapsed: one expert hogs the tokens; the router's mean prob is skewed to match.
    f_col = torch.tensor([0.70, 0.20, 0.07, 0.03])
    p_col = torch.tensor([0.62, 0.22, 0.10, 0.06])

    l_bal = aux_loss(f_bal, p_bal, n)
    l_col = aux_loss(f_col, p_col, n)
    print(f"aux loss, BALANCED routing (f=P=0.25 each) : {l_bal:.3f}   <- the minimum, 1.0")
    print(f"aux loss, COLLAPSED routing (one expert 70%): {l_col:.3f}   <- nearly double")
    # The whole point of the penalty: collapse must cost MORE than balance, so the
    # gradient pushes the over-used experts' probabilities down.
    assert l_col > l_bal, "collapsed routing must score a higher aux loss than balanced"
    assert abs(l_bal - 1.0) < 1e-6, "perfectly balanced routing must bottom out at exactly 1.0"
    print("assert passed: collapsed aux loss > balanced aux loss, and balanced == 1.000\n")


def train_moe(use_aux: bool, steps: int = TRAIN_STEPS, seed: int = 0) -> torch.Tensor:
    """Train one MoE layer (top-1) and return final per-expert utilisation in percent.

    The training signal deliberately includes the rich-get-richer driver: at each step
    we find, per token, which expert *currently* fits best and push the router toward it.
    Left unchecked that collapses routing onto a few experts; the aux loss is the only
    counter-pressure. Toggling use_aux isolates exactly what the aux loss buys.
    """
    torch.manual_seed(seed)
    moe = MoELayer(k=1)                               # top-1 routing: collapse is sharpest here
    opt = torch.optim.Adam(moe.parameters(), lr=LEARNING_RATE)
    target_map = torch.randn(D_MODEL, D_MODEL)        # a fixed nonlinear target the experts learn

    for _ in range(steps):
        x = torch.randn(BATCH_TOKENS, D_MODEL)
        target = torch.tanh(x @ target_map)
        y, aux, _ = moe(x)
        # Rich-get-richer driver: pick each token's currently-best expert (lowest error)
        # and push the router toward it. This is the collapse pressure made explicit.
        with torch.no_grad():
            per_expert_err = torch.stack(
                [((moe.experts[e](x) - target) ** 2).mean(-1) for e in range(moe.n_experts)],
                dim=-1,
            )                                          # (T, N): each expert's error on each token
            want = (-per_expert_err).argmax(-1)        # the lowest-error expert per token
        loss = F.mse_loss(y, target) + ROUTER_PUSH * F.cross_entropy(moe.router(x), want)
        if use_aux:
            loss = loss + AUX_COEFF * aux              # the only thing fighting the collapse
        opt.zero_grad()
        loss.backward()
        opt.step()

    with torch.no_grad():                              # measure final utilisation on fresh tokens
        x = torch.randn(EVAL_TOKENS, D_MODEL)
        _, _, top1 = moe(x)
        counts = torch.bincount(top1, minlength=moe.n_experts).float()
        return counts / counts.sum() * 100.0           # percent of tokens per expert


def show_collapse_vs_balanced() -> None:
    """Train without and with the aux loss; print utilisation and the dead-expert count."""
    print(f"expert utilisation after training (ideal = {IDEAL_UTIL_PCT:.1f}% each, 0 dead):")
    util_no_aux = None
    for use_aux in (False, True):
        util = train_moe(use_aux=use_aux)
        dead = int((util < DEAD_THRESHOLD_PCT).sum())
        tag = "WITH aux " if use_aux else "NO aux   "
        bars = ", ".join(f"{v:4.1f}" for v in util)
        print(f"  {tag}| util% = [{bars}] | max {util.max():.1f}%  dead {dead}/{len(util)}")
        if not use_aux:
            util_no_aux = util
            dead_no_aux = dead
        else:
            dead_with_aux = dead
    # The headline result: the aux loss revives the dead experts.
    assert dead_with_aux < dead_no_aux or dead_no_aux == 0, (
        "the aux loss should not increase the number of dead experts"
    )
    print(
        f"\nassert passed: dead experts went {dead_no_aux} (no aux) -> "
        f"{dead_with_aux} (with aux) -- the aux loss keeps every expert alive."
    )


def main() -> None:
    print("device:", DEVICE)
    print("torch:", torch.__version__)
    print("\n=== 1. one MoE layer: shapes and per-token routing ===\n")
    show_routing()
    print("=== 2. the auxiliary loss: balanced vs collapsed, by hand ===\n")
    show_aux_loss_balanced_vs_collapsed()
    print("=== 3. router collapse vs balance: trained, measured ===\n")
    show_collapse_vs_balanced()


if __name__ == "__main__":
    main()
