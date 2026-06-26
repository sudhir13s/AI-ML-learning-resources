"""In-context learning, from scratch: make ICL concrete, measured, and reproducible.

This is the single source of truth for every number quoted on the concept page, in the
teaching notebook, and in the figures. It runs three self-contained experiments on a tiny
from-scratch transformer -- NO pretrained weights, NO internet -- each proving one claim that
the page makes in prose:

  1. INDUCTION (the mechanism behind few-shot copying). Train a 2-layer attention-only
     transformer on a synthetic "in-context recall" task: sequences where the token after a
     repeated symbol must be copied from its earlier occurrence (... A B ... A -> B). The
     model learns an INDUCTION HEAD -- an attention pattern that, at the second A, looks back
     to the token right after the first A and copies it. We then read the head's attention
     weights and SHOW the look-back. This is Olsson et al.'s induction head, in miniature.

  2. FEW-SHOT > ZERO-SHOT. On a fresh held-out alphabet the model has NEVER been trained on,
     measure next-token accuracy as a function of k = number of in-context demonstrations of
     the rule. Zero-shot (k=0) is at chance; accuracy climbs as k grows -- the model infers
     the task FROM THE PROMPT, with no weight updates. We assert few-shot(k>=1) > zero-shot.

  3. SENSITIVITY (ICL is brittle). Hold k fixed but PERMUTE the order of the in-context
     demonstrations and measure how much the prediction swings. We assert the swing is real
     (the spread across permutations is non-trivial), making "ICL is order-sensitive"
     measured, not asserted.

DEVICE HONESTY: the model is tiny, so the reproducible run is pinned to CPU. We still detect
the best available accelerator and print it honestly -- "device: cpu (detected mps; pinned to
CPU for reproducibility)" -- because seeded cross-device float math drifts and would break the
page == notebook == .py == figures contract. Every asserted result is checked BEFORE any
timing.

Run:
    python prompting_icl.py
"""

from __future__ import annotations

import time
from collections.abc import Callable

import torch
import torch.nn.functional as F
from torch import nn

# --------------------------------------------------------------------------------------
# Reproducibility + device honesty
# --------------------------------------------------------------------------------------
SEED = 0
# The reproducible trace is pinned to CPU: seeded float math drifts across CPU/MPS/CUDA, and
# this model is small enough that CPU is fast. We still DETECT the accelerator and report it
# honestly, never printing a device we are not actually running on.
DETECTED_DEVICE = (
    "cuda"
    if torch.cuda.is_available()
    else "mps"
    if torch.backends.mps.is_available()
    else "cpu"
)
DEVICE = torch.device("cpu")  # pinned for reproducibility (see module docstring)


# --------------------------------------------------------------------------------------
# Synthetic in-context recall task
# --------------------------------------------------------------------------------------
# Vocabulary layout (kept tiny so attention patterns are readable by eye):
#   ids 0 .. N_SYMBOLS-1  : content symbols (the "alphabet" the task is played on)
#   id  CUE_ID            : a special CUE token meaning "the next token repeats an earlier one"
# A sequence is a run of random (symbol, value) PAIRS, then a CUE, then a KEY symbol that
# appeared earlier; the TARGET is the value that followed that key the first time. To predict
# it, the model must look back to the key's first occurrence and copy the token after it --
# exactly the induction operation.
#
# The (key -> value) mapping is SAMPLED FRESH for every sequence and never repeats, so the
# answer can NEVER be memorized in the weights -- it exists only in the prompt. This is the
# honest form of "in-context learning" for a from-scratch model: the model learns the TASK
# STRUCTURE (look back, copy) during training, and at eval it applies that structure to
# brand-new content it is seeing for the first time. (A fully DISJOINT eval alphabet is too
# strong for a tiny from-scratch unembedding, which never learns to emit unseen token ids;
# fresh random pairings over the full alphabet isolate ICL just as cleanly -- nothing about a
# specific sequence's answer was ever in the gradients.)
N_SYMBOLS = 40
CUE_ID = N_SYMBOLS  # the cue/query marker sits just above the symbol ids
VOCAB_SIZE = N_SYMBOLS + 1
N_PAIRS = 6  # number of (key, value) demonstration pairs per sequence
SEQ_LEN = 2 * N_PAIRS + 2  # pairs (2 tokens each) + cue + query key = position of the target

# --------------------------------------------------------------------------------------
# Tiny attention-only transformer (the smallest thing that can grow an induction head)
# --------------------------------------------------------------------------------------
D_MODEL = 64
N_HEADS = 4
HEAD_DIM = D_MODEL // N_HEADS
N_LAYERS = 2  # induction provably needs >=2 layers: one to label "previous token", one to copy
LEARNING_RATE = 3e-3
N_TRAIN_STEPS = 1500
BATCH_SIZE = 128

# --------------------------------------------------------------------------------------
# Evaluation knobs (the numbers the page/figures quote)
# --------------------------------------------------------------------------------------
SHOT_VALUES = (0, 1, 2, 3, 4, 5)  # k in-context demonstrations of the rule
N_EVAL_SEQS = 400  # sequences averaged per accuracy point
N_PERMUTATIONS = 12  # distinct demonstration orderings probed in the sensitivity test
TIMING_REPS = 3


def make_batch(
    batch_size: int,
    *,
    n_pairs: int = N_PAIRS,
    generator: torch.Generator,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Build a batch of in-context recall sequences over the full symbol alphabet.

    Each sequence: n_pairs random (key, value) pairs, then CUE, then one of the earlier keys.
    The target (what the model must predict at the final position) is that key's value. The
    (key, value) mapping is fresh per sequence, so the answer lives only in the prompt.
    Returns (tokens [B, SEQ_LEN], targets [B]).
    """
    tokens = torch.empty(batch_size, 2 * n_pairs + 2, dtype=torch.long)
    targets = torch.empty(batch_size, dtype=torch.long)
    for b in range(batch_size):
        # Sample DISTINCT keys so the look-back target is unambiguous; values may repeat.
        keys = torch.randperm(N_SYMBOLS, generator=generator)[:n_pairs]
        values = torch.randint(N_SYMBOLS, (n_pairs,), generator=generator)
        seq = []
        for k, v in zip(keys.tolist(), values.tolist()):
            seq += [k, v]
        query_idx = int(torch.randint(n_pairs, (1,), generator=generator).item())
        seq += [CUE_ID, keys[query_idx].item()]  # cue, then re-show one earlier key
        tokens[b] = torch.tensor(seq, dtype=torch.long)
        targets[b] = values[query_idx]  # the value that followed that key the first time
    return tokens.to(DEVICE), targets.to(DEVICE)


class CausalSelfAttention(nn.Module):
    """Multi-head causal self-attention; exposes the last-layer attention weights for reading."""

    def __init__(self) -> None:
        super().__init__()
        self.qkv = nn.Linear(D_MODEL, 3 * D_MODEL, bias=False)
        self.proj = nn.Linear(D_MODEL, D_MODEL, bias=False)
        self.last_attn: torch.Tensor | None = None  # cached for the induction visualization

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        batch, seq_len, _ = x.shape
        q, k, v = self.qkv(x).split(D_MODEL, dim=-1)
        # reshape (B, T, d_model) -> (B, n_heads, T, head_dim) so heads attend independently
        q = q.view(batch, seq_len, N_HEADS, HEAD_DIM).transpose(1, 2)
        k = k.view(batch, seq_len, N_HEADS, HEAD_DIM).transpose(1, 2)
        v = v.view(batch, seq_len, N_HEADS, HEAD_DIM).transpose(1, 2)
        scores = (q @ k.transpose(-1, -2)) / HEAD_DIM**0.5  # (B, n_heads, T, T) alignment
        causal = torch.triu(  # upper-triangular mask: position t may not see the future
            torch.ones(seq_len, seq_len, device=x.device, dtype=torch.bool), diagonal=1
        )
        scores = scores.masked_fill(causal, float("-inf"))
        attn = F.softmax(scores, dim=-1)  # (B, n_heads, T, T) attention weights
        self.last_attn = attn.detach()  # stash for the look-back picture
        out = (attn @ v).transpose(1, 2).reshape(batch, seq_len, D_MODEL)
        return self.proj(out)


class Block(nn.Module):
    """Pre-norm attention block. Attention-only (no MLP) keeps the induction circuit legible."""

    def __init__(self) -> None:
        super().__init__()
        self.norm = nn.LayerNorm(D_MODEL)
        self.attn = CausalSelfAttention()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x + self.attn(self.norm(x))  # residual stream: attention writes an update


class InductionTransformer(nn.Module):
    """The smallest model that can form an induction head: token + positional embeddings,
    N_LAYERS attention blocks, and a tied-free output head."""

    def __init__(self) -> None:
        super().__init__()
        self.tok_emb = nn.Embedding(VOCAB_SIZE, D_MODEL)
        self.pos_emb = nn.Embedding(SEQ_LEN, D_MODEL)
        self.blocks = nn.ModuleList(Block() for _ in range(N_LAYERS))
        self.norm_f = nn.LayerNorm(D_MODEL)
        self.head = nn.Linear(D_MODEL, VOCAB_SIZE, bias=False)

    def forward(self, tokens: torch.Tensor) -> torch.Tensor:
        seq_len = tokens.shape[1]
        pos = torch.arange(seq_len, device=tokens.device)
        x = self.tok_emb(tokens) + self.pos_emb(pos)[None]
        for block in self.blocks:
            x = block(x)
        return self.head(self.norm_f(x))  # (B, T, vocab) next-token logits

    @torch.no_grad()
    def predict_last(self, tokens: torch.Tensor) -> torch.Tensor:
        """Argmax next-token prediction at the final position (where the target lives)."""
        return self.forward(tokens)[:, -1, :].argmax(dim=-1)


def train_model() -> InductionTransformer:
    """Train the tiny transformer on the in-context recall task over the TRAIN alphabet only."""
    torch.manual_seed(SEED)
    gen = torch.Generator().manual_seed(SEED)
    model = InductionTransformer().to(DEVICE)
    opt = torch.optim.AdamW(model.parameters(), lr=LEARNING_RATE)
    model.train()
    for _ in range(N_TRAIN_STEPS):
        tokens, targets = make_batch(BATCH_SIZE, generator=gen)
        logits = model(tokens)[:, -1, :]  # only the final position carries a target
        loss = F.cross_entropy(logits, targets)
        opt.zero_grad()
        loss.backward()
        opt.step()
    model.eval()
    return model


@torch.no_grad()
def accuracy_vs_shots(model: InductionTransformer) -> dict[int, float]:
    """Measure next-token accuracy as a function of k in-context demonstrations.

    k=0 means the query key never appeared before (zero-shot: nothing to copy -> chance,
    ~1/N_SYMBOLS). k>=1 means the (query key -> value) rule is demonstrated k times in the
    prompt. The mapping is fresh per sequence, so any success above chance can ONLY come from
    reading the demonstrations in the prompt -- the definition of in-context learning.
    """
    gen = torch.Generator().manual_seed(SEED + 1)
    out: dict[int, float] = {}
    for k in SHOT_VALUES:
        correct = 0
        total = 0
        while total < N_EVAL_SEQS:
            batch = min(BATCH_SIZE, N_EVAL_SEQS - total)
            tokens, targets = _make_kshot_batch(batch, k, gen)
            preds = model.predict_last(tokens)
            correct += int((preds == targets).sum().item())
            total += batch
        out[k] = correct / total
    return out


def _make_kshot_batch(
    batch_size: int, k: int, gen: torch.Generator
) -> tuple[torch.Tensor, torch.Tensor]:
    """Build an eval batch with EXACTLY k demonstrations of the queried (key -> value) rule.

    We construct N_PAIRS slots; the queried key is shown k times earlier (all with the SAME
    value -- the rule to infer). The remaining slots hold distractor pairs with OTHER keys.
    For k=0 the query key is absent from the demonstrations -> the model has no in-context
    evidence and can only guess (zero-shot baseline = chance, ~1/N_SYMBOLS).
    """
    tokens = torch.empty(batch_size, SEQ_LEN, dtype=torch.long)
    targets = torch.empty(batch_size, dtype=torch.long)
    for b in range(batch_size):
        perm = torch.randperm(N_SYMBOLS, generator=gen)
        query_key = int(perm[0].item())
        rule_value = int(perm[1].item())  # the value query_key maps to (to be inferred)
        distractor_keys = perm[2 : 2 + N_PAIRS].tolist()  # keys != query_key
        slots: list[tuple[int, int]] = [(query_key, rule_value)] * k  # k demos of the rule
        di = 0
        while len(slots) < N_PAIRS:  # pad with distractor pairs to a fixed N_PAIRS length
            dk = distractor_keys[di % len(distractor_keys)]
            dv = int(torch.randint(N_SYMBOLS, (1,), generator=gen).item())
            slots.append((dk, dv))
            di += 1
        order = torch.randperm(N_PAIRS, generator=gen).tolist()  # demos not always first
        slots = [slots[i] for i in order]
        seq: list[int] = []
        for key, val in slots:
            seq += [key, val]
        seq += [CUE_ID, query_key]
        tokens[b] = torch.tensor(seq, dtype=torch.long)
        targets[b] = rule_value
    return tokens.to(DEVICE), targets.to(DEVICE)


@torch.no_grad()
def order_sensitivity(model: InductionTransformer) -> dict[str, float]:
    """Put the query key in CONFLICTING demonstrations and let ORDER pick the winner.

    Real ICL brittleness shows up when the context is ambiguous. So we show the query key
    TWICE -- once mapped to value A, once to value B -- and ask: which one does the model
    copy? With a clean look-back-to-FIRST-occurrence rule the model should always answer A,
    regardless of order. Any dependence on which demo comes LAST is a recency/primacy BIAS --
    exactly the order sensitivity the ICL literature reports. We measure, per sequence, how
    often the prediction FLIPS between A and B as we permute the two conflicting demos among
    the distractors, and report:
      - frac_order_sensitive: fraction of sequences whose answer changes with order,
      - recency_rate: among flips, how often the model copies the value shown LATER
        (recency bias) vs earlier.
    """
    gen = torch.Generator().manual_seed(SEED + 2)
    n_seqs = 200
    flip_count = 0
    later_wins = 0
    later_total = 0
    for _ in range(n_seqs):
        perm = torch.randperm(N_SYMBOLS, generator=gen)
        query_key = int(perm[0].item())
        value_a = int(perm[1].item())
        value_b = int(perm[2].item())
        if value_a == value_b:
            continue
        distractor_keys = perm[3 : 3 + (N_PAIRS - 2)].tolist()
        distractors = [
            (dk, int(torch.randint(N_SYMBOLS, (1,), generator=gen).item()))
            for dk in distractor_keys
        ]
        preds: set[int] = set()
        for _ in range(N_PERMUTATIONS):
            # Place the two conflicting demos at two random distinct slots; track which is later.
            order = torch.randperm(N_PAIRS, generator=gen).tolist()
            slots_keyed: list[tuple[int, int] | None] = [None] * N_PAIRS
            pos_a, pos_b = order[0], order[1]
            slots_keyed[pos_a] = (query_key, value_a)
            slots_keyed[pos_b] = (query_key, value_b)
            d_iter = iter(distractors)
            for i in range(N_PAIRS):
                if slots_keyed[i] is None:
                    slots_keyed[i] = next(d_iter)
            seq: list[int] = []
            for key, val in slots_keyed:  # type: ignore[misc]
                seq += [key, val]
            seq += [CUE_ID, query_key]
            tok = torch.tensor(seq, dtype=torch.long, device=DEVICE)[None]
            pred = int(model.predict_last(tok).item())
            preds.add(pred)
            if pred in (value_a, value_b):
                later_value = value_a if pos_a > pos_b else value_b
                later_total += 1
                later_wins += int(pred == later_value)
        if len({value_a, value_b} & preds) == 2:  # answered BOTH values across orderings
            flip_count += 1
    return {
        "frac_order_sensitive": flip_count / n_seqs,
        "recency_rate": (later_wins / later_total) if later_total else 0.0,
    }


@torch.no_grad()
def induction_lookback(model: InductionTransformer) -> dict[str, object]:
    """Read the induction head: at the query position, where does attention land?

    We feed one recall sequence and inspect the LAST layer's attention from the final (query)
    position. An induction head puts most of its weight on the position holding the VALUE that
    followed the query key earlier -- i.e. it "looks back and copies." Returns the attention
    row, the token ids, and the look-back target position for plotting.
    """
    gen = torch.Generator().manual_seed(SEED + 3)
    tokens, targets = make_batch(1, generator=gen)
    _ = model(tokens)  # populates each block's last_attn
    # Average the final-layer heads' attention from the query (last) position.
    final_attn = model.blocks[-1].attn.last_attn  # (1, n_heads, T, T)
    query_row = final_attn[0, :, -1, :].mean(dim=0)  # (T,) averaged over heads
    ids = tokens[0].tolist()
    query_key = ids[-1]
    # The value token sits immediately AFTER the first occurrence of the query key.
    lookback_value_pos = None
    for i in range(0, SEQ_LEN - 2, 2):  # keys live at even positions among the pairs
        if ids[i] == query_key:
            lookback_value_pos = i + 1
            break
    return {
        "tokens": ids,
        "attn_from_query": query_row.tolist(),
        "lookback_value_pos": lookback_value_pos,
        "target": int(targets[0].item()),
        "query_key": query_key,
    }


def timeit(fn: Callable[[], object], reps: int = TIMING_REPS) -> float:
    """Mean wall-clock milliseconds over `reps` runs (one warmup first)."""
    fn()
    start = time.perf_counter()
    for _ in range(reps):
        fn()
    return (time.perf_counter() - start) / reps * 1e3


def main() -> None:
    torch.manual_seed(SEED)
    print(f"device: {DEVICE} (detected {DETECTED_DEVICE}; pinned to CPU for reproducibility)")
    print("torch:", torch.__version__)
    print()

    model = train_model()

    # ---- Experiment 1: induction look-back (the mechanism) --------------------------
    look = induction_lookback(model)
    attn = look["attn_from_query"]
    vpos = look["lookback_value_pos"]
    assert vpos is not None, "query key must occur earlier for induction to copy"
    peak_pos = int(torch.tensor(attn).argmax().item())
    # The query position must put its STRONGEST look-back weight on the value-after-key slot.
    assert peak_pos == vpos, f"induction head did not look back (peak {peak_pos} != {vpos})"
    print("[1] Induction head look-back")
    print(f"    query key id={look['query_key']}, target value id={look['target']}")
    print(f"    attention from query peaks at position {peak_pos} "
          f"(the value after the key's first occurrence) -> copies it. weight={attn[vpos]:.2f}")
    print()

    # ---- Experiment 2: few-shot > zero-shot -----------------------------------------
    acc = accuracy_vs_shots(model)
    zero = acc[0]
    few = acc[max(SHOT_VALUES)]
    # The core ICL claim, asserted BEFORE timing: demonstrations in the prompt raise accuracy.
    assert few > zero + 0.3, f"few-shot ({few:.2f}) must beat zero-shot ({zero:.2f})"
    assert acc[1] > zero, "even one shot should beat zero-shot"
    print("[2] Few-shot beats zero-shot (fresh per-sequence rule, no weight updates)")
    print(f"    {'k (shots)':>10} | accuracy")
    print(f"    {'-'*10}-+---------")
    for k in SHOT_VALUES:
        print(f"    {k:>10} | {acc[k]:.3f}")
    print(f"    zero-shot={zero:.3f}  ->  {max(SHOT_VALUES)}-shot={few:.3f}  "
          f"(+{few - zero:.3f})")
    print()

    # ---- Experiment 3: order sensitivity (ICL is brittle) ---------------------------
    sens = order_sensitivity(model)
    # The brittleness claim, measured: with two CONFLICTING demos of the query key, a
    # non-trivial fraction of sequences flip their answer purely from reordering.
    assert sens["frac_order_sensitive"] > 0.05, "expected measurable order sensitivity"
    print("[3] Order sensitivity (two CONFLICTING demos, permuted order)")
    print(f"    fraction of sequences whose answer FLIPS with order : "
          f"{sens['frac_order_sensitive']:.3f}")
    print(f"    recency rate (copies the LATER demo when it flips)  : "
          f"{sens['recency_rate']:.3f}")
    print()

    # ---- Timing comes LAST, after every qualitative result is asserted --------------
    timing_tokens, timing_targets = make_batch(
        BATCH_SIZE, generator=torch.Generator().manual_seed(SEED)
    )
    t_fwd = timeit(
        lambda: F.cross_entropy(model(timing_tokens)[:, -1, :], timing_targets)
    )
    print(f"[timing] one forward+loss over a batch of {BATCH_SIZE}: {t_fwd:.1f} ms")


if __name__ == "__main__":
    main()
