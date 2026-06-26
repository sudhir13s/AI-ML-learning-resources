"""Language modeling objectives, from scratch: cross-entropy by hand, then watch a tiny LM learn.

Three self-contained demos that make the causal language-modeling objective concrete:
  1. Cross-entropy and perplexity computed BY HAND on one toy prediction, so the arithmetic
     is checkable against -log(p) and exp(loss).
  2. The causal mask printed as a lower-triangular matrix -- position t may not see the future.
  3. A tiny causal LM trained a few steps so loss and perplexity visibly DROP toward their
     floor (~1.0 when memorizing a single sentence), which is maximum likelihood at work.

This is the same verified demo embedded in the concept page and the teaching notebook.
Verified on Python 3.12 / torch 2.x. Device-agnostic (CUDA / MPS / CPU); the by-hand numbers
are exact on any device; the training curve drops toward its floor the same way everywhere
(the exact mid-curve digits are device-dependent, so the trace is run on CPU).

Run:
    python language_modeling_objectives.py
"""

from __future__ import annotations

import math

import torch
import torch.nn as nn
import torch.nn.functional as F

# --- Toy problem setup -------------------------------------------------------
VOCAB = ["the", "cat", "sat", "on", "mat"]  # a 5-token toy vocabulary
VOCAB_SIZE = len(VOCAB)
TRUE_NEXT_TOKEN_ID = VOCAB.index("sat")  # the token that actually came next: "sat" (id 2)

# The model's predicted distribution over the next token for the by-hand demo.
# Sums to 1.0; it puts 0.50 on the true token "sat".
PREDICTED_PROBS = (0.10, 0.20, 0.50, 0.15, 0.05)
CONFIDENT_WRONG_PROB = 0.05  # a confident-but-wrong guess, to show how it's punished

# --- Tiny-LM training hyperparameters (named, not magic) ---------------------
MASK_SEQ_LEN = 4  # sequence length used only to print the causal mask
EMBED_DIM = 16  # token embedding width for the tiny model
LEARNING_RATE = 0.05
TOTAL_STEPS = 200
STEPS_PER_REPORT = 40  # train in chunks of this many steps, then print one row
SEED = 0

# One repeated training sentence: "the cat sat on mat" as token ids, shape [1, 5].
TRAIN_SENTENCE_IDS = [[0, 1, 2, 3, 4]]

# Run on the best available accelerator; CPU is the universal fallback.
DEVICE = (
    "cuda"
    if torch.cuda.is_available()
    else "mps"
    if torch.backends.mps.is_available()
    else "cpu"
)

# train the trace on CPU so the printed loss curve matches the page/notebook on every machine
# (MPS/CUDA reorder float ops and shift the mid-curve digits)
TRACE_DEVICE = "cpu"


def cross_entropy_by_hand(prob_of_true_token: float) -> tuple[float, float]:
    """Return (cross_entropy_nats, perplexity) for a single next-token prediction.

    Cross-entropy is -log(p) of the true token; perplexity is exp of that loss. This is the
    exact per-position quantity the training loss averages over -- computed here by hand so the
    arithmetic is checkable.
    """
    loss = -math.log(prob_of_true_token)  # cross-entropy = negative log-likelihood of the truth
    perplexity = math.exp(loss)  # perplexity = exp(loss): the "effective branching factor"
    return loss, perplexity


def causal_mask(seq_len: int) -> torch.Tensor:
    """Return the (seq_len, seq_len) lower-triangular causal mask: 1 = may attend, 0 = blocked.

    Row t (a query position) has 1s in columns 0..t and 0s above the diagonal -- so position t
    can attend to itself and the past, never the future. This is what makes prediction honest.
    """
    return torch.tril(torch.ones(seq_len, seq_len))  # tril keeps the lower triangle (incl. diagonal)


class TinyCausalLM(nn.Module):
    """A minimal causal language model: embed -> nonlinearity -> per-token vocab logits.

    Deliberately tiny (no real attention) -- enough to demonstrate that the next-token
    cross-entropy objective drives loss and perplexity down, which is the point of the demo.
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


def causal_lm_loss(logits: torch.Tensor, token_ids: torch.Tensor) -> torch.Tensor:
    """Mean next-token cross-entropy with the standard shift.

    SHIFT: position t predicts token t+1, so logits[:, :-1] line up with labels[:, 1:]. Getting
    this shift wrong silently measures the wrong thing -- the single most common LM-loss bug.
    """
    shifted_logits = logits[:, :-1].reshape(-1, VOCAB_SIZE)  # drop the last position (no next token)
    shifted_labels = token_ids[:, 1:].reshape(-1)  # drop the first position (it's never a target)
    # F.cross_entropy fuses log_softmax + NLL in one numerically-stable op; never compute
    # log(softmax(logits)) yourself — it overflows for large logits
    return F.cross_entropy(shifted_logits, shifted_labels)  # averages -log p(true) over positions


def demo_cross_entropy_by_hand() -> None:
    """Show cross-entropy and perplexity for one prediction, and the cost of being confidently wrong."""
    prob_true = PREDICTED_PROBS[TRUE_NEXT_TOKEN_ID]
    loss, perplexity = cross_entropy_by_hand(prob_true)
    wrong_loss, _ = cross_entropy_by_hand(CONFIDENT_WRONG_PROB)
    print(f"p(true='{VOCAB[TRUE_NEXT_TOKEN_ID]}') = {prob_true:.2f}")
    print(f"cross-entropy = -log({prob_true:.2f}) = {loss:.4f} nats")
    print(f"perplexity    = exp({loss:.4f})    = {perplexity:.4f}")
    print(f"if it had said p={CONFIDENT_WRONG_PROB:.2f} instead: loss = {wrong_loss:.4f} (much worse)\n")


def demo_causal_mask() -> None:
    """Print the lower-triangular causal mask: each row sees only itself and the past."""
    mask = causal_mask(MASK_SEQ_LEN)
    print("causal mask (1=can attend, 0=blocked):")
    print(mask.int().numpy(), "\n")


def demo_training() -> float:
    """Train the tiny LM and print loss / perplexity dropping toward ~1.0 (memorizing one sentence)."""
    torch.manual_seed(SEED)
    # Build/run the trace on CPU (TRACE_DEVICE) so the printed loss curve is reproducible on
    # every machine; the detected DEVICE is still reported in main() for correctness work.
    sentence = torch.tensor(TRAIN_SENTENCE_IDS, device=TRACE_DEVICE)  # [1, 5] token ids
    model = TinyCausalLM(VOCAB_SIZE, EMBED_DIM).to(TRACE_DEVICE)
    optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE)

    print(f"{'step':>4} | {'loss':>7} | {'perplexity':>10}")
    print("-" * 28)
    for step in range(0, TOTAL_STEPS + 1, STEPS_PER_REPORT):
        # On the first report (step 0) take one step so we print the near-untrained loss;
        # afterward run a full chunk of STEPS_PER_REPORT updates between prints.
        steps_this_chunk = STEPS_PER_REPORT if step else 1
        loss = torch.tensor(float("nan"))
        for _ in range(steps_this_chunk):
            logits = model(sentence)  # [1, 5, vocab]
            loss = causal_lm_loss(logits, sentence)  # mean next-token cross-entropy (shifted)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
        perplexity = math.exp(loss.item())  # perplexity = exp(loss), the readable twin
        print(f"{step:>4} | {loss.item():>7.4f} | {perplexity:>10.4f}")
    return loss.item()


def main() -> None:
    print(f"compute device available: {DEVICE} (the tiny training trace runs on CPU for a reproducible curve)")
    print("torch:", torch.__version__, "\n")

    # 1. Cross-entropy / perplexity by hand on one toy prediction.
    demo_cross_entropy_by_hand()
    # The 0.50-probability case must give exactly -log(0.5) and exp of that.
    loss_half, ppl_half = cross_entropy_by_hand(0.5)
    assert abs(loss_half - math.log(2)) < 1e-9, "cross-entropy of p=0.5 must be log(2)"
    assert abs(ppl_half - 2.0) < 1e-9, "perplexity of a fair-coin prediction must be 2.0"

    # 2. The causal mask is strictly lower-triangular.
    demo_causal_mask()
    mask = causal_mask(MASK_SEQ_LEN)
    assert mask.equal(torch.tril(mask)), "causal mask must be lower-triangular"
    assert mask[0].sum().item() == 1, "position 0 may attend to exactly one token (itself)"

    # 3. Training drives perplexity toward its floor (~1.0 for one memorized sentence).
    final_loss = demo_training()
    # loose on purpose: the exact final digit (~1e-4) is device-dependent; <0.1 holds on CPU/MPS/CUDA alike
    assert final_loss < 0.1, "memorizing one sentence should drive the loss well below 0.1"


if __name__ == "__main__":
    main()
