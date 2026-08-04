"""From-scratch seq2seq encoder-decoder: the fixed-context bottleneck, and attention as the fix.

Single seeded source of truth for the chapter "08 Sequence-to-Sequence and Encoder-Decoder".
Both the teaching notebook and the figure generator (`make_figures_08.py`) import the functions
defined here, so the prose, the notebook output, and every embedded figure come from the SAME
verified code and cannot silently drift apart.

What it builds, from scratch (no nn.Transformer, no HF):

  * a bidirectional GRU ENCODER that reads a digit string into per-position hidden states;
  * a NO-ATTENTION decoder, conditioned only on the single final context vector c = h_S
    (this is the bottleneck -- the whole source squeezed through one fixed vector);
  * a BAHDANAU-ATTENTION decoder that, every step, computes a fresh context
    c_t = sum_j alpha_tj h_j as a softmax-weighted blend of ALL encoder states;
  * training with teacher forcing; free-running (autoregressive) exact-match evaluation;
  * an alignment-matrix extractor (the attention weights as soft word-alignment);
  * a greedy vs beam-search decoder to show beam recovers the higher-probability sequence.

The toy task is COPY (output = input): the easiest possible seq2seq task -- no reordering, no
vocabulary change -- so any failure is purely the model's capacity to carry information through
the context channel. That isolates the bottleneck cleanly: the no-attention model collapses with
length while the attention model holds.

Device-agnostic (CUDA / MPS / CPU): DEVICE is computed once and threaded through every tensor and
module via `.to(DEVICE)` / `device=`. The absolute milliseconds and exact percentages are
device-dependent, but the qualitative results -- attention >> no-attention, a clean alignment
diagonal, accuracy degrading less with length under attention -- hold on any device, and the run
is seeded for reproducibility. Verified on Python 3.12 / torch 2.12.0 / numpy 2.4.6, CPU.

Run:
    python seq2seq.py
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np
import torch
import torch.nn as nn

# ---- Vocabulary -------------------------------------------------------------------------------
# Digits 0-9 are the "content" symbols; PAD/BOS/EOS are the three special tokens every seq2seq
# model needs (pad batches to equal length, start the decoder, and let it learn WHEN to stop).
N_DIGITS = 10           # content vocabulary: the digits 0..9
PAD = 10                # padding (ignored by the loss and the encoder embedding)
BOS = 11                # beginning-of-sequence: the decoder's first input
EOS = 12                # end-of-sequence: the token the decoder learns to emit to stop
VOCAB_SIZE = 13         # 10 digits + PAD + BOS + EOS
HIDDEN = 128            # GRU hidden width H (also the embedding width, for compactness)

TRAIN_MAX_LEN = 16      # training strings have length sampled uniformly from 1..TRAIN_MAX_LEN
DEFAULT_SEED = 0

# Run on the best available accelerator; CPU is the universal fallback. Computed once, then
# threaded through every tensor and module so the printed device matches what actually executes.
DEVICE = (
    "cuda"
    if torch.cuda.is_available()
    else "mps"
    if torch.backends.mps.is_available()
    else "cpu"
)


def set_seed(seed: int = DEFAULT_SEED) -> None:
    """Seed Python/NumPy/torch so a fresh run reproduces the same numbers (given the same device)."""
    np.random.seed(seed)
    torch.manual_seed(seed)


# ---- Data: random digit strings to COPY -------------------------------------------------------
def make_batch(
    batch_size: int, min_len: int, max_len: int, *, device: str = DEVICE
) -> tuple[torch.Tensor, torch.Tensor]:
    """Return (src, tgt) for a COPY task: tgt is src wrapped in BOS ... EOS.

    src[b] = d_1 .. d_L EOS PAD...      (the digits, then EOS, padded to the batch max)
    tgt[b] = BOS d_1 .. d_L EOS PAD...  (teacher-forcing target: same digits, BOS-prefixed)

    Lengths vary within the batch, so we pad to the longest. PAD positions are masked out of the
    loss (ignore_index) and the embedding (padding_idx), so they contribute nothing to learning.
    """
    lengths = np.random.randint(min_len, max_len + 1, size=batch_size)
    max_in_batch = int(lengths.max())
    src = np.full((batch_size, max_in_batch + 1), PAD, dtype=np.int64)        # +1 for EOS
    tgt = np.full((batch_size, max_in_batch + 2), PAD, dtype=np.int64)        # +2 for BOS and EOS
    for b in range(batch_size):
        length = int(lengths[b])
        digits = np.random.randint(0, N_DIGITS, size=length)
        src[b, :length] = digits
        src[b, length] = EOS
        tgt[b, 0] = BOS
        tgt[b, 1 : 1 + length] = digits
        tgt[b, 1 + length] = EOS
    return (
        torch.tensor(src, device=device),
        torch.tensor(tgt, device=device),
    )


# ---- Encoder ----------------------------------------------------------------------------------
class Encoder(nn.Module):
    """Bidirectional GRU encoder: read the source into per-position states AND one context vector.

    Returns BOTH so a single encoder can feed either decoder:
      * `states` (B, S, H) -- every encoder hidden state, what attention reads from;
      * `context` (1, B, H) -- the single final summary c = h_S, what the bottleneck decoder gets.
    Bidirectional means each h_j sees both left and right context (word meaning often depends on
    words *after* it); a learned `bridge` projects the concatenated 2H back to H.
    """

    def __init__(self, hidden: int = HIDDEN) -> None:
        super().__init__()
        self.embedding = nn.Embedding(VOCAB_SIZE, hidden, padding_idx=PAD)
        self.rnn = nn.GRU(hidden, hidden, batch_first=True, bidirectional=True)
        self.bridge = nn.Linear(2 * hidden, hidden)  # fuse forward+backward 2H back down to H

    def forward(self, src: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        embedded = self.embedding(src)                       # (B, S, H)
        outputs, final = self.rnn(embedded)                  # outputs (B, S, 2H); final (2, B, H)
        states = self.bridge(outputs)                        # (B, S, H) -- per-position states
        # final[0]=last forward state, final[1]=last backward state; concat -> one summary vector.
        context = self.bridge(torch.cat([final[0], final[1]], dim=-1)).unsqueeze(0)  # (1, B, H)
        return states, context


# ---- Decoders ---------------------------------------------------------------------------------
class DecoderNoAttention(nn.Module):
    """The bottleneck decoder: conditioned ONLY on the single context vector c.

    The encoder's whole output is squeezed into the GRU's initial hidden state (= context). From
    there the decoder runs free -- it never gets to look back at the source again. This is the
    vanilla Sutskever/Cho design, and the structural reason it collapses with length.
    """

    def __init__(self, hidden: int = HIDDEN) -> None:
        super().__init__()
        self.embedding = nn.Embedding(VOCAB_SIZE, hidden, padding_idx=PAD)
        self.rnn = nn.GRU(hidden, hidden, batch_first=True)
        self.out = nn.Linear(hidden, VOCAB_SIZE)

    def forward(
        self, tgt_in: torch.Tensor, states: torch.Tensor, context: torch.Tensor
    ) -> torch.Tensor:
        # `states` is accepted but unused -- the whole point is that this decoder cannot read it.
        del states
        embedded = self.embedding(tgt_in)                    # (B, T, H)
        outputs, _ = self.rnn(embedded, context)             # init hidden = the single context
        return self.out(outputs)                             # (B, T, VOCAB_SIZE) logits


class DecoderBahdanau(nn.Module):
    """Bahdanau (additive) attention decoder: a FRESH context c_t over all encoder states, per step.

    At each step t the decoder query s_{t-1} scores every encoder state h_j with a small MLP,
    softmax-normalizes the scores into alignment weights alpha_tj, blends the states into
    c_t = sum_j alpha_tj h_j, and decodes from [embed(y_{t-1}); c_t]. `return_attention=True`
    also returns the (B, T, S) stack of alignment weights -- the soft word-alignment matrix.
    """

    def __init__(self, hidden: int = HIDDEN) -> None:
        super().__init__()
        self.embedding = nn.Embedding(VOCAB_SIZE, hidden, padding_idx=PAD)
        self.rnn = nn.GRU(2 * hidden, hidden, batch_first=True)   # input = [embed; context]
        # Additive score e_tj = v^T tanh(W s_{t-1} + U h_j): W on the query, U on each key, v scores.
        self.attn_query = nn.Linear(hidden, hidden, bias=False)   # W_a
        self.attn_keys = nn.Linear(hidden, hidden, bias=False)    # U_a
        self.attn_v = nn.Linear(hidden, 1, bias=False)            # v_a
        self.out = nn.Linear(hidden, VOCAB_SIZE)

    def forward(
        self,
        tgt_in: torch.Tensor,
        states: torch.Tensor,
        context: torch.Tensor,
        *,
        return_attention: bool = False,
    ) -> torch.Tensor | tuple[torch.Tensor, torch.Tensor]:
        embedded = self.embedding(tgt_in)                    # (B, T, H)
        hidden = context                                     # (1, B, H) decoder GRU state, s_0 = c
        projected_keys = self.attn_keys(states)              # (B, S, H) U_a h_j, computed once
        logits_steps = []
        attn_steps = []
        for t in range(embedded.shape[1]):
            query = hidden[-1].unsqueeze(1)                  # (B, 1, H) -- s_{t-1}
            # e_tj = v^T tanh(W s_{t-1} + U h_j): broadcast query (B,1,H) over keys (B,S,H).
            scores = self.attn_v(torch.tanh(self.attn_query(query) + projected_keys))  # (B, S, 1)
            alpha = torch.softmax(scores, dim=1)             # (B, S, 1) alignment over source
            step_context = (alpha * states).sum(dim=1, keepdim=True)  # (B, 1, H) -- c_t
            rnn_in = torch.cat([embedded[:, t : t + 1], step_context], dim=-1)  # (B,1,2H)
            output, hidden = self.rnn(rnn_in, hidden)        # advance one step
            logits_steps.append(self.out(output))            # (B, 1, VOCAB_SIZE)
            attn_steps.append(alpha.squeeze(-1))             # (B, S)
        logits = torch.cat(logits_steps, dim=1)              # (B, T, VOCAB_SIZE)
        if return_attention:
            attention = torch.stack(attn_steps, dim=1)       # (B, T, S) -- the alignment matrix
            return logits, attention
        return logits


# ---- Training ---------------------------------------------------------------------------------
@dataclass
class TrainedModel:
    """An (encoder, decoder) pair plus a tag, so callers can keep both models straight."""

    encoder: Encoder
    decoder: nn.Module
    uses_attention: bool
    tag: str


def train_model(
    *,
    attention: bool,
    steps: int = 4000,
    batch_size: int = 96,
    learning_rate: float = 1e-3,
    seed: int = DEFAULT_SEED,
    device: str = DEVICE,
    log_every: int = 0,
) -> TrainedModel:
    """Train one encoder-decoder on the copy task with teacher forcing; return the trained pair.

    `attention=False` builds the bottleneck (single-context) model; `attention=True` builds the
    Bahdanau model. Everything else (size, steps, optimizer) is identical, so the ONLY difference
    in the results is the attention mechanism -- a clean controlled comparison.
    """
    set_seed(seed)
    encoder = Encoder().to(device)
    decoder: nn.Module = (DecoderBahdanau() if attention else DecoderNoAttention()).to(device)
    params = list(encoder.parameters()) + list(decoder.parameters())
    optimizer = torch.optim.Adam(params, lr=learning_rate)
    loss_fn = nn.CrossEntropyLoss(ignore_index=PAD)          # PAD positions don't count
    encoder.train()
    decoder.train()
    for step in range(1, steps + 1):
        src, tgt = make_batch(batch_size, 1, TRAIN_MAX_LEN, device=device)
        states, context = encoder(src)
        logits = decoder(tgt[:, :-1], states, context)       # teacher forcing: feed GOLD prefix
        # Predict tgt[:, 1:] (everything after BOS) from the gold-shifted input tgt[:, :-1].
        loss = loss_fn(logits.reshape(-1, VOCAB_SIZE), tgt[:, 1:].reshape(-1))
        optimizer.zero_grad()
        loss.backward()
        nn.utils.clip_grad_norm_(params, 1.0)                # tame exploding RNN gradients
        optimizer.step()
        if log_every and (step % log_every == 0 or step == 1):
            print(f"  step {step:>4}/{steps}  loss {loss.item():.4f}")
    encoder.eval()
    decoder.eval()
    tag = "Bahdanau attention" if attention else "no attention (1 vector)"
    return TrainedModel(encoder, decoder, attention, tag)


# ---- Free-running (autoregressive) evaluation -------------------------------------------------
@torch.no_grad()
def exact_match_accuracy(
    model: TrainedModel, length: int, *, n_samples: int = 300, seed: int | None = None,
    device: str = DEVICE,
) -> float:
    """Free-running exact-match accuracy: generate the copy autoregressively, check it equals src.

    This is the HONEST test -- the decoder feeds on its OWN previous output (no teacher forcing),
    exactly as at inference. A sample counts only if the WHOLE string is reproduced, which is why
    the bottleneck model's score falls off a cliff: one wrong digit fails the whole sequence.
    """
    if seed is not None:
        set_seed(seed)
    src_np = np.full((n_samples, length + 1), PAD, dtype=np.int64)
    gold = []
    for b in range(n_samples):
        digits = np.random.randint(0, N_DIGITS, size=length)
        src_np[b, :length] = digits
        src_np[b, length] = EOS
        gold.append(digits)
    src = torch.tensor(src_np, device=device)
    states, context = model.encoder(src)
    generated = torch.full((n_samples, 1), BOS, dtype=torch.long, device=device)
    for _ in range(length):                                  # generate exactly `length` digits
        logits = model.decoder(generated, states, context)  # re-run on the running prefix
        next_token = logits[:, -1].argmax(dim=-1, keepdim=True)  # greedy pick
        generated = torch.cat([generated, next_token], dim=1)
    predictions = generated[:, 1:].cpu().numpy()             # drop the BOS
    correct = sum(
        np.array_equal(predictions[b], gold[b]) for b in range(n_samples)
    )
    return correct / n_samples


@torch.no_grad()
def accuracy_vs_length(
    model: TrainedModel, lengths: tuple[int, ...], *, n_samples: int = 300, seed: int = 123,
    device: str = DEVICE,
) -> list[float]:
    """Exact-match accuracy at each length -- the bottleneck curve (and attention's flat one)."""
    out = []
    for i, length in enumerate(lengths):
        # Re-seed per length so each point draws fresh-but-reproducible strings.
        out.append(
            exact_match_accuracy(model, length, n_samples=n_samples, seed=seed + i, device=device)
        )
    return out


# ---- Attention alignment matrix ---------------------------------------------------------------
@torch.no_grad()
def alignment_matrix(
    model: TrainedModel, digits: list[int], *, device: str = DEVICE
) -> np.ndarray:
    """Free-run the attention model on one source and return its (T, S) alignment matrix.

    Row t is the softmax distribution over source positions when generating target token t -- the
    soft, unsupervised word-alignment. On the copy task a well-trained model gives a bright
    diagonal band: because the query that produces target token t is the decoder state s_{t-1},
    the brightest cells sit on (or just below) the main diagonal -- target position t reads source
    position ~t-1..t. Requires the attention decoder.
    """
    if not model.uses_attention:
        raise ValueError("alignment_matrix requires the attention model")
    length = len(digits)
    src_np = np.full((1, length + 1), PAD, dtype=np.int64)
    src_np[0, :length] = digits
    src_np[0, length] = EOS
    src = torch.tensor(src_np, device=device)
    states, context = model.encoder(src)
    generated = torch.full((1, 1), BOS, dtype=torch.long, device=device)
    rows = []
    for _ in range(length + 1):                              # +1 to also capture the EOS step
        logits, attention = model.decoder(
            generated, states, context, return_attention=True
        )
        rows.append(attention[0, -1].cpu().numpy())          # (S+1,) weights for the newest step
        next_token = logits[:, -1].argmax(dim=-1, keepdim=True)
        generated = torch.cat([generated, next_token], dim=1)
    return np.stack(rows, axis=0)                            # (T, S) alignment matrix


def diagonal_band_strength(matrix: np.ndarray, band: int = 1) -> float:
    """Mean attention mass within +/- `band` of the main diagonal -- the alignment's sharpness.

    The Bahdanau query for target token t is the decoder state s_{t-1}, so the bright cells lie on
    a narrow band around the diagonal, not exactly on it. A copy-task alignment concentrates almost
    all its mass in this band; a diffuse (untrained / failing) alignment spreads it out.
    """
    rows, cols = matrix.shape
    n = min(rows, cols)
    captured = sum(
        float(matrix[t, j])
        for t in range(n)
        for j in range(max(0, t - band), min(cols, t + band + 1))
    )
    return captured / n


# ---- Greedy vs beam search (sequence vs per-token) --------------------------------------------
def beam_search_demo(step1: dict[str, float], after: dict[str, dict[str, float]], beam_width: int = 2):
    """A tiny, deterministic beam-search illustration on a 2-step toy distribution.

    Not the model -- a hand-set probability tree (the page's "Worked example: greedy can lose to
    beam") so the mechanism is reproducible and the numbers on the page are exactly these. Returns
    (greedy_seq, greedy_prob, ranked_beam) where ranked_beam is [(seq, prob), ...] best-first.
    """
    # Greedy: take the argmax at step 1, then the argmax of its continuations.
    first = max(step1, key=lambda k: step1[k])
    second = max(after[first], key=lambda k: after[first][k])
    greedy_seq = f"{first} {second}"
    greedy_prob = step1[first] * after[first][second]
    # Beam: keep the top `beam_width` first tokens, expand all continuations, rank full sequences.
    top_first = sorted(step1, key=lambda k: step1[k], reverse=True)[:beam_width]
    full = []
    for a in top_first:
        for b, pb in after[a].items():
            full.append((f"{a} {b}", step1[a] * pb))
    full.sort(key=lambda pair: pair[1], reverse=True)
    return greedy_seq, greedy_prob, full


# ---- Hand-computed single attention step (matches the page's worked example) ------------------
def attention_step_by_hand() -> dict[str, object]:
    """Reproduce the page's by-hand attention step: 3 toy states, one query, dot-product scores.

    Returns the scores, the softmax alignment, and the resulting context vector, so the notebook
    can assert the page's numbers ([0.212, 0.212, 0.576] and c=[0.788, 0.788]) are exactly right.
    """
    h = np.array([[1.0, 0.0], [0.0, 1.0], [1.0, 1.0]])       # h_1, h_2, h_3
    s = np.array([1.0, 1.0])                                  # decoder query
    scores = h @ s                                            # dot-product scores e_j = s . h_j
    weights = np.exp(scores) / np.exp(scores).sum()           # softmax
    context = (weights[:, None] * h).sum(axis=0)              # c = sum_j alpha_j h_j
    return {"scores": scores, "weights": weights, "context": context}


# ---- Script entry point -----------------------------------------------------------------------
def main(*, device: str = "cpu") -> None:
    # The module is device-agnostic: DEVICE auto-detects the best accelerator (printed below).
    # The published table is PINNED to CPU so the exact percentages are reproducible across
    # machines -- RNG streams differ between CPU and MPS/CUDA, so a fixed device is what makes
    # "run it and get these numbers" true. The qualitative gap (attention >> bottleneck) holds
    # on any device.
    print(f"detected device (DEVICE): {DEVICE}   |   running this table on: {device}")
    print(f"torch: {torch.__version__}  numpy: {np.__version__}")
    print("(CPU pin: published numbers were generated and verified on CPU.)\n")

    print("Training no-attention (bottleneck) model ...")
    no_attn = train_model(attention=False, device=device)
    print("Training Bahdanau-attention model ...")
    attn = train_model(attention=True, device=device)

    # 1) The headline: attention holds, the single context vector collapses.
    print("\n--- exact-match accuracy (free-running) ---")
    print(f"{'model':>23} | {'short (L=6)':>11} | {'long (L=18)':>11}")
    print("-" * 53)
    for model in (no_attn, attn):
        short = exact_match_accuracy(model, 6, seed=7, device=device)
        long = exact_match_accuracy(model, 18, seed=8, device=device)
        # The bottleneck model must do far worse; the attention model must hold up.
        print(f"{model.tag:>23} | {short * 100:9.1f}% | {long * 100:9.1f}%")
    short_no = exact_match_accuracy(no_attn, 6, seed=7, device=device)
    short_attn = exact_match_accuracy(attn, 6, seed=7, device=device)
    assert short_attn > short_no + 0.3, "attention should beat the bottleneck by a wide margin"

    # 2) The alignment matrix is a clean diagonal on the copy task.
    print("\n--- attention alignment on '3 1 4 1 5 9 2' ---")
    matrix = alignment_matrix(attn, [3, 1, 4, 1, 5, 9, 2], device=device)
    band = diagonal_band_strength(matrix, band=1)
    print(f"alignment matrix shape (T, S): {matrix.shape}")
    print(f"diagonal-band mass (+/-1): {band:.2f}  (a clean copy alignment is near 1.0)")
    assert band > 0.8, "copy-task attention should concentrate on the diagonal band"

    # 3) The accuracy-vs-length sweep (the bottleneck curve).
    print("\n--- accuracy vs length (the bottleneck curve) ---")
    sweep_lengths = (2, 4, 6, 8, 10, 12, 16)
    acc_no = accuracy_vs_length(no_attn, sweep_lengths, device=device)
    acc_attn = accuracy_vs_length(attn, sweep_lengths, device=device)
    print(f"{'length':>7} | {'no-attn':>8} | {'attention':>9}")
    print("-" * 32)
    for length, a_no, a_attn in zip(sweep_lengths, acc_no, acc_attn):
        print(f"{length:>7} | {a_no * 100:6.1f}% | {a_attn * 100:7.1f}%")
    assert acc_attn[-1] > acc_no[-1], "attention should degrade less with length"

    # 4) Greedy vs beam search: beam recovers the higher-probability sequence.
    print("\n--- greedy vs beam search (toy 2-step distribution) ---")
    step1 = {"A": 0.5, "B": 0.5}
    after = {"A": {"X": 0.4, "Y": 0.6}, "B": {"X": 0.9, "Y": 0.1}}
    greedy_seq, greedy_prob, ranked = beam_search_demo(step1, after, beam_width=2)
    print(f"greedy: '{greedy_seq}' (p={greedy_prob:.2f})")
    print(f"beam best: '{ranked[0][0]}' (p={ranked[0][1]:.2f})")
    assert ranked[0][1] > greedy_prob, "beam should find a higher-probability sequence than greedy"

    # 5) The by-hand attention step matches the page's worked example.
    hand = attention_step_by_hand()
    print("\n--- attention step by hand (matches worked example) ---")
    print(f"scores:  {hand['scores']}")
    print(f"weights: {np.round(hand['weights'], 3)}")
    print(f"context: {np.round(hand['context'], 3)}")
    assert math.isclose(hand["weights"][2], 0.576, abs_tol=2e-3)
    print("\nall assertions passed.")


if __name__ == "__main__":
    main()
