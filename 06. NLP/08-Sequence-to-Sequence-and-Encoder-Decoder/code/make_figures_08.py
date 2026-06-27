"""Reproducible figure generator for 08-Sequence-to-Sequence-and-Encoder-Decoder.

Produces every embedded PNG for the chapter from the SAME backend used on the page and in the
notebook -- the trained encoder-decoders, the free-running accuracy sweep, the attention alignment
matrix, and the by-hand attention step are all IMPORTED from `seq2seq.py`, so the figures cannot
silently drift from the prose or the demo. Run:

    python make_figures_08.py

Each figure is written to ../../images/ (the shared chapter image dir, "06. NLP/images/") at
150 dpi, prefixed `s2s_`. The palette matches the chapter's Mermaid diagrams (muted, white text on
fills).

Figures produced (measured = from the live trained models; illustrative = a labelled schematic):
  s2s_unrolled.png            -- illustrative: the encoder -> context vector -> decoder unrolled
  s2s_context_capacity.png    -- illustrative: bits-to-store vs a fixed-vector channel capacity
  s2s_bottleneck.png          -- measured: exact-match accuracy vs length, attention vs no-attention
  s2s_attention_step.png      -- measured: one attention step (scores -> softmax -> blended context)
  s2s_alignment.png           -- measured: the attention alignment heatmap (the mandatory diagonal)
  s2s_bahdanau_vs_luong.png   -- illustrative: additive vs multiplicative score functions
  s2s_teacher_forcing.png     -- illustrative: teacher forcing vs free-running error compounding
  s2s_beam_vs_greedy.png      -- measured: the greedy-vs-beam probability tree (worked example)
  s2s_attention_cost.png      -- illustrative: per-decode cost, no-attention O(T H) vs attention O(T S H)

The two trained models are reused across every measured figure (trained once on import), so all
numbers come from a single seeded run. Verified on Python 3.12 / torch 2.12.0 / numpy 2.4.6, CPU,
deterministic. The published figures are generated on CPU for reproducibility.
"""

from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")  # headless backend: write files, never open a window
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch

from seq2seq import (
    alignment_matrix,
    attention_step_by_hand,
    beam_search_demo,
    train_model,
)

# ---- Palette (matches the chapter Mermaid classDefs) ------------------------------------------
BLUE = "#3A6B96"
PURPLE = "#5D4A8A"
GREEN = "#2E7A5A"
RED = "#8B3B4A"
SLATE = "#4A5B6E"
AMBER = "#7A6528"
NAVY = "#2A5B80"
INK = "#1C2530"  # near-black for axis text
GRID = "#D4D9DF"

OUT_DIR = Path(__file__).resolve().parent.parent.parent / "images"
DPI = 150
PUBLISH_DEVICE = "cpu"  # generate the published figures on CPU so numbers are reproducible

# Train both models ONCE on import; every measured figure reuses them (one seeded run).
_NO_ATTN = None
_ATTN = None


def _models():
    """Train (and cache) the two encoder-decoders the measured figures share."""
    global _NO_ATTN, _ATTN
    if _NO_ATTN is None:
        print("training no-attention (bottleneck) model ...")
        _NO_ATTN = train_model(attention=False, device=PUBLISH_DEVICE)
        print("training Bahdanau-attention model ...")
        _ATTN = train_model(attention=True, device=PUBLISH_DEVICE)
    return _NO_ATTN, _ATTN


def _style_axis(ax: plt.Axes) -> None:
    """Consistent muted styling: light grid, no top/right spines, ink-coloured labels."""
    ax.grid(True, color=GRID, linewidth=0.8, zorder=0)
    ax.set_axisbelow(True)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    for side in ("left", "bottom"):
        ax.spines[side].set_color(SLATE)
    ax.tick_params(colors=INK)
    ax.xaxis.label.set_color(INK)
    ax.yaxis.label.set_color(INK)
    ax.title.set_color(INK)


def _save(fig: plt.Figure, name: str) -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    path = OUT_DIR / name
    fig.savefig(path, dpi=DPI, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    print(f"wrote images/{name}")


def _box(ax, xy, w, h, text, color, *, fontsize=10, text_color="white"):
    """Draw a rounded filled box with centered white text."""
    x, y = xy
    box = FancyBboxPatch(
        (x, y), w, h, boxstyle="round,pad=0.02,rounding_size=0.04",
        linewidth=1.4, edgecolor="white", facecolor=color, zorder=3,
    )
    ax.add_patch(box)
    ax.text(x + w / 2, y + h / 2, text, ha="center", va="center",
            color=text_color, fontsize=fontsize, zorder=4, weight="bold")


def _arrow(ax, start, end, color=SLATE, *, lw=1.8, style="-|>"):
    ax.add_patch(FancyArrowPatch(
        start, end, arrowstyle=style, mutation_scale=14,
        linewidth=lw, color=color, zorder=2,
    ))


# ---- Figure 1: the encoder-decoder unrolled (illustrative) ------------------------------------
def fig_unrolled() -> None:
    """Illustrative: encoder reads the source into ONE context vector, decoder unrolls the target.

    The whole architecture in one picture: three encoder steps compress to a single context vector
    c, which conditions a decoder generating the target token by token (each output fed to the next
    step). The lone c bridging the two halves is the bottleneck this chapter is about.
    """
    fig, ax = plt.subplots(figsize=(9.6, 4.6))
    ax.set_xlim(0, 12)
    ax.set_ylim(0, 5.4)
    ax.axis("off")
    src = ["le", "chat", "noir"]
    tgt_in = ["<bos>", "the", "black"]
    tgt_out = ["the", "black", "cat"]
    # Encoder row (bottom-left).
    enc_y = 1.3
    enc_x = [0.4, 2.0, 3.6]
    for i, (x, w) in enumerate(zip(enc_x, src)):
        _box(ax, (x, 0.2), 1.2, 0.6, w, BLUE, fontsize=10)             # source token
        _box(ax, (x, enc_y), 1.2, 0.8, f"enc\nh{i+1}", PURPLE, fontsize=9)  # encoder state
        _arrow(ax, (x + 0.6, 0.8), (x + 0.6, enc_y))
        if i > 0:
            _arrow(ax, (enc_x[i-1] + 1.2, enc_y + 0.4), (x, enc_y + 0.4))
    # Context vector (center).
    _box(ax, (5.1, enc_y - 0.05), 1.5, 0.9, "context\nc = h3", RED, fontsize=9)
    _arrow(ax, (enc_x[-1] + 1.2, enc_y + 0.4), (5.1, enc_y + 0.4), color=RED, lw=2.4)
    ax.text(5.85, enc_y + 1.15, "the bottleneck:\nALL of the source,\none fixed vector",
            ha="center", va="bottom", color=RED, fontsize=8.5, style="italic")
    # Decoder row (right).
    dec_y = 1.3
    dec_x = [7.0, 8.6, 10.2]
    for i, (x, ti, to) in enumerate(zip(dec_x, tgt_in, tgt_out)):
        _box(ax, (x, dec_y), 1.2, 0.8, f"dec\ns{i+1}", GREEN, fontsize=9)
        _box(ax, (x, 0.2), 1.2, 0.6, ti, SLATE, fontsize=9)            # input (prev token)
        _box(ax, (x, 3.4), 1.2, 0.6, to, NAVY, fontsize=9)            # output token
        _arrow(ax, (x + 0.6, 0.8), (x + 0.6, dec_y))
        _arrow(ax, (x + 0.6, dec_y + 0.8), (x + 0.6, 3.4))
        if i > 0:
            _arrow(ax, (dec_x[i-1] + 1.2, dec_y + 0.4), (x, dec_y + 0.4))
            # output feeds next input (autoregressive)
            _arrow(ax, (dec_x[i-1] + 0.6, 4.0), (x + 0.6, 0.8), color=AMBER, lw=1.2, style="-|>")
    _arrow(ax, (6.6, enc_y + 0.4), (dec_x[0], dec_y + 0.4), color=RED, lw=2.4)
    ax.text(3.0, 5.0, "ENCODER  (read the source)", ha="center", color=PURPLE, fontsize=11, weight="bold")
    ax.text(9.4, 5.0, "DECODER  (generate the target, autoregressively)",
            ha="center", color=GREEN, fontsize=11, weight="bold")
    ax.text(8.6, 4.55, "amber = each output is fed back as the next input",
            ha="center", color=AMBER, fontsize=8)
    _save(fig, "s2s_unrolled.png")


# ---- Figure 2: context-vector capacity (illustrative) -----------------------------------------
def fig_context_capacity() -> None:
    """Illustrative: bits needed to store a length-S digit string vs a fixed channel's capacity.

    A length-S digit string carries S * log2(10) ~= 3.32 S bits. A fixed-size context vector under
    a trained, noisy RNN reliably recovers only a bounded number of bits B (flat line). Where the
    rising demand crosses the flat ceiling is S* -- past it, lossless copy is impossible, which is
    the bottleneck's information-theoretic root.
    """
    fig, ax = plt.subplots(figsize=(7.4, 4.8))
    S = np.arange(0, 24)
    bits_needed = S * np.log2(10)
    ceiling = 20.0  # an illustrative "reliably recoverable bits" B for a fixed vector
    ax.plot(S, bits_needed, color=BLUE, linewidth=2.6, label="bits to store the string  (3.32 S)")
    ax.axhline(ceiling, color=RED, linewidth=2.4, linestyle="--",
               label="fixed context vector's reliable capacity B")
    s_star = ceiling / np.log2(10)
    ax.axvline(s_star, color=SLATE, linewidth=1.4, linestyle=":")
    ax.scatter([s_star], [ceiling], color=AMBER, s=120, zorder=5, edgecolor="white")
    ax.annotate(f"S* ~= {s_star:.1f}\nbeyond here the\nbucket overflows",
                xy=(s_star, ceiling), xytext=(s_star + 1.5, ceiling - 9),
                color=INK, fontsize=9,
                arrowprops=dict(arrowstyle="->", color=SLATE))
    ax.fill_between(S, bits_needed, ceiling, where=(bits_needed > ceiling),
                    color=RED, alpha=0.12)
    ax.set_xlabel("source length S (digits)")
    ax.set_ylabel("information (bits)")
    ax.set_title("Why one fixed vector must fail with length (illustrative)\n"
                 "demand grows linearly; the channel's reliable capacity is bounded", fontsize=11)
    ax.legend(frameon=False, loc="upper left", fontsize=9)
    ax.set_xlim(0, 23)
    ax.set_ylim(0, 50)
    _style_axis(ax)
    _save(fig, "s2s_context_capacity.png")


# ---- Figure 3: the bottleneck, measured -------------------------------------------------------
def fig_bottleneck() -> None:
    """Measured: free-running exact-match accuracy vs source length, attention vs no-attention.

    The two trained models, evaluated at growing lengths. The no-attention model (single context
    vector) collapses almost immediately; the attention model holds at ~100% across the whole
    training range -- the bottleneck, measured, on the simplest seq2seq task (copy).
    """
    no_attn, attn = _models()
    from seq2seq import accuracy_vs_length

    lengths = (1, 2, 3, 4, 5, 6, 8, 10, 12, 14, 16, 18, 20)
    acc_no = np.array(accuracy_vs_length(no_attn, lengths, device=PUBLISH_DEVICE)) * 100
    acc_attn = np.array(accuracy_vs_length(attn, lengths, device=PUBLISH_DEVICE)) * 100
    fig, ax = plt.subplots(figsize=(7.6, 4.8))
    ax.plot(lengths, acc_attn, color=GREEN, linewidth=2.8, marker="o", markersize=6,
            markeredgecolor="white", label="WITH attention  (reads all states each step)")
    ax.plot(lengths, acc_no, color=RED, linewidth=2.8, marker="s", markersize=6,
            markeredgecolor="white", label="NO attention  (one fixed context vector)")
    ax.axvspan(1, 16, color=BLUE, alpha=0.06)
    ax.text(8.5, 8, "training range (lengths 1-16)", ha="center", color=SLATE, fontsize=8.5)
    ax.set_xlabel("source length (digits to copy)")
    ax.set_ylabel("exact-match accuracy (%)")
    ax.set_title("Attention holds; a single context vector collapses (measured)\n"
                 "copy task -- output = input, so failure is purely capacity to carry the source",
                 fontsize=11)
    ax.legend(frameon=False, loc="center right", fontsize=9)
    ax.set_ylim(-3, 105)
    ax.set_xlim(0, 21)
    _style_axis(ax)
    _save(fig, "s2s_bottleneck.png")


# ---- Figure 4: one attention step, measured ---------------------------------------------------
def fig_attention_step() -> None:
    """Measured: one attention step -- scores -> softmax weights -> blended context vector.

    The by-hand worked example from the page (3 toy encoder states, one query, dot-product scores),
    computed by the SAME function the page cites. Left: the three alignment weights. Right: the
    blended context as a weighted sum of the states. This is attention's core arithmetic, isolated.
    """
    hand = attention_step_by_hand()
    weights = hand["weights"]
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(9.6, 4.2))
    # Left: the alignment weights over the 3 source states.
    labels = ["h1 = [1,0]", "h2 = [0,1]", "h3 = [1,1]"]
    colors = [BLUE, PURPLE, GREEN]
    bars = ax1.bar(labels, weights, color=colors, edgecolor="white", linewidth=1.4)
    for bar, w in zip(bars, weights):
        ax1.text(bar.get_x() + bar.get_width() / 2, w + 0.02, f"{w:.3f}",
                 ha="center", color=INK, fontsize=10, weight="bold")
    ax1.set_ylabel("alignment weight  alpha_j = softmax(s . h_j)")
    ax1.set_title("Step 1: score each state, softmax to weights\nquery s = [1,1]", fontsize=10.5)
    ax1.set_ylim(0, 0.72)
    _style_axis(ax1)
    # Right: the blended context vector as a stacked contribution.
    ax2.set_xlim(-0.3, 1.4)
    ax2.set_ylim(-0.3, 1.4)
    states = np.array([[1, 0], [0, 1], [1, 1]])
    running = np.array([0.0, 0.0])
    for i, (h, w, c) in enumerate(zip(states, weights, colors)):
        contribution = w * h
        ax2.add_patch(FancyArrowPatch(running, running + contribution, arrowstyle="-|>",
                                      mutation_scale=14, color=c, linewidth=2.4, zorder=3))
        running = running + contribution
    ax2.scatter([running[0]], [running[1]], color=RED, s=160, zorder=5, edgecolor="white")
    ax2.annotate(f"context c = [{running[0]:.3f}, {running[1]:.3f}]",
                 xy=(running[0], running[1]), xytext=(running[0] - 0.55, running[1] + 0.18),
                 color=RED, fontsize=9.5, weight="bold")
    ax2.set_title("Step 2: blend states by weight -> context c_t\nc = sum_j alpha_j h_j", fontsize=10.5)
    ax2.set_xlabel("dim 1")
    ax2.set_ylabel("dim 2")
    _style_axis(ax2)
    _save(fig, "s2s_attention_step.png")


# ---- Figure 5: the alignment heatmap, measured (MANDATORY) ------------------------------------
def fig_alignment() -> None:
    """Measured: the attention alignment matrix on the copy source '3 1 4 1 5 9 2'.

    Rows = target steps, columns = source positions; each cell is the attention weight. A bright
    diagonal band means the model learned 'to emit output token t, look at source position ~t' --
    the copy mapping, discovered from scratch. On a TRANSLATION task this diagonal would bend and
    cross wherever the languages reorder; here the task is monotonic so it stays straight.
    """
    _, attn = _models()
    digits = [3, 1, 4, 1, 5, 9, 2]
    matrix = alignment_matrix(attn, digits, device=PUBLISH_DEVICE)
    src_labels = [str(d) for d in digits] + ["<eos>"]
    tgt_labels = [str(d) for d in digits] + ["<eos>"]
    # matrix is (T, S); rows are target steps. Trim/pad labels to the matrix shape.
    n_t, n_s = matrix.shape
    fig, ax = plt.subplots(figsize=(6.8, 6.0))
    im = ax.imshow(matrix, cmap="magma", aspect="auto", vmin=0, vmax=1)
    ax.set_xticks(range(n_s))
    ax.set_xticklabels(src_labels[:n_s])
    ax.set_yticks(range(n_t))
    ax.set_yticklabels(tgt_labels[:n_t])
    ax.set_xlabel("source position (input digit)")
    ax.set_ylabel("target step (output digit)")
    ax.set_title("Measured attention alignment on a copy (the diagonal IS the copy rule)\n"
                 "source '3 1 4 1 5 9 2' -> the model learns: output t reads source ~t", fontsize=10.5)
    # Annotate the brightest cell per row so the diagonal band is unmistakable.
    for t in range(n_t):
        j = int(matrix[t].argmax())
        ax.text(j, t, f"{matrix[t, j]:.2f}", ha="center", va="center",
                color="white", fontsize=7.5, weight="bold")
    cbar = fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    cbar.set_label("attention weight", color=INK)
    _save(fig, "s2s_alignment.png")


# ---- Figure 6: Bahdanau vs Luong score functions (illustrative) -------------------------------
def fig_bahdanau_vs_luong() -> None:
    """Illustrative: the two classic score functions side by side -- additive vs multiplicative.

    Bahdanau (additive): a small MLP, v^T tanh(W s + U h) -- expressive, works across dims.
    Luong (multiplicative): a (scaled) dot product, s^T W h -- cheaper, the Transformer's ancestor.
    Both feed the SAME softmax -> blend; only the scoring box differs.
    """
    fig, ax = plt.subplots(figsize=(9.4, 4.4))
    ax.set_xlim(0, 12)
    ax.set_ylim(0, 5)
    ax.axis("off")
    # Shared inputs.
    _box(ax, (0.3, 2.0), 1.5, 0.8, "query s\n(decoder)", PURPLE, fontsize=9)
    _box(ax, (0.3, 0.4), 1.5, 0.8, "keys h_j\n(encoder)", BLUE, fontsize=9)
    # Bahdanau branch (top).
    _box(ax, (2.6, 3.0), 3.4, 1.1, "BAHDANAU (additive)\n e = v^T tanh(W s + U h)", AMBER, fontsize=9)
    _arrow(ax, (1.8, 2.4), (2.6, 3.5))
    _arrow(ax, (1.8, 0.8), (2.6, 3.2))
    ax.text(4.3, 4.25, "one hidden layer -> expressive, dims may differ",
            ha="center", color=AMBER, fontsize=8)
    # Luong branch (bottom).
    _box(ax, (2.6, 0.6), 3.4, 1.1, "LUONG (multiplicative)\n e = s^T W h", GREEN, fontsize=9)
    _arrow(ax, (1.8, 2.2), (2.6, 1.4))
    _arrow(ax, (1.8, 0.8), (2.6, 1.1))
    ax.text(4.3, 0.35, "just a matmul -> cheaper, ancestor of QK^T/sqrt(d)",
            ha="center", color=GREEN, fontsize=8)
    # Shared tail: softmax -> blend.
    _box(ax, (6.8, 1.9), 2.0, 1.0, "softmax\nover j", SLATE, fontsize=9)
    _arrow(ax, (6.0, 3.5), (6.8, 2.6))
    _arrow(ax, (6.0, 1.1), (6.8, 2.2))
    _box(ax, (9.4, 1.9), 2.2, 1.0, "context c_t =\nsum_j alpha_j h_j", RED, fontsize=9)
    _arrow(ax, (8.8, 2.4), (9.4, 2.4))
    ax.text(6.0, 4.7, "Same softmax-and-blend; only the SCORE box differs",
            ha="center", color=INK, fontsize=11, weight="bold")
    _save(fig, "s2s_bahdanau_vs_luong.png")


# ---- Figure 7: teacher forcing vs free-running (illustrative) ---------------------------------
def fig_teacher_forcing() -> None:
    """Illustrative: teacher forcing feeds the gold prefix; free-running feeds the model's own.

    Top track (training, teacher forcing): every step gets the GOLD previous token, so a single
    wrong prediction doesn't propagate -- clean gradients. Bottom track (inference, free-running):
    each step gets the model's OWN previous token, so one mistake lands on an unfamiliar prefix and
    errors COMPOUND. The gap between the tracks is exposure bias.
    """
    fig, ax = plt.subplots(figsize=(9.6, 4.6))
    ax.set_xlim(0, 12)
    ax.set_ylim(0, 5.2)
    ax.axis("off")
    steps = ["the", "black", "cat", "<eos>"]
    # Teacher forcing (top): gold fed each step.
    ax.text(0.2, 4.7, "TRAINING -- teacher forcing (feed the GOLD token)",
            color=GREEN, fontsize=11, weight="bold")
    for i, w in enumerate(steps):
        x = 1.0 + i * 2.7
        _box(ax, (x, 3.4), 1.4, 0.8, f"gold:\n{w}", GREEN, fontsize=9)
        if i < len(steps) - 1:
            _arrow(ax, (x + 1.4, 3.8), (x + 2.7, 3.8), color=GREEN)
    ax.text(6.0, 3.0, "one wrong prediction does NOT propagate -- next input is still gold",
            ha="center", color=GREEN, fontsize=8.5, style="italic")
    # Free-running (bottom): own token fed each step, error compounds.
    own = ["the", "black", "dog", "barked"]
    ax.text(0.2, 2.2, "INFERENCE -- free-running (feed the model's OWN token)",
            color=RED, fontsize=11, weight="bold")
    for i, w in enumerate(own):
        x = 1.0 + i * 2.7
        col = RED if i >= 2 else SLATE
        _box(ax, (x, 0.8), 1.4, 0.8, f"own:\n{w}", col, fontsize=9)
        if i < len(own) - 1:
            _arrow(ax, (x + 1.4, 1.2), (x + 2.7, 1.2), color=(RED if i >= 1 else SLATE))
    ax.text(7.5, 0.35, "step-3 slip 'dog' -> unfamiliar prefix -> errors COMPOUND",
            ha="center", color=RED, fontsize=8.5, style="italic")
    ax.add_patch(FancyArrowPatch((6.0, 3.3), (6.0, 1.7), arrowstyle="<->", mutation_scale=14,
                                 color=INK, linewidth=1.4, linestyle="--"))
    ax.text(6.25, 2.5, "exposure\nbias", color=INK, fontsize=9, weight="bold")
    _save(fig, "s2s_teacher_forcing.png")


# ---- Figure 8: greedy vs beam search, measured ------------------------------------------------
def fig_beam_vs_greedy() -> None:
    """Measured: the greedy-vs-beam probability tree from the page's worked example.

    A 2-step toy distribution where greedy's locally-tied first choice (A) shuts the door on B's
    much better continuation. Beam width 2 keeps both, expands all four, and finds 'B X' at 0.45 --
    50% more probable than greedy's 'A Y' at 0.30. The numbers are computed by seq2seq.beam_search_demo.
    """
    step1 = {"A": 0.5, "B": 0.5}
    after = {"A": {"X": 0.4, "Y": 0.6}, "B": {"X": 0.9, "Y": 0.1}}
    greedy_seq, greedy_prob, ranked = beam_search_demo(step1, after, beam_width=2)
    best_seq, best_prob = ranked[0]
    fig, ax = plt.subplots(figsize=(8.4, 5.0))
    ax.set_xlim(0, 10)
    ax.set_ylim(0, 6.2)
    ax.axis("off")
    # Root.
    _box(ax, (0.3, 2.6), 1.2, 0.8, "<bos>", SLATE, fontsize=10)
    # Level 1: A, B.
    lvl1 = {"A": (3.0, 4.0, 0.5), "B": (3.0, 1.2, 0.5)}
    for tok, (x, y, p) in lvl1.items():
        _box(ax, (x, y), 1.0, 0.8, tok, PURPLE, fontsize=10)
        _arrow(ax, (1.5, 3.0), (x, y + 0.4))
        ax.text((1.5 + x) / 2, (3.0 + y + 0.4) / 2 + 0.12, f"{p:.1f}", color=INK, fontsize=8.5)
    # Level 2 continuations, sequence probs.
    seqs = {
        ("A", "X"): (6.0, 4.8), ("A", "Y"): (6.0, 3.4),
        ("B", "X"): (6.0, 1.8), ("B", "Y"): (6.0, 0.4),
    }
    for (a, b), (x, y) in seqs.items():
        prob = step1[a] * after[a][b]
        is_best = (f"{a} {b}" == best_seq)
        is_greedy = (f"{a} {b}" == greedy_seq)
        col = GREEN if is_best else (AMBER if is_greedy else SLATE)
        _box(ax, (x, y), 1.2, 0.7, f"{a} {b}\np={prob:.2f}", col, fontsize=8.5)
        ax_, ay_ = lvl1[a][0] + 1.0, lvl1[a][1] + 0.4
        _arrow(ax, (ax_, ay_), (x, y + 0.35))
        ax.text((ax_ + x) / 2, (ay_ + y + 0.35) / 2 + 0.1, f"{after[a][b]:.1f}",
                color=INK, fontsize=8)
    ax.text(0.2, 5.55, f"GREEN = beam best ('{best_seq}', p={best_prob:.2f})",
            ha="left", color=GREEN, fontsize=8.5, weight="bold")
    ax.text(0.2, 0.15, f"AMBER = greedy's pick ('{greedy_seq}', p={greedy_prob:.2f})",
            ha="left", color=AMBER, fontsize=8.5, weight="bold")
    ax.set_title("Greedy can lose to beam search (measured worked example)\n"
                 "greedy's tied first choice (A) misses B's far better continuation X", fontsize=11)
    _save(fig, "s2s_beam_vs_greedy.png")


# ---- Figure 9: attention's cost (illustrative) ------------------------------------------------
def fig_attention_cost() -> None:
    """Illustrative: per-decode cost, no-attention O(T H) vs attention O(T S H).

    Attention scores the query against ALL S encoder states every step, adding a factor of S to the
    decode cost. Plotted as relative work vs source length S (fixed T, H): the no-attention line is
    flat in S; the attention line grows linearly. The quality win (previous figures) is almost
    always worth this linear cost -- and it's exactly the cross-attention cost in a Transformer.
    """
    fig, ax = plt.subplots(figsize=(7.4, 4.6))
    S = np.arange(1, 41)
    no_attn_cost = np.ones_like(S, dtype=float)         # O(T H), flat in S (normalized to 1)
    attn_cost = S / 1.0                                  # O(T S H), linear in S (normalized)
    ax.plot(S, attn_cost, color=GREEN, linewidth=2.8, label="attention  O(T . S . H)  (per-step blend over all states)")
    ax.plot(S, no_attn_cost, color=RED, linewidth=2.8, label="no attention  O(T . H)  (one fixed context)")
    ax.fill_between(S, no_attn_cost, attn_cost, color=GREEN, alpha=0.10)
    ax.set_xlabel("source length S")
    ax.set_ylabel("relative decode work (normalized)")
    ax.set_title("Attention costs an extra factor of S per decode step (illustrative)\n"
                 "the price of reading all encoder states each step",
                 fontsize=11)
    ax.legend(frameon=False, loc="upper left", fontsize=9)
    ax.set_xlim(0, 40)
    _style_axis(ax)
    _save(fig, "s2s_attention_cost.png")


def main() -> None:
    print(f"writing figures to {OUT_DIR}")
    fig_unrolled()
    fig_context_capacity()
    fig_bottleneck()
    fig_attention_step()
    fig_alignment()
    fig_bahdanau_vs_luong()
    fig_teacher_forcing()
    fig_beam_vs_greedy()
    fig_attention_cost()
    print("all figures written.")


if __name__ == "__main__":
    main()
