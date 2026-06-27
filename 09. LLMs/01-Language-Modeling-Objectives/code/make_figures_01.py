"""Reproducible figure generator for 01-Language-Modeling-Objectives.

Produces every embedded PNG for the chapter from the SAME numbers used on the page and in the
notebook -- the constants and seeded functions are IMPORTED from language_modeling_objectives.py,
so the figures cannot silently drift from the prose or the demo. Run:

    python make_figures_01.py

Each figure is written to ../../images/ (the shared chapter image dir) at 150 dpi, prefixed lm_.
The palette matches the chapter's Mermaid diagrams (muted, white text on coloured fills).

Figures produced:
  lm_next_token_distribution.png  -- the by-hand next-token softmax (0.50 on "sat"); the true
                                     token highlighted, with the cross-entropy it incurs.
  lm_causal_vs_masked_attention.png-- causal (lower-triangular, GPT) vs bidirectional (full, BERT)
                                     attention masks side by side -- the one structural difference.
  lm_teacher_forcing.png          -- the teacher-forcing / SHIFT setup: position t's logits scored
                                     against label t+1, cross-entropy at every position.
  lm_loss_perplexity_curve.png    -- the tiny-LM training trace (loss + perplexity dropping toward
                                     1.0), the exact seeded numbers from training_trace().
  lm_causal_vs_mlm_positions.png  -- which positions are scored: causal scores EVERY next-token
                                     prediction; MLM scores only the ~15% masked positions.
  lm_cross_entropy_cost.png       -- the -log(p) curve, marking p=0.50 (loss 0.69) vs the
                                     confident-wrong p=0.05 (loss 3.00): why wrong-and-sure hurts.

Verified on Python 3.12 / torch 2.x / matplotlib 3.x.
"""

from __future__ import annotations

import math
from pathlib import Path

import matplotlib

matplotlib.use("Agg")  # headless backend: write files, never open a window
import matplotlib.pyplot as plt
import numpy as np

from language_modeling_objectives import (
    CONFIDENT_WRONG_PROB,
    PREDICTED_PROBS,
    TRUE_NEXT_TOKEN_ID,
    VOCAB,
    causal_and_bidirectional_masks,
    cross_entropy_curve,
    divergent_training_perplexity,
    predicted_next_token_probs,
    training_trace,
)

# ---- Palette (matches the chapter Mermaid classDefs) --------------------------------------
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
    print(f"wrote {path}")


def fig_next_token_distribution() -> None:
    """The by-hand next-token softmax over the toy vocab: 0.50 on the true token 'sat'.

    The true token is highlighted; the rest are context. The caption number -- cross-entropy
    -log(0.50) = 0.6931 nats -- is exactly what the page and notebook compute by hand.
    """
    vocab, probs, true_id = predicted_next_token_probs()
    x = np.arange(len(vocab))
    colors = [GREEN if i == true_id else SLATE for i in range(len(vocab))]
    alphas = [1.0 if i == true_id else 0.45 for i in range(len(vocab))]
    fig, ax = plt.subplots(figsize=(8.4, 4.6))
    for i in range(len(vocab)):
        ax.bar(x[i], probs[i], color=colors[i], alpha=alphas[i], edgecolor="white",
               linewidth=0.8, zorder=3)
        ax.text(x[i], probs[i] + 0.012, f"{probs[i]:.2f}", ha="center", va="bottom",
                fontsize=10, color=INK)
    loss = -math.log(probs[true_id])
    ax.annotate(
        f"true next token = '{vocab[true_id]}'\n"
        f"p = {probs[true_id]:.2f}  →  cross-entropy = −log {probs[true_id]:.2f} = {loss:.4f} nats",
        xy=(true_id, probs[true_id]), xytext=(true_id + 0.6, probs[true_id] - 0.04),
        textcoords="data", fontsize=10, color=GREEN, fontweight="bold",
        arrowprops=dict(arrowstyle="->", color=GREEN),
    )
    ax.set_xticks(x)
    ax.set_xticklabels([f"'{t}'" for t in vocab], fontsize=11)
    ax.set_ylabel("p(next token)")
    ax.set_ylim(0, 0.62)
    ax.set_title("The next-token distribution the loss scores: only −log p(true) matters")
    _style_axis(ax)
    fig.tight_layout()
    _save(fig, "lm_next_token_distribution.png")


def fig_causal_vs_masked_attention() -> None:
    """Causal (lower-triangular, GPT) vs bidirectional (full, BERT) attention masks side by side."""
    seq_len = 6
    causal, bidirectional = causal_and_bidirectional_masks(seq_len)
    tokens = ["The", "cat", "sat", "on", "the", "mat"][:seq_len]
    fig, axes = plt.subplots(1, 2, figsize=(11.6, 5.4))
    panels = [
        (causal.numpy(), "Causal mask — GPT (causal LM)\nposition t sees only 0..t (the past)", BLUE),
        (bidirectional.numpy(), "Bidirectional — BERT (masked LM)\nevery position sees every other", PURPLE),
    ]
    for ax, (mask, title, on_color) in zip(axes, panels):
        for i in range(seq_len):          # query rows (top = position 0)
            for j in range(seq_len):      # key columns
                allowed = mask[i, j] > 0.5
                fc = on_color if allowed else "#EAECEF"
                alpha = 0.92 if allowed else 1.0
                ax.add_patch(plt.Rectangle((j, seq_len - 1 - i), 0.94, 0.94, facecolor=fc,
                                           alpha=alpha, edgecolor="white", linewidth=1.0, zorder=2))
        ax.set_xlim(-0.2, seq_len + 0.2)
        ax.set_ylim(-0.2, seq_len + 0.2)
        ax.set_xticks(np.arange(seq_len) + 0.47)
        ax.set_xticklabels(tokens, fontsize=9)
        ax.set_yticks(np.arange(seq_len) + 0.47)
        ax.set_yticklabels(tokens[::-1], fontsize=9)
        ax.set_xlabel("key (attended-to) position")
        ax.set_ylabel("query (predicting) position")
        ax.set_title(title, fontsize=11, color=INK)
        ax.set_aspect("equal")
        for side in ("top", "right", "left", "bottom"):
            ax.spines[side].set_visible(False)
        ax.tick_params(length=0, colors=INK)
    n_causal = int(causal.sum().item())
    n_bidir = int(bidirectional.sum().item())
    fig.suptitle(
        f"One mask flag is the whole difference: {n_causal} allowed cells (causal) vs "
        f"{n_bidir} (bidirectional)",
        fontsize=13, color=INK,
    )
    fig.tight_layout(rect=(0, 0, 1, 0.95))
    _save(fig, "lm_causal_vs_masked_attention.png")


def fig_teacher_forcing() -> None:
    """Teacher forcing + the SHIFT: position t's prediction is scored against the true token t+1.

    Visualises the 'the cat sat on mat' sequence: each input position points to the next-token
    target it is scored against, with cross-entropy summed at every position (the causal-LM loss).
    """
    tokens = ["the", "cat", "sat", "on", "mat"]
    n = len(tokens)
    fig, ax = plt.subplots(figsize=(10.2, 4.6))
    y_in, y_out = 1.0, 0.0
    # input row (positions 0..n-1) and target row (tokens 1..n-1), illustrating the shift
    for i, tok in enumerate(tokens):
        ax.add_patch(plt.Rectangle((i, y_in), 0.86, 0.6, facecolor=BLUE, edgecolor="white",
                                   linewidth=1.0, zorder=3))
        ax.text(i + 0.43, y_in + 0.3, tok, ha="center", va="center", color="#fff",
                fontsize=11, fontweight="bold")
    for i in range(1, n):  # targets are tokens 1..n-1 (the SHIFT)
        ax.add_patch(plt.Rectangle((i, y_out), 0.86, 0.6, facecolor=GREEN, edgecolor="white",
                                   linewidth=1.0, zorder=3))
        ax.text(i + 0.43, y_out + 0.3, tokens[i], ha="center", va="center", color="#fff",
                fontsize=11, fontweight="bold")
        # arrow: position i-1 predicts target at i
        ax.annotate("", xy=(i + 0.18, y_out + 0.6), xytext=(i - 1 + 0.66, y_in),
                    arrowprops=dict(arrowstyle="->", color=AMBER, linewidth=1.6), zorder=4)
    ax.text(-0.15, y_in + 0.3, "input\n(true prefix)", ha="right", va="center", fontsize=10,
            color=BLUE, fontweight="bold")
    ax.text(-0.15, y_out + 0.3, "target\n(next token)", ha="right", va="center", fontsize=10,
            color=GREEN, fontweight="bold")
    ax.text(n / 2.0, y_out - 0.55,
            "cross-entropy at every position, averaged  →  the causal-LM loss\n"
            "SHIFT: logits[:, :−1] line up with labels[:, 1:]   (n tokens → n−1 predictions)",
            ha="center", va="center", fontsize=10, color=INK)
    ax.set_xlim(-1.6, n + 0.2)
    ax.set_ylim(-1.0, 1.9)
    ax.set_title("Teacher forcing: condition on the true prefix, predict each next token in parallel",
                 fontsize=12.5, color=INK)
    ax.axis("off")
    fig.tight_layout()
    _save(fig, "lm_teacher_forcing.png")


def fig_loss_perplexity_curve() -> None:
    """The tiny-LM training trace: loss and perplexity dropping toward 1.0 (the seeded numbers).

    Two stacked panels (one per quantity) rather than a dual axis: because perplexity = exp(loss)
    the two curves are visually near-identical on a shared plot, so separate panels keep each
    independently readable. The realistic >1.0 floor (ambiguous 'mat' vs 'rug' data) is marked.
    """
    steps, losses, perplexities = training_trace()
    floor_ppl = divergent_training_perplexity()  # the realistic >1.0 floor (ambiguous data)
    fig, (ax_loss, ax_ppl) = plt.subplots(2, 1, figsize=(9.2, 6.4), sharex=True)

    ax_loss.plot(steps, losses, "-o", color=PURPLE, linewidth=2.6, markersize=7, zorder=4)
    ax_loss.set_ylabel("loss (nats)")
    ax_loss.set_ylim(-0.12, max(losses) * 1.15)
    _style_axis(ax_loss)
    ax_loss.annotate(f"start: loss {losses[0]:.4f}",
                     xy=(steps[0], losses[0]), xytext=(steps[0] + 16, losses[0] - 0.05),
                     fontsize=10, color=PURPLE, fontweight="bold",
                     arrowprops=dict(arrowstyle="->", color=PURPLE))
    ax_loss.annotate(f"end: loss {losses[-1]:.4f}\n(memorized the one sentence)",
                     xy=(steps[-1], losses[-1]), xytext=(steps[-1] - 96, max(losses) * 0.55),
                     fontsize=10, color=PURPLE, fontweight="bold",
                     arrowprops=dict(arrowstyle="->", color=PURPLE))
    ax_loss.set_title("Training drives loss and perplexity down — maximum likelihood at work")

    ax_ppl.plot(steps, perplexities, "-s", color=GREEN, linewidth=2.6, markersize=7, zorder=4)
    ax_ppl.set_ylabel("perplexity = exp(loss)")
    ax_ppl.set_xlabel("training step")
    ax_ppl.set_ylim(0.7, max(perplexities) * 1.12)
    _style_axis(ax_ppl)
    ax_ppl.annotate(f"start: PPL {perplexities[0]:.2f}\n(uncertain among ~6 tokens)",
                    xy=(steps[0], perplexities[0]), xytext=(steps[0] + 16, perplexities[0] - 1.4),
                    fontsize=10, color=GREEN, fontweight="bold",
                    arrowprops=dict(arrowstyle="->", color=GREEN))
    ax_ppl.annotate(f"end: PPL {perplexities[-1]:.4f}",
                    xy=(steps[-1], perplexities[-1]), xytext=(steps[-1] - 70, perplexities[0] * 0.5),
                    fontsize=10, color=GREEN, fontweight="bold",
                    arrowprops=dict(arrowstyle="->", color=GREEN))
    ax_ppl.axhline(floor_ppl, color=RED, linestyle="--", linewidth=1.8, zorder=3)
    ax_ppl.text(steps[-1], floor_ppl + 0.12,
                f"realistic floor ≈ {floor_ppl:.2f}  (ambiguous data: 'mat' vs 'rug')",
                ha="right", va="bottom", color=RED, fontsize=9.5)

    fig.tight_layout()
    _save(fig, "lm_loss_perplexity_curve.png")


def fig_causal_vs_mlm_positions() -> None:
    """Which positions are scored: causal scores EVERY next-token prediction; MLM scores ~15%.

    A 20-token strip for each objective. Causal: 19 of 20 positions are training signals (every
    position predicts the next; the last has no next). MLM: only the ~15% masked positions (here 3)
    are scored. The visual reason causal LM is more sample-efficient per token.
    """
    n = 20
    rng = np.random.default_rng(0)  # seeded so the masked positions are reproducible
    masked = sorted(rng.choice(n, size=max(1, round(0.15 * n)), replace=False).tolist())
    fig, axes = plt.subplots(2, 1, figsize=(11.0, 4.4))

    # Causal row: every position except the last is a scored next-token prediction.
    ax = axes[0]
    for i in range(n):
        scored = i < n - 1
        fc = GREEN if scored else "#EAECEF"
        ax.add_patch(plt.Rectangle((i, 0), 0.92, 0.9, facecolor=fc, edgecolor="white",
                                   linewidth=1.0, zorder=2))
    n_causal = n - 1
    ax.set_title(f"Causal LM (GPT): {n_causal} of {n} positions are scored — every next-token "
                 f"prediction is a training signal", fontsize=11, color=INK, loc="left")
    ax.set_xlim(-0.2, n + 0.2)
    ax.set_ylim(-0.1, 1.0)
    ax.axis("off")

    # MLM row: only the ~15% masked positions are scored.
    ax = axes[1]
    for i in range(n):
        scored = i in masked
        fc = PURPLE if scored else "#EAECEF"
        ax.add_patch(plt.Rectangle((i, 0), 0.92, 0.9, facecolor=fc, edgecolor="white",
                                   linewidth=1.0, zorder=2))
        if scored:
            ax.text(i + 0.46, 0.45, "M", ha="center", va="center", color="#fff",
                    fontsize=9, fontweight="bold")
    ax.set_title(f"Masked LM (BERT): only the {len(masked)} masked (~15%) positions are scored — "
                 f"less signal per token", fontsize=11, color=INK, loc="left")
    ax.set_xlim(-0.2, n + 0.2)
    ax.set_ylim(-0.1, 1.0)
    ax.axis("off")

    fig.suptitle("Scored positions per sentence: causal LM extracts a signal at (almost) every "
                 "token; MLM at ~15%", fontsize=13, color=INK)
    fig.tight_layout(rect=(0, 0, 1, 0.93))
    _save(fig, "lm_causal_vs_mlm_positions.png")


def fig_cross_entropy_cost() -> None:
    """The −log(p) cost curve, marking p=0.50 (loss 0.69) vs confident-wrong p=0.05 (loss 3.00)."""
    probs, losses = cross_entropy_curve()
    true_p = PREDICTED_PROBS[TRUE_NEXT_TOKEN_ID]  # 0.50
    fig, ax = plt.subplots(figsize=(9.2, 5.0))
    ax.plot(probs, losses, color=PURPLE, linewidth=2.8, zorder=4)
    for p, color, note in [
        (true_p, GREEN, "model's guess: p(true)=0.50\n→ loss 0.69 (a fair coin)"),
        (CONFIDENT_WRONG_PROB, RED, "confident & wrong: p(true)=0.05\n→ loss 3.00 (brutal)"),
    ]:
        loss = -math.log(p)
        ax.plot([p], [loss], "o", color=color, markersize=10, zorder=5)
        ax.annotate(note, xy=(p, loss), xytext=(p + 0.12, loss + 0.4), fontsize=10,
                    color=color, fontweight="bold", arrowprops=dict(arrowstyle="->", color=color))
    ax.set_xlabel("probability the model assigned to the TRUE token")
    ax.set_ylabel("cross-entropy loss = −log p  (nats)")
    ax.set_xlim(0, 1.0)
    ax.set_ylim(0, 5.0)
    ax.set_title("Why confident-and-wrong is punished: −log p explodes as p(true) → 0")
    _style_axis(ax)
    fig.tight_layout()
    _save(fig, "lm_cross_entropy_cost.png")


def main() -> None:
    fig_next_token_distribution()
    fig_causal_vs_masked_attention()
    fig_teacher_forcing()
    fig_loss_perplexity_curve()
    fig_causal_vs_mlm_positions()
    fig_cross_entropy_cost()
    print("all figures written to", OUT_DIR)


if __name__ == "__main__":
    main()
