"""Animated (GIF) intuition figure for 11-RAG-Evaluation.

Companion to the static PNGs. It brings faithfulness to life -- the DECOMPOSE -> check-each-claim ->
aggregate mechanism: the fluent-but-unfaithful answer is split into claims, each claim is checked
against the retrieved context one at a time (its support cosine appears and it's marked
supported/hallucinated), and the running faithfulness fraction assembles claim by claim -- ending at
2/3 = 0.67 because the 'solar panels' claim has no support in the context.

    python make_animation_11.py

The GIF is written to ../../images/ and uses matplotlib's PillowWriter (no ffmpeg needed). Every
claim, its support cosine, and the final faithfulness score are the REAL rag_evaluation.py output
over the ch5 all-MiniLM encoder; only the claim-split and the support JUDGE are illustrative
stand-ins for an LLM (stated in the honesty footer).

Produced:
  rag11_faithfulness_check.gif -- claims checked one-by-one against context; faithfulness assembles.

Verified on Python 3.12 / matplotlib 3.x / numpy 2.x / Pillow (PillowWriter) / sentence-transformers.
"""

from __future__ import annotations

import textwrap
from pathlib import Path

import matplotlib

matplotlib.use("Agg")  # headless: render frames to a file, never open a window
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation, PillowWriter

from rag_evaluation import (
    QUESTION,
    SUPPORT_THRESHOLD,
    UNFAITHFUL_ANSWER,
    DenseRetriever,
    faithfulness,
    full_corpus,
)

# ---- Palette (matches the chapter's muted Mermaid classDefs) -------------------------------
BLUE = "#3A6B96"
PURPLE = "#5D4A8A"
GREEN = "#2E7A5A"
RED = "#8B3B4A"
SLATE = "#4A5B6E"
AMBER = "#7A6528"
INK = "#1C2530"
GRID = "#D4D9DF"

OUT_DIR = Path(__file__).resolve().parent.parent.parent / "images"
DPI = 100
FPS = 12


def _wrap(text: str, width: int) -> str:
    return "\n".join(textwrap.wrap(text, width))


def build_animation() -> None:
    corpus = full_corpus()
    dense = DenseRetriever(corpus)
    ranked = dense.search(QUESTION, k=len(corpus)).indices
    context_chunks = tuple(corpus[i] for i in ranked[:3])
    context_text = " ".join(context_chunks)
    result = faithfulness(dense, UNFAITHFUL_ANSWER, context_text)
    n_claims = len(result.claims)

    # timeline: intro -> reveal each claim's check in turn -> hold on the final score
    frames_per_claim = 12
    intro = 8
    hold = 22
    total = intro + n_claims * frames_per_claim + hold

    fig, ax = plt.subplots(figsize=(10.6, 6.8))
    fig.subplots_adjust(left=0.03, right=0.97, top=0.9, bottom=0.04)

    def update(frame: int):
        ax.clear()
        ax.axis("off")
        ax.set_xlim(0, 1)
        ax.set_ylim(0, 1)

        # how many claims have been checked so far
        if frame < intro:
            checked = 0
        elif frame >= intro + n_claims * frames_per_claim:
            checked = n_claims
        else:
            checked = (frame - intro) // frames_per_claim + 1
        in_hold = frame >= intro + n_claims * frames_per_claim

        fig.suptitle("Faithfulness: decompose the answer, check each claim against the context",
                     fontsize=13, color=INK, y=0.965, fontweight="bold")

        # context box (top) -- what claims are checked against
        ax.text(0.02, 0.9, "retrieved context:", ha="left", fontsize=8.8, color=BLUE, fontweight="bold")
        ax.text(0.02, 0.855, _wrap(context_text, 96), ha="left", va="top", fontsize=7.6, color=INK,
                bbox=dict(boxstyle="round,pad=0.4", facecolor=BLUE, alpha=0.08, edgecolor=BLUE))

        # claim rows -- each appears with its support + verdict once "checked"
        top = 0.62
        row_h = 0.15
        supported_so_far = 0
        for i in range(n_claims):
            y = top - i * row_h
            revealed = i < checked
            claim = result.claims[i]
            support = result.supports[i]
            ok = result.supported[i]
            face = (GREEN if ok else RED) if revealed else SLATE
            alpha = 0.9 if revealed else 0.18
            ax.add_patch(plt.Rectangle((0.03, y - row_h * 0.42), 0.66, row_h * 0.8, facecolor=face,
                         alpha=alpha * 0.3, edgecolor=face if revealed else GRID, linewidth=1.2))
            ax.text(0.05, y, f"claim {i+1}: " + _wrap(claim, 58).split("\n")[0]
                    + ("…" if len(claim) > 58 else ""), ha="left", va="center", fontsize=8.0,
                    color=INK if revealed else SLATE)
            if revealed:
                verdict = "✓ SUPPORTED" if ok else "✗ HALLUCINATED"
                ax.text(0.71, y, f"cos {support:.3f}   {verdict}", ha="left", va="center", fontsize=8.6,
                        color=GREEN if ok else RED, fontweight="bold")
                if ok:
                    supported_so_far += 1

        # threshold reminder
        ax.text(0.71, top + 0.11, f"support ≥ {SUPPORT_THRESHOLD} = supported", ha="left", fontsize=7.6,
                color=AMBER, style="italic")

        # running faithfulness fraction assembling
        checked_supported = sum(result.supported[:checked])
        frac = f"{checked_supported}/{checked}" if checked else "0/0"
        score = (checked_supported / checked) if checked else 0.0
        ax.text(0.5, 0.10, f"faithfulness = supported / total = {frac} = {score:.3f}",
                ha="center", fontsize=13, color=GREEN if not in_hold else GREEN, fontweight="bold",
                bbox=dict(boxstyle="round,pad=0.5", facecolor=GREEN, alpha=0.12, edgecolor=GREEN))
        if in_hold:
            ax.text(0.5, 0.03, "the hallucinated 'solar panels' claim drags faithfulness below 1.0 — "
                    "a fluent answer, caught",
                    ha="center", fontsize=8.8, color=RED, style="italic", fontweight="bold")

        # honesty footer
        ax.text(0.5, 0.005, "claims, support cosines and the score are REAL rag_evaluation output; "
                "the claim-split + support JUDGE are illustrative LLM stand-ins",
                ha="center", va="bottom", fontsize=6.9, color=SLATE, style="italic")
        # per-frame progress tick keeps every frame distinct so the GIF writer preserves pacing
        prog = (frame + 1) / total
        ax.add_patch(plt.Rectangle((0.90, 0.008), 0.08 * prog, 0.007, facecolor=GREEN, edgecolor="none"))
        return []

    anim = FuncAnimation(fig, update, frames=total, interval=1000 / FPS, blit=False)
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    out_path = OUT_DIR / "rag11_faithfulness_check.gif"
    anim.save(out_path, writer=PillowWriter(fps=FPS), dpi=DPI)
    plt.close(fig)
    print(f"wrote {out_path}")


def main() -> None:
    build_animation()


if __name__ == "__main__":
    main()
