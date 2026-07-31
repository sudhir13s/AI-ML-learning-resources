"""Animated (GIF) intuition figure for 13-Citations-and-Attribution.

Companion to the static PNGs. It brings post-hoc attribution to life -- the DECOMPOSE -> attribute
-> assign/flag mechanism: the answer is split into claims, and one at a time each claim lights up and
draws a line to the retrieved passage that best supports it (green, with the real cosine on the line)
-- until the fabricated claim, which matches NO passage, flashes RED and is flagged UNCITABLE. The
reader watches the two grounded claims find their source and the hallucination fail to.

    python make_animation_13.py

The GIF is written to ../../images/ and uses matplotlib's PillowWriter (no ffmpeg needed). Every
claim, its citation, and its support cosine are the REAL citations_attribution.py output over the
ch5 all-MiniLM encoder; only the claim-split and the generator are illustrative stand-ins for an LLM
(stated in the honesty footer).

Produced:
  rag13_attribution_reveal.gif -- claims attach to their source one-by-one; the uncitable one flashes red.

Verified on Python 3.12 / matplotlib 3.x / numpy 2.x / Pillow (PillowWriter) / sentence-transformers.
"""

from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")  # headless: render frames to a file, never open a window
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation, PillowWriter

from citations_attribution import (
    ANSWER,
    SUPPORT_THRESHOLD,
    DenseRetriever,
    attribute_claims,
    full_corpus,
    retrieve_passages,
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


def _short(text: str, n: int) -> str:
    return text if len(text) <= n else text[: n - 1] + "…"


def build_animation() -> None:
    corpus = full_corpus()
    dense = DenseRetriever(corpus)
    passages, _ = retrieve_passages(dense, corpus, k=3)
    attributions = attribute_claims(dense, ANSWER, passages)
    n_claims = len(attributions)
    n_pass = len(passages)

    # timeline: intro -> reveal each claim's attribution in turn -> hold on the final cited answer
    frames_per_claim = 14
    intro = 8
    hold = 24
    total = intro + n_claims * frames_per_claim + hold

    claim_x, claim_w = 0.02, 0.34
    pass_x, pass_w = 0.60, 0.39
    claim_ys = [0.70 - i * 0.24 for i in range(n_claims)]
    pass_ys = [0.72 - i * 0.24 for i in range(n_pass)]

    fig, ax = plt.subplots(figsize=(11.6, 6.6))
    fig.subplots_adjust(left=0.02, right=0.98, top=0.9, bottom=0.04)

    def update(frame: int):
        ax.clear()
        ax.axis("off")
        ax.set_xlim(0, 1)
        ax.set_ylim(0, 1)

        # how many claims have been revealed so far
        if frame < intro:
            revealed = 0
        elif frame >= intro + n_claims * frames_per_claim:
            revealed = n_claims
        else:
            revealed = (frame - intro) // frames_per_claim + 1
        in_hold = frame >= intro + n_claims * frames_per_claim

        fig.suptitle("Post-hoc attribution: attach each claim to its source, one at a time",
                     fontsize=13, color=INK, y=0.965, fontweight="bold")
        ax.text(0.18, 0.86, "ANSWER claims", ha="center", fontsize=9.6, color=INK, fontweight="bold")
        ax.text(0.78, 0.86, "RETRIEVED passages", ha="center", fontsize=9.6, color=BLUE, fontweight="bold")

        # passages on the right (always visible)
        pass_centers = {}
        for i, (y, passage) in enumerate(zip(pass_ys, passages), start=1):
            ax.add_patch(plt.Rectangle((pass_x, y - 0.055), pass_w, 0.09, facecolor=BLUE, alpha=0.10,
                         edgecolor=BLUE, linewidth=1.1))
            ax.text(pass_x + 0.012, y - 0.01, f"[{i}] " + _short(passage, 52), ha="left", va="center",
                    fontsize=7.6, color=INK)
            pass_centers[i] = (pass_x, y - 0.01)

        # claims on the left, revealed one-by-one with their attribution line
        for i, (attr, y) in enumerate(zip(attributions, claim_ys)):
            is_revealed = i < revealed
            cited = attr.citation is not None
            if is_revealed:
                face = GREEN if cited else RED
                alpha = 0.16
            else:
                face = SLATE
                alpha = 0.06
            ax.add_patch(plt.Rectangle((claim_x, y - 0.055), claim_w, 0.09, facecolor=face, alpha=alpha,
                         edgecolor=face if is_revealed else GRID, linewidth=1.3))
            ax.text(claim_x + 0.012, y - 0.01, _short(attr.claim, 44), ha="left", va="center",
                    fontsize=7.6, color=INK if is_revealed else SLATE)
            if is_revealed:
                marker = f"[{attr.citation}]" if cited else "[?]"
                ax.text(claim_x + claim_w - 0.012, y - 0.01, marker, ha="right", va="center",
                        fontsize=8.8, color=face, fontweight="bold")
                if cited:
                    px, py = pass_centers[attr.citation]
                    ax.annotate("", xy=(px, py), xytext=(claim_x + claim_w, y - 0.01),
                                arrowprops=dict(arrowstyle="->", color=GREEN, lw=1.8, alpha=0.85))
                    ax.text(claim_x + claim_w + 0.03, y - 0.01, f"cos {attr.best_score:.2f}", ha="left",
                            va="center", fontsize=7.6, color=GREEN, fontweight="bold",
                            bbox=dict(boxstyle="round,pad=0.16", facecolor="white", edgecolor=GREEN,
                                      alpha=0.92, linewidth=0.8))
                else:
                    # the uncitable claim: no line, a red "no source" flag
                    ax.text(claim_x + claim_w + 0.03, y - 0.01,
                            f"UNCITABLE — best cos {attr.best_score:.2f} < {SUPPORT_THRESHOLD}", ha="left",
                            va="center", fontsize=7.6, color=RED, fontweight="bold", style="italic")

        # running tally
        cited_so_far = sum(1 for a in attributions[:revealed] if a.citation is not None)
        flagged = sum(1 for a in attributions[:revealed] if a.citation is None)
        ax.text(0.5, 0.075, f"cited to a source: {cited_so_far}    flagged uncitable: {flagged}",
                ha="center", fontsize=11.5, color=INK, fontweight="bold",
                bbox=dict(boxstyle="round,pad=0.4", facecolor=GREEN, alpha=0.10, edgecolor=GREEN))
        if in_hold:
            ax.text(0.5, 0.02, "the fabricated 'solar panels' claim matches NO passage → flagged [?]. "
                    "a fluent hallucination, caught.",
                    ha="center", fontsize=8.6, color=RED, style="italic", fontweight="bold")

        # honesty footer
        ax.text(0.5, -0.005, "claims, citations and cosines are REAL citations_attribution output; "
                "the claim-split + generator are illustrative LLM stand-ins",
                ha="center", va="bottom", fontsize=6.8, color=SLATE, style="italic")
        # per-frame progress tick keeps every frame distinct so the GIF writer preserves pacing
        prog = (frame + 1) / total
        ax.add_patch(plt.Rectangle((0.90, 0.005), 0.08 * prog, 0.006, facecolor=GREEN, edgecolor="none"))
        return []

    anim = FuncAnimation(fig, update, frames=total, interval=1000 / FPS, blit=False)
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    out_path = OUT_DIR / "rag13_attribution_reveal.gif"
    anim.save(out_path, writer=PillowWriter(fps=FPS), dpi=DPI)
    plt.close(fig)
    print(f"wrote {out_path}")


def main() -> None:
    build_animation()


if __name__ == "__main__":
    main()
