"""Animated (GIF) intuition figure for 08-Advanced-RAG-Parent-Doc-Fusion-Self-RAG.

Companion to the static PNGs. It brings the core parent-document idea to life -- "retrieve small,
read large": a query arrives, the SINGLE best child sentence lights up (sharp retrieval on a small
unit), and then that child EXPANDS to reveal its full parent section -- the surrounding sentences of
context materializing around it -- which is what actually gets fed to the LLM. You watch the
context-starved fragment grow into the self-contained section that answers the query.

    python make_animation_08.py

The GIF is written to ../../images/ and uses matplotlib's PillowWriter (no ffmpeg needed). The
retrieved child, its parent section, and the cosine score are the chapter's OWN -- computed by the
real all-MiniLM child index in advanced_rag.py over the ch2 Helios-7 document.

Produced:
  rag08_retrieve_small_read_large.gif -- the #1 child sentence lights up, then expands to its full
                                         parent section (retrieve small, read large).

Verified on Python 3.12 / matplotlib 3.x / numpy 2.x / Pillow (PillowWriter) / sentence-transformers.
"""

from __future__ import annotations

import textwrap
from pathlib import Path

import matplotlib

matplotlib.use("Agg")  # headless: render frames to a file, never open a window
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.animation import FuncAnimation, PillowWriter

from advanced_rag import ParentDocumentRetriever, build_hierarchy

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


def build_animation() -> None:
    parents, children = build_hierarchy()
    retriever = ParentDocumentRetriever(parents, children)
    query = "What keeps the local solar time of each pass roughly constant?"
    result = retriever.retrieve(query, k=3)
    child = children[result.child_indices[0]]
    parent = parents[child.parent_id]
    cos = result.child_scores[0]

    # parent section sentences (the context that materializes around the child)
    parent_sentences = [s.strip() for s in parent.text.split("\n") if s.strip() and not s.startswith("#")]
    child_text = child.text

    # phase timeline: query in -> child lights up -> child expands to parent -> hold
    n_query = 8
    n_light = 10
    n_expand = 18
    n_hold = 16
    total = n_query + n_light + n_expand + n_hold

    fig, ax = plt.subplots(figsize=(9.2, 6.2))
    fig.subplots_adjust(left=0.04, right=0.96, top=0.86, bottom=0.06)

    def wrap(text, width=58):
        return "\n".join(textwrap.wrap(text, width))

    def update(frame: int):
        ax.clear()
        ax.axis("off")
        ax.set_xlim(0, 1)
        ax.set_ylim(0, 1)

        # phase and an eased "expansion" fraction e in [0,1]
        if frame < n_query:
            phase, e = "query", 0.0
        elif frame < n_query + n_light:
            phase, e = "light", 0.0
        elif frame < n_query + n_light + n_expand:
            t = (frame - n_query - n_light) / max(n_expand - 1, 1)
            phase, e = "expand", 0.5 - 0.5 * np.cos(np.pi * t)  # smooth ease 0->1
        else:
            phase, e = "hold", 1.0

        # header
        titles = {
            "query": ("A query arrives — search the SMALL child sentences", SLATE),
            "light": ("Sharp retrieval: the #1 child sentence lights up", AMBER),
            "expand": ("Retrieve small, READ LARGE: the child expands to its parent section", GREEN),
            "hold": ("The full parent section — self-contained context — is fed to the LLM", GREEN),
        }
        title, tcol = titles[phase]
        fig.suptitle(title, fontsize=13, color=tcol, y=0.955, fontweight="bold")

        # query box
        ax.text(0.5, 0.9, f"query: “{query}”", ha="center", va="center", fontsize=9.5, color=INK,
                style="italic",
                bbox=dict(boxstyle="round,pad=0.4", facecolor=SLATE, alpha=0.14, edgecolor=SLATE))

        # the child box grows from a single-sentence height to the full parent-section height.
        # child sits at a fixed centre; the parent box expands symmetrically around it.
        cx, cy = 0.5, 0.44
        child_h = 0.12
        parent_h = child_h + e * 0.44  # grows with the expansion fraction
        box_w = 0.86

        # parent (context) box — appears/expands behind the child (only once expansion begins)
        if e > 0.01:
            ax.add_patch(plt.Rectangle((cx - box_w / 2, cy - parent_h / 2), box_w, parent_h,
                         facecolor=GREEN, alpha=0.10 + 0.12 * e, edgecolor=GREEN, linewidth=1.4))
            ax.text(cx - box_w / 2 + 0.02, cy + parent_h / 2 - 0.03, f"PARENT  # {parent.heading}",
                    ha="left", va="top", fontsize=9.5, color=GREEN, fontweight="bold", alpha=e)
            # the surrounding context sentences fade in above/below the child
            others = [s for s in parent_sentences if child_text not in s]
            for j, sent in enumerate(others):
                y = cy + parent_h / 2 - 0.11 - j * 0.10
                if y > cy + child_h / 2 + 0.01 or True:
                    ax.text(cx, max(y, cy - parent_h / 2 + 0.06), wrap(sent, 60), ha="center", va="center",
                            fontsize=8.2, color=INK, alpha=min(1.0, e * 1.2))

        # the child box (always visible, highlighted once it "lights up")
        lit = phase in ("light", "expand", "hold")
        child_face = AMBER if lit else BLUE
        ax.add_patch(plt.Rectangle((cx - box_w / 2 + 0.02, cy - child_h / 2), box_w - 0.04, child_h,
                     facecolor=child_face, alpha=0.9 if lit else 0.5, edgecolor=INK, linewidth=1.1))
        ax.text(cx, cy, wrap(f"child[{result.child_indices[0]}]: " + child_text, 56),
                ha="center", va="center", fontsize=8.4, color="white", fontweight="bold")
        if lit:
            ax.text(cx, cy - child_h / 2 - 0.02, f"retrieved  (cos {cos:.3f})", ha="center", va="top",
                    fontsize=8.2, color=AMBER, fontweight="bold")

        # the fragment-vs-context caption
        if phase == "light":
            ax.text(0.5, 0.08, "…but alone it opens with “This orbit” — which orbit? a context-starved FRAGMENT",
                    ha="center", fontsize=8.6, color=RED, style="italic")
        elif phase in ("expand", "hold"):
            ax.text(0.5, 0.05, "the parent supplies the referent (“sun-synchronous orbit”) the fragment lacked",
                    ha="center", fontsize=8.6, color=GREEN, style="italic", alpha=e)
        # honesty footer
        ax.text(0.5, 0.005, "child index, parent section, and cosine are REAL all-MiniLM outputs over the ch2 document",
                ha="center", fontsize=7.2, color=SLATE, style="italic")
        return []

    anim = FuncAnimation(fig, update, frames=total, interval=1000 / FPS, blit=False)
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    out_path = OUT_DIR / "rag08_retrieve_small_read_large.gif"
    anim.save(out_path, writer=PillowWriter(fps=FPS), dpi=DPI)
    plt.close(fig)
    print(f"wrote {out_path}")


def main() -> None:
    build_animation()


if __name__ == "__main__":
    main()
