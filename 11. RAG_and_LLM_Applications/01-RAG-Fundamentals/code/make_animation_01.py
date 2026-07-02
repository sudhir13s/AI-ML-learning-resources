"""Animated (GIF) intuition figure for 01-RAG-Fundamentals -- driven by the REAL pipeline.

Companion to the static PNGs. Where those show final states, this brings the *mechanism* to life:
the real query vector sweeps the real embedding space, lights up its actual top-3 retrieved
Wikipedia passages, those passages flow into the augmented prompt, and the real grounded answer
(from the HF Inference API) appears -- the retrieve -> augment -> generate loop, stepping.

    python make_animation_01.py

The geometry and the answer are the chapter's OWN: they come from `_anim_cache.json`, written by
`make_figures_01.py` from the real FAISS index + a real LLM call. Run `make_figures_01.py` first
(or this script will build the cache itself, which loads the models and calls the network).

Produced:
  rag01_retrieve_augment_generate.gif -- the query finds its real nearest passages, they flow into
                                          the augmented prompt, and the real grounded answer appears.

Verified on Python 3.12 / matplotlib 3.x / numpy 2.x / Pillow (PillowWriter).
"""

from __future__ import annotations

import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")  # headless: render frames to a file, never open a window
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.animation import FuncAnimation, PillowWriter

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
ANIM_CACHE = Path(__file__).resolve().parent / "_anim_cache.json"
DPI = 95
FPS = 12
HOLD_FRAMES = 16  # frames to dwell on the final grounded answer before looping


def _load_cache() -> dict:
    """Load the real geometry+answer cache, building it via make_figures_01 if it's missing."""
    if not ANIM_CACHE.exists():
        import make_figures_01 as figs

        from rag_fundamentals import RagPipeline, load_corpus

        passages, _ = load_corpus()
        figs.write_anim_cache(RagPipeline(passages))
    return json.loads(ANIM_CACHE.read_text())


def _wrap(text: str, width: int = 40) -> str:
    words, lines, line = text.split(), [], ""
    for word in words:
        if len(line) + len(word) + 1 > width:
            lines.append(line)
            line = word
        else:
            line = f"{line} {word}".strip()
    lines.append(line)
    return "\n".join(lines)


def build_animation() -> None:
    cache = _load_cache()
    doc_xy = np.array(cache["doc_xy"])
    q_xy = np.array(cache["q_xy"])
    n_top = int(cache["n_top"])
    query = cache["query"]
    top = cache["top"]  # list of {id, text, cos}
    # Keep the displayed answer short and real (trim to the first sentence for the panel).
    answer = cache["answer"].split("\n")[0].strip()

    fig, (ax_space, ax_flow) = plt.subplots(1, 2, figsize=(11.4, 5.4))
    fig.suptitle("RAG on real Wikipedia: retrieve the right passages, then answer grounded in them",
                 fontsize=12.5, color=INK, y=0.99)

    # ---- left panel: the real embedding space the query sweeps --------------------------
    ax_space.set_xlim(doc_xy[:, 0].min() - 0.12, doc_xy[:, 0].max() + 0.12)
    ax_space.set_ylim(doc_xy[:, 1].min() - 0.12, doc_xy[:, 1].max() + 0.12)
    ax_space.set_title("1. retrieve -- real nearest passages", fontsize=11, color=INK)
    ax_space.set_xticks([])
    ax_space.set_yticks([])
    for spine in ax_space.spines.values():
        spine.set_color(GRID)

    # ---- right panel: the prompt the retrieved passages flow into -----------------------
    ax_flow.set_xlim(0, 1)
    ax_flow.set_ylim(0, 1)
    ax_flow.axis("off")
    ax_flow.set_title("2. augment + generate -- real grounded answer", fontsize=11, color=INK)

    # static scatter of all passages (dim until retrieved); top-k are the first n_top rows.
    doc_artists = []
    for i, (x, y) in enumerate(doc_xy):
        is_top = i < n_top
        artist = ax_space.scatter(x, y, s=150 if is_top else 90,
                                  color=SLATE, alpha=0.5, edgecolors=INK, linewidths=0.9, zorder=3)
        if is_top:
            ax_space.annotate(f"doc[{top[i]['id']}]", (x, y), fontsize=7.5, color=INK, ha="left",
                              va="top", xytext=(7, -7), textcoords="offset points")
        doc_artists.append(artist)
    query_artist = ax_space.scatter(*q_xy, s=380, marker="*", color=AMBER, edgecolors=INK,
                                    linewidths=1.2, zorder=5)
    ax_space.annotate("query", q_xy, fontsize=9.5, color=INK, ha="center", va="bottom",
                      xytext=(0, 12), textcoords="offset points", fontweight="bold")
    beam_lines: list = []

    passage_texts = [t["text"] for t in top]

    n_retrieve_frames = n_top
    total = 1 + n_retrieve_frames + 2 + HOLD_FRAMES
    flow_texts: list = []

    def update(frame: int):
        retrieve_start = 1
        augment_frame = retrieve_start + n_retrieve_frames
        generate_frame = augment_frame + 1

        if frame == 0:
            return [query_artist]

        if retrieve_start <= frame < augment_frame:
            lit_count = frame - retrieve_start + 1
            for rank in range(lit_count):
                doc_artists[rank].set_color(GREEN)
                doc_artists[rank].set_alpha(0.95)
                doc_artists[rank].set_sizes([260])
                if rank >= len(beam_lines):
                    x, y = doc_xy[rank]
                    (line,) = ax_space.plot([q_xy[0], x], [q_xy[1], y], color=GREEN,
                                            linewidth=1.6, alpha=0.55, zorder=2)
                    beam_lines.append(line)
            return doc_artists + beam_lines

        if frame == augment_frame:
            ax_flow.text(0.5, 0.95, "augmented prompt", ha="center", va="top", fontsize=10.5,
                         fontweight="bold", color=PURPLE, transform=ax_flow.transAxes)
            y = 0.85
            for rank, passage in enumerate(passage_texts, start=1):
                snippet = passage[:150] + ("..." if len(passage) > 150 else "")
                txt = ax_flow.text(0.03, y, f"[{rank}] {_wrap(snippet)}", ha="left", va="top",
                                   fontsize=7.6, color=INK, family="monospace",
                                   transform=ax_flow.transAxes,
                                   bbox=dict(boxstyle="round,pad=0.3", facecolor=GREEN, alpha=0.14,
                                             edgecolor=GREEN))
                flow_texts.append(txt)
                y -= 0.185
            q_txt = ax_flow.text(0.03, y, f"Q: {_wrap(query)}", ha="left", va="top",
                                 fontsize=8, color=INK, family="monospace",
                                 transform=ax_flow.transAxes,
                                 bbox=dict(boxstyle="round,pad=0.3", facecolor=BLUE, alpha=0.14,
                                           edgecolor=BLUE))
            flow_texts.append(q_txt)
            return flow_texts

        if frame >= generate_frame:
            ans = ax_flow.text(0.5, 0.06, f"grounded answer:\n{_wrap(answer, 46)}",
                               ha="center", va="bottom", fontsize=9, color=INK,
                               fontweight="bold", transform=ax_flow.transAxes,
                               bbox=dict(boxstyle="round,pad=0.4", facecolor=GREEN, alpha=0.22,
                                         edgecolor=GREEN))
            return [ans]

        return [query_artist]

    anim = FuncAnimation(fig, update, frames=total, interval=1000 / FPS, blit=False)
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    out_path = OUT_DIR / "rag01_retrieve_augment_generate.gif"
    anim.save(out_path, writer=PillowWriter(fps=FPS), dpi=DPI)
    plt.close(fig)
    print(f"wrote {out_path}")


def main() -> None:
    build_animation()


if __name__ == "__main__":
    main()
