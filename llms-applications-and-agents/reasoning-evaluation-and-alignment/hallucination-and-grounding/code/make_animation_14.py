"""Animated (GIF) intuition figure for 14-Guardrails-and-Hallucination-Mitigation.

Companion to the static PNGs. It brings the guardrail STACK to life -- a request flowing through both
rails, stage by stage: three retrieved passages hit the INPUT rail (the injected and PII ones flash
red and are dropped; the clean one passes green); the surviving context reaches GENERATE; then the
answer hits the OUTPUT rail's grounding gate -- a grounded answer clears the bar and is emitted, an
ungrounded one falls below it and is refused with "I don't know". The reader watches the attack get
stripped and the hallucination get refused, one stage at a time.

    python make_animation_14.py

The GIF is written to ../../images/ and uses matplotlib's PillowWriter (no ffmpeg needed). Every
verdict (blocked/pass) and grounding score is the REAL guardrails.py output over the ch5 all-MiniLM
encoder; only the generator and a trained ML safety classifier are illustrative LLM stand-ins (stated
in the honesty footer).

Produced:
  rag14_guardrail_flow.gif -- a request flows through the input rail (attacks dropped) then the output
                              rail (ungrounded answer refused); the stack stops the attack AND the hallucination.

Verified on Python 3.12 / matplotlib 3.x / numpy 2.x / Pillow (PillowWriter) / sentence-transformers.
"""

from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")  # headless: render frames to a file, never open a window
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation, PillowWriter

from guardrails import (
    GROUNDING_THRESHOLD,
    HALLUCINATED_ANSWER,
    INJECTED_PASSAGE,
    PII_PASSAGE,
    DenseRetriever,
    answer_grounding,
    guarded_corpus,
    screen_passage,
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
    corpus = guarded_corpus()
    dense = DenseRetriever(corpus)
    clean_passage = "Helios-7 carries a hyperspectral imager with a ground resolution of 4 meters."
    passages = (clean_passage, INJECTED_PASSAGE, PII_PASSAGE)
    verdicts = [screen_passage(p) for p in passages]
    # the UNGROUNDED answer is graded against the surviving clean context -> low grounding -> abstain
    ungrounded_grounding = answer_grounding(dense, HALLUCINATED_ANSWER, (clean_passage,))
    abstained = ungrounded_grounding < GROUNDING_THRESHOLD

    # timeline: intro -> reveal input verdicts one by one -> generate -> output grounding -> hold
    intro = 8
    per_passage = 12
    gen = 14
    out = 16
    hold = 24
    total = intro + len(passages) * per_passage + gen + out + hold

    fig, ax = plt.subplots(figsize=(11.6, 6.8))
    fig.subplots_adjust(left=0.02, right=0.98, top=0.9, bottom=0.04)

    def update(frame: int):
        ax.clear()
        ax.axis("off")
        ax.set_xlim(0, 1)
        ax.set_ylim(0, 1)

        # progress through stages
        input_done = min(max((frame - intro) // per_passage, 0), len(passages))
        after_input = frame >= intro + len(passages) * per_passage
        generating = after_input and frame < intro + len(passages) * per_passage + gen
        after_gen = frame >= intro + len(passages) * per_passage + gen
        show_output = frame >= intro + len(passages) * per_passage + gen + out

        fig.suptitle("The guardrail stack: input rail strips the attack, output rail refuses the hallucination",
                     fontsize=12.5, color=INK, y=0.965, fontweight="bold")

        # ---- INPUT rail: three passages revealed one by one ----
        ax.text(0.02, 0.88, "INPUT rail — screen each retrieved passage", ha="left", fontsize=9.4,
                color=RED, fontweight="bold")
        ys = [0.78, 0.66, 0.54]
        for i, (y, passage, verdict) in enumerate(zip(ys, passages, verdicts)):
            revealed = i < input_done
            blocked = verdict.blocked
            if revealed:
                face = RED if blocked else GREEN
                alpha = 0.16
            else:
                face, alpha = SLATE, 0.05
            ax.add_patch(plt.Rectangle((0.03, y - 0.045), 0.62, 0.08, facecolor=face, alpha=alpha,
                         edgecolor=face if revealed else GRID, linewidth=1.2))
            ax.text(0.05, y, _short(passage, 62), ha="left", va="center", fontsize=7.4,
                    color=INK if revealed else SLATE)
            if revealed:
                if verdict.injection:
                    tag = "✗ BLOCKED · injection"
                elif verdict.pii_types:
                    tag = f"✗ BLOCKED · PII {list(verdict.pii_types)}"
                else:
                    tag = "✓ PASS · clean"
                ax.text(0.67, y, tag, ha="left", va="center", fontsize=8.0,
                        color=RED if blocked else GREEN, fontweight="bold")

        # ---- GENERATE (only the clean passage survived) ----
        if after_input:
            gen_col = AMBER if generating else PURPLE
            ax.add_patch(plt.Rectangle((0.25, 0.40), 0.5, 0.07, facecolor=gen_col, alpha=0.9, edgecolor=INK, linewidth=1.0))
            ax.text(0.5, 0.435, "GENERATE over 1 clean passage" + (" …" if generating else ""),
                    ha="center", va="center", fontsize=8.6, color="white", fontweight="bold")

        # ---- OUTPUT rail: grounding gate ----
        if after_gen:
            ax.text(0.02, 0.33, "OUTPUT rail — grounding gate", ha="left", fontsize=9.4, color=GREEN, fontweight="bold")
            # number line
            ax.plot([0.08, 0.92], [0.23, 0.23], color=SLATE, linewidth=1.6)
            tx = 0.08 + 0.84 * GROUNDING_THRESHOLD
            ax.plot([tx, tx], [0.20, 0.26], color=AMBER, linewidth=2.2)
            # τ label to the RIGHT of its tick so it never collides with the grounding-score label
            ax.text(tx + 0.02, 0.235, f"τ = {GROUNDING_THRESHOLD}", ha="left", va="center", fontsize=8.0,
                    color=AMBER, fontweight="bold")
            ax.text(0.14, 0.155, "ABSTAIN", ha="center", fontsize=7.8, color=RED, fontweight="bold")
            ax.text(0.82, 0.155, "EMIT", ha="center", fontsize=7.8, color=GREEN, fontweight="bold")
            if show_output:
                px = 0.08 + 0.84 * ungrounded_grounding
                ax.scatter([px], [0.23], s=170, color=RED, edgecolor=INK, linewidth=1.2, zorder=5)
                ax.text(px, 0.30, f"answer grounding {ungrounded_grounding:.3f}", ha="center", fontsize=7.8,
                        color=RED, fontweight="bold")

        # ---- final verdict banner ----
        if show_output:
            msg = "REFUSED: “I don't know based on the provided context.”" if abstained else "EMITTED (grounded)"
            ax.text(0.5, 0.075, msg, ha="center", fontsize=11.0, color=RED if abstained else GREEN,
                    fontweight="bold", bbox=dict(boxstyle="round,pad=0.4",
                    facecolor=RED if abstained else GREEN, alpha=0.12, edgecolor=RED if abstained else GREEN))

        # honesty footer
        ax.text(0.5, -0.005, "verdicts + grounding are REAL guardrails.py output over all-MiniLM; "
                "the generator and a trained ML safety classifier are illustrative LLM stand-ins",
                ha="center", va="bottom", fontsize=6.8, color=SLATE, style="italic")
        prog = (frame + 1) / total
        ax.add_patch(plt.Rectangle((0.90, 0.005), 0.08 * prog, 0.006, facecolor=GREEN, edgecolor="none"))
        return []

    anim = FuncAnimation(fig, update, frames=total, interval=1000 / FPS, blit=False)
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    out_path = OUT_DIR / "rag14_guardrail_flow.gif"
    anim.save(out_path, writer=PillowWriter(fps=FPS), dpi=DPI)
    plt.close(fig)
    print(f"wrote {out_path}")


def main() -> None:
    build_animation()


if __name__ == "__main__":
    main()
