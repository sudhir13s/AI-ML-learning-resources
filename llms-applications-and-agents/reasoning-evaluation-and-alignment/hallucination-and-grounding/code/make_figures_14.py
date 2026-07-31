"""Static figure generator for 14-Guardrails-and-Hallucination-Mitigation.

Imports the SAME canonical functions the page and notebook use (guardrails.py, which reuses ch13/ch5)
so every plotted number is the chapter's own -- no hand-typed values. Writes muted-palette PNGs to the
shared chapter image dir (../../images/) with the per-chapter prefix `rag14_`.

    python make_figures_14.py

Figures produced:
  rag14_guardrail_stack.png    -- the stack: query -> INPUT rail (injection/PII/topic) -> retrieve ->
                                  generate -> OUTPUT rail (grounding -> abstain / safety) -> answer or
                                  refusal. Schematic mechanism diagram.
  rag14_input_rail.png         -- the input rail flagging a retrieved injection + a PII passage while a
                                  clean passage passes (real screen_passage verdicts).
  rag14_abstain_gate.png       -- the abstention gate on the grounding number line: a grounded answer
                                  (0.848) passes, an ungrounded one (0.060) is refused, threshold marked.
  rag14_tradeoff.png           -- the false-refuse / false-allow tradeoff as the grounding threshold
                                  sweeps (the real risk-coverage curve from the pipeline).
  rag14_case_grounding.png     -- every abstention case's grounding score vs the threshold: which pass,
                                  which abstain, and the topically-near hallucination that fools the gate.

Verified on Python 3.12 / matplotlib 3.x / numpy 2.x / sentence-transformers (CPU). Headless (Agg).
"""

from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")  # headless: render straight to PNG, never open a window
import matplotlib.pyplot as plt
import numpy as np

from guardrails import (
    GROUNDED_ANSWER,
    GROUNDING_THRESHOLD,
    HALLUCINATED_ANSWER,
    INJECTED_PASSAGE,
    PII_PASSAGE,
    DenseRetriever,
    abstention_rates,
    answer_grounding,
    build_abstention_cases,
    guarded_corpus,
    screen_passage,
)

# ---- Palette (matches the chapter's muted Mermaid classDefs) -------------------------------
BLUE = "#3A6B96"  # data / retrieval / clean
PURPLE = "#5D4A8A"  # process / generation
GREEN = "#2E7A5A"  # pass / grounded / good
RED = "#8B3B4A"  # blocked / ungrounded / abstain / bad
SLATE = "#4A5B6E"  # neutral
AMBER = "#7A6528"  # threshold / highlight
INK = "#1C2530"  # labels
GRID = "#D4D9DF"  # gridlines

OUT_DIR = Path(__file__).resolve().parent.parent.parent / "images"
DPI = 110


def _style_axis(ax: plt.Axes) -> None:
    """Consistent muted styling: light grid, no top/right spines, ink-coloured labels."""
    ax.grid(True, color=GRID, linewidth=0.7, alpha=0.8)
    ax.set_axisbelow(True)
    for spine in ("top", "right"):
        ax.spines[spine].set_visible(False)
    for spine in ("left", "bottom"):
        ax.spines[spine].set_color(GRID)
    ax.tick_params(colors=INK, labelsize=9)
    ax.xaxis.label.set_color(INK)
    ax.yaxis.label.set_color(INK)
    ax.title.set_color(INK)


def _save(fig: plt.Figure, name: str) -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    path = OUT_DIR / name
    fig.savefig(path, dpi=DPI, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    print(f"wrote {path}")


def _box(ax, x, y, w, h, text, color, tcol="white", fs=8.6):
    """A filled rounded box with centred text -- the flow-diagram primitive."""
    ax.add_patch(plt.Rectangle((x, y), w, h, facecolor=color, alpha=0.92, edgecolor=INK, linewidth=1.0))
    ax.text(x + w / 2, y + h / 2, text, ha="center", va="center", color=tcol, fontsize=fs, fontweight="bold")


def _short(text: str, n: int) -> str:
    return text if len(text) <= n else text[: n - 1] + "…"


# ================================================================================================
# Figure 1 -- the guardrail stack
# ================================================================================================


def fig_guardrail_stack() -> None:
    """The stack: query -> INPUT rail -> retrieve -> generate -> OUTPUT rail -> answer or refusal.

    Schematic mechanism diagram (labelled). The point: guardrails wrap the RAG pipeline at BOTH ends
    -- input rails sanitize before retrieval/generation, output rails verify grounding and can ABSTAIN.
    Matches the page's Mermaid diagram exactly.
    """
    fig, ax = plt.subplots(figsize=(13.0, 5.6))
    ax.axis("off")
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.text(0.5, 0.95, "Guardrails wrap the RAG pipeline at both ends: input rails in, output rails out",
            ha="center", fontsize=12.5, fontweight="bold", color=INK)

    # main pipeline row
    _box(ax, 0.01, 0.58, 0.11, 0.16, "query", SLATE, fs=9.0)
    _box(ax, 0.15, 0.58, 0.16, 0.16, "INPUT rail\ninjection · PII\n· topic", RED, fs=7.8)
    _box(ax, 0.34, 0.58, 0.13, 0.16, "RETRIEVE", BLUE, fs=8.6)
    _box(ax, 0.50, 0.58, 0.13, 0.16, "GENERATE", PURPLE, fs=8.6)
    _box(ax, 0.66, 0.58, 0.17, 0.16, "OUTPUT rail\ngrounding →\nabstain · safety", GREEN, fs=7.6)
    _box(ax, 0.86, 0.58, 0.12, 0.16, "answer", GREEN, fs=9.0)
    for x0, x1 in ((0.12, 0.15), (0.31, 0.34), (0.47, 0.50), (0.63, 0.66), (0.83, 0.86)):
        ax.annotate("", xy=(x1, 0.66), xytext=(x0, 0.66), arrowprops=dict(arrowstyle="->", color=INK, lw=1.6))

    # branch down from each rail to what it does
    ax.annotate("", xy=(0.23, 0.34), xytext=(0.23, 0.57), arrowprops=dict(arrowstyle="->", color=RED, lw=1.5))
    _box(ax, 0.10, 0.20, 0.26, 0.14, "BLOCK injected / PII passage\nbefore it reaches the prompt", RED, fs=7.8)
    ax.annotate("", xy=(0.745, 0.34), xytext=(0.745, 0.57), arrowprops=dict(arrowstyle="->", color=AMBER, lw=1.5))
    _box(ax, 0.62, 0.20, 0.24, 0.14, "grounding < τ ?\nABSTAIN 'I don't know'", AMBER, fs=7.8)

    # the refusal exit
    ax.annotate("", xy=(0.74, 0.10), xytext=(0.74, 0.19), arrowprops=dict(arrowstyle="->", color=RED, lw=1.5))
    _box(ax, 0.66, 0.02, 0.16, 0.08, "refusal", RED, fs=8.4)

    ax.text(0.5, 0.005, "input rails stop the ATTACK; output rails stop the HALLUCINATION — "
            "'I don't know' beats a confident wrong answer",
            ha="center", fontsize=9.0, color=INK, style="italic")
    _save(fig, "rag14_guardrail_stack.png")


# ================================================================================================
# Figure 2 -- the input rail flagging injection + PII
# ================================================================================================


def fig_input_rail() -> None:
    """The input rail: a clean passage passes; an injected one and a PII one are BLOCKED.

    Runs the REAL screen_passage verdicts. Three retrieved passages as rows: green = pass, red =
    blocked, with the reason (injection phrase / PII types) shown. The attacker's directive never
    reaches the prompt.
    """
    passages = (
        "Helios-7 carries a hyperspectral imager with a ground resolution of 4 meters.",
        INJECTED_PASSAGE,
        PII_PASSAGE,
    )
    verdicts = [screen_passage(p) for p in passages]

    fig, ax = plt.subplots(figsize=(13.0, 4.8))
    ax.axis("off")
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.text(0.5, 0.93, "Input rail: screen every retrieved passage before it reaches the LLM",
            ha="center", fontsize=12.5, fontweight="bold", color=INK)

    ys = [0.68, 0.44, 0.20]
    for y, passage, verdict in zip(ys, passages, verdicts):
        blocked = verdict.blocked
        face = RED if blocked else GREEN
        ax.add_patch(plt.Rectangle((0.03, y - 0.075), 0.72, 0.13, facecolor=face, alpha=0.13,
                     edgecolor=face, linewidth=1.5))
        ax.text(0.05, y, _short(passage, 78), ha="left", va="center", fontsize=8.2, color=INK)
        if verdict.injection:
            tag, reason = "BLOCKED", f"injection: “{verdict.injection_evidence}”"
        elif verdict.pii_types:
            tag, reason = "BLOCKED", f"PII: {', '.join(verdict.pii_types)}"
        else:
            tag, reason = "PASS", "clean → reaches the generator"
        ax.text(0.77, y + 0.018, tag, ha="left", va="center", fontsize=10.0, color=face, fontweight="bold")
        ax.text(0.77, y - 0.028, reason, ha="left", va="center", fontsize=7.8, color=face, style="italic")

    ax.text(0.5, 0.03, "the injected 'ignore previous instructions' and the PII line are dropped; "
            "only the clean passage survives",
            ha="center", fontsize=8.8, color=INK, style="italic")
    _save(fig, "rag14_input_rail.png")


# ================================================================================================
# Figure 3 -- the abstention gate on the grounding number line
# ================================================================================================


def fig_abstain_gate(dense: DenseRetriever, corpus: tuple[str, ...]) -> None:
    """The abstention gate: a grounded answer passes, an ungrounded one is refused, on the number line.

    Runs the REAL grounding scores. A horizontal grounding axis [0,1] with the threshold τ marked; the
    grounded answer's score sits to the right (emit, green), the hallucination's to the left (abstain,
    red). The gate is just 'is the answer's support above τ?'.
    """
    imager_ctx = tuple(p for p in corpus if "ground resolution" in p)
    offtopic_ctx = tuple(p for p in corpus if "chessboard" in p or "Eiffel" in p)[:1]
    g_grounded = answer_grounding(dense, GROUNDED_ANSWER, imager_ctx)
    g_hallucinated = answer_grounding(dense, HALLUCINATED_ANSWER, offtopic_ctx)

    fig, ax = plt.subplots(figsize=(12.0, 4.4))
    _style_axis(ax)
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.set_yticks([])
    ax.set_xlabel("answer grounding = max support cosine to the retrieved context", fontsize=10.0)
    ax.set_title("The abstention gate: emit if grounding ≥ τ, else refuse 'I don't know'",
                 fontsize=12.0, color=INK, fontweight="bold", pad=12)

    # shaded regions
    ax.axvspan(0, GROUNDING_THRESHOLD, color=RED, alpha=0.07)
    ax.axvspan(GROUNDING_THRESHOLD, 1, color=GREEN, alpha=0.07)
    ax.axvline(GROUNDING_THRESHOLD, color=AMBER, linewidth=2.2, linestyle="--")
    ax.text(GROUNDING_THRESHOLD, 0.93, f"τ = {GROUNDING_THRESHOLD}", ha="center", fontsize=9.5,
            color=AMBER, fontweight="bold", bbox=dict(boxstyle="round,pad=0.25", facecolor="white", edgecolor=AMBER))
    ax.text(GROUNDING_THRESHOLD / 2, 0.12, "ABSTAIN\n'I don't know'", ha="center", fontsize=9.0, color=RED, fontweight="bold")
    ax.text((1 + GROUNDING_THRESHOLD) / 2, 0.12, "EMIT the answer", ha="center", fontsize=9.0, color=GREEN, fontweight="bold")

    for score, label, col in ((g_grounded, "grounded answer", GREEN), (g_hallucinated, "hallucinated answer", RED)):
        ax.scatter([score], [0.55], s=260, color=col, edgecolor=INK, linewidth=1.3, zorder=5)
        ax.annotate(f"{label}\n{score:.3f}", (score, 0.55), xytext=(0, 34 if col == GREEN else 34),
                    textcoords="offset points", ha="center", fontsize=9.0, color=col, fontweight="bold")
    _save(fig, "rag14_abstain_gate.png")


# ================================================================================================
# Figure 4 -- the false-refuse / false-allow tradeoff
# ================================================================================================


def fig_tradeoff(dense: DenseRetriever, corpus: tuple[str, ...]) -> None:
    """The false-refuse / false-allow tradeoff as the grounding threshold sweeps (real risk-coverage).

    Runs the REAL abstention_rates over a fine threshold sweep. false-refuse (coverage lost) rises with
    τ; false-allow (hallucinations let through) falls. They cross -- you cannot minimize both. The
    default τ=0.5 is marked.
    """
    cases = build_abstention_cases(dense, corpus)
    taus = np.round(np.arange(0.30, 0.76, 0.05), 2)
    frs, fas = [], []
    for tau in taus:
        fr, fa = abstention_rates(dense, cases, float(tau))
        frs.append(fr)
        fas.append(fa)

    fig, ax = plt.subplots(figsize=(10.4, 6.0))
    _style_axis(ax)
    ax.plot(taus, frs, "-o", color=RED, linewidth=2.2, markersize=6, label="false-refuse (over-refusal; coverage lost)")
    ax.plot(taus, fas, "-s", color=BLUE, linewidth=2.2, markersize=6, label="false-allow (hallucination let through)")
    ax.axvline(GROUNDING_THRESHOLD, color=AMBER, linewidth=1.8, linestyle="--")
    ax.text(GROUNDING_THRESHOLD + 0.005, 0.9, f"default τ = {GROUNDING_THRESHOLD}", color=AMBER,
            fontsize=8.6, fontweight="bold", rotation=90, va="top")
    ax.set_xlabel("grounding threshold τ", fontsize=10.5)
    ax.set_ylabel("rate over the labelled cases", fontsize=10.5)
    ax.set_ylim(-0.05, 1.12)
    ax.set_title("Raising τ trades coverage for safety: false-refuse ↑, false-allow ↓ (measured)",
                 fontsize=11.5, color=INK, fontweight="bold", pad=10)
    ax.legend(fontsize=8.6, loc="upper center", bbox_to_anchor=(0.5, 1.0), framealpha=0.95)
    # annotation in the empty lower-left band (both curves are away from there mid-sweep)
    ax.annotate("medical / legal:\nfavour high τ\n(abstain rather than err)", xy=(0.72, 0.02),
                xytext=(0.40, 0.18), fontsize=8.0, color=INK, ha="center",
                arrowprops=dict(arrowstyle="->", color=SLATE, lw=1.2),
                bbox=dict(boxstyle="round,pad=0.3", facecolor="white", edgecolor=SLATE, alpha=0.95))
    _save(fig, "rag14_tradeoff.png")


# ================================================================================================
# Figure 5 -- every case's grounding vs the threshold (which pass, which abstain)
# ================================================================================================


def fig_case_grounding(dense: DenseRetriever, corpus: tuple[str, ...]) -> None:
    """Each abstention case's grounding score, sorted, vs the threshold -- the gate's decisions laid bare.

    Runs the REAL grounding for every case. Bars coloured by GOLD (green = should-answer, red =
    should-abstain); the threshold line shows the gate's cut. The topically-near hallucination that
    sits ABOVE the line (a false-allow at τ=0.5) is the cosine != entailment gap, made visible.
    """
    cases = build_abstention_cases(dense, corpus)
    scored = sorted(
        ((answer_grounding(dense, c.answer, c.passages), c.should_answer, c.label) for c in cases),
        key=lambda t: t[0],
    )
    scores = [s for s, _, _ in scored]
    golds = [g for _, g, _ in scored]
    labels = [lab for _, _, lab in scored]

    fig, ax = plt.subplots(figsize=(11.6, 6.2))
    _style_axis(ax)
    y = np.arange(len(scores))
    colors = [GREEN if g else RED for g in golds]
    bars = ax.barh(y, scores, color=colors, edgecolor=INK, linewidth=0.9, height=0.62, alpha=0.85)
    for bar, s in zip(bars, scores):
        ax.text(bar.get_width() + 0.012, bar.get_y() + bar.get_height() / 2, f"{s:.3f}",
                va="center", fontsize=8.6, color=INK, fontweight="bold")
    ax.axvline(GROUNDING_THRESHOLD, color=AMBER, linewidth=2.2, linestyle="--")
    ax.text(GROUNDING_THRESHOLD, len(scores) - 0.3, f"τ = {GROUNDING_THRESHOLD}", ha="center", fontsize=9.0,
            color=AMBER, fontweight="bold", bbox=dict(boxstyle="round,pad=0.2", facecolor="white", edgecolor=AMBER))
    ax.set_yticks(y)
    ax.set_yticklabels([_short(lab, 40) for lab in labels], fontsize=8.0)
    ax.set_xlim(0, 1.12)
    ax.set_xlabel("answer grounding (max support cosine)", fontsize=10.5, labelpad=8)
    ax.set_title("Every case's grounding vs τ: green should ANSWER, red should ABSTAIN",
                 fontsize=11.5, color=INK, fontweight="bold", pad=10)
    # legend proxies (Patch handles so the swatches match the bar colours exactly)
    from matplotlib.patches import Patch  # local import: only this figure needs legend proxies

    handles = [
        Patch(facecolor=GREEN, edgecolor=INK, label="should answer (grounded)"),
        Patch(facecolor=RED, edgecolor=INK, label="should abstain (ungrounded)"),
    ]
    ax.legend(handles=handles, fontsize=8.6, loc="lower right", framealpha=0.95)
    ax.text(0.5, -0.16, "a red bar RIGHT of τ = a false-allow (topically-near hallucination the cosine "
            "gate can't catch → why entailment/NLI is the next step)",
            transform=ax.transAxes, ha="center", fontsize=8.0, color=INK, style="italic")
    _save(fig, "rag14_case_grounding.png")


def main() -> None:
    corpus = guarded_corpus()
    dense = DenseRetriever(corpus)
    print(f"corpus: {len(corpus)} passages | dense lens: {dense.backend}")
    fig_guardrail_stack()
    fig_input_rail()
    fig_abstain_gate(dense, corpus)
    fig_tradeoff(dense, corpus)
    fig_case_grounding(dense, corpus)


if __name__ == "__main__":
    main()
