"""Reproducible figure generator for 09-Sequence-Labeling-POS-and-NER.

Produces every embedded PNG for the chapter from the SAME backend used on the page and in the
notebook -- the HMM tables, the Viterbi delta trellis, the greedy-vs-Viterbi path probabilities,
the CRF transition matrix, the illegal-transition decode, the span-consistency decode, and the
token-accuracy-vs-entity-F1 gap are all IMPORTED from `sequence_labeling.py`, so the figures cannot
silently drift from the prose or the demo. Run:

    python make_figures_09.py

Each figure is written to ../../images/ (the shared chapter image dir, "06. NLP/images/") at 150 dpi,
prefixed `sl_`. The palette matches the chapter's Mermaid diagrams (muted, white text on fills).

Figures produced (measured = from the live backend; illustrative = a labelled schematic):
  sl_bio_spans.png            -- illustrative: BIO tagging of a sentence -> two recovered spans
  sl_tagging_schemes.png      -- illustrative: BIO vs BIOES on the same sentence (begin/inside/end/single)
  sl_viterbi_trellis.png      -- MEASURED: delta_t(j) trellis for 'fish sleep', best path highlighted
  sl_greedy_vs_viterbi.png    -- MEASURED: path probabilities, greedy (2nd-best) vs Viterbi (global best)
  sl_illegal_transition.png   -- MEASURED: the O->I-PER transition ruled out; CRF backs off to O,B-LOC
  sl_crf_vs_independent.png   -- MEASURED: emission heatmap + independent-argmax break vs CRF consistent span
  sl_forward_vs_viterbi.png   -- illustrative: same trellis, max (Viterbi) vs sum (forward / partition Z)
  sl_bilstm_crf.png           -- illustrative: char+word -> biLSTM -> emissions -> CRF layer -> Viterbi
  sl_entity_vs_token.png      -- MEASURED: token accuracy 0.875 vs entity P/R/F1 0.50 (one dropped token)

Verified on Python 3.12 / numpy 2.x / matplotlib 3.x, CPU, deterministic (numpy seeded in the backend).
"""

from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")  # headless backend: write files, never open a window

import matplotlib.patches as mpatches
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch

from sequence_labeling import (
    HMM_TAGS,
    NER_TAGS,
    WORKED_EMIT,
    WORKED_START,
    WORKED_TRANS,
    bio_transition_matrix,
    crf_sequence_score,
    crf_viterbi_decode,
    entity_prf,
    greedy_argmax,
    path_probability,
    token_accuracy,
    viterbi,
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
LIGHT_GREEN = "#D6E6DC"
LIGHT_BLUE = "#D3DFEA"

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
    print(f"wrote images/{name}")


def _token_box(ax, x, y, w, h, text, facecolor, textcolor="white", fontsize=12, weight="bold"):
    box = FancyBboxPatch(
        (x, y), w, h, boxstyle="round,pad=0.02,rounding_size=0.04",
        facecolor=facecolor, edgecolor="white", linewidth=1.4, zorder=3,
    )
    ax.add_patch(box)
    ax.text(x + w / 2, y + h / 2, text, ha="center", va="center",
            color=textcolor, fontsize=fontsize, weight=weight, zorder=4)


# =============================================================================================
# Figure 1 -- BIO tagging of a sentence -> recovered spans (illustrative)
# =============================================================================================
def fig_bio_spans() -> None:
    tokens = ["Jane", "Smith", "flew", "to", "New", "York", "City", "."]
    tags = ["B-PER", "I-PER", "O", "O", "B-LOC", "I-LOC", "I-LOC", "O"]
    tag_color = {"B-PER": GREEN, "I-PER": BLUE, "B-LOC": GREEN, "I-LOC": BLUE, "O": SLATE}
    n = len(tokens)
    fig, ax = plt.subplots(figsize=(12.5, 5.6))
    ax.set_xlim(0, n)
    ax.set_ylim(0, 4.2)
    ax.axis("off")
    bw, gap = 0.86, 0.07
    for i, (tok, tag) in enumerate(zip(tokens, tags)):
        x = i + gap
        _token_box(ax, x, 3.0, bw, 0.85, tok, SLATE, fontsize=13)             # word row
        ax.add_patch(FancyArrowPatch((x + bw / 2, 3.0), (x + bw / 2, 2.55),
                     arrowstyle="-|>", mutation_scale=12, color=INK, lw=1.2, zorder=2))
        _token_box(ax, x, 1.7, bw, 0.85, tag, tag_color[tag], fontsize=12)    # tag row
    # span brackets
    def span_bracket(i0, i1, label, color, light):
        x0, x1 = i0 + gap, i1 + gap + bw
        ax.add_patch(FancyBboxPatch((x0, 0.35), x1 - x0, 0.7,
                     boxstyle="round,pad=0.02,rounding_size=0.05",
                     facecolor=light, edgecolor=color, linewidth=1.6, zorder=2))
        ax.text((x0 + x1) / 2, 0.7, f"⟦ {label} ⟧", ha="center", va="center",
                color=color, fontsize=12.5, weight="bold")
        mid = i0 + gap + bw / 2
        ax.add_patch(FancyArrowPatch((mid, 1.7), (mid, 1.05),
                     arrowstyle="-", color=color, lw=1.4, zorder=1))
    span_bracket(0, 1, "PERSON  (Jane Smith)", GREEN, LIGHT_GREEN)
    span_bracket(4, 6, "LOCATION  (New York City)", BLUE, LIGHT_BLUE)
    ax.set_title("BIO tagging: B- begins a span, I- continues it, O is outside — "
                 "spans = merged B…I runs",
                 fontsize=13.5, weight="bold", color=INK, pad=12)
    ax.text(n / 2, -0.05, "eight tokens, eight tags, two entities — every token gets a label",
            ha="center", va="center", color=SLATE, fontsize=11, style="italic")
    _save(fig, "sl_bio_spans.png")


# =============================================================================================
# Figure 2 -- BIO vs BIOES on the same sentence (illustrative)
# =============================================================================================
def fig_tagging_schemes() -> None:
    tokens = ["Jane", "Smith", "flew", "to", "Paris"]
    bio = ["B-PER", "I-PER", "O", "O", "B-LOC"]
    bioes = ["B-PER", "E-PER", "O", "O", "S-LOC"]
    color = {"B-PER": GREEN, "I-PER": BLUE, "E-PER": PURPLE, "S-LOC": AMBER,
             "B-LOC": GREEN, "O": SLATE}
    n = len(tokens)
    fig, ax = plt.subplots(figsize=(11.0, 4.4))
    ax.set_xlim(0, n)
    ax.set_ylim(0, 3.4)
    ax.axis("off")
    bw, gap = 0.9, 0.05
    ax.text(-0.05, 2.85, "word", ha="right", va="center", color=INK, fontsize=11, weight="bold")
    ax.text(-0.05, 1.75, "BIO", ha="right", va="center", color=INK, fontsize=11, weight="bold")
    ax.text(-0.05, 0.65, "BIOES", ha="right", va="center", color=INK, fontsize=11, weight="bold")
    for i, tok in enumerate(tokens):
        x = i + gap
        _token_box(ax, x, 2.45, bw, 0.8, tok, NAVY, fontsize=12)
        _token_box(ax, x, 1.35, bw, 0.8, bio[i], color[bio[i]], fontsize=11)
        _token_box(ax, x, 0.25, bw, 0.8, bioes[i], color[bioes[i]], fontsize=11)
    ax.set_title("BIO vs BIOES — BIOES adds explicit End and Single tags so span boundaries "
                 "are learned directly",
                 fontsize=13.5, weight="bold", color=INK, pad=10)
    ax.text(n / 2, -0.15,
            "'Smith' ends the person span (I-PER → E-PER); 'Paris' is a one-token span (B-LOC → S-LOC)",
            ha="center", va="center", color=SLATE, fontsize=10.5, style="italic")
    _save(fig, "sl_tagging_schemes.png")


# =============================================================================================
# Figure 3 -- the MANDATORY Viterbi delta trellis (MEASURED from viterbi())
# =============================================================================================
def fig_viterbi_trellis() -> None:
    # the trellis matches the page's hand-worked derivation: the 2-word "fish sleep" HMM
    words = ["fish", "sleep"]
    best_path, best_prob, delta = viterbi(
        words, HMM_TAGS, WORKED_START, WORKED_TRANS, WORKED_EMIT)  # delta measured from the backend
    trans_tbl = WORKED_TRANS
    tags = HMM_TAGS
    n = len(words)
    # two columns, generous spacing so transition labels sit cleanly between them
    xpos = {0: 0.0, 1: 4.0}
    ypos = {"N": 3.0, "V": 0.4}
    rx, ry = 0.5, 0.6                                       # node radii (small enough to leave gaps)
    fig, ax = plt.subplots(figsize=(11.0, 6.6))
    ax.set_xlim(-1.0, 5.0)
    ax.set_ylim(-1.0, 4.2)
    ax.axis("off")
    state_color = {"N": GREEN, "V": PURPLE}
    best_idx = [(t, tags.index(best_path[t])) for t in range(n)]
    best_edges = {(t, best_path[t - 1], best_path[t]) for t in range(1, n)}
    # edges first (so nodes draw on top)
    for t in range(1, n):
        for src in tags:
            for dst in tags:
                on_best = (t, src, dst) in best_edges
                col = GREEN if on_best else "#C4CAD2"
                lw = 3.6 if on_best else 1.2
                x0, x1 = xpos[t - 1] + rx, xpos[t] - rx
                ax.add_patch(FancyArrowPatch(
                    (x0, ypos[src]), (x1, ypos[dst]),
                    arrowstyle="-|>", mutation_scale=15, color=col, lw=lw,
                    zorder=4 if on_best else 1, alpha=1.0 if on_best else 0.85))
                # transition label nudged off the line; cross-edges shifted toward their
                # source column (up-going left, down-going right) so the two diagonals never collide
                if src == dst:
                    mx = (x0 + x1) / 2
                    my = ypos[src] + (0.42 if src == "N" else -0.42)   # straight edges: above/below
                else:
                    frac = 0.30 if ypos[dst] > ypos[src] else 0.70     # up-going hugs source, down-going hugs dest
                    mx = x0 + frac * (x1 - x0)
                    my = ypos[src] + frac * (ypos[dst] - ypos[src])
                ax.text(mx, my, f"a={trans_tbl[(src, dst)]}", ha="center", va="center",
                        color=GREEN if on_best else SLATE,
                        fontsize=10.5 if on_best else 9,
                        weight="bold" if on_best else "normal", zorder=6,
                        bbox=dict(boxstyle="round,pad=0.12", fc="white", ec="none", alpha=0.85))
    # nodes
    for t in range(n):
        for j, tag in enumerate(tags):
            is_best = (t, j) in best_idx
            ec = GREEN if is_best else "white"
            ew = 3.2 if is_best else 1.4
            circ = mpatches.Ellipse((xpos[t], ypos[tag]), 2 * rx, 2 * ry,
                                    facecolor=state_color[tag], edgecolor=ec,
                                    linewidth=ew, zorder=5)
            ax.add_patch(circ)
            ax.text(xpos[t], ypos[tag], tag, ha="center", va="center", color="white",
                    fontsize=15, weight="bold", zorder=6)
            dv = delta[t, j]
            ax.text(xpos[t], ypos[tag] - ry - 0.26, f"δ={dv:.4f}", ha="center", va="center",
                    color=GREEN if is_best else SLATE,
                    fontsize=10.5, weight="bold" if is_best else "normal", zorder=6)
    # column headers
    for t, w in enumerate(words):
        ax.text(xpos[t], ypos["N"] + ry + 0.45, f"t={t + 1}   '{w}'", ha="center", va="center",
                color=INK, fontsize=12.5, weight="bold")
    ax.text((xpos[0] + xpos[1]) / 2, -0.85,
            f"best path:  {' → '.join(best_path)}   (δ* = {best_prob:.4f})",
            ha="center", va="center", color=GREEN, fontsize=13, weight="bold")
    ax.set_title("Viterbi trellis on the 2-tag HMM for 'fish sleep' — each cell keeps "
                 "δ, the best path probability into that state",
                 fontsize=13.5, weight="bold", color=INK, pad=14)
    _save(fig, "sl_viterbi_trellis.png")


# =============================================================================================
# Figure 4 -- greedy vs Viterbi path probabilities (MEASURED)
# =============================================================================================
def fig_greedy_vs_viterbi() -> None:
    words = ["time", "flies", "fast"]
    v_path, v_prob, _ = viterbi(words)
    g_path = greedy_argmax(words)
    v_score = path_probability(words, v_path)
    g_score = path_probability(words, g_path)
    fig, ax = plt.subplots(figsize=(8.4, 5.2))
    labels = [f"greedy\n{'·'.join(g_path)}", f"Viterbi\n{'·'.join(v_path)}"]
    scores = [g_score, v_score]
    colors = [RED, GREEN]
    bars = ax.bar(labels, scores, color=colors, edgecolor="white", linewidth=1.5, width=0.55, zorder=3)
    for bar, s in zip(bars, scores):
        ax.text(bar.get_x() + bar.get_width() / 2, s + max(scores) * 0.015,
                f"p(y,x) = {s:.5f}", ha="center", va="bottom", color=INK, fontsize=11.5, weight="bold")
    ax.set_ylabel("joint path probability  p(y, x)")
    ax.set_ylim(0, max(scores) * 1.18)
    ax.set_title("Greedy per-token argmax lands on the 2nd-best path; "
                 "Viterbi finds the global best",
                 fontsize=13, weight="bold", color=INK, pad=10)
    _style_axis(ax)
    ax.text(0.5, -0.16,
            "greedy commits to V at 'flies' (b_V=0.7) and is locked in; Viterbi scores the whole sequence",
            transform=ax.transAxes, ha="center", va="center", color=SLATE, fontsize=10, style="italic")
    _save(fig, "sl_greedy_vs_viterbi.png")


# =============================================================================================
# Figure 5 -- the illegal O -> I-PER transition, ruled out (MEASURED)
# =============================================================================================
def fig_illegal_transition() -> None:
    A = bio_transition_matrix()
    emissions = np.array([
        [2.0, 0.1, 0.1, 0.5, 0.1],
        [0.5, 0.4, 2.2, 1.9, 0.2],
    ])
    greedy_tags = [NER_TAGS[int(np.argmax(emissions[t]))] for t in range(2)]
    crf_tags, _ = crf_viterbi_decode(emissions, A)
    s_illegal = crf_sequence_score(emissions, ["O", "I-PER"], A)
    s_crf = crf_sequence_score(emissions, crf_tags, A)
    fig, ax = plt.subplots(figsize=(11.5, 5.8))
    ax.set_xlim(0, 10)
    ax.set_ylim(0, 6)
    ax.axis("off")
    # two columns of states for tokens 0 and 1; draw O at t0 and {I-PER, B-LOC} at t1
    _token_box(ax, 0.6, 2.6, 1.7, 0.9, "O\n(token 0:\n'downtown')", SLATE, fontsize=11)
    # candidate next states
    _token_box(ax, 5.8, 4.1, 2.0, 0.95, "I-PER  (emit 2.2)", RED, fontsize=12)
    _token_box(ax, 5.8, 1.1, 2.0, 0.95, "B-LOC  (emit 1.9)", GREEN, fontsize=12)
    # illegal edge O -> I-PER
    ax.add_patch(FancyArrowPatch((2.3, 3.05), (5.8, 4.55), arrowstyle="-|>", mutation_scale=18,
                 color=RED, lw=2.6, linestyle=(0, (4, 3)), zorder=2))
    ax.text(4.0, 4.25, "O → I-PER\ntransition = −∞  (ILLEGAL)", ha="center", va="center",
            color=RED, fontsize=11, weight="bold")
    ax.plot([3.9, 4.3], [3.78, 3.78], color=RED, lw=3)  # strike mark
    ax.text(4.1, 3.55, "✗", ha="center", va="center", color=RED, fontsize=20, weight="bold")
    # legal edge O -> B-LOC
    ax.add_patch(FancyArrowPatch((2.3, 2.9), (5.8, 1.6), arrowstyle="-|>", mutation_scale=18,
                 color=GREEN, lw=2.8, zorder=2))
    ax.text(4.0, 1.65, "O → B-LOC\ntransition = legal  ✓", ha="center", va="center",
            color=GREEN, fontsize=11, weight="bold")
    # decode summary
    ax.text(5.0, 0.25,
            f"per-token argmax: {greedy_tags}  (ILLEGAL)        "
            f"CRF Viterbi: {crf_tags}  (score {s_crf:.1f}  vs  {s_illegal:.0e})",
            ha="center", va="center", color=INK, fontsize=11, weight="bold")
    ax.set_title("Global scoring rules out the illegal transition: the top per-token emission "
                 "is I-PER, but O→I-PER is impossible",
                 fontsize=13, weight="bold", color=INK, pad=10)
    _save(fig, "sl_illegal_transition.png")


# =============================================================================================
# Figure 6 -- CRF span consistency vs independent classification (MEASURED)
# =============================================================================================
def fig_crf_vs_independent() -> None:
    A = bio_transition_matrix()
    words = ["Dr", "Jane", "Smith"]
    em3 = np.array([
        [0.2, 2.5, 0.1, 0.3, 0.1],
        [0.2, 1.0, 2.2, 0.3, 0.1],
        [0.2, 1.5, 1.4, 0.3, 0.1],
    ])
    indep = [NER_TAGS[int(np.argmax(em3[t]))] for t in range(3)]
    crf3, _ = crf_viterbi_decode(em3, A)
    fig, (axh, axt) = plt.subplots(1, 2, figsize=(13.2, 5.2), gridspec_kw={"width_ratios": [1.35, 1]})
    # left: emission heatmap
    axh.imshow(em3.T, cmap="BuGn", aspect="auto", vmin=0, vmax=em3.max())
    axh.set_xticks(range(3))
    axh.set_xticklabels(words, fontsize=11)
    axh.set_yticks(range(len(NER_TAGS)))
    axh.set_yticklabels(NER_TAGS, fontsize=10)
    for t in range(3):
        for j in range(len(NER_TAGS)):
            axh.text(t, j, f"{em3[t, j]:.1f}", ha="center", va="center",
                     color="white" if em3[t, j] > 1.3 else INK, fontsize=10, weight="bold")
    axh.set_title("biLSTM emission scores P(tag | token)", fontsize=12, weight="bold", color=INK)
    axh.tick_params(colors=INK)
    # right: the two decodings (label ABOVE each row of boxes so nothing overlaps)
    axt.set_xlim(0, 3)
    axt.set_ylim(0, 3.4)
    axt.axis("off")
    axt.text(1.5, 3.2, "two ways to read the emissions", ha="center", color=INK,
             fontsize=12, weight="bold")
    color = {"B-PER": GREEN, "I-PER": BLUE, "O": SLATE}
    rows = [
        ("independent per-token argmax  →  span breaks", indep, RED),
        ("CRF Viterbi (global score)  →  consistent span", crf3, GREEN),
    ]
    for row, (label, tags_seq, col) in enumerate(rows):
        y_box = 2.25 - row * 1.35
        axt.text(0.1, y_box + 0.78, label, ha="left", va="center", color=col,
                 fontsize=10.5, weight="bold")
        for t, tag in enumerate(tags_seq):
            _token_box(axt, 0.1 + t * 0.95, y_box, 0.85, 0.55, tag, color[tag], fontsize=10.5)
    axt.text(1.5, 0.05,
             "independent argmax puts a stray B-PER on 'Smith' → two broken spans;\n"
             "the CRF keeps one consistent 3-token PERSON span",
             ha="center", va="center", color=SLATE, fontsize=9.6, style="italic")
    fig.suptitle("CRF sequence scoring keeps a span consistent where per-token classification fragments it",
                 fontsize=13.5, weight="bold", color=INK, y=1.02)
    fig.tight_layout()
    _save(fig, "sl_crf_vs_independent.png")


# =============================================================================================
# Figure 7 -- forward (sum) vs Viterbi (max) on the same trellis (illustrative)
# =============================================================================================
def fig_forward_vs_viterbi() -> None:
    fig, axes = plt.subplots(1, 2, figsize=(12.5, 4.8))
    for ax, (title, op, color, caption) in zip(axes, [
        ("Viterbi — max-product", "max", GREEN,
         "keeps the SINGLE best path into each state\n→ decoding: argmax_y p(y | x)"),
        ("Forward — sum-product", "Σ", BLUE,
         "sums over ALL paths into each state\n→ partition function Z(x) / total p(x)"),
    ]):
        ax.set_xlim(0, 3)
        ax.set_ylim(0, 3)
        ax.axis("off")
        # 2 states x 2 steps mini-trellis
        pos = {(0, 0): (0.6, 2.2), (0, 1): (0.6, 0.8), (1, 0): (2.4, 2.2), (1, 1): (2.4, 0.8)}
        for (t, j), (x, y) in pos.items():
            ax.add_patch(mpatches.Ellipse((x, y), 0.5, 0.6, facecolor=SLATE,
                         edgecolor="white", linewidth=1.4, zorder=3))
            ax.text(x, y, "N" if j == 0 else "V", ha="center", va="center",
                    color="white", fontsize=13, weight="bold", zorder=4)
        # edges into (1,0)
        for src in (0, 1):
            on = (op == "max" and src == 0)  # highlight just one for the max panel
            ax.add_patch(FancyArrowPatch(pos[(0, src)], pos[(1, 0)], arrowstyle="-|>",
                         mutation_scale=14, color=color if (op == "Σ" or on) else "#C4CAD2",
                         lw=2.8 if (op == "Σ" or on) else 1.4, zorder=2))
        ax.text(1.5, 2.55, op, ha="center", va="center", color=color, fontsize=22, weight="bold")
        ax.set_title(title, fontsize=12.5, weight="bold", color=color)
        ax.text(1.5, 0.05, caption, ha="center", va="center", color=INK, fontsize=10)
    fig.suptitle("Same trellis, different semiring: Viterbi (max) decodes the best tags; "
                 "forward (sum) computes Z(x) for training",
                 fontsize=13, weight="bold", color=INK, y=1.04)
    fig.tight_layout()
    _save(fig, "sl_forward_vs_viterbi.png")


# =============================================================================================
# Figure 8 -- biLSTM-CRF architecture (illustrative)
# =============================================================================================
def fig_bilstm_crf() -> None:
    fig, ax = plt.subplots(figsize=(11.5, 6.6))
    ax.set_xlim(0, 10)
    ax.set_ylim(0, 7)
    ax.axis("off")
    tokens = ["Jane", "Smith", "flew"]
    xs = [2.6, 5.4, 8.2]
    # layer y-centers
    y_char, y_word, y_lstm, y_emit, y_crf = 0.6, 1.8, 3.2, 4.6, 6.0
    for x, tok in zip(xs, tokens):
        _token_box(ax, x - 0.85, y_char - 0.32, 1.7, 0.62, f"chars({tok})", AMBER, fontsize=9.5)
        _token_box(ax, x - 0.85, y_word - 0.32, 1.7, 0.62, f"emb({tok})", NAVY, fontsize=9.5)
        _token_box(ax, x - 0.85, y_lstm - 0.32, 1.7, 0.62, "biLSTM hᵢ", PURPLE, fontsize=10)
        _token_box(ax, x - 0.85, y_emit - 0.32, 1.7, 0.62, "emissions Pᵢ", BLUE, fontsize=9.5)
        # vertical arrows up the stack
        for y0, y1 in [(y_char + 0.3, y_word - 0.32), (y_word + 0.3, y_lstm - 0.32),
                       (y_lstm + 0.3, y_emit - 0.32), (y_emit + 0.3, y_crf - 0.32)]:
            ax.add_patch(FancyArrowPatch((x, y0), (x, y1), arrowstyle="-|>",
                         mutation_scale=11, color=INK, lw=1.2, zorder=2))
    # biLSTM horizontal recurrence arrows
    for i in range(len(xs) - 1):
        ax.add_patch(FancyArrowPatch((xs[i] + 0.85, y_lstm + 0.1), (xs[i + 1] - 0.85, y_lstm + 0.1),
                     arrowstyle="-|>", mutation_scale=10, color=PURPLE, lw=1.5, zorder=2))
        ax.add_patch(FancyArrowPatch((xs[i + 1] - 0.85, y_lstm - 0.1), (xs[i] + 0.85, y_lstm - 0.1),
                     arrowstyle="-|>", mutation_scale=10, color=PURPLE, lw=1.5, zorder=2))
    # CRF layer spanning the top
    ax.add_patch(FancyBboxPatch((1.7, y_crf - 0.4), 7.4, 0.8,
                 boxstyle="round,pad=0.02,rounding_size=0.06", facecolor=GREEN,
                 edgecolor="white", linewidth=1.6, zorder=3))
    ax.text(5.4, y_crf, "CRF layer:  + transition scores A(yᵢ₋₁→yᵢ),  Viterbi-decode",
            ha="center", va="center", color="white", fontsize=10.5, weight="bold", zorder=4)
    # left-side layer labels (short, kept clear of the leftmost column at x≈1.75)
    for y, lab in [(y_char, "chars"), (y_word, "word emb"), (y_lstm, "context"),
                   (y_emit, "emissions"), (y_crf, "transitions")]:
        ax.text(0.1, y, lab, ha="left", va="center", color=SLATE, fontsize=9.5, style="italic")
    ax.set_title("biLSTM-CRF: char+word features → biLSTM context → per-token emissions\n"
                 "→ CRF transition layer → Viterbi decode",
                 fontsize=11.5, weight="bold", color=INK, pad=10)
    ax.text(5.4, -0.15,
            "the biLSTM scores WHICH TAG FITS each token; the CRF scores WHICH TAG MAY FOLLOW WHICH",
            ha="center", va="center", color=SLATE, fontsize=10, style="italic")
    _save(fig, "sl_bilstm_crf.png")


# =============================================================================================
# Figure 9 -- token accuracy vs entity P/R/F1 (MEASURED)
# =============================================================================================
def fig_entity_vs_token() -> None:
    gold = ["B-PER", "I-PER", "O", "B-LOC", "I-LOC", "I-LOC", "O", "O"]
    pred = ["B-PER", "I-PER", "O", "B-LOC", "I-LOC", "O", "O", "O"]
    acc = token_accuracy(gold, pred)
    precision, recall, f1 = entity_prf(gold, pred)
    fig, ax = plt.subplots(figsize=(8.8, 5.2))
    labels = ["token\naccuracy", "entity\nprecision", "entity\nrecall", "entity\nF1"]
    vals = [acc, precision, recall, f1]
    colors = [SLATE, BLUE, PURPLE, RED]
    bars = ax.bar(labels, vals, color=colors, edgecolor="white", linewidth=1.5, width=0.6, zorder=3)
    for bar, v in zip(bars, vals):
        ax.text(bar.get_x() + bar.get_width() / 2, v + 0.02, f"{v:.3f}",
                ha="center", va="bottom", color=INK, fontsize=12, weight="bold")
    ax.axhline(acc, color=SLATE, lw=1.0, linestyle="--", alpha=0.6, zorder=1)
    ax.set_ylim(0, 1.08)
    ax.set_ylabel("score")
    ax.set_title("One dropped token of a 3-token entity: token accuracy barely moves (0.875), "
                 "entity F1 halves (0.50)",
                 fontsize=12.5, weight="bold", color=INK, pad=10)
    _style_axis(ax)
    ax.text(0.5, -0.17,
            "gold 'New York City' = LOC; the model drops 'City' → a wrong span. "
            "Entity F1 is the honest metric.",
            transform=ax.transAxes, ha="center", va="center", color=SLATE, fontsize=9.8, style="italic")
    _save(fig, "sl_entity_vs_token.png")


def main() -> None:
    print(f"writing figures to {OUT_DIR}")
    fig_bio_spans()
    fig_tagging_schemes()
    fig_viterbi_trellis()
    fig_greedy_vs_viterbi()
    fig_illegal_transition()
    fig_crf_vs_independent()
    fig_forward_vs_viterbi()
    fig_bilstm_crf()
    fig_entity_vs_token()
    print("done.")


if __name__ == "__main__":
    main()
