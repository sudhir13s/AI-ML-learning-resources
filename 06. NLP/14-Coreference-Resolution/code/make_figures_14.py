"""Reproducible figure generator for 14-Coreference-Resolution.

Produces EVERY embedded PNG for the chapter from the SAME backend used on the page and in the
notebook -- the mention-ranking softmax, the metric numbers (MUC / B-cubed / CEAF-phi4), the
B-cubed over-split/over-merge symmetry, and the Winograd flip are all IMPORTED from
`coreference.py`, so the figures cannot silently drift from the prose or the demo.

    python make_figures_14.py

Each figure is written to ../../images/ (the shared chapter image dir, "06. NLP/images/") at
150 dpi, prefixed `coref_`. The palette matches the chapter's Mermaid diagrams (muted, white
text on fills).

Figures produced (measured = computed by the live backend; illustrative = a labelled schematic):
  coref_clusters.png        -- illustrative: passage with colored mention spans linked into chains
  coref_model_families.png  -- illustrative: mention-pair vs mention-ranking vs entity-level
  coref_e2e_arch.png        -- illustrative: end-to-end span-ranking architecture (Lee et al. 2017)
  coref_metrics.png         -- MEASURED: MUC / B-cubed / CEAF-phi4 on one gold-vs-pred clustering
  coref_mention_ranking.png -- MEASURED: softmax P(antecedent) for 'him' over candidates incl. epsilon
  coref_bcubed_symmetry.png -- MEASURED: B-cubed P/R for over-split vs over-merge (the two failures)
  coref_winograd.png        -- MEASURED: world-knowledge feature flips 'it' when the adjective flips
  coref_field_progress.png  -- illustrative: CoNLL F1 over the field's model-family arc

Verified on Python 3.12 / numpy 2.x / matplotlib 3.x, CPU, deterministic (numpy seeded in the
backend). spaCy mention detection for figure 1 is printed when this script runs so the page's
worked example matches.
"""

from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")  # headless backend: write files, never open a window

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch

from coreference import (
    METRIC_GOLD,
    METRIC_PRED,
    MENTIONS,
    antecedent_distribution,
    bcubed,
    ceaf_phi4,
    muc,
    runtime_banner,
    spacy_mentions,
    winograd_resolve,
)

# ---- Palette (matches the chapter Mermaid classDefs) ------------------------------------------
BLUE = "#3A6B96"
PURPLE = "#5D4A8A"
GREEN = "#2E7A5A"
RED = "#8B3B4A"
SLATE = "#4A5B6E"
AMBER = "#7A6528"
NAVY = "#2A5B80"
INK = "#1C2530"
GRID = "#D4D9DF"
MUTED = "#9AA3AD"

OUT_DIR = Path(__file__).resolve().parent.parent.parent / "images"
DPI = 150

plt.rcParams.update({"font.size": 11, "font.family": "DejaVu Sans"})


def _save(fig: plt.Figure, name: str) -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    path = OUT_DIR / name
    fig.savefig(path, dpi=DPI, bbox_inches="tight")
    plt.close(fig)
    print("wrote", path.name)


def _style_axis(ax: plt.Axes) -> None:
    ax.grid(True, color=GRID, linewidth=0.8, zorder=0)
    ax.set_axisbelow(True)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    for side in ("left", "bottom"):
        ax.spines[side].set_color(SLATE)
    ax.tick_params(colors=INK)
    ax.title.set_color(INK)


def _box(ax, x, y, w, h, text, fc, fontsize=10, tc="#fff", weight="normal"):
    ax.add_patch(
        FancyBboxPatch(
            (x, y), w, h,
            boxstyle="round,pad=0.012,rounding_size=0.02",
            fc=fc, ec="#222", lw=1.0, mutation_aspect=1.0,
        )
    )
    ax.text(x + w / 2, y + h / 2, text, ha="center", va="center",
            fontsize=fontsize, color=tc, weight=weight, wrap=True)


# --------------------------------------------------------------------------------------
# Figure 1 — passage with colored mention spans linked into coreference clusters
# --------------------------------------------------------------------------------------
def fig_clusters() -> None:
    fig, ax = plt.subplots(figsize=(11.0, 5.0))
    ax.set_xlim(0, 11)
    ax.set_ylim(0, 5)
    ax.axis("off")
    ax.set_title("Coreference clustering: linking mentions to the entity they refer to",
                 fontsize=14, fontweight="bold", pad=10)

    line1 = [("John", 0), ("told", None), ("his", 0), ("manager", 1),
             ("that", None), ("he", 0), ("would", None), ("finish", None),
             ("the report", None), (".", None)]
    line2 = [("She", 1), ("thanked", None), ("him", 0), ("for", None),
             ("the update", None), (".", None)]

    cmap = {0: BLUE, 1: GREEN, None: None}
    y1, y2 = 3.4, 1.7
    centers: dict = {}

    def lay(line, y):
        x = 0.3
        for i, (tok, cid) in enumerate(line):
            w = 0.20 + 0.115 * len(tok)
            if cid is not None:
                _box(ax, x, y, w, 0.55, tok, cmap[cid], fontsize=11, weight="bold")
                centers[(y, i)] = (x + w / 2, y)
            else:
                ax.text(x + w / 2, y + 0.27, tok, ha="center", va="center",
                        fontsize=11, color="#333")
            x += w + 0.18

    lay(line1, y1)
    lay(line2, y2)

    def link(a, b, col, rad=0.35):
        xa, ya = centers[a]
        xb, yb = centers[b]
        ax.add_patch(FancyArrowPatch((xa, ya + 0.55), (xb, yb + 0.55),
                     connectionstyle=f"arc3,rad={rad}", color=col, lw=2.0,
                     arrowstyle="-|>", mutation_scale=14, alpha=0.9))

    link((y1, 0), (y1, 2), BLUE)            # John -> his
    link((y1, 2), (y1, 5), BLUE)            # his -> he
    link((y1, 5), (y2, 2), BLUE, rad=-0.18)  # he -> him
    link((y1, 3), (y2, 0), GREEN, rad=-0.30)  # manager -> She

    ax.text(0.3, 4.55, "Cluster 1 (entity = John):  John · his · he · him",
            fontsize=11, color=BLUE, weight="bold")
    ax.text(0.3, 4.18, "Cluster 2 (entity = the manager):  manager · She",
            fontsize=11, color=GREEN, weight="bold")
    ax.text(0.3, 0.75,
            "Arrows = antecedent → anaphor links; a mention's cluster is the entity it refers to. "
            "Note 'his manager' nests two mentions: the possessive 'his' (John) inside the NP 'manager'.",
            fontsize=9.5, color="#444", style="italic")
    _save(fig, "coref_clusters.png")


# --------------------------------------------------------------------------------------
# Figure 2 — mention-pair vs mention-ranking vs entity/cluster model (schematic)
# --------------------------------------------------------------------------------------
def fig_model_families() -> None:
    fig, axes = plt.subplots(1, 3, figsize=(13.5, 4.6))
    fig.suptitle("Three model families: what each one scores when resolving a mention",
                 fontsize=14, fontweight="bold", y=1.00)

    def strip(ax, hi=None, y=3.5, x0=0.25, w=0.78, gap=0.20):
        toks = ["John", "his", "he", "him"]
        cols = [BLUE, SLATE, SLATE, NAVY]
        x = x0
        centers = []
        for i, t in enumerate(toks):
            fc = cols[i] if (hi is None or i in hi) else MUTED
            _box(ax, x, y, w, 0.55, t, fc, fontsize=10, weight="bold")
            centers.append(x + w / 2)
            x += w + gap
        return centers, y, x

    # (a) mention-pair
    ax = axes[0]
    ax.set_xlim(0, 5)
    ax.set_ylim(0, 5)
    ax.axis("off")
    ax.set_title("Mention-pair (binary)", fontsize=11.5, fontweight="bold", color=BLUE)
    cx, y, _ = strip(ax)
    for srci, col, rad in [(0, GREEN, 0.55), (1, AMBER, 0.62), (2, RED, 0.7)]:
        ax.add_patch(FancyArrowPatch((cx[3], y + 0.55), (cx[srci], y + 0.55),
                     connectionstyle=f"arc3,rad={rad}", color=col, lw=1.8,
                     arrowstyle="-|>", mutation_scale=12))
    ax.text(2.5, 2.0,
            "for EACH (mention, antecedent) pair\nemit P(coref) ∈ {0,1} independently.\n"
            "Then cluster the positive pairs.\n\n⚠ pairwise decisions can violate\ntransitivity "
            "(A=B, B=C, but A≠C).",
            ha="center", va="center", fontsize=9.3, color="#333")

    # (b) mention-ranking
    ax = axes[1]
    ax.set_xlim(0, 5)
    ax.set_ylim(0, 5)
    ax.axis("off")
    ax.set_title("Mention-ranking (softmax)", fontsize=11.5, fontweight="bold", color=PURPLE)
    cx, y, nx = strip(ax)
    _box(ax, nx, y, 0.50, 0.55, "ε", AMBER, fontsize=11, weight="bold")
    eps_cx = nx + 0.25
    src = (cx[3], y)
    cands = [(cx[0], 0.40, GREEN), (cx[1], 0.10, SLATE), (cx[2], 0.35, SLATE), (eps_cx, 0.15, AMBER)]
    for tx, p, col in cands:
        rad = 0.55 if tx < src[0] else -0.55
        ax.add_patch(FancyArrowPatch((src[0], y + 0.55), (tx, y + 0.55),
                     connectionstyle=f"arc3,rad={rad}", color=col, lw=1.0 + 4 * p,
                     arrowstyle="-|>", mutation_scale=11, alpha=0.85))
    ax.text(2.5, 2.0,
            "for the mention 'him', RANK all\ncandidate antecedents (incl. ε = 'no\n"
            "antecedent') with ONE softmax.\n\n✓ picks the single best antecedent;\n"
            "consistent, end-to-end trainable.",
            ha="center", va="center", fontsize=9.3, color="#333")

    # (c) entity/cluster
    ax = axes[2]
    ax.set_xlim(0, 5)
    ax.set_ylim(0, 5)
    ax.axis("off")
    ax.set_title("Entity / cluster-level", fontsize=11.5, fontweight="bold", color=GREEN)
    _box(ax, 0.4, 3.3, 2.0, 0.8, "partial cluster\n{John, his, he}", BLUE, fontsize=9.5, weight="bold")
    _box(ax, 3.2, 3.45, 1.0, 0.55, "him", NAVY, fontsize=10, weight="bold")
    ax.add_patch(FancyArrowPatch((3.2, 3.72), (2.4, 3.72), color=GREEN, lw=2.2,
                 arrowstyle="-|>", mutation_scale=13))
    ax.text(2.5, 1.9,
            "score the mention against the WHOLE\npartial cluster built so far,\nnot a single antecedent.\n\n"
            "✓ uses global entity-level features\n(e.g. a cluster already has a name).",
            ha="center", va="center", fontsize=9.3, color="#333")

    fig.tight_layout(rect=(0, 0, 1, 0.96))
    _save(fig, "coref_model_families.png")


# --------------------------------------------------------------------------------------
# Figure 3 — end-to-end span-ranking architecture (Lee et al. 2017)
# --------------------------------------------------------------------------------------
def fig_e2e_arch() -> None:
    fig, ax = plt.subplots(figsize=(11.5, 6.2))
    ax.set_xlim(0, 11.5)
    ax.set_ylim(0, 6.2)
    ax.axis("off")
    ax.set_title("End-to-end span-ranking coreference (Lee et al. 2017)",
                 fontsize=14, fontweight="bold", pad=8)

    _box(ax, 0.3, 5.2, 10.9, 0.62,
         "Document tokens:  John  told  his  manager  that  he ...", BLUE, fontsize=11)
    _box(ax, 0.3, 4.25, 10.9, 0.62,
         "Encoder (biLSTM 2017 → SpanBERT 2020): contextual token vectors", PURPLE, fontsize=10.5)
    _box(ax, 0.3, 3.30, 10.9, 0.62,
         "Enumerate ALL spans up to width L → span rep  gᵢ = [x_start ; x_end ; x̂_attn ; φ(width)]",
         NAVY, fontsize=10)

    _box(ax, 0.3, 2.20, 5.1, 0.70, "mention score  sₘ(i)\nkeep top-λT spans (prune)", AMBER, fontsize=10)
    _box(ax, 6.1, 2.20, 5.1, 0.70, "antecedent score  sₐ(i,j)\nover kept candidates + ε", GREEN, fontsize=10)
    ax.add_patch(FancyArrowPatch((5.4, 2.55), (6.1, 2.55), color="#333", lw=1.6,
                 arrowstyle="-|>", mutation_scale=13))

    ax.text(5.75, 1.55,
            r"$s(i,j) = s_m(i) + s_m(j) + s_a(i,j)$,    $s(i,\epsilon)=0$",
            ha="center", va="center", fontsize=13, color="#111")
    ax.text(5.75, 0.95,
            r"$P(y_i = j) = \mathrm{softmax}_j\, s(i,j)$  over candidate antecedents $j \in \{\epsilon, 1, \ldots, i-1\}$",
            ha="center", va="center", fontsize=11.5, color="#333")
    ax.text(5.75, 0.40,
            "Loss: maximize the marginal probability of landing on ANY gold antecedent in i's cluster.",
            ha="center", va="center", fontsize=10, color="#555", style="italic")

    for y0 in (5.2, 4.25, 3.30):
        ax.add_patch(FancyArrowPatch((5.75, y0), (5.75, y0 - 0.33), color="#333",
                     lw=1.6, arrowstyle="-|>", mutation_scale=13))
    _save(fig, "coref_e2e_arch.png")


# --------------------------------------------------------------------------------------
# Figure 4 — MUC / B-cubed / CEAF-phi4 on ONE example (MEASURED via coreference.py)
# --------------------------------------------------------------------------------------
def fig_metrics() -> tuple:
    mP, mR, mF = muc(METRIC_GOLD, METRIC_PRED)
    bP, bR, bF = bcubed(METRIC_GOLD, METRIC_PRED)
    cP, cR, cF = ceaf_phi4(METRIC_GOLD, METRIC_PRED)
    conll = (mF + bF + cF) / 3
    print("MEASURED metrics on gold={{a,b,c},{d,e}} pred={{a,b},{c},{d,e}}:")
    print(f"  MUC       P={mP:.3f} R={mR:.3f} F1={mF:.3f}")
    print(f"  B-cubed   P={bP:.3f} R={bR:.3f} F1={bF:.3f}")
    print(f"  CEAF-phi4 P={cP:.3f} R={cR:.3f} F1={cF:.3f}")
    print(f"  CoNLL avg F1 = {conll:.3f}")

    fig, (axL, axR) = plt.subplots(1, 2, figsize=(12.5, 4.8), gridspec_kw={"width_ratios": [1, 1.25]})

    axL.set_xlim(0, 6)
    axL.set_ylim(0, 6)
    axL.axis("off")
    axL.set_title("One example: gold vs predicted clustering", fontsize=12, fontweight="bold")
    axL.text(0.2, 5.4, "GOLD", fontsize=11, weight="bold", color="#111")
    _box(axL, 0.4, 4.4, 2.6, 0.7, "{ a · b · c }", BLUE, fontsize=11, weight="bold")
    _box(axL, 3.3, 4.4, 1.8, 0.7, "{ d · e }", GREEN, fontsize=11, weight="bold")
    axL.text(0.2, 3.5, "PREDICTED", fontsize=11, weight="bold", color="#111")
    _box(axL, 0.4, 2.5, 1.8, 0.7, "{ a · b }", BLUE, fontsize=11, weight="bold")
    _box(axL, 2.5, 2.5, 1.0, 0.7, "{ c }", RED, fontsize=11, weight="bold")
    _box(axL, 3.7, 2.5, 1.8, 0.7, "{ d · e }", GREEN, fontsize=11, weight="bold")
    axL.text(0.2, 1.5,
             "Error: the predictor split mention 'c' out of the {a,b,c} entity\n"
             "(a recall miss on one link). Each metric punishes this differently →",
             fontsize=9.6, color="#444", style="italic")

    axR.set_title("Why no single metric suffices (same error, different scores)",
                  fontsize=12, fontweight="bold")
    metrics = ["MUC", "B³", "CEAF-φ4", "CoNLL\navg"]
    Ps = [mP, bP, cP, np.nan]
    Rs = [mR, bR, cR, np.nan]
    Fs = [mF, bF, cF, conll]
    x = np.arange(len(metrics))
    w = 0.26
    axR.bar(x - w, [p if not np.isnan(p) else 0 for p in Ps], w, label="Precision", color=BLUE)
    axR.bar(x, [r if not np.isnan(r) else 0 for r in Rs], w, label="Recall", color=AMBER)
    axR.bar(x + w, Fs, w, label="F1", color=GREEN)
    for i, f in enumerate(Fs):
        axR.text(x[i] + w, f + 0.02, f"{f:.2f}", ha="center", fontsize=8.5, color=GREEN)
    axR.set_ylim(0, 1.15)
    axR.set_xticks(x)
    axR.set_xticklabels(metrics, fontsize=10)
    axR.set_ylabel("score")
    axR.legend(frameon=False, fontsize=9, ncol=3, loc="upper center")
    _style_axis(axR)
    _save(fig, "coref_metrics.png")
    return (mP, mR, mF), (bP, bR, bF), (cP, cR, cF), conll


# --------------------------------------------------------------------------------------
# Figure 5 — mention-ranking softmax over candidates for 'him' (MEASURED)
# --------------------------------------------------------------------------------------
def fig_mention_ranking() -> None:
    him_index = next(i for i, m in enumerate(MENTIONS) if m["text"] == "him")
    keys, probs = antecedent_distribution(MENTIONS, him_index)
    labels = ["ε\n(new entity)" if k == "ε" else MENTIONS[int(k)]["text"] for k in keys]
    # color the winner green, gender-clash candidates red, the rest slate; epsilon amber
    colors = []
    for k in keys:
        if k == "ε":
            colors.append(AMBER)
        else:
            colors.append(SLATE)
    winner = int(probs.argmax())
    colors[winner] = GREEN
    # mark gender-clashing candidates (She) in red
    for idx, k in enumerate(keys):
        if k != "ε" and MENTIONS[int(k)]["gender"] == "fem":
            colors[idx] = RED

    fig, ax = plt.subplots(figsize=(9.5, 4.6))
    bars = ax.bar(labels, probs, color=colors, edgecolor="#222", linewidth=0.8, zorder=3)
    for b, p in zip(bars, probs):
        ax.text(b.get_x() + b.get_width() / 2, p + 0.01, f"{p:.2f}",
                ha="center", fontsize=10, color=INK)
    ax.set_ylabel("P(antecedent of 'him')")
    ax.set_ylim(0, max(probs) * 1.25)
    ax.set_title("Mention-ranking: one softmax over every candidate antecedent of 'him' (incl. ε)",
                 fontsize=12.5, fontweight="bold")
    ax.text(0.5, 0.93,
            "green = chosen antecedent  ·  red = gender clash (crushed)  ·  amber = ε 'no antecedent'",
            transform=ax.transAxes, ha="center", fontsize=9.5, color="#444", style="italic")
    _style_axis(ax)
    _save(fig, "coref_mention_ranking.png")


# --------------------------------------------------------------------------------------
# Figure 6 — B-cubed over-split vs over-merge symmetry (MEASURED)
# --------------------------------------------------------------------------------------
def fig_bcubed_symmetry() -> None:
    # gold: one 4-mention entity {w,x,y,z}
    gold = [["w", "x", "y", "z"]]
    over_split = [["w", "x"], ["y", "z"]]          # splits one entity into two
    over_merge_gold = [["w", "x"], ["y", "z"]]      # two gold entities
    over_merge_pred = [["w", "x", "y", "z"]]        # merged into one

    sp = bcubed(gold, over_split)
    mg = bcubed(over_merge_gold, over_merge_pred)

    fig, ax = plt.subplots(figsize=(9.0, 4.8))
    groups = ["Over-SPLIT\n(one entity → two)", "Over-MERGE\n(two entities → one)"]
    x = np.arange(len(groups))
    w = 0.28
    precision = [sp[0], mg[0]]
    recall = [sp[1], mg[1]]
    f1 = [sp[2], mg[2]]
    ax.bar(x - w, precision, w, label="B³ Precision", color=BLUE, zorder=3)
    ax.bar(x, recall, w, label="B³ Recall", color=AMBER, zorder=3)
    ax.bar(x + w, f1, w, label="B³ F1", color=GREEN, zorder=3)
    for xi, vals in zip(x, [(sp), (mg)]):
        ax.text(xi - w, vals[0] + 0.02, f"{vals[0]:.2f}", ha="center", fontsize=9, color=BLUE)
        ax.text(xi, vals[1] + 0.02, f"{vals[1]:.2f}", ha="center", fontsize=9, color=AMBER)
        ax.text(xi + w, vals[2] + 0.02, f"{vals[2]:.2f}", ha="center", fontsize=9, color=GREEN)
    ax.set_xticks(x)
    ax.set_xticklabels(groups, fontsize=10.5)
    ax.set_ylim(0, 1.2)
    ax.set_ylabel("B³ score")
    ax.set_title("B³ recall punishes over-splitting; precision punishes over-merging (same F1)",
                 fontsize=12, fontweight="bold")
    ax.legend(frameon=False, fontsize=9.5, ncol=3, loc="upper center")
    _style_axis(ax)
    _save(fig, "coref_bcubed_symmetry.png")


# --------------------------------------------------------------------------------------
# Figure 7 — Winograd flip: world knowledge moves the antecedent (MEASURED)
# --------------------------------------------------------------------------------------
def fig_winograd() -> None:
    fig, axes = plt.subplots(1, 2, figsize=(12.0, 4.4))
    fig.suptitle("Winograd schema: flip one adjective and the antecedent of 'it' flips",
                 fontsize=13.5, fontweight="bold", y=1.02)

    for ax, adj in zip(axes, ("big", "small")):
        winner, scores = winograd_resolve(adj)
        cands = list(scores.keys())
        vals = [scores[c] for c in cands]
        colors = [GREEN if c == winner else MUTED for c in cands]
        bars = ax.bar(cands, vals, color=colors, edgecolor="#222", linewidth=0.8, zorder=3)
        for b, v in zip(bars, vals):
            ax.text(b.get_x() + b.get_width() / 2,
                    v + (0.06 if v >= 0 else -0.14),
                    f"{v:+.0f}", ha="center", fontsize=11, color=INK)
        ax.axhline(0, color=SLATE, lw=1.0)
        ax.set_ylim(-1.6, 1.6)
        ax.set_title(f"“...because it is too {adj}.”  →  it = {winner}",
                     fontsize=11.5, fontweight="bold",
                     color=(RED if adj == "big" else BLUE))
        ax.set_ylabel("world-knowledge score" if adj == "big" else "")
        _style_axis(ax)

    axes[0].text(0.5, -0.30,
                 "Structure & agreement give both candidates 0 (a tie); only the commonsense feature decides.",
                 transform=axes[0].transAxes, ha="left", fontsize=9.3, color="#444", style="italic")
    fig.tight_layout(rect=(0, 0, 1, 0.95))
    _save(fig, "coref_winograd.png")


# --------------------------------------------------------------------------------------
# Figure 8 — the field's accuracy arc over model families (illustrative)
# --------------------------------------------------------------------------------------
def fig_field_progress() -> None:
    # Representative CoNLL-2012 (OntoNotes English) test F1 by era. Illustrative reference
    # points from the literature, not recomputed here -- labelled illustrative on the page.
    systems = [
        ("Rule/feature\n(pre-2015)", 55, SLATE),
        ("Mention-pair\n(Soon 2001-era)", 60, NAVY),
        ("Mention-ranking\n(Clark-Manning 2016)", 65, AMBER),
        ("End-to-end\n(Lee 2017)", 67, PURPLE),
        ("+ ELMo\n(Lee 2018)", 73, BLUE),
        ("SpanBERT\n(Joshi 2020)", 80, GREEN),
    ]
    labels = [s[0] for s in systems]
    vals = [s[1] for s in systems]
    cols = [s[2] for s in systems]

    fig, ax = plt.subplots(figsize=(11.0, 4.8))
    x = np.arange(len(systems))
    bars = ax.bar(x, vals, color=cols, edgecolor="#222", linewidth=0.8, zorder=3, width=0.62)
    ax.plot(x, vals, color="#444", lw=1.3, marker="o", markersize=5, zorder=4, alpha=0.7)
    for b, v in zip(bars, vals):
        ax.text(b.get_x() + b.get_width() / 2, v + 0.8, f"{v}", ha="center", fontsize=10, color=INK)
    ax.set_xticks(x)
    ax.set_xticklabels(labels, fontsize=9.0)
    ax.set_ylim(40, 90)
    ax.set_ylabel("CoNLL F1 (OntoNotes English)")
    ax.set_title("Coreference accuracy rose with the encoder: the field's model-family arc",
                 fontsize=12.5, fontweight="bold")
    ax.text(0.015, 0.97,
            "Illustrative reference points from the literature — the architecture stayed\n"
            "(span-ranking); only the encoder improved (biLSTM → ELMo → SpanBERT).",
            transform=ax.transAxes, ha="left", va="top", fontsize=9.0, color="#555", style="italic")
    _style_axis(ax)
    _save(fig, "coref_field_progress.png")


def _print_spacy_mentions() -> None:
    det = spacy_mentions()
    print("\nspaCy mention detection on the running passage:")
    print("  NER         :", det["ner"])
    print("  pronouns    :", det["pronouns"])
    print("  noun chunks :", det["noun_chunks"])


if __name__ == "__main__":
    print(runtime_banner())
    fig_clusters()
    fig_model_families()
    fig_e2e_arch()
    fig_metrics()
    fig_mention_ranking()
    fig_bcubed_symmetry()
    fig_winograd()
    fig_field_progress()
    _print_spacy_mentions()
    print("\nWrote 8 PNGs to", OUT_DIR)
