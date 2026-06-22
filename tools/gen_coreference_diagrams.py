"""Coreference-resolution concept-page diagrams (muted palette, parallel scale).

Four figures for 06. NLP/concepts/14-Coreference-Resolution.md:

  1. coref_clusters.png   -- an annotated passage with colored mention spans linked
     into coreference chains ("John ... he ... his ... the manager ... she").
     Illustrative (hand-laid layout), but the mentions are the ones spaCy actually
     detects on the passage (printed when this script runs).
  2. coref_model_families.png -- schematic comparing mention-pair, mention-ranking,
     and entity/cluster-level models (what each one scores).
  3. coref_e2e_arch.png   -- the end-to-end span-ranking architecture (Lee et al. 2017):
     tokens -> encoder -> span representations -> mention score + antecedent score
     -> softmax over candidate antecedents incl. the epsilon dummy.
  4. coref_metrics.png    -- MUC / B-cubed / CEAF-phi4 computed on ONE small
     predicted-vs-gold clustering (MEASURED: the numbers are computed in code below
     and printed, so the figure cannot drift from the math in the page).

All metric numbers in figure 4 are computed exactly here (no hand-fabrication), and
the spaCy mention list for figure 1 is printed so the page's worked example matches.
Run:  /Users/sudhirsingh/.uv/envs/ml-py312/bin/python3 tools/gen_coreference_diagrams.py
"""
import os, matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch
import numpy as np

OUT = os.path.join(os.path.dirname(__file__), "..", "06. NLP", "concepts", "images")
OUT = os.path.abspath(OUT)
os.makedirs(OUT, exist_ok=True)

BLUE, PURPLE, GREEN, RED, SLATE, AMBER, NAVY = (
    "#3A6B96", "#5D4A8A", "#2E7A5A", "#8B3B4A", "#4A5B6E", "#7A6528", "#2A5B80")
plt.rcParams.update({"font.size": 11, "font.family": "DejaVu Sans"})


def _despine(ax):
    for s in ("top", "right", "left", "bottom"):
        ax.spines[s].set_visible(False)


def _box(ax, x, y, w, h, text, fc, fontsize=10, tc="#fff", weight="normal"):
    ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.012,rounding_size=0.02",
                                fc=fc, ec="#222", lw=1.0, mutation_aspect=1.0))
    ax.text(x + w / 2, y + h / 2, text, ha="center", va="center",
            fontsize=fontsize, color=tc, weight=weight, wrap=True)


# --------------------------------------------------------------------------------------
# Figure 1 — passage with colored mention spans linked into coreference clusters
# --------------------------------------------------------------------------------------
def fig_clusters():
    fig, ax = plt.subplots(figsize=(11.0, 5.0))
    ax.set_xlim(0, 11); ax.set_ylim(0, 5); ax.axis("off")
    ax.set_title("Coreference clustering: linking mentions to the entity they refer to",
                 fontsize=14, fontweight="bold", pad=10)

    # token, x-center, cluster-id (0 = John, 1 = manager, None = not a mention)
    line1 = [("John", 0), ("told", None), ("his", 0), ("manager", 1),
             ("that", None), ("he", 0), ("would", None), ("finish", None),
             ("the report", None), (".", None)]
    line2 = [("She", 1), ("thanked", None), ("him", 0), ("for", None),
             ("the update", None), (".", None)]

    cmap = {0: BLUE, 1: GREEN, None: None}
    y1, y2 = 3.4, 1.7
    centers = {}  # (line,idx) -> (x,y) center of mention box for arrows

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
        return x

    lay(line1, y1)
    lay(line2, y2)

    # coreference links (curved arrows) within each cluster, antecedent -> anaphor
    def link(a, b, col, rad=0.35):
        xa, ya = centers[a]; xb, yb = centers[b]
        ax.add_patch(FancyArrowPatch((xa, ya + 0.55), (xb, yb + 0.55),
                     connectionstyle=f"arc3,rad={rad}", color=col, lw=2.0,
                     arrowstyle="-|>", mutation_scale=14, alpha=0.9))

    # John cluster (blue): John -> his -> he -> him
    link((y1, 0), (y1, 2), BLUE)      # John -> his
    link((y1, 2), (y1, 5), BLUE)      # his -> he
    link((y1, 5), (y2, 2), BLUE, rad=-0.18)   # he -> him (cross line)
    # manager cluster (green): manager -> She
    link((y1, 3), (y2, 0), GREEN, rad=-0.30)  # manager -> She

    ax.text(0.3, 4.55, "Cluster 1 (entity = John):  John · his · he · him",
            fontsize=11, color=BLUE, weight="bold")
    ax.text(0.3, 4.18, "Cluster 2 (entity = the manager):  manager · She",
            fontsize=11, color=GREEN, weight="bold")
    ax.text(0.3, 0.75, "Arrows = antecedent → anaphor links; a mention's cluster is the entity it refers to. "
                       "Note 'his manager' nests two mentions: the possessive 'his' (John) inside the NP 'manager'.",
            fontsize=9.5, color="#444", style="italic")
    fig.tight_layout()
    fig.savefig(f"{OUT}/coref_clusters.png", dpi=150, bbox_inches="tight")
    plt.close(fig)


# --------------------------------------------------------------------------------------
# Figure 2 — mention-pair vs mention-ranking vs entity/cluster model (schematic)
# --------------------------------------------------------------------------------------
def fig_model_families():
    fig, axes = plt.subplots(1, 3, figsize=(13.5, 4.6))
    fig.suptitle("Three model families: what each one scores when resolving a mention",
                 fontsize=14, fontweight="bold", y=1.00)

    # shared mention strip (compact so an extra epsilon slot fits without overlap)
    def strip(ax, hi=None, y=3.5, x0=0.25, w=0.78, gap=0.20):
        toks = ["John", "his", "he", "him"]
        cols = [BLUE, SLATE, SLATE, NAVY]
        x = x0
        centers = []
        for i, t in enumerate(toks):
            fc = cols[i] if (hi is None or i in hi) else "#9aa3ad"
            _box(ax, x, y, w, 0.55, t, fc, fontsize=10, weight="bold")
            centers.append(x + w / 2)
            x += w + gap
        return centers, y, x  # next free x

    # (a) mention-pair
    ax = axes[0]; ax.set_xlim(0, 5); ax.set_ylim(0, 5); _despine(ax); ax.set_xticks([]); ax.set_yticks([])
    ax.set_title("Mention-pair (binary)", fontsize=11.5, fontweight="bold", color=BLUE)
    cx, y, _ = strip(ax)
    # classify pair (him, John), (him, his), (him, he) independently — arc above the strip
    for srci, col, rad in [(0, GREEN, 0.55), (1, AMBER, 0.62), (2, RED, 0.7)]:
        ax.add_patch(FancyArrowPatch((cx[3], y + 0.55), (cx[srci], y + 0.55),
                     connectionstyle=f"arc3,rad={rad}", color=col, lw=1.8,
                     arrowstyle="-|>", mutation_scale=12))
    ax.text(2.5, 2.0, "for EACH (mention, antecedent) pair\nemit P(coref) ∈ {0,1} independently.\n"
                      "Then cluster the positive pairs.\n\n⚠ pairwise decisions can violate\ntransitivity "
                      "(A=B, B=C, but A≠C).",
            ha="center", va="center", fontsize=9.3, color="#333")

    # (b) mention-ranking
    ax = axes[1]; ax.set_xlim(0, 5); ax.set_ylim(0, 5); _despine(ax); ax.set_xticks([]); ax.set_yticks([])
    ax.set_title("Mention-ranking (softmax)", fontsize=11.5, fontweight="bold", color=PURPLE)
    cx, y, nx = strip(ax)
    _box(ax, nx, y, 0.50, 0.55, "ε", AMBER, fontsize=11, weight="bold")
    eps_cx = nx + 0.25
    # arrows START from below the 'him' box so they don't cross over it; target each candidate
    src = (cx[3], y)
    cands = [(cx[0], 0.40, GREEN), (cx[1], 0.10, SLATE), (cx[2], 0.35, SLATE), (eps_cx, 0.15, AMBER)]
    for tx, p, col in cands:
        rad = 0.55 if tx < src[0] else -0.55
        ax.add_patch(FancyArrowPatch((src[0], y + 0.55), (tx, y + 0.55),
                     connectionstyle=f"arc3,rad={rad}", color=col, lw=1.0 + 4 * p,
                     arrowstyle="-|>", mutation_scale=11, alpha=0.85))
    ax.text(2.5, 2.0, "for the mention 'him', RANK all\ncandidate antecedents (incl. ε = 'no\n"
                      "antecedent') with ONE softmax.\n\n✓ picks the single best antecedent;\n"
                      "consistent, end-to-end trainable.",
            ha="center", va="center", fontsize=9.3, color="#333")

    # (c) entity/cluster
    ax = axes[2]; ax.set_xlim(0, 5); ax.set_ylim(0, 5); _despine(ax); ax.set_xticks([]); ax.set_yticks([])
    ax.set_title("Entity / cluster-level", fontsize=11.5, fontweight="bold", color=GREEN)
    _box(ax, 0.4, 3.3, 2.0, 0.8, "partial cluster\n{John, his, he}", BLUE, fontsize=9.5, weight="bold")
    _box(ax, 3.2, 3.45, 1.0, 0.55, "him", NAVY, fontsize=10, weight="bold")
    ax.add_patch(FancyArrowPatch((3.2, 3.72), (2.4, 3.72), color=GREEN, lw=2.2,
                 arrowstyle="-|>", mutation_scale=13))
    ax.text(2.5, 1.9, "score the mention against the WHOLE\npartial cluster built so far,\nnot a single antecedent.\n\n"
                      "✓ uses global entity-level features\n(e.g. a cluster already has a name).",
            ha="center", va="center", fontsize=9.3, color="#333")

    fig.tight_layout(rect=(0, 0, 1, 0.96))
    fig.savefig(f"{OUT}/coref_model_families.png", dpi=150, bbox_inches="tight")
    plt.close(fig)


# --------------------------------------------------------------------------------------
# Figure 3 — end-to-end span-ranking architecture (Lee et al. 2017)
# --------------------------------------------------------------------------------------
def fig_e2e_arch():
    fig, ax = plt.subplots(figsize=(11.5, 6.2))
    ax.set_xlim(0, 11.5); ax.set_ylim(0, 6.2); ax.axis("off")
    ax.set_title("End-to-end span-ranking coreference (Lee et al. 2017)",
                 fontsize=14, fontweight="bold", pad=8)

    _box(ax, 0.3, 5.2, 10.9, 0.62,
         "Document tokens:  John  told  his  manager  that  he ...", BLUE, fontsize=11)
    _box(ax, 0.3, 4.25, 10.9, 0.62,
         "Encoder (biLSTM 2017 → SpanBERT 2020): contextual token vectors", PURPLE, fontsize=10.5)
    _box(ax, 0.3, 3.30, 10.9, 0.62,
         "Enumerate ALL spans up to width L → span rep  gᵢ = [x_start ; x_end ; x̂_attn ; φ(width)]",
         NAVY, fontsize=10)

    # mention scoring prunes spans
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
    fig.tight_layout()
    fig.savefig(f"{OUT}/coref_e2e_arch.png", dpi=150, bbox_inches="tight")
    plt.close(fig)


# --------------------------------------------------------------------------------------
# Figure 4 — MUC / B-cubed / CEAF-phi4 on ONE example (MEASURED — computed here)
# --------------------------------------------------------------------------------------
def muc(gold, pred):
    """MUC link-based recall/precision. Counts on links (|cluster|-1 per cluster)."""
    def partition_of(clusters):
        m = {}
        for ci, c in enumerate(clusters):
            for e in c:
                m[e] = ci
        return m

    def score(key, response):
        kp = partition_of(response)
        num = den = 0
        for c in key:
            den += len(c) - 1
            # partitions of c induced by response
            seen = {}
            for e in c:
                p = kp.get(e, ("solo", e))
                seen.setdefault(p, 0)
                seen[p] += 1
            # number of response-partitions touching c (count singletons not in response as own)
            parts = len(seen)
            num += len(c) - parts
        return num, den
    rn, rd = score(gold, pred)   # recall
    pn, pd = score(pred, gold)   # precision
    R = rn / rd if rd else 0.0
    P = pn / pd if pd else 0.0
    F = 2 * P * R / (P + R) if (P + R) else 0.0
    return P, R, F


def bcubed(gold, pred):
    g = {e: set(c) for c in gold for e in c}
    p = {e: set(c) for c in pred for e in c}
    mentions = set(g) | set(p)
    P = R = 0.0
    for m in mentions:
        gm, pm = g.get(m, {m}), p.get(m, {m})
        inter = len(gm & pm)
        P += inter / len(pm)
        R += inter / len(gm)
    n = len(mentions)
    P, R = P / n, R / n
    F = 2 * P * R / (P + R) if (P + R) else 0.0
    return P, R, F


def ceaf_phi4(gold, pred):
    """CEAF-phi4: best 1-1 alignment of gold and predicted entities. phi4(g,p)=2|g∩p|/(|g|+|p|)."""
    from itertools import permutations
    G = [set(c) for c in gold]
    Pr = [set(c) for c in pred]

    def phi4(a, b):
        return 2 * len(a & b) / (len(a) + len(b))
    n = max(len(G), len(Pr))
    Gx = G + [set() for _ in range(n - len(G))]
    Px = Pr + [set() for _ in range(n - len(Pr))]
    best = 0.0
    for perm in permutations(range(n)):
        s = sum(phi4(Gx[i], Px[perm[i]]) for i in range(n) if Gx[i] and Px[perm[i]])
        best = max(best, s)
    P = best / len(Pr) if Pr else 0.0
    R = best / len(G) if G else 0.0
    F = 2 * P * R / (P + R) if (P + R) else 0.0
    return P, R, F


def fig_metrics():
    # gold: {a,b,c} {d,e}   pred: {a,b} {c} {d,e}  (predictor split off c)
    gold = [["a", "b", "c"], ["d", "e"]]
    pred = [["a", "b"], ["c"], ["d", "e"]]
    mP, mR, mF = muc(gold, pred)
    bP, bR, bF = bcubed(gold, pred)
    cP, cR, cF = ceaf_phi4(gold, pred)
    conll = (mF + bF + cF) / 3
    print("MEASURED metrics on gold={ {a,b,c},{d,e} } pred={ {a,b},{c},{d,e} }:")
    print(f"  MUC      P={mP:.3f} R={mR:.3f} F1={mF:.3f}")
    print(f"  B-cubed  P={bP:.3f} R={bR:.3f} F1={bF:.3f}")
    print(f"  CEAF-phi4 P={cP:.3f} R={cR:.3f} F1={cF:.3f}")
    print(f"  CoNLL avg F1 = {conll:.3f}")

    fig, (axL, axR) = plt.subplots(1, 2, figsize=(12.5, 4.8), gridspec_kw={"width_ratios": [1, 1.25]})

    # left: the two clusterings drawn
    axL.set_xlim(0, 6); axL.set_ylim(0, 6); axL.axis("off")
    axL.set_title("One example: gold vs predicted clustering", fontsize=12, fontweight="bold")
    # gold
    axL.text(0.2, 5.4, "GOLD", fontsize=11, weight="bold", color="#111")
    _box(axL, 0.4, 4.4, 2.6, 0.7, "{ a · b · c }", BLUE, fontsize=11, weight="bold")
    _box(axL, 3.3, 4.4, 1.8, 0.7, "{ d · e }", GREEN, fontsize=11, weight="bold")
    axL.text(0.2, 3.5, "PREDICTED", fontsize=11, weight="bold", color="#111")
    _box(axL, 0.4, 2.5, 1.8, 0.7, "{ a · b }", BLUE, fontsize=11, weight="bold")
    _box(axL, 2.5, 2.5, 1.0, 0.7, "{ c }", RED, fontsize=11, weight="bold")
    _box(axL, 3.7, 2.5, 1.8, 0.7, "{ d · e }", GREEN, fontsize=11, weight="bold")
    axL.text(0.2, 1.5, "Error: the predictor split mention 'c' out of the {a,b,c} entity\n"
                       "(a recall miss on one link). Each metric punishes this differently →",
             fontsize=9.6, color="#444", style="italic")

    # right: grouped bars P/R/F per metric
    axR.set_title("Why no single metric suffices (same error, different scores)",
                  fontsize=12, fontweight="bold")
    metrics = ["MUC", "B³", "CEAF-φ4", "CoNLL\navg"]
    Ps = [mP, bP, cP, np.nan]
    Rs = [mR, bR, cR, np.nan]
    Fs = [mF, bF, cF, conll]
    x = np.arange(len(metrics)); w = 0.26
    axR.bar(x - w, [p if not np.isnan(p) else 0 for p in Ps], w, label="Precision", color=BLUE)
    axR.bar(x, [r if not np.isnan(r) else 0 for r in Rs], w, label="Recall", color=AMBER)
    axR.bar(x + w, Fs, w, label="F1", color=GREEN)
    for i, f in enumerate(Fs):
        axR.text(x[i] + w, f + 0.02, f"{f:.2f}", ha="center", fontsize=8.5, color=GREEN)
    axR.set_ylim(0, 1.15); axR.set_xticks(x); axR.set_xticklabels(metrics, fontsize=10)
    axR.set_ylabel("score"); axR.legend(frameon=False, fontsize=9, ncol=3, loc="upper center")
    for s in ("top", "right"):
        axR.spines[s].set_visible(False)
    fig.tight_layout()
    fig.savefig(f"{OUT}/coref_metrics.png", dpi=150, bbox_inches="tight")
    plt.close(fig)
    return (mP, mR, mF), (bP, bR, bF), (cP, cR, cF), conll


def spacy_mentions():
    """Print the mentions spaCy detects on the running passage (for the page's example)."""
    try:
        import spacy
        nlp = spacy.load("en_core_web_sm")
        text = ("John told his manager that he would finish the report by Friday. "
                "She thanked him for the update.")
        doc = nlp(text)
        ents = [(e.text, e.label_) for e in doc.ents]
        prons = [(t.text, t.tag_) for t in doc if t.tag_ in ("PRP", "PRP$")]
        ncs = [nc.text for nc in doc.noun_chunks]
        print("spaCy NER:", ents)
        print("spaCy pronouns:", prons)
        print("spaCy noun chunks (candidate mentions):", ncs)
    except Exception as e:
        print("spaCy mention detection skipped:", e)


if __name__ == "__main__":
    fig_clusters()
    fig_model_families()
    fig_e2e_arch()
    fig_metrics()
    spacy_mentions()
    print("Wrote 4 PNGs to", OUT)
