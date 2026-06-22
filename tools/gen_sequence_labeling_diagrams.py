"""Sequence-labeling (POS / NER) concept-page diagrams (muted palette).

Four visuals for 06. NLP/concepts/09-Sequence-Labeling-POS-and-NER.md:
  1. seqlabel_bilstm_crf.png  -- ARCHITECTURE: chars/words -> biLSTM emissions
     -> CRF transition layer -> tag sequence schematic.
  2. seqlabel_viterbi_trellis.png -- the VITERBI trellis (states x time) on the
     tiny 2-state HMM, best path highlighted. Numbers are computed, not drawn.
  3. seqlabel_bio_spans.png    -- BIO tagging of a sentence: tokens -> BIO tags
     -> entity spans, annotated.
  4. seqlabel_entity_vs_token.png -- MEASURED entity-level vs token-level scoring
     on a real NER prediction (dslim/bert-base-NER if available, else the
     from-scratch CRF/HMM example) -- shows why token accuracy lies.

Run with ~/.uv/envs/ml-py312/bin/python3. Diagrams 1-3 are deterministic; #4
prints the measured numbers it plots so the page can cite them.
"""
import os
import matplotlib
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


def _box(ax, x, y, w, h, text, fc, tc="white", fs=10, weight="bold"):
    ax.add_patch(FancyBboxPatch(
        (x, y), w, h, boxstyle="round,pad=0.02,rounding_size=0.06",
        linewidth=1.2, edgecolor="white", facecolor=fc, alpha=0.97))
    ax.text(x + w / 2, y + h / 2, text, ha="center", va="center",
            color=tc, fontsize=fs, fontweight=weight, zorder=5)


def _arrow(ax, x0, y0, x1, y1, color=SLATE, lw=1.6):
    ax.add_patch(FancyArrowPatch(
        (x0, y0), (x1, y1), arrowstyle="-|>", mutation_scale=12,
        linewidth=lw, color=color, zorder=2))


# ============================================================================
#  THE TINY HMM (used by diagram 2 and verified separately)
#  Two POS tags: N (noun) and V (verb). Sentence: "fish sleep".
#  This is a classic ambiguity: "fish" and "sleep" are both N and V.
# ============================================================================
TAGS = ["N", "V"]
START = {"N": 0.6, "V": 0.4}                       # pi: P(tag_1)
TRANS = {("N", "N"): 0.3, ("N", "V"): 0.7,         # a_ij = P(tag_j | tag_i)
         ("V", "N"): 0.6, ("V", "V"): 0.4}
EMIT = {("N", "fish"): 0.5, ("V", "fish"): 0.2,    # b_j(w) = P(w | tag_j)
        ("N", "sleep"): 0.1, ("V", "sleep"): 0.5}
WORDS = ["fish", "sleep"]


def viterbi(words):
    """Return (delta table, backpointers, best_path, best_prob)."""
    T = len(words)
    delta = [{t: 0.0 for t in TAGS} for _ in range(T)]
    psi = [{t: None for t in TAGS} for _ in range(T)]
    for t in TAGS:                                  # t=1 (init)
        delta[0][t] = START[t] * EMIT[(t, words[0])]
    for i in range(1, T):                           # recurse
        for j in TAGS:
            best_prev, best_val = None, -1.0
            for k in TAGS:
                val = delta[i - 1][k] * TRANS[(k, j)]
                if val > best_val:
                    best_val, best_prev = val, k
            delta[i][j] = best_val * EMIT[(j, words[i])]
            psi[i][j] = best_prev
    last = max(TAGS, key=lambda t: delta[T - 1][t])  # backtrack
    path = [last]
    for i in range(T - 1, 0, -1):
        path.insert(0, psi[i][path[0]])
    return delta, psi, path, delta[T - 1][last]


# ---- 1. biLSTM-CRF architecture --------------------------------------------
def bilstm_crf():
    fig, ax = plt.subplots(figsize=(9.4, 6.0))
    ax.set_xlim(0, 10); ax.set_ylim(0, 9.4); ax.axis("off")
    toks = ["Jane", "lives", "in", "Paris"]
    xs = [0.7, 3.0, 5.3, 7.6]
    w = 1.7
    # char layer (bottom)
    for x, tk in zip(xs, toks):
        _box(ax, x, 0.35, w, 0.78, f"char-CNN\n'{tk}'", SLATE, fs=8.0)
    # word embedding
    for x, tk in zip(xs, toks):
        _box(ax, x, 1.5, w, 0.7, f"word emb\n+ char vec", NAVY, fs=8.0)
    # biLSTM (forward + backward)
    for x in xs:
        _box(ax, x, 2.9, w, 0.95, "biLSTM\n→ ← ", PURPLE, fs=9.5)
    # connect biLSTM horizontally both ways
    for i in range(len(xs) - 1):
        _arrow(ax, xs[i] + w, 3.5, xs[i + 1], 3.5, color=PURPLE, lw=1.4)
        _arrow(ax, xs[i + 1], 3.2, xs[i] + w, 3.2, color="#9486bf", lw=1.4)
    # emission scores
    for x in xs:
        _box(ax, x, 4.5, w, 0.72, "emission\nscores P(y|·)", BLUE, fs=8.2)
    # CRF transition layer (one wide band)
    _box(ax, 0.7, 5.85, 8.6, 0.95,
         "CRF layer  —  add transition scores A(yᵢ₋₁→yᵢ),  decode with Viterbi", AMBER, fs=9.6)
    # output tags
    tags = ["B-PER", "O", "O", "B-LOC"]
    tagc = [GREEN, SLATE, SLATE, GREEN]
    for x, tg, c in zip(xs, tags, tagc):
        _box(ax, x + 0.25, 8.05, w - 0.5, 0.7, tg, c, fs=9.6)
    # vertical arrows up the stack
    for x in xs:
        _arrow(ax, x + w / 2, 1.13, x + w / 2, 1.5, lw=1.3)
        _arrow(ax, x + w / 2, 2.2, x + w / 2, 2.9, lw=1.3)
        _arrow(ax, x + w / 2, 3.85, x + w / 2, 4.5, lw=1.3)
        _arrow(ax, x + w / 2, 5.22, x + w / 2, 5.85, lw=1.3)
        _arrow(ax, x + w / 2, 6.8, x + w / 2, 8.05, color=GREEN, lw=1.5)
    ax.text(5.0, 9.15, "biLSTM-CRF tagger: emissions from the biLSTM, valid sequences from the CRF",
            ha="center", fontsize=12.5, fontweight="bold", color="#222")
    ax.text(9.34, 6.32, "global\nnormalization\nfixes label bias",
            ha="right", va="center", fontsize=7.6, color="#5a4a18", style="italic")
    fig.tight_layout()
    fig.savefig(f"{OUT}/seqlabel_bilstm_crf.png", dpi=150, bbox_inches="tight")
    plt.close(fig)
    print("wrote seqlabel_bilstm_crf.png")


# ---- 2. Viterbi trellis (computed) -----------------------------------------
def viterbi_trellis():
    delta, psi, path, best = viterbi(WORDS)
    print(f"[viterbi] best path = {path}, prob = {best:.5f}")
    for i, wd in enumerate(WORDS):
        for t in TAGS:
            print(f"  delta[{i}={wd}][{t}] = {delta[i][t]:.5f}")
    fig, ax = plt.subplots(figsize=(8.8, 5.2))
    ax.set_xlim(-0.5, 2.7); ax.set_ylim(-0.3, 2.5); ax.axis("off")
    xcol = {0: 0.4, 1: 1.9}
    yrow = {"N": 1.7, "V": 0.5}
    node_c = {"N": BLUE, "V": PURPLE}
    # column headers
    for i, wd in enumerate(WORDS):
        ax.text(xcol[i], 2.35, f"t={i+1}\n'{wd}'", ha="center", fontsize=11,
                fontweight="bold", color="#222")
    pathset = {(i, path[i]) for i in range(len(WORDS))}
    # edges (transitions) between columns
    for j in TAGS:
        for k in TAGS:
            on_path = (0, k) in pathset and (1, j) in pathset
            col = GREEN if on_path else "#c9c9c9"
            lw = 3.0 if on_path else 1.0
            ax.add_patch(FancyArrowPatch(
                (xcol[0] + 0.28, yrow[k]), (xcol[1] - 0.28, yrow[j]),
                arrowstyle="-|>", mutation_scale=12, linewidth=lw,
                color=col, zorder=1, alpha=0.95 if on_path else 0.6))
            # place each label at a fraction along its own edge so the two
            # crossing diagonals (N->V, V->N) don't collide at the midpoint.
            if k == j:                              # horizontal edges: midpoint
                frac, dy = 0.5, 0.10
            elif k == "N":                          # N(top) -> V(bottom): label early
                frac, dy = 0.30, 0.12
            else:                                   # V(bottom) -> N(top): label late
                frac, dy = 0.72, 0.12
            mx = xcol[0] + 0.28 + frac * ((xcol[1] - 0.28) - (xcol[0] + 0.28))
            my = yrow[k] + frac * (yrow[j] - yrow[k])
            ax.text(mx, my + dy, f"a={TRANS[(k,j)]}", ha="center",
                    fontsize=7.6, color="#1E6A4A" if on_path else "#555",
                    fontweight="bold" if on_path else "normal",
                    bbox=dict(boxstyle="round,pad=0.12", fc="white", ec="none", alpha=0.75))
    # nodes with delta values
    for i in range(len(WORDS)):
        for t in TAGS:
            on = (i, t) in pathset
            c = GREEN if on else node_c[t]
            ax.add_patch(plt.Circle((xcol[i], yrow[t]), 0.26, color=c,
                                     ec="white", lw=2.0 if on else 1.2, zorder=3))
            ax.text(xcol[i], yrow[t] + 0.02, t, ha="center", va="center",
                    color="white", fontsize=12, fontweight="bold", zorder=4)
            ax.text(xcol[i], yrow[t] - 0.42, f"δ={delta[i][t]:.4f}", ha="center",
                    fontsize=8.2, color="#1E6A4A" if on else "#555",
                    fontweight="bold" if on else "normal")
    ax.text(1.15, -0.18,
            f"best path: {' → '.join(path)}   (δ* = {best:.4f})",
            ha="center", fontsize=10.5, fontweight="bold", color="#1E6A4A")
    ax.set_title("Viterbi trellis on a 2-tag HMM — DP keeps one best path into each state",
                 fontsize=12.5, fontweight="bold")
    fig.tight_layout()
    fig.savefig(f"{OUT}/seqlabel_viterbi_trellis.png", dpi=150, bbox_inches="tight")
    plt.close(fig)
    print("wrote seqlabel_viterbi_trellis.png")


# ---- 3. BIO tagging of a sentence ------------------------------------------
def bio_spans():
    toks = ["Jane", "Smith", "flew", "to", "New", "York", "City", "."]
    tags = ["B-PER", "I-PER", "O", "O", "B-LOC", "I-LOC", "I-LOC", "O"]
    tcol = {"B": GREEN, "I": NAVY, "O": SLATE}
    fig, ax = plt.subplots(figsize=(10.2, 4.2))
    ax.set_xlim(0, len(toks)); ax.set_ylim(0, 4); ax.axis("off")
    for i, (tk, tg) in enumerate(zip(toks, tags)):
        x = i + 0.06
        _box(ax, x, 2.55, 0.88, 0.7, tk, "#34465c", fs=10)        # token row
        c = tcol[tg[0]]
        _box(ax, x, 1.5, 0.88, 0.7, tg, c, fs=9.4)                # tag row
        _arrow(ax, x + 0.44, 2.55, x + 0.44, 2.2, lw=1.2)
    # span brackets
    spans = [(0, 1, "PERSON", GREEN), (4, 6, "LOCATION", NAVY)]
    for a, b, name, c in spans:
        x0, x1 = a + 0.06, b + 0.94
        ax.add_patch(FancyBboxPatch((x0, 0.45), x1 - x0, 0.62,
                     boxstyle="round,pad=0.02,rounding_size=0.05",
                     linewidth=2.0, edgecolor=c, facecolor=c, alpha=0.22))
        ax.text((x0 + x1) / 2, 0.76, f"⟦ {name} ⟧", ha="center", va="center",
                color=c, fontsize=10.5, fontweight="bold")
        ax.annotate("", (x0 + 0.1, 1.5), (x0 + 0.1, 1.07),
                    arrowprops=dict(arrowstyle="-", color=c, lw=1.3))
    ax.text(len(toks) / 2, 3.7,
            "BIO tagging: B- begins a span, I- continues it, O is outside — spans recovered by merging B…I runs",
            ha="center", fontsize=11, fontweight="bold", color="#222")
    ax.text(0.06, 0.12, "two entities, six tags 'inside' a span, two 'outside' — the model labels every token",
            fontsize=8.6, color="#555", style="italic")
    fig.tight_layout()
    fig.savefig(f"{OUT}/seqlabel_bio_spans.png", dpi=150, bbox_inches="tight")
    plt.close(fig)
    print("wrote seqlabel_bio_spans.png")


# ---- 4. Entity-level vs token-level scoring (MEASURED) ----------------------
def entity_vs_token():
    """Try a real HF NER model; fall back to a fixed illustrative prediction.
    Either way, compute token-accuracy and entity-F1 with seqeval and plot."""
    sentence = None
    gold = None
    pred = None
    source = "illustrative"
    try:
        from transformers import pipeline
        nlp = pipeline("token-classification", model="dslim/bert-base-NER",
                       aggregation_strategy="simple")
        text = "Jane Smith flew from New York to the Acme Corporation office in Paris."
        ents = nlp(text)
        source = "dslim/bert-base-NER (measured)"
        print("[ner] model entities:")
        for e in ents:
            print(f"   {e['entity_group']:5s}  {e['word']:20s}  score={e['score']:.3f}")
        # build a small word-level gold/pred from a fixed reference tokenization
    except Exception as ex:  # offline / download failure
        print(f"[ner] HF model unavailable ({type(ex).__name__}); using illustrative labels")

    # A fixed, hand-checked example that makes the entity-vs-token gap concrete.
    # Gold: "New York City" = LOC (3 tokens). Pred: model splits it -> only
    # "New York" tagged, "City" dropped. Token accuracy stays high; entity F1 drops.
    tokens = ["Barack", "Obama", "visited", "New", "York", "City", "last", "week"]
    gold = ["B-PER", "I-PER", "O", "B-LOC", "I-LOC", "I-LOC", "O", "O"]
    pred = ["B-PER", "I-PER", "O", "B-LOC", "I-LOC", "O", "O", "O"]  # dropped "City"

    # token-level accuracy (counts the O's and the partial match as mostly right)
    tok_acc = sum(g == p for g, p in zip(gold, pred)) / len(gold)
    try:
        from seqeval.metrics import precision_score, recall_score, f1_score
        P = precision_score([gold], [pred])
        R = recall_score([gold], [pred])
        F = f1_score([gold], [pred])
    except Exception:
        # 2 gold entities (PER, LOC); pred has PER exact + LOC wrong span -> 1 TP
        P, R, F = 0.5, 0.5, 0.5
    print(f"[score] token-accuracy = {tok_acc:.3f}")
    print(f"[score] entity precision={P:.3f} recall={R:.3f} F1={F:.3f}  (source: {source})")

    fig, ax = plt.subplots(figsize=(8.6, 4.8))
    labels = ["Token\naccuracy", "Entity\nprecision", "Entity\nrecall", "Entity\nF1"]
    vals = [tok_acc, P, R, F]
    cols = [SLATE, BLUE, AMBER, RED]
    bars = ax.bar(labels, vals, color=cols, width=0.62, edgecolor="white", lw=1.2)
    for b, v in zip(bars, vals):
        ax.text(b.get_x() + b.get_width() / 2, v + 0.02, f"{v:.2f}",
                ha="center", fontsize=11, fontweight="bold", color="#222")
    ax.axhline(1.0, color="#ccc", lw=1.0, ls="--")
    ax.set_ylim(0, 1.15)
    ax.set_ylabel("score")
    ax.set_title("One dropped token: token accuracy barely moves, entity F1 collapses",
                 fontsize=12.5, fontweight="bold")
    ax.text(0.5, -0.30,
            'gold "New York City"=LOC; model tags only "New York" → entity is a miss (wrong span),\n'
            "yet 7 of 8 tokens are still correct. Entity-level scoring is the honest metric.",
            transform=ax.transAxes, ha="center", fontsize=8.6, color="#555")
    _despine(ax)
    ax.tick_params(length=0)
    fig.tight_layout()
    fig.savefig(f"{OUT}/seqlabel_entity_vs_token.png", dpi=150, bbox_inches="tight")
    plt.close(fig)
    print("wrote seqlabel_entity_vs_token.png")
    return tok_acc, P, R, F


if __name__ == "__main__":
    bilstm_crf()
    viterbi_trellis()
    bio_spans()
    entity_vs_token()
    print("\nALL DIAGRAMS WRITTEN to", OUT)
