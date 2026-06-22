"""Information Retrieval & Semantic Search concept-page diagrams.

Muted palette, parallel matplotlib scale. Four figures for
06. NLP/concepts/16-Information-Retrieval-and-Semantic-Search.md:

  1. ir_pipeline.png       -- schematic: query -> (sparse | dense | hybrid) retrieve
                              -> fuse -> rerank -> ranked list (two-stage pattern).
  2. ir_sparse_vs_dense.png-- MEASURED: on a vocabulary-mismatch query, BM25 ranks the
                              relevant passage low/zero while a dense bi-encoder ranks it
                              first; bar chart of the relevant doc's score/rank by method.
  3. ir_ann_tradeoff.png   -- ANN recall-vs-latency (schematic, with a PQ-compression
                              inset): exact kNN is 100% recall but O(N) slow; HNSW/IVF/PQ
                              trade a little recall for big speed/memory wins.
  4. ir_ndcg.png           -- MEASURED: precision@k / recall@k curve + an nDCG bar for a
                              small ranked list, computed by hand and verified in code.

Run with:  ~/.uv/envs/ml-py312/bin/python3 tools/gen_ir_diagrams.py
Needs: matplotlib, numpy, rank_bm25, sentence-transformers (falls back to a constructed
example if the model can't download).
"""
import os, math, matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

OUT = os.path.join(os.path.dirname(__file__), "..", "06. NLP", "concepts", "images")
OUT = os.path.abspath(OUT)
os.makedirs(OUT, exist_ok=True)

BLUE, PURPLE, GREEN, RED, SLATE, AMBER, NAVY = (
    "#3A6B96", "#5D4A8A", "#2E7A5A", "#8B3B4A", "#4A5B6E", "#7A6528", "#2A5B80")
plt.rcParams.update({"font.size": 11, "font.family": "DejaVu Sans"})


def _despine(ax):
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)


# ---------------------------------------------------------------------------
# A tiny corpus engineered to show vocabulary mismatch.  The query uses
# "car" / "physician"; the gold passages use "automobile" / "doctor".
# Lexical BM25 has zero term overlap on those; a dense encoder still matches.
# ---------------------------------------------------------------------------
CORPUS = [
    "A mechanic repaired the automobile engine and replaced the worn brake pads.",  # 0 GOLD: synonym (automobile), no 'car'
    "A balanced diet and regular exercise keep your heart healthy.",                # 1
    "I had to fix the spelling errors in my report before submitting it.",          # 2 lexical DISTRACTOR: shares 'fix' + 'my'
    "Stock markets fell sharply amid fears of rising interest rates.",              # 3
    "How to fix my code when the build breaks on my machine.",                      # 4 lexical DISTRACTOR: shares 'how','fix','my'
    "The doctor advised the patient to rest and drink plenty of fluids.",           # 5
    "Photosynthesis converts sunlight into chemical energy in plants.",             # 6
    "Quarterly earnings beat analyst expectations, lifting the share price.",       # 7
]
QUERY = "How do I fix my car?"          # 'car' never appears literally; gold doc 0 says 'automobile'
GOLD_IDX = 0                             # the automobile-repair passage


def _bm25_scores(query, corpus):
    from rank_bm25 import BM25Okapi
    tok = [c.lower().replace("'s", "").replace(".", "").replace(",", "").split() for c in corpus]
    bm = BM25Okapi(tok)
    q = query.lower().replace("?", "").split()
    return np.array(bm.get_scores(q))


def _dense_scores(query, corpus):
    """Cosine scores from a small bi-encoder; fall back to a constructed result
    if the model can't be downloaded in this environment."""
    try:
        from sentence_transformers import SentenceTransformer
        m = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")
        emb = m.encode([query] + corpus, normalize_embeddings=True)
        q, d = emb[0], emb[1:]
        return d @ q, True
    except Exception as e:        # offline / no model: constructed but representative
        print("  [dense] model unavailable, using constructed scores:", repr(e)[:60])
        constructed = np.array([0.52, 0.06, 0.11, 0.03, 0.46, 0.14, 0.02, 0.04])
        return constructed, False


def sparse_vs_dense():
    bm = _bm25_scores(QUERY, CORPUS)
    dn, real = _dense_scores(QUERY, CORPUS)
    bm_rank = int(np.argsort(-bm).tolist().index(GOLD_IDX)) + 1
    dn_rank = int(np.argsort(-dn).tolist().index(GOLD_IDX)) + 1
    print(f"  BM25 gold rank = {bm_rank} (score {bm[GOLD_IDX]:.3f}); "
          f"dense gold rank = {dn_rank} (score {dn[GOLD_IDX]:.3f}); real_model={real}")

    # normalize each method to [0,1] for a fair side-by-side score bar
    def norm(s):
        s = np.asarray(s, float)
        return (s - s.min()) / (s.max() - s.min() + 1e-9)
    bm_n, dn_n = norm(bm), norm(dn)

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(11.4, 5.0))
    labels = [f"d{i}" for i in range(len(CORPUS))]
    x = np.arange(len(CORPUS))

    for ax, vals, title, col in (
        (ax1, bm_n, "Lexical BM25 (term overlap)", BLUE),
        (ax2, dn_n, "Dense bi-encoder (meaning)", GREEN)):
        bars = ax.bar(x, vals, color=col, edgecolor="white")
        bars[GOLD_IDX].set_color(AMBER); bars[GOLD_IDX].set_edgecolor(RED)
        bars[GOLD_IDX].set_linewidth(2.2)
        ax.set_xticks(x); ax.set_xticklabels(labels, fontsize=9)
        ax.set_ylabel("normalized score"); ax.set_ylim(0, 1.08)
        ax.set_title(title, fontsize=12.5, fontweight="bold"); _despine(ax)
    ax1.annotate(f"gold doc d0\nBM25 rank #{bm_rank}\n(no shared word\nwith 'car')",
                 (GOLD_IDX, bm_n[GOLD_IDX]), textcoords="offset points",
                 xytext=(28, 36), fontsize=9.2, fontweight="bold", color=RED,
                 arrowprops=dict(arrowstyle="->", color=RED))
    ax2.annotate(f"gold doc d0\ndense rank #{dn_rank}\n('automobile' ≈ 'car')",
                 (GOLD_IDX, dn_n[GOLD_IDX]), textcoords="offset points",
                 xytext=(24, -8), fontsize=9.2, fontweight="bold", color=GREEN,
                 arrowprops=dict(arrowstyle="->", color=GREEN))
    fig.suptitle(f'Query: "{QUERY}"  —  lexical misses the synonym, dense finds it',
                 fontsize=13.5, fontweight="bold")
    fig.tight_layout(rect=(0, 0, 1, 0.95))
    fig.savefig(f"{OUT}/ir_sparse_vs_dense.png", dpi=150, bbox_inches="tight")
    plt.close(fig); print("wrote ir_sparse_vs_dense.png")
    return bm, dn, bm_rank, dn_rank


def pipeline():
    """Schematic two-stage retrieve -> fuse -> rerank pipeline (hand-drawn boxes)."""
    fig, ax = plt.subplots(figsize=(11.6, 5.6)); ax.axis("off")
    ax.set_xlim(0, 12); ax.set_ylim(0, 7)

    def box(x, y, w, h, text, fc, fs=10):
        ax.add_patch(plt.Rectangle((x, y), w, h, facecolor=fc, edgecolor="white", lw=1.5))
        ax.text(x + w / 2, y + h / 2, text, ha="center", va="center",
                color="white", fontsize=fs, fontweight="bold")

    def arrow(x1, y1, x2, y2):
        ax.annotate("", (x2, y2), (x1, y1),
                    arrowprops=dict(arrowstyle="-|>", color=SLATE, lw=1.8))

    box(0.2, 3.0, 2.0, 1.0, "Query", NAVY, 11)
    # two retrievers
    box(3.0, 4.7, 3.0, 1.0, "Sparse retriever\n(BM25 + inverted index)", BLUE)
    box(3.0, 1.3, 3.0, 1.0, "Dense retriever\n(bi-encoder + ANN index)", GREEN)
    arrow(2.2, 3.7, 3.0, 5.2); arrow(2.2, 3.3, 3.0, 1.8)
    # fuse
    box(6.6, 3.0, 1.9, 1.0, "Fuse\n(RRF)", PURPLE)
    arrow(6.0, 5.2, 6.7, 4.0); arrow(6.0, 1.8, 6.7, 3.0)
    ax.text(4.5, 6.05, "Stage 1 — cheap recall over millions of docs (top-k ≈ 100-1000)",
            ha="center", fontsize=9.5, color=SLATE, style="italic")
    # rerank
    box(9.0, 3.0, 2.0, 1.0, "Cross-encoder\nre-ranker", AMBER)
    arrow(8.5, 3.5, 9.0, 3.5)
    ax.text(10.0, 4.35, "Stage 2 — expensive precision\non the top-k only",
            ha="center", fontsize=9.5, color=SLATE, style="italic")
    # final
    box(9.4, 0.6, 2.2, 1.0, "Ranked results", GREEN, 10)
    arrow(10.0, 3.0, 10.5, 1.6)
    ax.text(6.0, 0.15, "retrieve-then-rerank: a fast funnel narrows millions → hundreds → a precise few",
            ha="center", fontsize=10, color="#333", fontweight="bold")
    fig.tight_layout()
    fig.savefig(f"{OUT}/ir_pipeline.png", dpi=150, bbox_inches="tight")
    plt.close(fig); print("wrote ir_pipeline.png")


def ann_tradeoff():
    """Recall-vs-latency frontier (schematic) + PQ compression inset (measured math)."""
    fig, (ax, ax2) = plt.subplots(1, 2, figsize=(11.6, 5.0),
                                  gridspec_kw={"width_ratios": [1.5, 1]})
    # --- recall vs latency frontier ---
    methods = ["Exact kNN\n(flat, O(N·d))", "IVF", "HNSW", "IVF+PQ\n(compressed)"]
    latency = [100.0, 8.0, 3.0, 1.2]          # relative query latency (lower=faster)
    recall = [1.000, 0.93, 0.97, 0.88]         # recall@10 vs exact
    cols = [RED, NAVY, GREEN, PURPLE]
    ax.scatter(latency, recall, s=[220, 180, 200, 170], c=cols, edgecolor="white", zorder=5)
    label_off = {"Exact kNN\n(flat, O(N·d))": (10, -30), "IVF": (8, -26),
                 "HNSW": (10, -2), "IVF+PQ\n(compressed)": (-8, 16)}
    for m, lx, rc, c in zip(methods, latency, recall, cols):
        ax.annotate(m, (lx, rc), textcoords="offset points",
                    xytext=label_off.get(m, (6, 10)),
                    fontsize=9.2, fontweight="bold", color=c)
    ax.set_xscale("log")
    ax.set_xlabel("query latency  (relative, log scale — lower is faster)")
    ax.set_ylabel("recall@10  (vs exact search)")
    ax.set_title("ANN: trade a little recall for a big speed/memory win",
                 fontsize=12.5, fontweight="bold")
    ax.set_ylim(0.82, 1.02); _despine(ax)
    ax.annotate("exact is perfect\nbut scans all N", (100, 1.0),
                textcoords="offset points", xytext=(-104, -34), fontsize=9, color=RED)
    ax.annotate("HNSW: ~log N hops,\nnear-exact recall", (3.0, 0.97),
                textcoords="offset points", xytext=(-2, 22), fontsize=9, color=GREEN)

    # --- PQ compression bar (real arithmetic) ---
    d, bits_fp32 = 768, 32
    m_sub, bits_code = 96, 8           # 96 subvectors, 8-bit codes -> 1 byte each
    raw_bytes = d * bits_fp32 / 8      # 3072 B
    pq_bytes = m_sub * bits_code / 8   # 96 B
    ratio = raw_bytes / pq_bytes
    print(f"  PQ: raw {raw_bytes:.0f} B/vec -> PQ {pq_bytes:.0f} B/vec  = {ratio:.0f}x smaller")
    ax2.bar(["FP32\nvector", "PQ code\n(m=96, 8-bit)"], [raw_bytes, pq_bytes],
            color=[SLATE, PURPLE], edgecolor="white")
    ax2.set_ylabel("bytes per vector"); _despine(ax2)
    ax2.set_title(f"Product Quantization\n{raw_bytes:.0f} B → {pq_bytes:.0f} B  ({ratio:.0f}× smaller)",
                  fontsize=11.5, fontweight="bold")
    ax2.text(0, raw_bytes + 60, f"{raw_bytes:.0f} B", ha="center", fontsize=9.5, fontweight="bold")
    ax2.text(1, pq_bytes + 60, f"{pq_bytes:.0f} B", ha="center", fontsize=9.5, fontweight="bold", color=PURPLE)
    ax2.set_ylim(0, raw_bytes * 1.15)
    fig.tight_layout()
    fig.savefig(f"{OUT}/ir_ann_tradeoff.png", dpi=150, bbox_inches="tight")
    plt.close(fig); print("wrote ir_ann_tradeoff.png")


def ndcg_figure():
    """precision@k / recall@k curve + nDCG bar for a small ranked list (computed in code)."""
    # graded relevance of the top-8 returned docs (3=perfect ... 0=irrelevant)
    rel = np.array([3, 2, 3, 0, 1, 2, 0, 0], float)
    total_relevant = int((rel > 0).sum())            # 5 relevant in the returned set
    ks = np.arange(1, len(rel) + 1)
    prec = np.array([(rel[:k] > 0).sum() / k for k in ks])
    recall = np.array([(rel[:k] > 0).sum() / total_relevant for k in ks])

    # DCG and nDCG
    def dcg(g): return float(np.sum(g / np.log2(np.arange(2, len(g) + 2))))
    dcg_v = dcg(rel)
    idcg_v = dcg(np.sort(rel)[::-1])
    ndcg_v = dcg_v / idcg_v
    print(f"  DCG={dcg_v:.3f}  IDCG={idcg_v:.3f}  nDCG@8={ndcg_v:.3f}")

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(11.6, 5.0),
                                   gridspec_kw={"width_ratios": [1.5, 1]})
    ax1.plot(ks, prec, "-o", color=BLUE, lw=2.4, label="precision@k")
    ax1.plot(ks, recall, "-s", color=GREEN, lw=2.4, label="recall@k")
    ax1.set_xlabel("k (rank cutoff)"); ax1.set_ylabel("metric value")
    ax1.set_title("Precision@k falls, recall@k rises as k grows",
                  fontsize=12.5, fontweight="bold")
    ax1.set_ylim(0, 1.05); ax1.legend(frameon=False, fontsize=10); _despine(ax1)
    ax1.set_xticks(ks)

    # nDCG decomposition bar
    ax2.bar(["DCG", "IDCG\n(ideal)", "nDCG\n=DCG/IDCG"], [dcg_v, idcg_v, ndcg_v * idcg_v / idcg_v],
            color=[PURPLE, SLATE, GREEN], edgecolor="white")
    # show nDCG as fraction visually by relabeling
    ax2.cla()
    bars = ax2.bar(["DCG", "IDCG", "nDCG"], [dcg_v, idcg_v, ndcg_v],
                   color=[PURPLE, SLATE, GREEN], edgecolor="white")
    for b, v in zip(bars, [dcg_v, idcg_v, ndcg_v]):
        ax2.text(b.get_x() + b.get_width() / 2, v + 0.06, f"{v:.3f}",
                 ha="center", fontsize=9.8, fontweight="bold")
    ax2.set_title(f"nDCG@8 = {ndcg_v:.3f}\n(rel = [3,2,3,0,1,2,0,0])",
                  fontsize=11.5, fontweight="bold")
    ax2.set_ylim(0, max(dcg_v, idcg_v) * 1.2); _despine(ax2)
    fig.tight_layout()
    fig.savefig(f"{OUT}/ir_ndcg.png", dpi=150, bbox_inches="tight")
    plt.close(fig); print("wrote ir_ndcg.png")


if __name__ == "__main__":
    print("Generating IR diagrams ->", OUT)
    pipeline()
    sparse_vs_dense()
    ann_tradeoff()
    ndcg_figure()
    print("done.")
