"""Question-Answering concept-page diagrams (muted palette, parallel matplotlib scale).

Four visuals for 06. NLP/concepts/11-Question-Answering.md:
  1. qa_taxonomy.png        -- the MAP: QA by answer type (extractive / abstractive /
     multiple-choice / yes-no) x knowledge source (reading-comp / closed-book / open-domain).
  2. qa_span_head.png       -- the MODEL: BERT span head -- start & end probability
     distributions over real passage tokens (MEASURED from a HF SQuAD model).
  3. qa_retrieve_read.png   -- the PIPELINE: open-domain retrieve-then-read / RAG, schematic.
  4. qa_em_f1.png           -- the METRIC: Exact Match vs token-level F1 on worked
     prediction/gold pairs (computed here, exact).

Run:  ~/.uv/envs/ml-py312/bin/python3 tools/gen_qa_diagrams.py
If the SQuAD model can't be downloaded, diagram 2 falls back to hand-constructed
logits (noted in the page); diagrams 1, 3, 4 need no model.
"""
import os, warnings
warnings.filterwarnings("ignore")
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
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)


def _box(ax, x, y, w, h, text, fc, fs=10, weight="bold", tc="white"):
    ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.012,rounding_size=0.02",
                                fc=fc, ec="white", lw=1.4, zorder=2))
    ax.text(x + w / 2, y + h / 2, text, ha="center", va="center",
            fontsize=fs, color=tc, fontweight=weight, zorder=3)


def _arrow(ax, x0, y0, x1, y1, color="#444", lw=2.0):
    ax.add_patch(FancyArrowPatch((x0, y0), (x1, y1), arrowstyle="-|>",
                 mutation_scale=15, color=color, lw=lw, zorder=1))


# ---- 1. QA taxonomy: answer type x knowledge source -------------------------
def taxonomy():
    fig, ax = plt.subplots(figsize=(11.4, 6.0))
    ax.set_xlim(0, 11.4); ax.set_ylim(0, 6.6); ax.axis("off")
    ax.text(6.4, 6.35, "The QA design space: answer type x knowledge source",
            ha="center", fontsize=14, fontweight="bold", color="#222")

    # column headers = knowledge source (grid starts at x=2.5)
    sources = [("Reading comprehension\n(answer FROM a passage)", BLUE),
               ("Closed-book\n(answer from PARAMETERS)", PURPLE),
               ("Open-domain\n(retrieve from a CORPUS)", GREEN)]
    cx = [3.85, 6.85, 9.85]
    for (label, c), x in zip(sources, cx):
        _box(ax, x - 1.35, 5.2, 2.7, 0.78, label, c, fs=9.5)

    # rows = answer type, labelled to the LEFT of the grid (own narrow boxes)
    rows = [("Extractive\n(predict\na SPAN)", 4.05),
            ("Abstractive\n(GENERATE\ntext)", 2.9),
            ("Multiple-choice\nYes-No\n(CLASSIFY)", 1.75)]
    for label, y in rows:
        _box(ax, 0.15, y, 2.05, 0.85, label, SLATE, fs=8.6)

    # cells = canonical example at the intersection
    cells = {
        (0, 0): "SQuAD 1.1\nBERT span head", (0, 1): "(rare:\nno passage)", (0, 2): "DrQA / DPR\nretrieve+span",
        (1, 0): "narrative QA\n(gen from text)", (1, 1): "T5 / GPT\nclosed-book", (1, 2): "RAG\nretrieve+generate",
        (2, 0): "RACE / BoolQ\n(from passage)", (2, 1): "MMLU\n(parametric)", (2, 2): "open-domain\nMC (rare)"}
    colors = {0: BLUE, 1: PURPLE, 2: GREEN}
    for (ri, ci), txt in cells.items():
        x = cx[ci] - 1.35; y = rows[ri][1]
        faint = "(rare" in txt
        _box(ax, x, y, 2.7, 0.85, txt, "#cfd6dd" if faint else colors[ci],
             fs=8.4, tc="#444" if faint else "white", weight="normal" if faint else "bold")

    ax.text(6.4, 0.55, "Same task, three knowledge sources x four answer shapes -- the cell picks the architecture.",
            ha="center", fontsize=9.5, color="#555", style="italic")
    fig.tight_layout(); fig.savefig(f"{OUT}/qa_taxonomy.png", dpi=150, bbox_inches="tight")
    plt.close(fig); print("wrote qa_taxonomy.png")


# ---- 2. BERT span head: measured start/end distributions --------------------
def span_head():
    q = "Where is the Eiffel Tower located?"
    passage = ("The Eiffel Tower is a wrought-iron lattice tower on the Champ de Mars "
               "in Paris, France. It was completed in 1889.")
    tokens, p_start, p_end, measured = None, None, None, False
    try:
        from transformers import AutoTokenizer, AutoModelForQuestionAnswering
        import torch, torch.nn.functional as F
        name = "distilbert-base-cased-distilled-squad"
        tok = AutoTokenizer.from_pretrained(name)
        model = AutoModelForQuestionAnswering.from_pretrained(name)
        inp = tok(q, passage, return_tensors="pt")
        with torch.no_grad():
            out = model(**inp)
        ids = inp["input_ids"][0]
        toks = tok.convert_ids_to_tokens(ids)
        # restrict to the passage tokens (after first [SEP]) for a readable plot
        sep = [i for i, t in enumerate(toks) if t == "[SEP]"]
        lo = sep[0] + 1; hi = sep[1] if len(sep) > 1 else len(toks)
        ps = F.softmax(out.start_logits[0], dim=-1).numpy()[lo:hi]
        pe = F.softmax(out.end_logits[0], dim=-1).numpy()[lo:hi]
        tokens = [t.replace("##", "") for t in toks[lo:hi]]
        p_start, p_end, measured = ps, pe, True
    except Exception as ex:
        print("  (span_head) model unavailable, using constructed logits:", type(ex).__name__)
        tokens = ["The", "Eiffel", "Tower", "is", "on", "the", "Champ", "de", "Mars",
                  "in", "Paris", ",", "France", ".", "Completed", "1889", "."]
        rng = np.random.default_rng(0)
        sl = rng.normal(-3, 0.6, len(tokens)); el = rng.normal(-3, 0.6, len(tokens))
        sl[6] = 6.0; el[12] = 6.4   # span = "Champ de Mars in Paris, France"
        p_start = np.exp(sl) / np.exp(sl).sum(); p_end = np.exp(el) / np.exp(el).sum()

    n = len(tokens); x = np.arange(n)
    fig, ax = plt.subplots(figsize=(11.0, 4.9))
    w = 0.4
    ax.bar(x - w / 2, p_start, w, color=GREEN, label="P_start(i)")
    ax.bar(x + w / 2, p_end, w, color=RED, label="P_end(j)")
    si = int(np.argmax(p_start)); ei = int(np.argmax(p_end))
    ax.axvspan(si - 0.5, ei + 0.5, color=AMBER, alpha=0.13, zorder=0)
    ax.set_xticks(x); ax.set_xticklabels(tokens, rotation=45, ha="right", fontsize=8.6)
    ax.set_ylabel("probability over passage positions")
    src = "MEASURED (distilbert-squad)" if measured else "constructed logits"
    ax.set_title(f"BERT span head: softmax start/end over passage tokens -- {src}\n"
                 f'Q: "Where is the Eiffel Tower located?"  ->  predicted span shaded',
                 fontsize=12.5, fontweight="bold")
    ax.annotate("argmax start", (si, p_start[si]), textcoords="offset points",
                xytext=(-4, 14), fontsize=9, color=GREEN, fontweight="bold",
                arrowprops=dict(arrowstyle="->", color=GREEN))
    ax.annotate("argmax end", (ei, p_end[ei]), textcoords="offset points",
                xytext=(6, 14), fontsize=9, color=RED, fontweight="bold",
                arrowprops=dict(arrowstyle="->", color=RED))
    ax.legend(loc="upper right", frameon=False, fontsize=10); _despine(ax)
    ax.set_ylim(0, max(p_start.max(), p_end.max()) * 1.25)
    fig.tight_layout(); fig.savefig(f"{OUT}/qa_span_head.png", dpi=150, bbox_inches="tight")
    plt.close(fig); print(f"wrote qa_span_head.png (measured={measured})")
    return measured


# ---- 3. Retrieve-then-read / RAG pipeline -----------------------------------
def retrieve_read():
    fig, ax = plt.subplots(figsize=(10.6, 5.4))
    ax.set_xlim(0, 11); ax.set_ylim(0, 6); ax.axis("off")
    ax.text(5.5, 5.7, "Open-domain QA: retrieve-then-read (and RAG)",
            ha="center", fontsize=14, fontweight="bold", color="#222")

    _box(ax, 0.2, 3.2, 1.7, 0.9, "Question\n\"Who painted\nthe Mona Lisa?\"", BLUE, fs=8.6)
    _box(ax, 3.0, 4.4, 2.0, 0.9, "RETRIEVER\nBM25 (lexical) or\nDPR (dense)", PURPLE, fs=8.8)
    _box(ax, 3.0, 2.0, 2.0, 0.9, "Corpus index\n(Wikipedia,\nANN / FAISS)", SLATE, fs=8.8)
    _box(ax, 6.1, 3.2, 2.0, 1.0, "Top-k passages\n(evidence)", NAVY, fs=9)
    _box(ax, 8.7, 4.3, 2.1, 0.95, "READER (extractive)\nBERT span head", GREEN, fs=8.6)
    _box(ax, 8.7, 2.1, 2.1, 0.95, "GENERATOR (RAG)\nseq2seq, conditioned\non passages", AMBER, fs=8.2)
    _box(ax, 8.7, 0.55, 2.1, 0.8, "Answer\n+ citation", RED, fs=9)

    _arrow(ax, 1.9, 3.65, 3.0, 4.6)
    _arrow(ax, 1.9, 3.6, 3.0, 2.5)
    _arrow(ax, 4.0, 2.9, 4.0, 4.4)        # index -> retriever (lookup)
    _arrow(ax, 5.0, 4.7, 6.1, 3.9)        # retriever -> passages
    _arrow(ax, 8.1, 3.9, 8.7, 4.6)        # passages -> reader
    _arrow(ax, 8.1, 3.5, 8.7, 2.7)        # passages -> generator
    _arrow(ax, 9.75, 4.3, 9.75, 3.05, color=GREEN)   # reader -> answer
    _arrow(ax, 9.75, 2.1, 9.75, 1.35, color=AMBER)   # generator -> answer

    ax.text(5.5, 0.35, "Reader = extract a span from the evidence;  RAG = generate, conditioned on it "
            "(fresher knowledge, citations, fewer hallucinations).",
            ha="center", fontsize=9, color="#555", style="italic")
    fig.tight_layout(); fig.savefig(f"{OUT}/qa_retrieve_read.png", dpi=150, bbox_inches="tight")
    plt.close(fig); print("wrote qa_retrieve_read.png")


# ---- 4. Exact Match vs token-level F1 ---------------------------------------
def _normalize(s):
    import re, string
    s = s.lower()
    s = "".join(ch for ch in s if ch not in set(string.punctuation))
    s = re.sub(r"\b(a|an|the)\b", " ", s)
    return " ".join(s.split())


def _em_f1(pred, gold):
    pn, gn = _normalize(pred), _normalize(gold)
    em = 1.0 if pn == gn else 0.0
    pt, gt = pn.split(), gn.split()
    if not pt or not gt:
        return em, float(pt == gt)
    common = {}
    for t in pt:
        if t in gt and gt.count(t) > common.get(t, 0) and pt.count(t) > common.get(t, 0):
            common[t] = min(pt.count(t), gt.count(t))
    n_same = sum(common.values())
    if n_same == 0:
        return em, 0.0
    prec = n_same / len(pt); rec = n_same / len(gt)
    return em, 2 * prec * rec / (prec + rec)


def em_f1():
    pairs = [
        ("Paris, France", "Paris", "partial overlap"),
        ("the Champ de Mars", "Champ de Mars", "article dropped"),
        ("Leonardo da Vinci", "Leonardo da Vinci", "exact match"),
        ("1889", "in 1889", "extra token"),
        ("London", "Paris", "wrong answer"),
    ]
    labels, ems, f1s = [], [], []
    for pred, gold, _ in pairs:
        em, f1 = _em_f1(pred, gold)
        labels.append(f'"{pred}"\nvs "{gold}"'); ems.append(em); f1s.append(f1)
    y = np.arange(len(pairs))[::-1]
    fig, ax = plt.subplots(figsize=(10.2, 5.0))
    h = 0.36
    ax.barh(y + h / 2, ems, h, color=SLATE, label="Exact Match (0/1)")
    ax.barh(y - h / 2, f1s, h, color=GREEN, label="token-level F1")
    for yi, (em, f1) in zip(y, zip(ems, f1s)):
        ax.text(em + 0.02, yi + h / 2, f"{em:.0f}", va="center", fontsize=9, color=SLATE, fontweight="bold")
        ax.text(f1 + 0.02, yi - h / 2, f"{f1:.2f}", va="center", fontsize=9, color=GREEN, fontweight="bold")
    ax.set_yticks(y); ax.set_yticklabels(labels, fontsize=8.8)
    ax.set_xlim(0, 1.18); ax.set_xlabel("score")
    ax.set_title("Why SQuAD reports both: EM is all-or-nothing, F1 gives partial credit",
                 fontsize=13, fontweight="bold")
    ax.legend(loc="lower right", frameon=False, fontsize=10); _despine(ax)
    fig.tight_layout(); fig.savefig(f"{OUT}/qa_em_f1.png", dpi=150, bbox_inches="tight")
    plt.close(fig); print("wrote qa_em_f1.png")
    # print exact numbers for the page's worked example
    print("  EM/F1 numbers:")
    for pred, gold, desc in pairs:
        em, f1 = _em_f1(pred, gold)
        print(f"    {desc:18s}  pred={pred!r:20s} gold={gold!r:18s} EM={em:.0f} F1={f1:.3f}")


if __name__ == "__main__":
    taxonomy()
    measured = span_head()
    retrieve_read()
    em_f1()
    print("OUT:", OUT, "| span measured:", measured)
