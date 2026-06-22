"""Diagrams for 06. NLP / 18-NLP-Evaluation-Metrics.md

Generates four PNGs into ../06. NLP/concepts/images/ :
  1. nlpeval_taxonomy.png       — the metric taxonomy (overlap / embedding / model-LLM / human), schematic
  2. nlpeval_bleu_vs_bertscore.png — MEASURED BLEU vs ROUGE-L vs BERTScore on candidate sentences
  3. nlpeval_brevity_penalty.png — MEASURED BLEU brevity-penalty curve vs candidate length
  4. nlpeval_judge_bias.png      — LLM-judge position/verbosity bias + metric-human correlation, schematic

Run with the project env:
  /Users/sudhirsingh/.uv/envs/ml-py312/bin/python3 tools/gen_nlp_eval_diagrams.py

Numbers in #2 are computed live with sacrebleu / rouge-score / bert-score so the figure
and the by-hand walkthroughs in the page stay in sync. Verified on Python 3.12.
"""
import math
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch

# ---- muted palette (white text on fills), shared with the page's mermaid -------
BLUE = "#3A6B96"   # input / data / overlap
PURPLE = "#5D4A8A"  # process / embedding
GREEN = "#2E7A5A"   # output / model-LLM
RED = "#8B3B4A"     # danger / human-hard / bias
SLATE = "#4A5B6E"   # frozen / neutral
AMBER = "#7A6528"   # highlight
NAVY = "#2A5B80"    # alt

OUT = os.path.join(os.path.dirname(__file__), "..", "06. NLP", "concepts", "images")
os.makedirs(OUT, exist_ok=True)

plt.rcParams.update({
    "font.family": "DejaVu Sans",
    "font.size": 11,
    "axes.titlesize": 15,
    "axes.titleweight": "bold",
    "axes.labelsize": 11,
    "figure.dpi": 140,
})


def save(fig, name):
    path = os.path.join(OUT, name)
    fig.savefig(path, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    print("wrote", os.path.abspath(path))


# =====================================================================
# 1. The taxonomy of NLP evaluation metrics (schematic)
# =====================================================================
def taxonomy():
    fig, ax = plt.subplots(figsize=(11.5, 6.4))
    ax.axis("off")
    ax.set_xlim(0, 12)
    ax.set_ylim(0, 8)

    def box(x, y, w, h, text, fill, fs=10.5, tc="#fff", weight="normal"):
        ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.04,rounding_size=0.12",
                                    fc=fill, ec="white", lw=1.4))
        ax.text(x + w / 2, y + h / 2, text, ha="center", va="center",
                color=tc, fontsize=fs, fontweight=weight, wrap=True)

    box(3.6, 7.0, 4.8, 0.85, "How good is the output?\nNLP evaluation", SLATE, 12.5, weight="bold")

    # two top splits
    box(0.3, 5.5, 5.6, 0.8, "Reference-based\n(compare to a gold answer)", NAVY, 10.5, weight="bold")
    box(6.1, 5.5, 5.6, 0.8, "Reference-free\n(judge the output alone)", AMBER, 10.5, weight="bold")
    ax.plot([6.0, 3.1], [7.0, 6.3], color=SLATE, lw=1.4)
    ax.plot([6.0, 8.9], [7.0, 6.3], color=SLATE, lw=1.4)
    _ = NAVY  # connectors below reuse split centres ~3.1 / 8.9

    # reference-based children
    box(0.3, 3.4, 2.55, 1.7,
        "n-gram OVERLAP\n\nBLEU (MT)\nROUGE (summ.)\nMETEOR · chrF\nEM · token-F1 (QA)",
        BLUE, 9.6)
    box(3.05, 3.4, 2.55, 1.7,
        "EMBEDDING-based\n\nBERTScore\nMoverScore\n(cosine match of\ncontextual vectors)",
        PURPLE, 9.6)
    ax.plot([3.1, 1.55], [5.5, 5.1], color=NAVY, lw=1.3)
    ax.plot([3.1, 4.3], [5.5, 5.1], color=NAVY, lw=1.3)

    # reference-free children
    box(6.1, 3.4, 2.55, 1.7,
        "MODEL / LEARNED\n\nBLEURT · COMET\nperplexity (LM)\n(trained to predict\nhuman scores)",
        GREEN, 9.6)
    box(8.85, 3.4, 2.55, 1.7,
        "LLM-as-JUDGE\n\nstrong LLM rates\nor ranks outputs\nMT-Bench · Arena\n(beware biases)",
        RED, 9.6)
    ax.plot([8.9, 7.4], [5.5, 5.1], color=AMBER, lw=1.3)
    ax.plot([8.9, 10.1], [5.5, 5.1], color=AMBER, lw=1.3)

    # the gold standard banner
    box(1.4, 1.45, 9.2, 1.05,
        "HUMAN evaluation — the gold standard\n"
        "Likert · A/B · inter-annotator agreement (κ).\n"
        "Slow and costly, but the ground truth every automatic metric is validated against.",
        SLATE, 9.6, weight="bold")
    for x in (1.55, 4.3, 7.4, 10.1):
        ax.plot([x, 6.0], [3.4, 2.5], color=SLATE, lw=0.8, ls=":", alpha=0.5)

    ax.text(6, 0.7, "left → right:  cheaper & faster  →  closer to true meaning & human judgement",
            ha="center", color="#444", fontsize=10.5, style="italic")
    ax.set_title("A taxonomy of NLP evaluation metrics", pad=12)
    save(fig, "nlpeval_taxonomy.png")


# =====================================================================
# 2. MEASURED BLEU vs ROUGE-L vs BERTScore on candidate sentences
# =====================================================================
def bleu_vs_bertscore():
    import sacrebleu
    from rouge_score import rouge_scorer
    from bert_score import score as bscore

    ref = "the movie was absolutely fantastic"
    cands = {
        "exact\n(copy)": "the movie was absolutely fantastic",
        "paraphrase\n(same meaning)": "the film was incredibly wonderful",
        "negation\n(opposite!)": "the movie was absolutely terrible",
        "unrelated": "i need to buy groceries today",
    }
    scorer = rouge_scorer.RougeScorer(["rougeL"], use_stemmer=True)
    _, _, F = bscore(list(cands.values()), [ref] * len(cands),
                     lang="en", verbose=False, rescale_with_baseline=True)

    labels, bleu, rouge, bert = [], [], [], []
    for (name, c), f in zip(cands.items(), F.tolist()):
        labels.append(name)
        bleu.append(sacrebleu.sentence_bleu(c, [ref]).score)
        rouge.append(scorer.score(ref, c)["rougeL"].fmeasure * 100)
        bert.append(f * 100)

    fig, ax = plt.subplots(figsize=(10.5, 5.8))
    import numpy as np
    x = np.arange(len(labels))
    w = 0.26
    ax.bar(x - w, bleu, w, label="BLEU", color=BLUE)
    ax.bar(x, rouge, w, label="ROUGE-L (F1)", color=AMBER)
    ax.bar(x + w, bert, w, label="BERTScore (F1)", color=GREEN)
    for i in range(len(labels)):
        for off, v in ((-w, bleu[i]), (0, rouge[i]), (w, bert[i])):
            ax.text(x[i] + off, v + 1.5, f"{v:.0f}", ha="center", fontsize=9, color="#333")

    ax.set_xticks(x)
    ax.set_xticklabels(labels, fontsize=9.5)
    ax.set_ylabel("score  (0–100)")
    ax.set_ylim(0, 112)
    ax.set_title("Same reference, four candidates — where surface metrics fail", pad=10)
    ax.legend(loc="upper right", frameon=False)
    ax.text(1.0, 102, "↑ paraphrase: BLEU/ROUGE collapse,\nBERTScore stays high (meaning preserved)",
            ha="center", fontsize=8.6, color=GREEN)
    ax.text(2.0, 70, "↓ negation fools BERTScore too\n(high score, opposite meaning)",
            ha="center", fontsize=8.6, color=RED)
    ax.grid(axis="y", alpha=0.25)
    save(fig, "nlpeval_bleu_vs_bertscore.png")


# =====================================================================
# 3. MEASURED BLEU brevity-penalty curve vs candidate length
# =====================================================================
def brevity_penalty():
    import numpy as np
    r = 20  # reference length
    c = np.linspace(4, 36, 400)
    bp = np.where(c >= r, 1.0, np.exp(1 - r / c))

    fig, ax = plt.subplots(figsize=(9.6, 5.4))
    ax.plot(c, bp, color=BLUE, lw=2.6, label=r"$BP=\min(1,\,e^{1-r/c})$")
    ax.axvline(r, color=SLATE, ls="--", lw=1.4)
    ax.axhline(1.0, color=GREEN, ls=":", lw=1.2)
    ax.text(r + 0.4, 0.18, "c = r\n(no penalty)", color=SLATE, fontsize=9.5)

    for cc in (8, 12, 16):
        b = math.exp(1 - r / cc)
        ax.scatter([cc], [b], color=RED, zorder=5, s=42)
        ax.annotate(f"c={cc}\nBP={b:.2f}", (cc, b), textcoords="offset points",
                    xytext=(6, -22), fontsize=8.6, color=RED)

    ax.fill_between(c, bp, 1.0, where=(c < r), color=RED, alpha=0.10)
    ax.text(9, 0.93, "penalty region\n(candidate too short)", color=RED, fontsize=9.5, ha="center")

    ax.set_xlabel("candidate length  c   (reference length r = 20 words)")
    ax.set_ylabel("brevity penalty  BP")
    ax.set_ylim(0, 1.08)
    ax.set_xlim(4, 36)
    ax.set_title("BLEU's brevity penalty — why precision metrics must punish short output", pad=10)
    ax.legend(loc="lower right", frameon=False, fontsize=11)
    ax.grid(alpha=0.25)
    save(fig, "nlpeval_brevity_penalty.png")


# =====================================================================
# 4. LLM-judge biases + metric-vs-human correlation (schematic)
# =====================================================================
def judge_bias():
    import numpy as np
    fig, (axL, axR) = plt.subplots(1, 2, figsize=(12.2, 5.2))

    # left: position bias — same pair, swap order, judge flips
    axL.axis("off")
    axL.set_xlim(0, 10)
    axL.set_ylim(0, 10)
    axL.set_title("LLM-judge position bias", pad=8)

    def jbox(ax, x, y, w, h, t, fill, fs=9.5):
        ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.04,rounding_size=0.1",
                                    fc=fill, ec="white", lw=1.3))
        ax.text(x + w / 2, y + h / 2, t, ha="center", va="center", color="#fff",
                fontsize=fs, fontweight="bold")

    jbox(axL, 0.4, 7.2, 4.3, 1.5, "Prompt order 1:\n[A first, B second]", NAVY)
    jbox(axL, 5.3, 7.2, 4.3, 1.5, "Prompt order 2:\n[B first, A second]", NAVY)
    jbox(axL, 0.9, 4.8, 3.3, 1.5, "Judge picks\nthe FIRST one", AMBER)
    jbox(axL, 5.8, 4.8, 3.3, 1.5, "Judge picks\nthe FIRST one", AMBER)
    axL.annotate("", (2.5, 7.2), (2.5, 6.3), arrowprops=dict(arrowstyle="-|>", color=SLATE, lw=1.6))
    axL.annotate("", (7.4, 7.2), (7.4, 6.3), arrowprops=dict(arrowstyle="-|>", color=SLATE, lw=1.6))
    jbox(axL, 2.0, 2.0, 6.0, 1.6,
         "Verdict FLIPS with order →\nposition bias.\nFix: swap & require consistency,\nelse score a tie.", RED, 9.3)

    # right: metric-vs-human correlation bars (illustrative, literature-typical)
    metrics = ["BLEU", "ROUGE-L", "METEOR", "chrF", "BERTScore", "COMET", "LLM-judge"]
    corr = [0.31, 0.34, 0.42, 0.46, 0.61, 0.78, 0.85]
    colors = [BLUE, BLUE, BLUE, BLUE, PURPLE, GREEN, GREEN]
    y = np.arange(len(metrics))[::-1]
    axR.barh(y, corr, color=colors, height=0.62)
    for yi, v in zip(y, corr):
        axR.text(v + 0.01, yi, f"{v:.2f}", va="center", fontsize=9.5, color="#333")
    axR.set_yticks(y)
    axR.set_yticklabels(metrics, fontsize=10)
    axR.set_xlim(0, 1.0)
    axR.set_xlabel("correlation with human judgement  (illustrative — direction, not exact values)")
    axR.set_title("Agreement with humans rises with semantic depth", pad=8)
    axR.grid(axis="x", alpha=0.25)

    fig.tight_layout(pad=1.6)
    save(fig, "nlpeval_judge_bias.png")


if __name__ == "__main__":
    taxonomy()
    brevity_penalty()
    judge_bias()
    bleu_vs_bertscore()  # last: heavy (loads roberta-large)
    print("all NLP-eval diagrams done")
