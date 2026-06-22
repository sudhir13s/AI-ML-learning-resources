"""Machine Translation concept-page diagrams (muted palette, parallel matplotlib scale).

Four figures for 06. NLP/concepts/12-Machine-Translation.md:
  1. mt_evolution_timeline.png -- the MT paradigm arc (rule-based -> SMT -> NMT -> LLM):
     fluency rising while engineering complexity falls. Schematic.
  2. mt_alignment_matrix.png  -- a source x target word-alignment matrix (the latent
     variable IBM Model 1 learns). Illustrative, hand-set on a tiny FR->EN pair.
  3. mt_nmt_measured.png      -- a MEASURED NMT translation: a real opus-mt model
     translates 3 French sentences; bars show sentence chrF vs the reference.
     Falls back to a constructed example (clearly labelled) if the model can't download.
  4. mt_backtranslation_curve.png -- back-translation / data-augmentation: BLEU vs the
     amount of (real + synthetic) parallel data, showing the low-resource lift. Schematic
     but shaped from the qualitative curves reported by Sennrich et al. 2016.

Run with:  ~/.uv/envs/ml-py312/bin/python3 tools/gen_mt_diagrams.py
"""
import os
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyArrowPatch
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
# 1. MT evolution timeline: fluency up, hand-engineering down
# ---------------------------------------------------------------------------
def evolution_timeline():
    eras = ["Rule-based\n(1950s-80s)", "Statistical MT\n(1990-2014)",
            "Neural MT\n(2014-2017)", "Transformer NMT\n(2017+)", "LLM translators\n(2020+)"]
    x = np.arange(len(eras))
    fluency = [2.0, 4.5, 7.0, 8.6, 9.2]            # subjective adequacy/fluency, 0-10
    hand_eng = [9.5, 8.0, 4.0, 2.5, 1.0]           # human linguistic engineering needed, 0-10

    fig, ax = plt.subplots(figsize=(9.4, 5.2))
    ax.plot(x, fluency, color=GREEN, lw=3.0, marker="o", ms=9,
            markeredgecolor="white", label="Translation quality (fluency + adequacy)")
    ax.plot(x, hand_eng, color=RED, lw=3.0, marker="s", ms=8, ls="--",
            markeredgecolor="white", label="Hand-built linguistic machinery required")

    ann = {1: "noisy-channel:\nargmax p(f|e)·p(e)", 2: "seq2seq + attention\nfixes the bottleneck",
           3: "self-attention;\ninvented FOR translation", 4: "zero-shot, prompt-\ncontrolled style"}
    for i, t in ann.items():
        ax.annotate(t, (x[i], fluency[i]), textcoords="offset points",
                    xytext=(-6, 14), fontsize=8.5, color="#222", ha="center")

    ax.set_xticks(x)
    ax.set_xticklabels(eras, fontsize=9.5)
    ax.set_ylabel("relative score  (0-10, schematic)")
    ax.set_ylim(0, 11)
    ax.set_title("Machine translation through the eras: quality up, hand-engineering down",
                 fontsize=13.5, fontweight="bold")
    ax.legend(frameon=False, fontsize=9.5, loc="center right")
    _despine(ax)
    fig.tight_layout()
    fig.savefig(f"{OUT}/mt_evolution_timeline.png", dpi=150, bbox_inches="tight")
    plt.close(fig)
    print("wrote mt_evolution_timeline.png")


# ---------------------------------------------------------------------------
# 2. Word-alignment matrix (the latent variable in IBM Model 1)
# ---------------------------------------------------------------------------
def alignment_matrix():
    # French source -> English target. 1 = aligned (a translation pair).
    src = ["le", "chat", "noir", "dort"]              # FR
    tgt = ["the", "black", "cat", "sleeps"]           # EN  (note the reordering noir<->black/cat)
    # rows = target (EN), cols = source (FR)
    A = np.array([
        [1, 0, 0, 0],   # the   <- le
        [0, 0, 1, 0],   # black <- noir
        [0, 1, 0, 0],   # cat   <- chat
        [0, 0, 0, 1],   # sleeps<- dort
    ], dtype=float)

    fig, ax = plt.subplots(figsize=(6.4, 5.4))
    ax.imshow(A, cmap="Greens", vmin=0, vmax=1.4, aspect="equal")
    ax.set_xticks(range(len(src)))
    ax.set_xticklabels(src, fontsize=12)
    ax.set_yticks(range(len(tgt)))
    ax.set_yticklabels(tgt, fontsize=12)
    ax.set_xlabel("source  (French  f)", fontsize=11)
    ax.set_ylabel("target  (English  e)", fontsize=11)
    for i in range(len(tgt)):
        for j in range(len(src)):
            if A[i, j] > 0:
                ax.text(j, i, "a", ha="center", va="center", color="white",
                        fontsize=13, fontweight="bold")
    ax.set_xticks(np.arange(-.5, len(src), 1), minor=True)
    ax.set_yticks(np.arange(-.5, len(tgt), 1), minor=True)
    ax.grid(which="minor", color="#cccccc", lw=1)
    ax.tick_params(which="minor", length=0)
    ax.set_title("Word alignment: the latent variable MT must infer\n"
                 "(note 'noir' crosses to 'black', 'chat' to 'cat' -- reordering)",
                 fontsize=12, fontweight="bold")
    fig.tight_layout()
    fig.savefig(f"{OUT}/mt_alignment_matrix.png", dpi=150, bbox_inches="tight")
    plt.close(fig)
    print("wrote mt_alignment_matrix.png")


# ---------------------------------------------------------------------------
# 3. MEASURED NMT translation + chrF (real opus-mt model; fallback if offline)
# ---------------------------------------------------------------------------
def nmt_measured():
    srcs = [
        "Le chat noir dort sur le canapé.",
        "J'aime apprendre les langues étrangères.",
        "La traduction automatique a beaucoup progressé.",
    ]
    refs = [
        "The black cat is sleeping on the couch.",
        "I love learning foreign languages.",
        "Machine translation has progressed a lot.",
    ]
    measured = True
    note = ""
    try:
        from transformers import MarianMTModel, MarianTokenizer
        import sacrebleu
        name = "Helsinki-NLP/opus-mt-fr-en"
        tok = MarianTokenizer.from_pretrained(name)
        model = MarianMTModel.from_pretrained(name)
        batch = tok(srcs, return_tensors="pt", padding=True)
        gen = model.generate(**batch, num_beams=5, max_length=60)
        hyps = [tok.decode(g, skip_special_tokens=True) for g in gen]
        chrfs = [sacrebleu.sentence_chrf(h, [r]).score for h, r in zip(hyps, refs)]
        print("MEASURED opus-mt-fr-en translations:")
        for s, h, r, c in zip(srcs, hyps, refs, chrfs):
            print(f"  FR: {s}\n  -> {h}\n  ref: {r}\n  chrF={c:.1f}\n")
    except Exception as e:  # offline / download blocked -> constructed example
        measured = False
        note = "constructed example (model download unavailable in sandbox)"
        hyps = [
            "The black cat sleeps on the couch.",
            "I like to learn foreign languages.",
            "Machine translation has progressed a lot.",
        ]
        chrfs = [78.0, 71.0, 92.0]
        print("FALLBACK (constructed):", repr(e)[:120])

    labels = [f"sent {i+1}" for i in range(len(srcs))]
    y = np.arange(len(srcs))[::-1]
    fig, ax = plt.subplots(figsize=(9.2, 4.8))
    bars = ax.barh(y, chrfs, color=GREEN, height=0.5, edgecolor="white")
    for yi, c, h in zip(y, chrfs, hyps):
        ax.text(c + 1, yi, f"{c:.1f}", va="center", fontsize=10, fontweight="bold", color="#222")
        short = (h[:46] + "...") if len(h) > 48 else h
        ax.text(1, yi + 0.32, f'"{short}"', va="center", fontsize=8.2, color="#333", style="italic")
    ax.set_yticks(y)
    ax.set_yticklabels(labels, fontsize=10)
    ax.set_xlim(0, 105)
    ax.set_xlabel("chrF  (character n-gram F-score vs reference, 0-100)")
    title = "Measured NMT: opus-mt-fr-en translates, scored by chrF"
    if not measured:
        title = "NMT translation scored by chrF  (" + note + ")"
    ax.set_title(title, fontsize=12.5, fontweight="bold")
    _despine(ax)
    fig.tight_layout()
    fig.savefig(f"{OUT}/mt_nmt_measured.png", dpi=150, bbox_inches="tight")
    plt.close(fig)
    print("wrote mt_nmt_measured.png  (measured=%s)" % measured)
    return measured, list(zip(srcs, hyps, refs, chrfs))


# ---------------------------------------------------------------------------
# 4. Back-translation: BLEU vs parallel-data size, with the synthetic lift
# ---------------------------------------------------------------------------
def backtranslation_curve():
    # real parallel sentence pairs (log scale), schematic BLEU shaped like Sennrich 2016
    n = np.array([1e4, 3e4, 1e5, 3e5, 1e6, 3e6])
    bleu_real = 6 + 7.5 * np.log10(n / 1e4)               # rises with more real data
    bleu_bt = bleu_real + np.array([9.0, 8.0, 6.5, 5.0, 3.5, 2.2])  # synthetic lift, biggest when scarce

    fig, ax = plt.subplots(figsize=(9.2, 5.2))
    ax.semilogx(n, bleu_real, color=BLUE, lw=2.8, marker="o", ms=8,
                markeredgecolor="white", label="Real parallel data only")
    ax.semilogx(n, bleu_bt, color=GREEN, lw=2.8, marker="s", ms=8, ls="--",
                markeredgecolor="white", label="+ back-translated monolingual data")
    ax.fill_between(n, bleu_real, bleu_bt, color=GREEN, alpha=0.12)
    ax.annotate("biggest lift when\nparallel data is scarce\n(low-resource win)",
                (n[0], (bleu_real[0] + bleu_bt[0]) / 2), textcoords="offset points",
                xytext=(40, -6), fontsize=9.5, color=GREEN, fontweight="bold",
                arrowprops=dict(arrowstyle="->", color=GREEN))
    ax.set_xlabel("real parallel sentence pairs  (log scale)")
    ax.set_ylabel("BLEU  (schematic, shaped after Sennrich et al. 2016)")
    ax.set_title("Back-translation: synthetic data lifts low-resource MT most",
                 fontsize=13, fontweight="bold")
    ax.legend(frameon=False, fontsize=10, loc="lower right")
    _despine(ax)
    fig.tight_layout()
    fig.savefig(f"{OUT}/mt_backtranslation_curve.png", dpi=150, bbox_inches="tight")
    plt.close(fig)
    print("wrote mt_backtranslation_curve.png")


if __name__ == "__main__":
    evolution_timeline()
    alignment_matrix()
    measured, rows = nmt_measured()
    backtranslation_curve()
    print("\nDONE. measured NMT =", measured)
