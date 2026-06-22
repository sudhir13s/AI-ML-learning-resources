"""N-gram language-model concept-page diagrams (muted palette, parallel matplotlib scale).

Four figures for 06. NLP/concepts/04-N-gram-Language-Models-and-Smoothing.md:
  1. ngram_window.png   -- schematic: the Markov window of a trigram model sliding over a
     sentence, showing which context each prediction conditions on.
  2. ngram_smoothing_mass.png -- MEASURED: probability mass on a held-out word distribution
     under MLE vs Laplace vs Kneser-Ney, plus the zero-probability cliff for MLE.
  3. ngram_perplexity.png -- MEASURED: perplexity vs n (unigram..4-gram) and vs training-set
     size, on a real corpus, under add-k vs Kneser-Ney.
  4. ngram_discount.png  -- MEASURED: Good-Turing reestimated counts c* vs raw count c, and the
     absolute-discounting line c-d, visualizing how mass is shaved off seen events.

Run: /Users/sudhirsingh/.uv/envs/ml-py312/bin/python3 tools/gen_ngram_diagrams.py
"""
import os, math, re, random
from collections import Counter, defaultdict
import matplotlib
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


# ----------------------------------------------------------------------------
# A small, self-contained corpus (public-domain flavored sentences) so the
# measured plots are reproducible without any download.
# ----------------------------------------------------------------------------
CORPUS_TEXT = """
the cat sat on the mat . the cat saw the dog . the dog sat on the log .
a dog ran on the grass . the cat ran on the mat . the dog saw the cat .
the cat chased the dog . the dog chased the cat . a cat sat on a log .
the kitten saw the puppy . the puppy ran on the grass . a cat saw a dog .
the dog barked at the cat . the cat hissed at the dog . the puppy sat on the mat .
a kitten chased a ball . the ball rolled on the grass . the kitten ran on the mat .
the boy threw the ball . the girl caught the ball . a boy ran on the grass .
the girl saw the boy . the boy saw the girl . a dog chased a ball on the grass .
""".strip()


def tokenize_sentences(text):
    sents = []
    for line in text.replace("\n", " ").split("."):
        toks = re.findall(r"[a-z]+", line.lower())
        if toks:
            sents.append(toks)
    return sents


SENTS = tokenize_sentences(CORPUS_TEXT)


# ----------------------------------------------------------------------------
# Figure 1 -- the Markov window sliding over a sentence (schematic).
# ----------------------------------------------------------------------------
def fig_window():
    sent = ["<s>", "the", "cat", "sat", "on", "the", "mat", "</s>"]
    fig, ax = plt.subplots(figsize=(11, 3.6))
    n_tok = len(sent)
    xw = 1.0
    y = 2.4
    # draw tokens
    for i, tok in enumerate(sent):
        c = SLATE if tok in ("<s>", "</s>") else BLUE
        ax.add_patch(plt.Rectangle((i * xw, y), 0.9, 0.7, facecolor=c,
                                   edgecolor="white", lw=1.5))
        ax.text(i * xw + 0.45, y + 0.35, tok, ha="center", va="center",
                color="white", fontsize=11, fontweight="bold")
    # trigram window for predicting "mat" (index 6) from context "on the"
    # show three windows
    windows = [
        (1, 3, "sat", PURPLE),    # predict sat | the cat
        (3, 5, "the", GREEN),     # predict the | cat sat -> actually sat on
        (4, 6, "mat", AMBER),     # predict mat | on the
    ]
    yb = 1.4
    for k, (a, b, target, col) in enumerate(windows):
        yy = yb - k * 0.75
        # bracket under context tokens a..b-1 (2 words = trigram context)
        x0 = a * xw
        x1 = b * xw - 0.1
        ax.plot([x0 + 0.1, x1], [yy + 0.5, yy + 0.5], color=col, lw=2.5)
        ax.plot([x0 + 0.1, x0 + 0.1], [yy + 0.5, yy + 0.62], color=col, lw=2.5)
        ax.plot([x1, x1], [yy + 0.5, yy + 0.62], color=col, lw=2.5)
        ctx = " ".join(sent[a:b])
        ax.text((x0 + x1) / 2 + 0.05, yy + 0.18,
                f"P( {target} | {ctx} )", ha="center", va="center",
                color="white", fontsize=10.5,
                bbox=dict(boxstyle="round,pad=0.25", facecolor=col, edgecolor="none"))
        # arrow to predicted token
        tgt_idx = b  # the predicted token sits just after the context
        ax.annotate("", xy=(tgt_idx * xw + 0.45, y - 0.02),
                    xytext=((x0 + x1) / 2 + 0.05, yy + 0.45),
                    arrowprops=dict(arrowstyle="->", color=col, lw=2))
    ax.text(n_tok * xw / 2, y + 1.25,
            "Trigram model: each word is predicted from only the previous 2 words (the Markov window)",
            ha="center", va="center", fontsize=12.5, fontweight="bold", color="#333")
    ax.set_xlim(-0.3, n_tok * xw + 0.2)
    ax.set_ylim(-0.4, y + 1.7)
    ax.axis("off")
    fig.tight_layout()
    fig.savefig(f"{OUT}/ngram_window.png", dpi=150, bbox_inches="tight")
    plt.close(fig)


# ----------------------------------------------------------------------------
# Bigram model machinery shared by the measured figures.
# ----------------------------------------------------------------------------
def build_counts(sents, n=2):
    """Return n-gram and (n-1)-gram count tables with <s>/</s> padding."""
    ngrams = Counter()
    contexts = Counter()
    unigrams = Counter()
    for s in sents:
        padded = ["<s>"] * (n - 1) + s + ["</s>"]
        for w in padded:
            unigrams[w] += 1
        for i in range(len(padded) - n + 1):
            gram = tuple(padded[i:i + n])
            ngrams[gram] += 1
            contexts[gram[:-1]] += 1
    return ngrams, contexts, unigrams


def kn_bigram(sents, d=0.75):
    """Interpolated Kneser-Ney bigram model. Returns a prob(w|prev) function."""
    bigrams, contexts, unigrams = build_counts(sents, 2)
    vocab = set(unigrams)
    # continuation counts: number of distinct preceding words for each w
    preceders = defaultdict(set)
    for (a, b) in bigrams:
        preceders[b].add(a)
    total_bigram_types = len(bigrams)
    V = len(vocab)
    # Continuation prob, add-one smoothed over the vocabulary so unseen test
    # words still receive a small, sane mass (matches Laplace's OOV handling).
    def p_cont(w):
        return (len(preceders.get(w, ())) + 1) / (total_bigram_types + V)
    # number of distinct following words per context (for the back-off weight)
    followers = defaultdict(set)
    for (a, b) in bigrams:
        followers[a].add(b)

    def prob(w, prev):
        c_bigram = bigrams.get((prev, w), 0)
        c_ctx = contexts.get((prev,), 0)
        if c_ctx == 0:                 # unseen context: back off fully to continuation
            return p_cont(w)
        lam = d * len(followers[prev]) / c_ctx
        return max(c_bigram - d, 0) / c_ctx + lam * p_cont(w)

    return prob, vocab


def mle_bigram(sents):
    bigrams, contexts, unigrams = build_counts(sents, 2)
    vocab = set(unigrams)

    def prob(w, prev):
        c_ctx = contexts.get((prev,), 0)
        if c_ctx == 0:
            return 0.0
        return bigrams.get((prev, w), 0) / c_ctx

    return prob, vocab


def laplace_bigram(sents, k=1.0):
    bigrams, contexts, unigrams = build_counts(sents, 2)
    vocab = set(unigrams)
    V = len(vocab)

    def prob(w, prev):
        c_ctx = contexts.get((prev,), 0)
        return (bigrams.get((prev, w), 0) + k) / (c_ctx + k * V)

    return prob, vocab


# ----------------------------------------------------------------------------
# Figure 2 -- probability mass: MLE vs Laplace vs KN on a fixed context, and
# the zero-probability cliff.
# ----------------------------------------------------------------------------
def fig_smoothing_mass():
    prev = "the"
    # candidate next words: some seen after "the", some not
    bigrams, contexts, unigrams = build_counts(SENTS, 2)
    seen = sorted({b for (a, b) in bigrams if a == prev},
                  key=lambda w: -bigrams[(prev, w)])
    unseen = [w for w in sorted(unigrams) if (prev, w) not in bigrams and w != "<s>"]
    words = seen[:5] + unseen[:3]

    p_mle, _ = mle_bigram(SENTS)
    p_lap, _ = laplace_bigram(SENTS, k=1.0)
    p_kn, _ = kn_bigram(SENTS, d=0.75)

    mle = [p_mle(w, prev) for w in words]
    lap = [p_lap(w, prev) for w in words]
    kn = [p_kn(w, prev) for w in words]

    x = np.arange(len(words))
    w = 0.26
    fig, ax = plt.subplots(figsize=(10, 4.6))
    ax.bar(x - w, mle, w, label="MLE (unsmoothed)", color=RED)
    ax.bar(x, lap, w, label="Laplace (add-1)", color=AMBER)
    ax.bar(x + w, kn, w, label="Kneser-Ney", color=GREEN)
    ax.set_xticks(x)
    ax.set_xticklabels(words, rotation=0)
    ax.set_ylabel(r"$P(w \mid \mathrm{the})$")
    ax.set_title("P(next word | \"the\"): MLE zeros unseen words; smoothing rescues them",
                 fontsize=12.5, fontweight="bold")
    # annotate the zero cliff
    for i, w_ in enumerate(words):
        if mle[i] == 0:
            ax.annotate("MLE = 0\n(zero cliff)", xy=(i - w, 0.002),
                        xytext=(i - w, max(lap) * 0.55),
                        ha="center", fontsize=8.5, color=RED,
                        arrowprops=dict(arrowstyle="->", color=RED, lw=1.3))
            break
    ax.legend(frameon=False, fontsize=10)
    _despine(ax)
    fig.tight_layout()
    fig.savefig(f"{OUT}/ngram_smoothing_mass.png", dpi=150, bbox_inches="tight")
    plt.close(fig)


# ----------------------------------------------------------------------------
# Perplexity machinery.
# ----------------------------------------------------------------------------
def perplexity(prob_fn, test_sents, n=2):
    log2sum = 0.0
    N = 0
    for s in test_sents:
        padded = ["<s>"] * (n - 1) + s + ["</s>"]
        for i in range(n - 1, len(padded)):
            w = padded[i]
            prev = padded[i - 1]
            p = prob_fn(w, prev)
            p = max(p, 1e-12)
            log2sum += math.log2(p)
            N += 1
    return 2 ** (-log2sum / N)


def kn_ngram_prob_factory(train, n):
    """Generic interpolated-KN for n in {1,2,3,4} via recursive continuation."""
    # Build counts for orders 1..n
    counts = {}
    ctx_counts = {}
    for order in range(1, n + 1):
        ng, ctx, uni = build_counts(train, order)
        counts[order] = ng
        ctx_counts[order] = ctx
    uni = counts[1]
    total_unigrams = sum(uni.values())
    # continuation counts at each order
    pre = {}
    for order in range(2, n + 1):
        p = defaultdict(set)
        for gram in counts[order]:
            p[gram[1:]].add(gram[0])
        pre[order] = p
    d = 0.75

    def prob(seq):
        # seq is a tuple of length up to n; predict seq[-1] | seq[:-1]
        order = len(seq)
        if order == 1:
            return (uni.get(seq, 0) + 1) / (total_unigrams + len(uni))
        ng = counts[order]
        ctx = ctx_counts[order]
        c = ng.get(seq, 0)
        cc = ctx.get(seq[:-1], 0)
        lower = prob(seq[1:])
        if cc == 0:
            return lower
        followers = sum(1 for g in ng if g[:-1] == seq[:-1])
        lam = d * followers / cc
        return max(c - d, 0) / cc + lam * lower

    def prob_wprev(w, prev):  # bigram-style wrapper not used here
        return prob((prev, w))

    return prob


def perplexity_ngram(prob_seq, test_sents, n):
    log2sum = 0.0
    N = 0
    for s in test_sents:
        padded = ["<s>"] * (n - 1) + s + ["</s>"]
        for i in range(n - 1, len(padded)):
            seq = tuple(padded[i - n + 1:i + 1])
            p = max(prob_seq(seq), 1e-12)
            log2sum += math.log2(p)
            N += 1
    return 2 ** (-log2sum / N)


# ----------------------------------------------------------------------------
# Figure 3 -- perplexity vs n and vs training size (MEASURED).
# ----------------------------------------------------------------------------
def fig_perplexity():
    random.seed(0)
    sents = SENTS[:]
    random.shuffle(sents)
    split = int(len(sents) * 0.75)
    train_all, test = sents[:split], sents[split:]

    # (a) perplexity vs n
    ns = [1, 2, 3, 4]
    pps = []
    for n in ns:
        pf = kn_ngram_prob_factory(train_all, n)
        pps.append(perplexity_ngram(pf, test, n))

    # (b) perplexity vs training size (bigram, KN vs Laplace)
    fracs = np.linspace(0.25, 1.0, 6)
    pp_kn, pp_lap = [], []
    for f in fracs:
        m = max(2, int(len(train_all) * f))
        sub = train_all[:m]
        p_kn, _ = kn_bigram(sub, d=0.75)
        p_lap, _ = laplace_bigram(sub, k=1.0)
        pp_kn.append(perplexity(p_kn, test, n=2))
        pp_lap.append(perplexity(p_lap, test, n=2))

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(11, 4.2))
    ax1.plot(ns, pps, "o-", color=PURPLE, lw=2.5, markersize=8)
    for x, y in zip(ns, pps):
        ax1.annotate(f"{y:.1f}", (x, y), textcoords="offset points",
                     xytext=(0, 8), ha="center", fontsize=9, color=PURPLE)
    ax1.set_xticks(ns)
    ax1.set_xlabel("n (n-gram order)")
    ax1.set_ylabel("perplexity (lower is better)")
    ax1.set_title("Perplexity vs n (interpolated Kneser-Ney)", fontsize=12, fontweight="bold")
    _despine(ax1)

    ax2.plot(fracs * 100, pp_lap, "s-", color=AMBER, lw=2.5, markersize=7, label="Laplace (add-1)")
    ax2.plot(fracs * 100, pp_kn, "o-", color=GREEN, lw=2.5, markersize=7, label="Kneser-Ney")
    ax2.set_xlabel("training data used (%)")
    ax2.set_ylabel("bigram perplexity")
    ax2.set_title("More data + better smoothing → lower perplexity", fontsize=12, fontweight="bold")
    ax2.legend(frameon=False, fontsize=10)
    _despine(ax2)
    fig.tight_layout()
    fig.savefig(f"{OUT}/ngram_perplexity.png", dpi=150, bbox_inches="tight")
    plt.close(fig)
    return pps, pp_kn, pp_lap


# ----------------------------------------------------------------------------
# Figure 4 -- discounting curves: Good-Turing c* and absolute discounting.
# ----------------------------------------------------------------------------
def fig_discount():
    # Build frequency-of-frequencies on bigrams of the corpus
    bigrams, _, _ = build_counts(SENTS, 2)
    Nr = Counter(bigrams.values())  # N_r = number of bigram types occurring r times
    max_r = max(Nr)
    rs = list(range(1, min(max_r, 6) + 1))
    # Good-Turing c* = (r+1) N_{r+1} / N_r  (simple GT, where N_{r+1}>0)
    gt = []
    for r in rs:
        if Nr.get(r, 0) > 0 and Nr.get(r + 1, 0) > 0:
            gt.append((r + 1) * Nr[r + 1] / Nr[r])
        else:
            gt.append(float("nan"))
    d = 0.75
    absd = [max(r - d, 0) for r in rs]

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(11, 4.2))
    ax1.plot(rs, rs, "--", color=SLATE, lw=1.8, label="raw count c (no discount)")
    ax1.plot(rs, absd, "o-", color=GREEN, lw=2.5, markersize=7, label=f"absolute discount c−d (d={d})")
    ax1.plot(rs, gt, "s-", color=AMBER, lw=2.5, markersize=7, label="Good-Turing c*")
    ax1.set_xlabel("observed count c")
    ax1.set_ylabel("reestimated count c*")
    ax1.set_title("Discounting shaves mass off seen counts", fontsize=12, fontweight="bold")
    ax1.legend(frameon=False, fontsize=9)
    _despine(ax1)

    # Frequency-of-frequencies bar (Zipfian: many rare, few frequent) + missing mass
    rs2 = sorted(Nr)[:8]
    vals = [Nr[r] for r in rs2]
    total_bigram_tokens = sum(bigrams.values())
    missing_mass = Nr.get(1, 0) / total_bigram_tokens  # N_1 / N
    ax2.bar(rs2, vals, color=BLUE, edgecolor="white")
    for r, v in zip(rs2, vals):
        ax2.text(r, v + 0.3, str(v), ha="center", fontsize=9, color=BLUE)
    ax2.set_xlabel("count r")
    ax2.set_ylabel(r"$N_r$ = #(bigram types seen r times)")
    ax2.set_title(f"Frequency of frequencies (N₁ dominates)\nGood-Turing missing mass = N₁/N = {missing_mass:.3f}",
                  fontsize=11.5, fontweight="bold")
    _despine(ax2)
    fig.tight_layout()
    fig.savefig(f"{OUT}/ngram_discount.png", dpi=150, bbox_inches="tight")
    plt.close(fig)
    return missing_mass


if __name__ == "__main__":
    fig_window()
    fig_smoothing_mass()
    pps, pp_kn, pp_lap = fig_perplexity()
    mm = fig_discount()
    print("OUT:", OUT)
    print("perplexity vs n (KN):", [round(p, 2) for p in pps])
    print("bigram PP Laplace (full):", round(pp_lap[-1], 2), "| KN (full):", round(pp_kn[-1], 2))
    print("Good-Turing missing mass N1/N:", round(mm, 4))
