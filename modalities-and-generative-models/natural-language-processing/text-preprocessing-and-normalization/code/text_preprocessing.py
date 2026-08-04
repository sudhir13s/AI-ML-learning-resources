"""Text preprocessing & normalization, end to end -- the single source of truth.

Every number, table, and figure in the concept page and the teaching notebook is produced by
the functions in THIS file, so the prose, the notebook, and the figures cannot silently drift
apart. The module is pure Python + numpy + nltk + scikit-learn -- no tensors, no GPU. It is
deterministic: the only randomness is in the classifier's optimizer, which is seeded.

What it demonstrates, matching the page section by section:
  * `preprocess`            -- the full classical pipeline on one messy sentence (Worked ex. 1)
  * `stem_vs_lemma`         -- Porter stemming vs WordNet lemmatization on tricky words (ex. 2)
  * `vocab_shrink_stages`   -- vocabulary size after each cleaning step (the sparsity lever)
  * `classifier_sweep`      -- measured TF-IDF + LogReg accuracy/vocab on 20 Newsgroups (ex. 3)
  * `zipf_rank_frequency`   -- rank-frequency counts + the fitted Zipf slope (the math section)
  * `unicode_demo`          -- NFC/NFKC collapsing look-alike strings (the Unicode section)

Device note: this chapter has no tensors. We print an honest device line --
`device: cpu (pure-Python/numpy)` -- plus the numpy version (and the torch version if torch is
importable, only so the "no accelerator needed" claim is verifiable).

Run:
    python text_preprocessing.py
"""

from __future__ import annotations

import html
import re
import unicodedata
from collections import Counter

import nltk
import numpy as np

# 20 Newsgroups categories used for the measured classifier demo (4 well-separated topics).
NEWSGROUP_CATEGORIES = (
    "rec.sport.baseball",
    "sci.med",
    "comp.graphics",
    "talk.politics.guns",
)
# Rank window the Zipf log-log line is fit over: skip the noisy tail beyond rank 1000.
ZIPF_FIT_MIN_RANK = 1
ZIPF_FIT_MAX_RANK = 1000
CLASSIFIER_SEED = 0  # LogisticRegression is the only stochastic step; pin it for reproducibility
MIN_DOC_FREQUENCY = 2  # TF-IDF: ignore terms appearing in <2 docs (drops one-off noise)

# A regex that keeps only lowercase word tokens -- the canonical "word" definition reused
# across the vocab-shrink and classifier demos so their token counts are comparable.
WORD_TOKEN = re.compile(r"[a-z0-9]+")
# A stricter alphabetic-only token used by the classifier sweep (matches the page's table).
ALPHA_TOKEN = re.compile(r"[a-z]+")


def _ensure_nltk() -> None:
    """Download the small NLTK data packages the pipeline needs (idempotent, quiet)."""
    for package in ("stopwords", "wordnet", "omw-1.4"):
        nltk.download(package, quiet=True)


def device_line() -> str:
    """Return an honest one-line environment report for a pure-Python/numpy chapter.

    There are no tensors here, so the device is always CPU. We still surface the torch version
    when torch is importable, purely so the "no accelerator required" claim can be verified.
    """
    line = f"device: cpu (pure-Python/numpy)   numpy: {np.__version__}"
    try:
        import torch  # noqa: PLC0415 -- optional, only for the version string

        line += f"   torch: {torch.__version__} (imported for detection only, unused)"
    except ImportError:
        line += "   torch: not installed (not needed)"
    return line


def strip_accents(text: str) -> str:
    """Remove combining accent marks: NFD-decompose, then drop the 'Mn' (mark, nonspacing) chars.

    `cafes` and `cafés` collapse to one form. This is lossy on purpose -- only use it where an
    accent is surface noise (search-style matching), never where it carries meaning (resume vs
    resume, pena vs pena in Spanish).
    """
    decomposed = unicodedata.normalize("NFD", text)
    return "".join(c for c in decomposed if unicodedata.category(c) != "Mn")


def preprocess(text: str, *, stopwords_set: set[str]) -> list[str]:
    """Run the full classical cleaning pipeline on one string and return canonical tokens.

    The order is load-bearing (see the page): strip HTML *before* lowercasing so tag-matching
    is reliable; Unicode-normalize *before* tokenizing so `café`'s two spellings tokenize
    identically; remove stopwords *after* tokenizing (you need tokens to compare to the list).
    """
    text = re.sub(r"<[^>]+>", " ", text)  # strip HTML tags -- layout, never language
    text = html.unescape(text)  # &amp; -> & so entities don't survive as junk tokens
    text = re.sub(r"https?://\S+", " <URL> ", text)  # collapse every unique URL to one feature
    text = re.sub(r"[\U0001F000-\U0001FAFF☀-➿]", " <EMOJI> ", text)  # emoji -> placeholder
    text = unicodedata.normalize("NFKC", text).lower()  # fold compatibility variants + case
    text = strip_accents(text)  # cafés -> cafes so it matches plain `cafe`
    text = re.sub(r"[^a-z0-9<>\s]", " ", text)  # drop punctuation, keep placeholder brackets
    text = re.sub(r"\s+", " ", text).strip()  # collapse runs of whitespace
    tokens = text.split()
    return [t for t in tokens if t not in stopwords_set]  # finally drop function words


def stem_vs_lemma(
    words_with_pos: list[tuple[str, str]],
) -> list[tuple[str, str, str, str, bool]]:
    """Return (word, pos, porter_stem, wordnet_lemma, agree?) for each (word, pos) pair.

    These are the *real* NLTK outputs, not hand-picked -- the whole point is to show where a
    blind suffix-stripper (Porter) and a dictionary+POS lemmatizer (WordNet) disagree.
    """
    from nltk.stem import PorterStemmer, WordNetLemmatizer

    stemmer = PorterStemmer()
    lemmatizer = WordNetLemmatizer()
    rows: list[tuple[str, str, str, str, bool]] = []
    for word, pos in words_with_pos:
        stem = stemmer.stem(word)
        lemma = lemmatizer.lemmatize(word, pos=pos)
        rows.append((word, pos, stem, lemma, stem == lemma))
    return rows


def vocab_shrink_stages(documents: list[str], *, stopwords_set: set[str]) -> list[tuple[str, int]]:
    """Return (stage_name, distinct_vocabulary_size) after each cleaning step on a corpus.

    This is the quantitative heart of "preprocessing reduces sparsity": each step that collapses
    surface variation shrinks the number of distinct tokens the model must learn from.
    """
    from nltk.stem import PorterStemmer

    stemmer = PorterStemmer()

    def vocab_size(tokenized_docs: list[list[str]]) -> int:
        seen: set[str] = set()
        for toks in tokenized_docs:
            seen.update(toks)
        return len(seen)

    raw_case_sensitive = [d.split() for d in documents]  # whitespace split, case kept
    lowered = [d.lower().split() for d in documents]  # lowercase only
    tokenized = [WORD_TOKEN.findall(unicodedata.normalize("NFKC", d).lower()) for d in documents]
    de_stopped = [[t for t in toks if t not in stopwords_set] for toks in tokenized]
    stemmed = [[stemmer.stem(t) for t in toks] for toks in de_stopped]

    return [
        ("raw\n(case-sensitive)", vocab_size(raw_case_sensitive)),
        ("lowercase", vocab_size(lowered)),
        ("+ tokenize\n+ NFKC", vocab_size(tokenized)),
        ("+ stopword\nremoval", vocab_size(de_stopped)),
        ("+ Porter\nstemming", vocab_size(stemmed)),
    ]


def classifier_sweep(
    train_docs: list[str],
    train_labels: np.ndarray,
    test_docs: list[str],
    test_labels: np.ndarray,
    *,
    stopwords_set: set[str],
) -> list[tuple[str, int, float]]:
    """Train TF-IDF + LogReg under three cleaning configs; return (config, vocab_size, accuracy).

    The configs are cumulative: raw lowercase -> + stopword removal -> + Porter stemming. This is
    the measured evidence that heavy cleaning *helps* a classical model on a topic task.
    """
    from nltk.stem import PorterStemmer
    from sklearn.feature_extraction.text import TfidfVectorizer
    from sklearn.linear_model import LogisticRegression
    from sklearn.pipeline import make_pipeline

    stemmer = PorterStemmer()

    def clean(text: str, *, drop_stopwords: bool, do_stem: bool) -> str:
        words = ALPHA_TOKEN.findall(text.lower())
        if drop_stopwords:
            words = [w for w in words if w not in stopwords_set]
        if do_stem:
            words = [stemmer.stem(w) for w in words]
        return " ".join(words)

    configs = [
        ("raw (lowercase)", False, False),
        ("+ stopword removal", True, False),
        ("+ Porter stemming", True, True),
    ]
    results: list[tuple[str, int, float]] = []
    for name, drop_stopwords, do_stem in configs:
        x_train = [clean(t, drop_stopwords=drop_stopwords, do_stem=do_stem) for t in train_docs]
        x_test = [clean(t, drop_stopwords=drop_stopwords, do_stem=do_stem) for t in test_docs]
        model = make_pipeline(
            TfidfVectorizer(min_df=MIN_DOC_FREQUENCY),
            LogisticRegression(max_iter=1000, C=10, random_state=CLASSIFIER_SEED),
        ).fit(x_train, train_labels)
        vocab = len(model.named_steps["tfidfvectorizer"].vocabulary_)
        accuracy = model.score(x_test, test_labels)
        results.append((name, vocab, accuracy))
    return results


def zipf_rank_frequency(
    documents: list[str],
) -> tuple[np.ndarray, np.ndarray, float, float, list[tuple[str, int]]]:
    """Tokenize a corpus and return (ranks, frequencies, zipf_slope, intercept, top_terms).

    Zipf's law: the r-th most frequent word has frequency ~ C / r, i.e. on a log-log
    rank-frequency plot the points fall on a line of slope ~ -1. We fit that slope over ranks
    1..1000 (the tail is noisy) and return it so the page can quote the real measured value.
    """
    counts: Counter[str] = Counter()
    for doc in documents:
        counts.update(WORD_TOKEN.findall(doc.lower()))

    frequencies = np.array(sorted(counts.values(), reverse=True), dtype=float)
    ranks = np.arange(1, len(frequencies) + 1, dtype=float)

    fit_mask = (ranks >= ZIPF_FIT_MIN_RANK) & (ranks <= ZIPF_FIT_MAX_RANK)
    slope, intercept = np.polyfit(np.log10(ranks[fit_mask]), np.log10(frequencies[fit_mask]), 1)
    return ranks, frequencies, float(slope), float(intercept), counts.most_common(10)


def unicode_demo() -> dict[str, object]:
    """Show NFC/NFKC collapsing look-alike strings into one canonical form."""
    combining = "café"  # 'cafe' + COMBINING ACUTE ACCENT (U+0301): 5 code points
    precomposed = "café"  # 'café' with precomposed é (U+00E9): 4 code points
    return {
        "combining_codepoints": len(combining),
        "precomposed_codepoints": len(precomposed),
        "equal_raw": combining == precomposed,
        "equal_after_nfc": (
            unicodedata.normalize("NFC", combining) == unicodedata.normalize("NFC", precomposed)
        ),
        "nfkc_ligature_file": unicodedata.normalize("NFKC", "ﬁle"),  # ﬁ ligature -> 'fi'
        "nfkc_half": unicodedata.normalize("NFKC", "½"),  # ½ -> '1⁄2'
        "nfkc_roman_nine": unicodedata.normalize("NFKC", "Ⅸ"),  # Ⅸ -> 'IX'
        "nfkc_superscript_two": unicodedata.normalize("NFKC", "²"),  # ² -> '2'
    }


def load_newsgroups() -> tuple[list[str], np.ndarray, list[str], np.ndarray]:
    """Fetch the 4-category 20 Newsgroups train/test split (headers/footers/quotes removed)."""
    from sklearn.datasets import fetch_20newsgroups

    train = fetch_20newsgroups(
        subset="train", categories=list(NEWSGROUP_CATEGORIES), remove=("headers", "footers", "quotes")
    )
    test = fetch_20newsgroups(
        subset="test", categories=list(NEWSGROUP_CATEGORIES), remove=("headers", "footers", "quotes")
    )
    return train.data, train.target, test.data, test.target


def main() -> None:
    _ensure_nltk()
    from nltk.corpus import stopwords

    stopwords_set = set(stopwords.words("english"))
    print(device_line())
    print()

    # 1) Full pipeline on one messy sentence.
    messy = "<p>HELLO!!!</p> Visit https://x.io — I’m lovin’ cafés & NLP \U0001F600 #AI"
    tokens = preprocess(messy, stopwords_set=stopwords_set)
    assert "url" not in tokens and "<url>" in tokens, "URL should be a placeholder, not stripped"
    assert "i" not in tokens, "the pronoun 'i' is a stopword and should be removed"
    print("1) messy sentence ->", tokens)
    print()

    # 2) Stemming vs lemmatization on tricky words.
    pairs = [
        ("studies", "n"),
        ("studying", "v"),
        ("better", "a"),
        ("best", "a"),
        ("are", "v"),
        ("is", "v"),
        ("mice", "n"),
        ("organization", "n"),
        ("running", "v"),
    ]
    rows = stem_vs_lemma(pairs)
    disagreements = sum(1 for *_, agree in rows if not agree)
    assert disagreements >= 5, "the point is that stemmer and lemmatizer disagree on most irregulars"
    print("2) stem vs lemma (word: stem | lemma | agree?):")
    for word, _pos, stem, lemma, agree in rows:
        flag = "agree" if agree else "DIFFER"
        print(f"     {word:13s} {stem:10s} | {lemma:12s} {flag}")
    print()

    # 3) Stopword removal can flip sentiment -- the negation trap.
    naive = [t for t in "this movie is not good".split() if t not in stopwords_set]
    assert "not" not in naive, "'not' is on the standard stoplist -- removing it inverts sentiment"
    print("3) de-stopworded 'this movie is not good' ->", naive, "(negation destroyed)")
    print()

    # 4) Vocabulary shrinking, stage by stage.
    train_docs, train_labels, test_docs, test_labels = load_newsgroups()
    stages = vocab_shrink_stages(train_docs, stopwords_set=stopwords_set)
    sizes = [size for _, size in stages]
    assert sizes == sorted(sizes, reverse=True), "each cleaning step must not grow the vocabulary"
    print("4) vocabulary size after each cleaning step:")
    for name, size in stages:
        print(f"     {name.replace(chr(10), ' '):26s} {size:6d}")
    print()

    # 5) Measured classifier sweep.
    sweep = classifier_sweep(
        train_docs, train_labels, test_docs, test_labels, stopwords_set=stopwords_set
    )
    raw_vocab = sweep[0][1]
    stem_vocab = sweep[-1][1]
    assert stem_vocab < raw_vocab, "stemming must shrink the TF-IDF vocabulary"
    print("5) TF-IDF + LogReg on 20 Newsgroups (config: vocab | accuracy):")
    for name, vocab, accuracy in sweep:
        print(f"     {name:20s} vocab={vocab:6d}  acc={accuracy:.4f}")
    print()

    # 6) Zipf's law.
    ranks, freqs, slope, intercept, top = zipf_rank_frequency(train_docs)
    assert -1.2 < slope < -0.7, f"a natural corpus should show a Zipf slope near -1, got {slope:.3f}"
    print(f"6) Zipf slope over ranks 1..1000: {slope:.3f} (ideal -1); intercept {intercept:.3f}")
    print("     top terms:", [w for w, _ in top])
    print()

    # 7) Unicode normalization.
    uni = unicode_demo()
    assert uni["equal_raw"] is False and uni["equal_after_nfc"] is True, "NFC must merge the duplicates"
    print("7) unicode: 'café' (combining vs precomposed) equal raw?", uni["equal_raw"],
          "| equal after NFC?", uni["equal_after_nfc"])
    print("     NFKC('ﬁle') ->", repr(uni["nfkc_ligature_file"]),
          " NFKC('½') ->", repr(uni["nfkc_half"]),
          " NFKC('Ⅸ') ->", repr(uni["nfkc_roman_nine"]))


if __name__ == "__main__":
    main()
