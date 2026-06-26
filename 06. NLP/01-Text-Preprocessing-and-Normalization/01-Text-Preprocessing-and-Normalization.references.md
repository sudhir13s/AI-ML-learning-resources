---
id: "06-nlp/text-preprocessing/references"
topic: "Text Preprocessing & Normalization — References"
parent: "06-nlp/text-preprocessing"
type: references
updated: 2026-06-27
---

# Text Preprocessing & Normalization — references and further reading

> Companion link library for **[Text Preprocessing & Normalization](01-Text-Preprocessing-and-Normalization.md)** (the concept page). External sources *and* internal links to related pages on this platform, kept separate so it can be reused as a standalone reference list. Grouped by type, best-first. Every entry is from a primary author or a recognized deep explainer — chosen for depth on *this* topic, not popularity.

**Start here — suggested path**:
1. **See the pipeline whole** — read [An Introduction to NLP](https://victorzhou.com/blog/intro-to-nlp/) (**Victor Zhou**). *The end-to-end picture of what each preprocessing step does, before the tools.*
2. **Get the canonical definitions** — read [Speech and Language Processing, Ch. 2](https://web.stanford.edu/~jurafsky/slp3/2.pdf) (**Jurafsky & Martin**). *Normalization, tokenization, and edit distance in the standard textbook.*
3. **Nail stemming vs lemmatization** — read [Stemming and lemmatization](https://nlp.stanford.edu/IR-book/html/htmledition/stemming-and-lemmatization-1.html) (**Manning, IR Book Ch. 2**), then watch [Stemming and Lemmatization](https://www.youtube.com/watch?v=HHAilAC3cXw) (**codebasics**). *The precise distinction interviewers probe, then seen in code.*
4. **Understand Unicode normalization** — skim [UAX #15: Unicode Normalization Forms](https://unicode.org/reports/tr15/) (**Unicode Consortium**). *Why NFC vs NFKC matters and what each form does.*
5. **Run a real pipeline** — work through [NLTK Book Ch. 3](https://www.nltk.org/book/ch03.html) and [spaCy linguistic features](https://spacy.io/usage/linguistic-features). *Tokenize, stem, lemmatize on raw text with two production libraries.*

**Videos**:
- [Stemming and Lemmatization: NLP Tutorial For Beginners](https://www.youtube.com/watch?v=HHAilAC3cXw) — **codebasics** — NLTK stemming vs spaCy lemmatization side by side in code; the clearest short take.
- [NLP with spaCy & Python — Course for Beginners](https://www.youtube.com/watch?v=dIUTsFT2MeQ) — **freeCodeCamp.org (Dr. Sowmya Vajjala)** — modern spaCy-based preprocessing from scratch (tokenizing, lemmatizing, the pipeline).
- [Natural Language Processing with Python & NLTK](https://www.youtube.com/watch?v=X2vAabgKiuM) — **freeCodeCamp.org** — long, thorough walkthrough of tokenizing, stopwords, stemming, and lemmatizing.
- [Tokenization, Stemming, Lemmatization, Stopwords](https://www.youtube.com/watch?v=nxhCyeRR75Q) — **Machine Learning TV** — the full clean-then-classify preprocessing pipeline end to end.
- [Text Processing — Tokenization, Stop Words, Stemming, Lemmatization (CS50 / Natural Language Processing)](https://www.youtube.com/watch?v=8u66Ava9P-Y) — **freeCodeCamp.org (David J. Malan, Harvard CS50 AI)** — preprocessing within a rigorous, university-grade NLP lecture; situates each step in the larger pipeline.

**Interactive & tools**:
- [spaCy 101 — linguistic features (interactive)](https://spacy.io/usage/spacy-101) — **Explosion AI** — see tokenization, lemmatization, and POS tagging on your own text with the displaCy visualizer.
- [unicodedata — Unicode database](https://docs.python.org/3/library/unicodedata.html) — **Python docs** — the `normalize("NFKC", s)` one-liner from the page, plus character categories used for accent stripping.

**Courses (free)**:
- [Stanford CS224N — NLP with Deep Learning](https://web.stanford.edu/class/cs224n/) — **Stanford (Manning)** — situates preprocessing within the full modern NLP pipeline; lectures + notes free.
- [Working with Text Data — Applied Language Technology](https://applied-language-technology.mooc.fi/html/index.html) — **University of Helsinki** — free MOOC; hands-on text normalization and spaCy pipelines.

**Articles / blogs (free, no paywall)**:
- [An Introduction to NLP](https://victorzhou.com/blog/intro-to-nlp/) — **Victor Zhou** — clean, beginner-friendly overview of the preprocessing pipeline and why each step exists.
- [Stemming and lemmatization](https://nlp.stanford.edu/IR-book/html/htmledition/stemming-and-lemmatization-1.html) — **Manning, Raghavan & Schütze (IR Book Ch. 2)** — the precise, free textbook definitions and the Porter-stemmer discussion.
- [Dropping common terms: stop words](https://nlp.stanford.edu/IR-book/html/htmledition/dropping-common-terms-stop-words-1.html) — **IR Book Ch. 2** — the stopword trade-off (and why modern IR keeps them) from the canonical source.
- [The Porter Stemming Algorithm](https://tartarus.org/martin/PorterStemmer/) — **Martin Porter** — the original author's home page with the algorithm, rules, and reference implementations.
- [Snowball: A language for stemming algorithms](https://snowballstem.org/) — **Martin Porter** — the successor (“Porter2”) stemmers for many languages, with the rule definitions.
- [Linguistic Features](https://spacy.io/usage/linguistic-features) — **spaCy docs** — how a production library does tokenization, lemmatization, and POS tagging.
- [The Absolute Minimum Every Developer Must Know About Unicode](https://tonsky.me/blog/unicode/) — **Nikita Prokopov (tonsky)** — code points, combining marks, and normalization explained vividly; why “café” can be two strings.
- [Zipf's law: modeling the distribution of terms (IR Book §5.1.2)](https://nlp.stanford.edu/IR-book/html/htmledition/zipfs-law-modeling-the-distribution-of-terms-1.html) — **Manning, Raghavan & Schütze** — the rank-frequency power law $\text{cf}_i \propto 1/i$ and the $s\approx 1$ exponent; the source for the Zipf math on the page and the motivation for stopword removal.

**Key papers / specs**:
- [An algorithm for suffix stripping](https://tartarus.org/martin/PorterStemmer/def.txt) — **Porter (1980)** — the original Porter stemmer definition (the rules cited on the page: SSES→SS, IES→I, …).
- [UAX #15: Unicode Normalization Forms](https://unicode.org/reports/tr15/) — **Unicode Consortium** — the authoritative spec for NFC / NFD / NFKC / NFKD.
- [UTS #39: Unicode Security Mechanisms](https://www.unicode.org/reports/tr39/) — **Unicode Consortium** — confusables / homoglyph spoofing, the case normalization does *not* solve.
- [SentencePiece: A simple and language independent subword tokenizer](https://arxiv.org/abs/1808.06226) — **Kudo & Richardson (2018)** — raw-text-in tokenization that does normalization + segmentation jointly (why modern pipelines preprocess less).

**Books (free chapters)**:
- [Speech and Language Processing, 3rd ed. — Ch. 2 “Regular Expressions, Text Normalization, and Edit Distance”](https://web.stanford.edu/~jurafsky/slp3/2.pdf) — **Jurafsky & Martin** — the standard treatment of normalization, tokenization, and edit distance.
- [Natural Language Processing with Python — Ch. 3 “Processing Raw Text”](https://www.nltk.org/book/ch03.html) — **Bird, Klein & Loper** — the free NLTK book; hands-on normalization, stemming, and tokenizing.
- [Introduction to Information Retrieval — Ch. 2 “The term vocabulary and postings lists”](https://nlp.stanford.edu/IR-book/) — **Manning, Raghavan & Schütze** — tokenization, normalization, stemming, and stopwords from the IR perspective, free online.

**In this platform**:
- Concept page (full explanation): [Text Preprocessing & Normalization](01-Text-Preprocessing-and-Normalization.md)
- Runnable code (single source of truth): [text_preprocessing.py](code/text_preprocessing.py) — the pipeline, Zipf fit, stemmer/lemmatizer, classifier sweep, and Unicode demo as one verified module.
- Step-by-step teaching notebook: [01-Text-Preprocessing-and-Normalization.ipynb](code/01-Text-Preprocessing-and-Normalization.ipynb) — one idea per cell, each asserting its point before printing.
- Figure generator: [make_figures_01.py](code/make_figures_01.py) — regenerates every embedded figure from the same functions, so the prose and images can't drift.
- Next concept: [Tokenization & Subword Algorithms](../02-Tokenization-and-Subword-Algorithms/02-Tokenization-and-Subword-Algorithms.md) — where preprocessing meets modern subword vocabularies (BPE, WordPiece, SentencePiece, Unigram).
- Concept depth (the *why* of subwords): [AI-ML-intuition 1.15 Tokenization & BPE](../../../AI-ML-intuition/Module_1_Representation/1.15_Tokenization_and_BPE.md)
- The first models that consume preprocessed text: [Bag-of-Words & TF-IDF](../03-Bag-of-Words-and-TF-IDF/03-Bag-of-Words-and-TF-IDF.md)
- Why transformers want *minimal* preprocessing: [Contextual Embeddings (ELMo / BERT)](../06-Contextual-Embeddings-ELMo-BERT/06-Contextual-Embeddings-ELMo-BERT.md)
- Tasks where you must *keep* case and stopwords: [Sequence Labeling — POS & NER](../09-Sequence-Labeling-POS-and-NER/09-Sequence-Labeling-POS-and-NER.md)
