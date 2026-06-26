---
id: "06-nlp/ngram-language-models/references"
topic: "N-gram Language Models & Smoothing — References"
parent: "06-nlp/ngram-language-models"
type: references
updated: 2026-06-22
---

# N-gram Language Models and Smoothing — references and further reading

> Companion link library for **[N-gram Language Models and Smoothing](04-N-gram-Language-Models-and-Smoothing.md)** (the concept page). This file holds the curated links — external sources *and* internal links to related pages on this platform — kept separate so it can be reused as a standalone reference list. Grouped by type, best-first. Every entry is from a primary author (the inventors of these methods) or a recognized deep explainer — chosen for depth on *this* topic, not popularity.

**Start here — suggested path**:
1. **Build intuition** — watch [N-gram Language Models](https://www.youtube.com/watch?v=GiyMGBuu45w) (**Machine Learning TV / Dan Jurafsky-style walkthrough**). *See counts turn into next-word probabilities.*
2. **Get the definitive treatment** — read [Speech and Language Processing, 3rd ed., Ch. 3 "N-gram Language Models"](https://web.stanford.edu/~jurafsky/slp3/3.pdf) (**Jurafsky & Martin**). *The canonical chapter: chain rule, MLE, smoothing, Kneser-Ney, perplexity — the part interviews probe.*
3. **See why smoothing matters** — skim [An Empirical Study of Smoothing Techniques for Language Modeling](https://aclanthology.org/J99-1004/) (**Chen & Goodman 1999**). *The exhaustive comparison that crowned modified Kneser-Ney.*
4. **Connect perplexity to cross-entropy** — SLP3 §3.2–3.3 derives both. *The bridge to neural-LM evaluation.*
5. **Make it concrete** — code an n-gram LM with [NLTK Book Ch. 2](https://www.nltk.org/book/ch02.html) or [Dive into Deep Learning — Language Models](https://d2l.ai/chapter_recurrent-neural-networks/language-models-and-dataset.html). *Build and evaluate a working model.*

**Videos**:
- [N-gram Language Models](https://www.youtube.com/watch?v=GiyMGBuu45w) — **Machine Learning TV** — clear intro to counts and conditional probabilities.
- [Kneser-Ney Smoothing (NLP817 3.11)](https://www.youtube.com/watch?v=9SlJ76HtjoE) — **Herman Kamper** — a focused, rigorous lecture on absolute discounting and the continuation probability, the heart of the gold-standard smoother.
- [N-gram Language Modeling: Theory, Math, and Code](https://www.youtube.com/watch?v=Vc2C1NZkH0E) — **Learn With Yash Agrawal** — chain rule, MLE, and perplexity worked by hand plus a from-scratch implementation.
- [What is Perplexity?](https://www.youtube.com/watch?v=NURcDHhYe98) — **Hugging Face** — perplexity as branching factor and its link to entropy, explained concisely.

**Courses (free)**:
- [Stanford CS124 / CS224N — Language Modeling](https://web.stanford.edu/class/cs224n/) — **Stanford (Manning, Jurafsky)** — the language-modeling lectures that move from n-grams to neural LMs.
- [Dive into Deep Learning — Language Models and the Dataset](https://d2l.ai/chapter_recurrent-neural-networks/language-models-and-dataset.html) — **Zhang, Lipton, Li & Smola** — free chapter with runnable n-gram + neural-LM code and perplexity.

**Articles / blogs (free, no paywall)**:
- [N-gram Language Models (SLP3 Ch. 3, draft)](https://web.stanford.edu/~jurafsky/slp3/3.pdf) — **Jurafsky & Martin** — the free, definitive write-up of n-grams + smoothing + perplexity.
- [The Unreasonable Effectiveness of Recurrent Neural Networks](https://karpathy.github.io/2015/05/21/rnn-effectiveness/) — **Andrej Karpathy** — frames *why* n-gram limits motivate neural LMs, with a runnable char-level model.
- [NLTK Book, Ch. 2 — Accessing Text Corpora & Conditional Frequency Distributions](https://www.nltk.org/book/ch02.html) — **Bird, Klein & Loper** — conditional frequency distributions, the data structure behind n-gram counts.

**Key papers**:
- [A Mathematical Theory of Communication](https://people.math.harvard.edu/~ctm/home/text/others/shannon/entropy/entropy.pdf) — **Claude Shannon (1948)** — founds entropy, cross-entropy, and the prediction-of-English framing every LM inherits.
- [Estimation of Probabilities from Sparse Data for the Language Model Component of a Speech Recognizer](https://web.archive.org/web/20170825210849/http://l2r.cs.uiuc.edu/~danr/Teaching/CS546-09/Papers/Katz87.pdf) — **Slava Katz (1987)** — Katz backoff with Good-Turing discounting.
- [The Population Frequencies of Species and the Estimation of Population Parameters](https://www.jstor.org/stable/2333344) — **I. J. Good (1953)** — the Good-Turing estimator (the $N_1/N$ missing-mass idea).
- [Improved Backing-off for M-gram Language Modeling](https://ieeexplore.ieee.org/document/479394) — **Reinhard Kneser & Hermann Ney (1995)** — the original Kneser-Ney smoothing and continuation probability.
- [An Empirical Study of Smoothing Techniques for Language Modeling](https://aclanthology.org/J99-1004/) — **Stanley Chen & Joshua Goodman (1999)** — the definitive comparison; introduces *modified* Kneser-Ney and shows it wins.
- [Class-Based n-gram Models of Natural Language](https://aclanthology.org/J92-4003/) — **Brown, Della Pietra, deSouza, Lai & Mercer (1992)** — the IBM line: Brown clustering, the first big step toward sharing statistics across words (deleted interpolation grew from this group).
- [A Neural Probabilistic Language Model](https://www.jmlr.org/papers/volume3/bengio03a/bengio03a.pdf) — **Bengio, Ducharme, Vincent & Jauvin (2003)** — the paper that replaced n-gram counts with learned distributed representations.

**Books (free chapters)**:
- [Speech and Language Processing, 3rd ed. — Ch. 3 "N-gram Language Models"](https://web.stanford.edu/~jurafsky/slp3/3.pdf) — **Jurafsky & Martin** — chain rule, smoothing, backoff, Kneser-Ney, perplexity; the canonical reference.
- [Foundations of Statistical Natural Language Processing — Ch. 6 "Statistical Inference: n-gram Models over Sparse Data"](https://nlp.stanford.edu/fsnlp/) — **Manning & Schütze** — the classic deep treatment of smoothing over sparse counts.
- [Natural Language Processing with Python — Ch. 2](https://www.nltk.org/book/ch02.html) — **Bird, Klein & Loper** — conditional frequency distributions for building n-gram counts in code.

**In this platform**:
- Concept page (full explanation): [N-gram Language Models and Smoothing](04-N-gram-Language-Models-and-Smoothing.md)
- Precursor representation (counts → vectors): [Bag-of-Words and TF-IDF](../03-Bag-of-Words-and-TF-IDF/03-Bag-of-Words-and-TF-IDF.md) — the other count-based text model.
- The successor that fixed the semantics gap: [Word Embeddings: Word2Vec, GloVe, FastText](../05-Word-Embeddings-Word2Vec-GloVe-FastText/05-Word-Embeddings-Word2Vec-GloVe-FastText.md) — dense vectors so "blue" and "green" relate.
- Where perplexity is defined alongside the other metrics: [NLP Evaluation Metrics](../18-NLP-Evaluation-Metrics/18-NLP-Evaluation-Metrics.md).
- The neural successors to n-gram LMs: [Sequence-to-Sequence and Encoder–Decoder](../08-Sequence-to-Sequence-and-Encoder-Decoder/08-Sequence-to-Sequence-and-Encoder-Decoder.md), and the LLMs in [09. LLMs](../../09.%20LLMs/) (same chain-rule objective, learned estimator).
- The *why* behind autoregressive generation: [AI-ML-intuition 5.05 Autoregressive Generation & Sampling](../../../AI-ML-intuition/Module_5_Generation/5.05_Autoregressive_Generation_Sampling.md).
