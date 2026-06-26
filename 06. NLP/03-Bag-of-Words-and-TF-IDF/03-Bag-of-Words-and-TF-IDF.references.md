---
id: "06-nlp/bow-tfidf/references"
topic: "Bag-of-Words & TF-IDF — References"
parent: "06-nlp/bow-tfidf"
type: references
updated: 2026-06-22
---

# Bag-of-Words & TF-IDF — references and further reading

> Companion link library for **[Bag-of-Words & TF-IDF](03-Bag-of-Words-and-TF-IDF.md)** (the concept page). External sources *and* internal links to related pages on this platform, kept separate so it can be reused as a standalone reference list. Grouped by type, best-first. Every entry is from a primary author or a recognized deep explainer — chosen for depth on *this* topic, not popularity, and every link verified.

**Start here — suggested path**:
1. **Build intuition** — watch [Bag of Words](https://www.youtube.com/watch?v=irzVuSO8o4g) (**ritvikmath**). *See documents become count vectors before any weighting.*
2. **See why TF-IDF reweights** — watch [TF-IDF Explained](https://www.youtube.com/watch?v=zLMEnNbdh4Q) (**DataMListic**). *Why "the" gets crushed and topic words rise.*
3. **Get the math** — read [tf–idf weighting](https://nlp.stanford.edu/IR-book/html/htmledition/tf-idf-weighting-1.html) (**Stanford IR Book**). *The exact formula and the log-IDF justification.*
4. **Read the source** — [SLP3 Ch. 6 §6.5](https://web.stanford.edu/~jurafsky/slp3/6.pdf) (**Jurafsky & Martin**). *TF-IDF inside the vector-semantics chapter, leading into embeddings.*
5. **Go to BM25** — watch [BM25: The Most Important Text Metric](https://www.youtube.com/watch?v=ruBm9WywevM) (**ritvikmath**), then read [Okapi BM25](https://nlp.stanford.edu/IR-book/html/htmledition/okapi-bm25-a-non-binary-model-1.html) (**Stanford IR Book**). *The saturation + length-normalized upgrade real search engines use.*
6. **Make it concrete** — code it with [scikit-learn text feature extraction](https://scikit-learn.org/stable/modules/feature_extraction.html#text-feature-extraction). *`CountVectorizer` → `TfidfVectorizer` end to end.*

**Videos**:
- [Bag of Words : Natural Language Processing](https://www.youtube.com/watch?v=irzVuSO8o4g) — **ritvikmath** — crisp intuition for the count-vector representation, no jargon.
- [TF-IDF : Data Science Concepts](https://www.youtube.com/watch?v=OymqCnh-APA) — **ritvikmath** — the cleanest derivation of *why* TF-IDF reweights, on a whiteboard.
- [Term Frequency–Inverse Document Frequency (TF-IDF) Explained](https://www.youtube.com/watch?v=zLMEnNbdh4Q) — **DataMListic** — short, sharp explanation of the formula and the log-IDF.
- [BM25 : The Most Important Text Metric in Data Science](https://www.youtube.com/watch?v=ruBm9WywevM) — **ritvikmath** — derives BM25 from TF-IDF, the saturation and length knobs explained.
- [Calculate TF-IDF in NLP (Simple Example)](https://www.youtube.com/watch?v=vZAXpvHhQow) — **Data Science Garage** — a fully worked numeric example by hand, matching this page's style.
- [NLP: Bag of Words (BoW) | Sklearn CountVectorizer](https://www.youtube.com/watch?v=0VwKMu7N014) — **Practical Data Science and ML** — implement BoW with scikit-learn step by step.
- [Using TF-IDF to convert text to useful features](https://www.youtube.com/watch?v=hXNbFNCgPfY) — **Mike Bernico** — TF-IDF as the input to a real classifier pipeline.

**Courses (free)**:
- [Stanford CS276 — Information Retrieval and Web Search](https://web.stanford.edu/class/cs276/) — **Stanford (Manning & Tan)** — the definitive IR course; tf-idf, the vector space model, and BM25 in depth.
- [Stanford CS224N — NLP with Deep Learning](https://web.stanford.edu/class/cs224n/) — **Stanford (Manning)** — sets BoW/TF-IDF as the sparse baseline before dense vectors.
- [scikit-learn — Classification of text documents (20 newsgroups)](https://scikit-learn.org/stable/auto_examples/text/plot_document_classification_20newsgroups.html) — **scikit-learn** — free, hands-on BoW → TF-IDF → linear classifier pipeline.

**Articles / blogs (free, no paywall)**:
- [tf–idf weighting](https://nlp.stanford.edu/IR-book/html/htmledition/tf-idf-weighting-1.html) — **Stanford IR Book (Manning, Raghavan & Schütze)** — the canonical free derivation of the TF-IDF formula and the log-IDF justification.
- [Okapi BM25: a non-binary model](https://nlp.stanford.edu/IR-book/html/htmledition/okapi-bm25-a-non-binary-model-1.html) — **Stanford IR Book** — the BM25 ranking function derived from probabilistic retrieval, with the $k_1$/$b$ knobs.
- [Scoring, term weighting & the vector space model](https://nlp.stanford.edu/IR-book/html/htmledition/scoring-term-weighting-and-the-vector-space-model-1.html) — **Stanford IR Book** — the chapter index tying tf, idf, cosine, and the vector space model together.
- [Text feature extraction](https://scikit-learn.org/stable/modules/feature_extraction.html#text-feature-extraction) — **scikit-learn docs** — `CountVectorizer`/`TfidfVectorizer` with the *exact* smoothed-idf + L2 weighting this page reproduces.
- [An Introduction to NLP](https://victorzhou.com/blog/intro-to-nlp/) — **Victor Zhou** — places BoW in the broader pipeline with clear, worked examples.
- [tf–idf (Wikipedia)](https://en.wikipedia.org/wiki/Tf%E2%80%93idf) — well-sourced summary of every tf and idf variant in one table, with the probabilistic-idf derivation.
- [Okapi BM25 (Wikipedia)](https://en.wikipedia.org/wiki/Okapi_BM25) — the BM25 formula, its variants (BM25F, BM25+), and the parameter ranges, with primary citations.

**Papers**:
- [A Statistical Interpretation of Term Specificity (IDF)](https://www.staff.city.ac.uk/~sbrp622/idfpapers/ksj_orig.pdf) — **Karen Spärck Jones (1972)** — the original paper that introduced inverse document frequency and the term-specificity argument.
- [The Probabilistic Relevance Framework: BM25 and Beyond](https://www.staff.city.ac.uk/~sbrp622/papers/foundations_bm25_review.pdf) — **Robertson & Zaragoza (2009)** — the definitive derivation of BM25 from probabilistic retrieval theory, by its authors.

**Books (free chapters)**:
- [Introduction to Information Retrieval — Ch. 6 "Scoring, term weighting & the vector space model"](https://informationretrieval.org/) — **Manning, Raghavan & Schütze** — the IR-grade treatment of tf-idf, cosine, and BM25, free online in full.
- [Speech and Language Processing, 3rd ed. — Ch. 6 §6.5 "TF-IDF"](https://web.stanford.edu/~jurafsky/slp3/6.pdf) — **Jurafsky & Martin** — TF-IDF inside the vector-semantics chapter, the bridge to embeddings.

**In this platform**:
- Concept page (full explanation): [Bag-of-Words & TF-IDF](03-Bag-of-Words-and-TF-IDF.md)
- Feeds this (the clean tokens): [Text Preprocessing & Normalization](../01-Text-Preprocessing-and-Normalization/01-Text-Preprocessing-and-Normalization.md) · [Tokenization & Subword Algorithms](../02-Tokenization-and-Subword-Algorithms/02-Tokenization-and-Subword-Algorithms.md)
- Related representation idea: [N-gram Language Models & Smoothing](../04-N-gram-Language-Models-and-Smoothing/04-N-gram-Language-Models-and-Smoothing.md) — the same contiguous-window features, used for prediction.
- The dense successor: [Word Embeddings (Word2Vec, GloVe, FastText)](../05-Word-Embeddings-Word2Vec-GloVe-FastText/05-Word-Embeddings-Word2Vec-GloVe-FastText.md) · [Contextual Embeddings (ELMo, BERT)](../06-Contextual-Embeddings-ELMo-BERT/06-Contextual-Embeddings-ELMo-BERT.md)
- Puts it to work: [Text Classification & Sentiment Analysis](../10-Text-Classification-and-Sentiment-Analysis/10-Text-Classification-and-Sentiment-Analysis.md) · [Information Retrieval & Semantic Search](../16-Information-Retrieval-and-Semantic-Search/16-Information-Retrieval-and-Semantic-Search.md) · [Topic Modeling (LDA, NMF)](../15-Topic-Modeling-LDA-NMF/15-Topic-Modeling-LDA-NMF.md)
