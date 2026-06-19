---
id: "06-nlp/bow-tfidf"
topic: "Bag-of-Words & TF-IDF"
parent: "06-nlp"
level: beginner
prereqs: ["text-preprocessing", "linear-algebra"]
interview_frequency: high
updated: 2026-06-19
---

# Bag-of-Words & TF-IDF
> The first way to turn text into vectors: represent a document as **counts of its words**, ignoring
> order (bag-of-words), then reweight so common-everywhere words count less and rare-but-frequent-here
> words count more (**TF-IDF**). Sparse, interpretable, and still a strong baseline.

**Why it matters:** BoW/TF-IDF is the baseline every text-classification and search system is measured
against, and the interview staple before embeddings. Be ready to define **term frequency** and
**inverse document frequency**, write the TF-IDF formula, explain *why* the IDF log term down-weights
stopwords, and contrast sparse BoW with dense embeddings (no semantics, no word order, huge
dimensionality — but fast, interpretable, and hard to beat on small data).

**⭐ Start here — suggested path:**

1. **Build intuition** — watch [Bag of Words](https://www.youtube.com/watch?v=irzVuSO8o4g) (**ritvikmath**). *See documents become count vectors before any weighting.*
2. **See why TF-IDF reweights** — watch [TF-IDF Explained](https://www.youtube.com/watch?v=zLMEnNbdh4Q) (**DataMListic**). *Why "the" gets crushed and topic words rise.*
3. **Get the math** — read [tf–idf weighting](https://nlp.stanford.edu/IR-book/html/htmledition/tf-idf-weighting-1.html) (**Stanford IR Book**). *The exact formula and the log-IDF justification.*
4. **Read the reference** — [SLP3 Ch. 6 §6.5](https://web.stanford.edu/~jurafsky/slp3/6.pdf). *TF-IDF in the standard text, leading into embeddings.*
5. **Make it concrete** — code it with [scikit-learn text feature extraction](https://scikit-learn.org/stable/modules/feature_extraction.html#text-feature-extraction). *`CountVectorizer` → `TfidfVectorizer` end to end.*

## 🎓 Courses (free)
- [Stanford CS224N: NLP with Deep Learning](https://web.stanford.edu/class/cs224n/) — **Stanford (Manning)** — sets BoW/TF-IDF as the sparse baseline before dense vectors.
- [scikit-learn — Classification of text documents (20 newsgroups)](https://scikit-learn.org/stable/auto_examples/text/plot_document_classification_20newsgroups.html) — **scikit-learn** — free, hands-on BoW → TF-IDF → classifier pipeline.

## 🎥 Videos
- [Bag of Words](https://www.youtube.com/watch?v=irzVuSO8o4g) — **ritvikmath** — crisp intuition for the count-vector representation.
- [Term Frequency–Inverse Document Frequency (TF-IDF) Explained](https://www.youtube.com/watch?v=zLMEnNbdh4Q) — **DataMListic** — short, clear derivation of why TF-IDF reweights.
- [Calculate TF-IDF in NLP (Simple Example)](https://www.youtube.com/watch?v=vZAXpvHhQow) — **Data Science Garage** — worked numeric example by hand.
- [NLP: Bag of Words (BoW) | Sklearn Count Vectorizer](https://www.youtube.com/watch?v=0VwKMu7N014) — **Practical Data Science and ML** — implement BoW with scikit-learn.

## 📄 Key Papers
- [A Statistical Interpretation of Term Specificity (IDF)](https://www.staff.city.ac.uk/~sbrp622/idfpapers/ksj_orig.pdf) — **Karen Spärck Jones (1972)** — the original paper that introduced inverse document frequency.

## 📰 Articles / Blogs (free, no paywall)
- [tf–idf weighting](https://nlp.stanford.edu/IR-book/html/htmledition/tf-idf-weighting-1.html) — **Stanford IR Book** — the canonical free derivation of the TF-IDF formula.
- [Text feature extraction](https://scikit-learn.org/stable/modules/feature_extraction.html#text-feature-extraction) — **scikit-learn docs** — `CountVectorizer`/`TfidfVectorizer` with the exact weighting it uses.
- [An Introduction to NLP](https://victorzhou.com/blog/intro-to-nlp/) — **Victor Zhou** — places BoW in the broader pipeline with clear examples.

## 📚 Books (free, with chapters)
- [Speech and Language Processing, 3rd ed. — **Ch. 6 §6.5 "TF-IDF"**](https://web.stanford.edu/~jurafsky/slp3/6.pdf) — **Jurafsky & Martin** — TF-IDF inside the vector-semantics chapter.
- [Introduction to Information Retrieval — **Ch. 6 "Scoring, term weighting & the vector space model"**](https://nlp.stanford.edu/IR-book/html/htmledition/scoring-term-weighting-and-the-vector-space-model-1.html) — **Manning, Raghavan & Schütze** — the IR-grade treatment, free online.

## 🔗 In this platform
- Concept depth (the *why*): [AI-ML-intuition 1.17 BoW & TF-IDF](../../../AI-ML-intuition/Module_1_Representation/1.17_BoW_and_TF-IDF.md)
- Prior step: [01 Text Preprocessing](01-Text-Preprocessing-and-Normalization.md) — the cleaning that feeds the count vectors.
- Next: [05 Word Embeddings](05-Word-Embeddings-Word2Vec-GloVe-FastText.md) — the dense successor to sparse BoW.
