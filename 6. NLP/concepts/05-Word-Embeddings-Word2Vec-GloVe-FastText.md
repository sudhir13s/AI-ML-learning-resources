---
id: "06-nlp/word-embeddings"
topic: "Word Embeddings (Word2Vec, GloVe, FastText)"
parent: "06-nlp"
level: intermediate
prereqs: ["linear-algebra", "softmax", "logistic-regression"]
interview_frequency: very-high
updated: 2026-06-19
---

# Word Embeddings — Word2Vec · GloVe · FastText
> Dense vector representations of words learned from co-occurrence, so that semantically similar
> words land close together and meaning becomes arithmetic (`king − man + woman ≈ queen`).
> The bridge from sparse one-hot/TF-IDF to the distributed representations that power all modern NLP.

**Why it matters:** the canonical "explain word2vec" question — skip-gram vs CBOW,
negative sampling vs hierarchical softmax, why GloVe uses global co-occurrence, how FastText handles
OOV/morphology via subwords, and the limits (static, context-free) that motivate BERT.

**⭐ Start here — suggested path:**

1. **Build intuition** — watch [StatQuest: Word2Vec](https://www.youtube.com/watch?v=viZrOnJclY0), then read the ⭐ [Illustrated Word2Vec](https://jalammar.github.io/illustrated-word2vec/). *Fastest way to a correct mental model before any math.*
2. **See why it works** — [Computerphile: Vectoring Words](https://www.youtube.com/watch?v=gQddtTdmG_8). *Makes the "king − man + woman ≈ queen" geometry click.*
3. **Get the math** — [CS224N Lec 2](https://www.youtube.com/watch?v=ERibwqs9p38) + [SLP3 Ch. 6](https://web.stanford.edu/~jurafsky/slp3/6.pdf). *The objective, gradients, and negative sampling you'll be asked to derive.*
4. **Read the sources** — [word2vec](https://arxiv.org/abs/1301.3781) → [negative sampling](https://arxiv.org/abs/1310.4546) → [GloVe](https://nlp.stanford.edu/pubs/glove.pdf) → [FastText](https://arxiv.org/abs/1607.04606). *Follow the progression: prediction-based → global co-occurrence → subword/OOV.*
5. **Make it concrete** — code it with the [TensorFlow word2vec tutorial](https://www.tensorflow.org/text/tutorials/word2vec) or [d2l Ch. 15](https://d2l.ai/chapter_natural-language-processing-pretraining/index.html). *Implementing skip-gram + negative sampling cements it.*

## 🎓 Courses (free)
- [Stanford CS224N — Lec 1 & 2: Word Vectors](https://web.stanford.edu/class/cs224n/) — **Stanford (Manning)** — the definitive treatment (word2vec → GloVe), full lectures + notes free on YouTube.
- [Google ML Crash Course — Embeddings](https://developers.google.com/machine-learning/crash-course/embeddings) — **Google** — short, free, applied intro to why/how embeddings work.

## 🎥 Videos
- [Word Embedding and Word2Vec, Clearly Explained](https://www.youtube.com/watch?v=viZrOnJclY0) — **StatQuest (Josh Starmer)** — gentle, from-scratch intuition for skip-gram + training.
- [The Illustrated Word2vec — A Gentle Intro to Word Embeddings](https://www.youtube.com/watch?v=ISPId9Lhc1g) — **Jay Alammar** — the video companion to the Start-here article; richly visual.
- [Vectoring Words (Word Embeddings)](https://www.youtube.com/watch?v=gQddtTdmG_8) — **Computerphile (Rob Miles)** — vivid visual intuition for *why* vector arithmetic captures meaning.
- [CS224N Lecture 2 — Word Vector Representations: word2vec](https://www.youtube.com/watch?v=ERibwqs9p38) — **Stanford (Manning)** — the rigorous derivation (objective, gradients, negative sampling).

## 📄 Key Papers
- [Efficient Estimation of Word Representations (word2vec)](https://arxiv.org/abs/1301.3781) — **Mikolov et al. (2013)** — introduces CBOW + skip-gram.
- [Distributed Representations of Words and Phrases](https://arxiv.org/abs/1310.4546) — **Mikolov et al. (2013)** — negative sampling + hierarchical softmax (the training tricks).
- [GloVe: Global Vectors for Word Representation](https://nlp.stanford.edu/pubs/glove.pdf) — **Pennington, Socher & Manning (2014)** — co-occurrence-matrix factorization view.
- [Enriching Word Vectors with Subword Information (FastText)](https://arxiv.org/abs/1607.04606) — **Bojanowski et al. (2017)** — character n-grams → handles OOV & morphology.

## 📰 Articles / Blogs (free, no paywall)
- [On Word Embeddings — Part 1](https://www.ruder.io/word-embeddings-1/) — **Sebastian Ruder** — the definitive survey series (history, models, math), fully open.
- [Word2Vec Tutorial — The Skip-Gram Model](https://mccormickml.com/2016/04/19/word2vec-tutorial-the-skip-gram-model/) — **Chris McCormick** — step-by-step skip-gram with negative sampling, free.
- [GloVe project page](https://nlp.stanford.edu/projects/glove/) — **Stanford NLP** — paper, pretrained vectors, and the intuition in one place.
- [Word2Vec (TensorFlow tutorial)](https://www.tensorflow.org/text/tutorials/word2vec) — **TensorFlow** — implement skip-gram + negative sampling end-to-end.

## 📚 Books (free, with chapters)
- [Speech and Language Processing, 3rd ed. — **Ch. 6 "Vector Semantics and Embeddings"**](https://web.stanford.edu/~jurafsky/slp3/6.pdf) — **Jurafsky & Martin** — the standard reference chapter (TF-IDF → word2vec).
- [Dive into Deep Learning — **Ch. 15.1–15.7 (word2vec, approximate training, GloVe, subword)**](https://d2l.ai/chapter_natural-language-processing-pretraining/index.html) — **Zhang et al.** — free, with runnable code.
- [A Primer on Neural Network Models for NLP — **§5 (word embeddings)**](https://arxiv.org/abs/1510.00726) — **Yoav Goldberg** — concise, rigorous, free on arXiv.

## 🔗 In this platform
- Concept depth (the *why*): [AI-ML-intuition 1.02 Dense Embeddings](../../../AI-ML-intuition/Module_1_Representation/1.02_Dense_Embeddings.md) · [1.02a Word2Vec / Skip-Gram](../../../AI-ML-intuition/Module_1_Representation/1.02a_Word_Embeddings_Word2Vec_Skip-Gram.md)
- Next concepts: [06 Contextual Embeddings (ELMo/BERT)](06-Contextual-Embeddings-ELMo-BERT.md) · [02 Tokenization & Subword](02-Tokenization-and-Subword-Algorithms.md)
- Related domain: [8. LLMs](../../8.%20LLMs/README.md)
