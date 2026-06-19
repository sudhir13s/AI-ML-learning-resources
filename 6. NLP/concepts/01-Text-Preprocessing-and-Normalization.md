---
id: "06-nlp/text-preprocessing"
topic: "Text Preprocessing & Normalization (tokenize, stem, lemmatize, stopwords)"
parent: "06-nlp"
level: beginner
prereqs: ["regular-expressions"]
interview_frequency: medium
updated: 2026-06-19
---

# Text Preprocessing & Normalization — tokenize · stem · lemmatize · stopwords
> Turning messy raw text into clean, consistent units a model can use. The classic pipeline —
> normalize case/Unicode, split into tokens, drop stopwords, and reduce words to a base form by
> **stemming** (crude chop) or **lemmatization** (dictionary-aware) — plus sentence segmentation.

**Why it matters:** preprocessing decides what signal survives before any model sees it, and small
choices (lowercasing, removing punctuation, stemming vs lemmatizing) measurably move downstream
accuracy. Be ready to explain **stemming vs lemmatization** (speed vs correctness), when stopword
removal helps or hurts, and why modern transformer pipelines do *less* preprocessing (subword
tokenizers handle morphology) than classical bag-of-words pipelines.

**⭐ Start here — suggested path:**

1. **Build intuition** — read [Introduction to NLP](https://victorzhou.com/blog/intro-to-nlp/) (**Victor Zhou**) for the end-to-end picture of what each step does. *See the pipeline before the tools.*
2. **See it in code** — watch [Stemming and Lemmatization](https://www.youtube.com/watch?v=HHAilAC3cXw) (**codebasics**). *Watch NLTK stemming vs spaCy lemmatization side by side.*
3. **Get the definitions right** — read [Stemming and lemmatization](https://nlp.stanford.edu/IR-book/html/htmledition/stemming-and-lemmatization-1.html) (**Stanford IR Book**). *The precise distinction interviewers probe.*
4. **Read the reference** — [SLP3 Ch. 2](https://web.stanford.edu/~jurafsky/slp3/2.pdf). *Normalization, tokenization, and edit distance in the standard text.*
5. **Make it concrete** — work through [NLTK Book Ch. 3](https://www.nltk.org/book/ch03.html) or [spaCy linguistic features](https://spacy.io/usage/linguistic-features). *Run a real pipeline on raw text.*

## 🎓 Courses (free)
- [Working with Text Data — Applied Language Technology](https://applied-language-technology.mooc.fi/html/index.html) — **University of Helsinki** — free MOOC; hands-on text normalization and spaCy pipelines.
- [Stanford CS224N: NLP with Deep Learning](https://web.stanford.edu/class/cs224n/) — **Stanford (Manning)** — situates preprocessing within the modern NLP pipeline.

## 🎥 Videos
- [Stemming and Lemmatization: NLP Tutorial For Beginners](https://www.youtube.com/watch?v=HHAilAC3cXw) — **codebasics** — NLTK stemming vs spaCy lemmatization in code.
- [Stemming and Lemmatization explained with code](https://www.youtube.com/watch?v=jCY0SvsVTzc) — **InsightsByRish** — a second, concise take on the same distinction.
- [NLP — Text Preprocessing and Text Classification](https://www.youtube.com/watch?v=nxhCyeRR75Q) — **Machine Learning TV** — the full clean-then-classify pipeline end to end.
- [NLP Tutorial with Python & NLTK](https://www.youtube.com/watch?v=X2vAabgKiuM) — **freeCodeCamp.org** — long, thorough walkthrough of tokenizing, stopwords, stemming, lemmatizing.
- [NLP with spaCy & Python — Course for Beginners](https://www.youtube.com/watch?v=dIUTsFT2MeQ) — **freeCodeCamp.org** — modern spaCy-based preprocessing from scratch.

## 📄 Key Papers
- [Speech and Language Processing, 3rd ed. — Ch. 2 (text)](https://web.stanford.edu/~jurafsky/slp3/2.pdf) — **Jurafsky & Martin** — the canonical reference on normalization and tokenization (no single seminal paper; this chapter is the standard citation).

## 📰 Articles / Blogs (free, no paywall)
- [An Introduction to NLP](https://victorzhou.com/blog/intro-to-nlp/) — **Victor Zhou** — clean, beginner-friendly overview of the preprocessing pipeline.
- [Stemming and lemmatization](https://nlp.stanford.edu/IR-book/html/htmledition/stemming-and-lemmatization-1.html) — **Stanford IR Book** — the precise, free textbook definitions.
- [Linguistic Features](https://spacy.io/usage/linguistic-features) — **spaCy docs** — how a production library does tokenization, lemmatization, and tagging.

## 📚 Books (free, with chapters)
- [Speech and Language Processing, 3rd ed. — **Ch. 2 "Regular Expressions, Text Normalization, and Edit Distance"**](https://web.stanford.edu/~jurafsky/slp3/2.pdf) — **Jurafsky & Martin** — the standard treatment of normalization and tokenization.
- [Natural Language Processing with Python — **Ch. 3 "Processing Raw Text"**](https://www.nltk.org/book/ch03.html) — **Bird, Klein & Loper** — free NLTK book; hands-on normalization and stemming.

## 🔗 In this platform
- Next concept: [02 Tokenization & Subword](02-Tokenization-and-Subword-Algorithms.md) — where preprocessing meets modern subword vocabularies.
- Concept depth (the *why*): [AI-ML-intuition 1.15 Tokenization & BPE](../../../AI-ML-intuition/Module_1_Representation/1.15_Tokenization_and_BPE.md)
- Then: [03 Bag-of-Words & TF-IDF](03-Bag-of-Words-and-TF-IDF.md) — the first models that consume preprocessed text.
