---
id: "06-nlp/ngram-language-models"
topic: "N-gram Language Models & Smoothing"
parent: "06-nlp"
level: intermediate
prereqs: ["probability", "text-preprocessing"]
interview_frequency: high
updated: 2026-06-19
---

# N-gram Language Models & Smoothing
> The classical way to model language: estimate the probability of the next word from the previous
> *n−1* words using counts (the **Markov assumption**). Sparse counts force **smoothing** — moving
> probability mass to unseen n-grams — and the model is scored by **perplexity**.

**Why it matters:** n-gram LMs are where every core language-modeling idea is introduced cleanly —
the **chain rule**, the **Markov assumption**, **maximum-likelihood counts**, **perplexity**, and the
whole smoothing family (**add-k/Laplace**, **backoff**, **interpolation**, **Kneser-Ney**). Be ready
to explain why MLE assigns zero probability to unseen n-grams, how Kneser-Ney fixes it via
continuation counts, and how perplexity relates to cross-entropy — the same metric used for neural LMs.

**⭐ Start here — suggested path:**

1. **Build intuition** — watch [NLP: Understanding the N-gram Language Models](https://www.youtube.com/watch?v=GiyMGBuu45w) (**Machine Learning TV**). *See counts → next-word probabilities.*
2. **See the math worked out** — watch [N-gram Language Modeling: Theory, Math, Code](https://www.youtube.com/watch?v=Vc2C1NZkH0E) (**Yash Agrawal**). *Chain rule, MLE, and perplexity by hand.*
3. **Get smoothing right** — read [SLP3 Ch. 3](https://web.stanford.edu/~jurafsky/slp3/3.pdf). *Add-k, backoff, interpolation, and Kneser-Ney in the standard text — the part interviews probe.*
4. **Understand perplexity deeply** — same chapter §3.2–3.3 ties perplexity to cross-entropy. *The bridge to neural-LM evaluation.*
5. **Make it concrete** — code it with [NLTK Book Ch. 2 (conditional frequency)](https://www.nltk.org/book/ch02.html) or [d2l language models](https://d2l.ai/chapter_recurrent-neural-networks/language-models-and-dataset.html). *Build a working n-gram model.*

## 🎓 Courses (free)
- [Stanford CS224N: NLP with Deep Learning](https://web.stanford.edu/class/cs224n/) — **Stanford (Manning)** — the language-modeling lecture moves from n-grams to neural LMs.
- [Dive into Deep Learning — Language Models & the Dataset](https://d2l.ai/chapter_recurrent-neural-networks/language-models-and-dataset.html) — **Zhang et al.** — free chapter with runnable n-gram + neural-LM code.

## 🎥 Videos
- [NLP: Understanding the N-gram Language Models](https://www.youtube.com/watch?v=GiyMGBuu45w) — **Machine Learning TV** — clear intro to counts and conditional probabilities.
- [N-gram Language Modeling: Theory, Math, Code](https://www.youtube.com/watch?v=Vc2C1NZkH0E) — **Yash Agrawal** — derivation plus a from-scratch implementation.
- [Introduction to N-gram Language Model](https://www.youtube.com/watch?v=ULwwR67fIes) — **AKAdemy** — concise overview of the n-gram idea and assumptions.
- [N-Grams Models — Simple Example](https://www.youtube.com/watch?v=HHZ468ZUgdw) — **Dr. Rehan Choudhry** — a worked numeric example end to end.

## 📄 Key Papers
- [An Empirical Study of Smoothing Techniques for Language Modeling](https://aclanthology.org/P96-1041/) — **Chen & Goodman (1996)** — the definitive comparison of smoothing methods (Kneser-Ney wins).
- [A Statistical Interpretation of Term Specificity](https://www.staff.city.ac.uk/~sbrp622/idfpapers/ksj_orig.pdf) — **Spärck Jones (1972)** — foundational statistical-LM thinking on rare-event weighting.

## 📰 Articles / Blogs (free, no paywall)
- [N-gram Language Models (SLP3 draft chapter)](https://web.stanford.edu/~jurafsky/slp3/3.pdf) — **Jurafsky & Martin** — the free, definitive write-up of n-grams + smoothing + perplexity.
- [The Unreasonable Effectiveness of Recurrent Neural Networks](https://karpathy.github.io/2015/05/21/rnn-effectiveness/) — **Andrej Karpathy** — frames *why* n-gram limits motivate neural LMs (still free).

## 📚 Books (free, with chapters)
- [Speech and Language Processing, 3rd ed. — **Ch. 3 "N-gram Language Models"**](https://web.stanford.edu/~jurafsky/slp3/3.pdf) — **Jurafsky & Martin** — chain rule, smoothing, backoff, Kneser-Ney, perplexity.
- [Natural Language Processing with Python — **Ch. 2 "Accessing Text Corpora and Lexical Resources"**](https://www.nltk.org/book/ch02.html) — **Bird, Klein & Loper** — conditional frequency distributions for n-gram counts.

## 🔗 In this platform
- Concept depth (the *why*): [AI-ML-intuition 5.05 Autoregressive Generation & Sampling](../../../AI-ML-intuition/Module_5_Generation/5.05_Autoregressive_Generation_Sampling.md)
- Related metric: [18 NLP Evaluation Metrics](18-NLP-Evaluation-Metrics.md) — where perplexity is defined precisely.
- Next: [08 Sequence-to-Sequence & Encoder–Decoder](08-Sequence-to-Sequence-and-Encoder-Decoder.md) — the neural successors to n-gram LMs.
