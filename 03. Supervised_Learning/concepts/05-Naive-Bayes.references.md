---
id: "03-supervised-learning/naive-bayes/references"
topic: "Naive Bayes — References"
parent: "03-supervised-learning/naive-bayes"
type: references
updated: 2026-06-22
---

# Naive Bayes — references and further reading

> Companion link library for **[Naive Bayes](05-Naive-Bayes.md)** (the concept page). External sources *and* internal links to related pages on this platform, kept separate so it can be reused as a standalone reference list. Grouped by type, best-first. Every entry is from a primary author or a recognized deep explainer — chosen for depth on *this* topic.

**Start here — suggested path**:
1. **Build intuition** — watch [Naive Bayes, Clearly Explained](https://www.youtube.com/watch?v=O2L2Uv9pdDA) (**StatQuest**). *The spam-filter example: per-word probabilities multiplied into a class score.*
2. **See the continuous case** — watch [Gaussian Naive Bayes](https://www.youtube.com/watch?v=H3EjCKtlVog) (**StatQuest**). *Handling continuous features by assuming a Gaussian per class.*
3. **Get the math** — read [SLP3 Ch. 4](https://web.stanford.edu/~jurafsky/slp3/4.pdf). *The bag-of-words model, MLE counts, log-space, and Laplace smoothing.*
4. **Understand why it works** — skim [Domingos & Pazzani (1997)](https://gwern.net/doc/ai/1997-domingos.pdf) and [Ng & Jordan (2002)](https://ai.stanford.edu/~ang/papers/nips01-discriminativegenerative.pdf). *Why the independence assumption can be violated yet classification stays accurate, and the NB-vs-logistic-regression tradeoff.*
5. **Make it concrete** — build a text classifier with [scikit-learn Naive Bayes](https://scikit-learn.org/stable/modules/naive_bayes.html); tune `alpha`, compare to logistic regression.

**Videos**:
- [Naive Bayes, Clearly Explained!!!](https://www.youtube.com/watch?v=O2L2Uv9pdDA) — **StatQuest (Josh Starmer)** — the gentle, from-scratch intuition with the spam-filter example.
- [Gaussian Naive Bayes, Clearly Explained!!!](https://www.youtube.com/watch?v=H3EjCKtlVog) — **StatQuest (Josh Starmer)** — extending Naive Bayes to continuous features.
- [The Math Behind Bayesian Classifiers Clearly Explained!](https://www.youtube.com/watch?v=lFJbZ6LVxN8) — **Normalized Nerd** — a clean visual derivation of the posterior and the decision rule.
- [Naive Bayes Classifier: A Friendly Approach](https://www.youtube.com/watch?v=Q8l0Vip5YUw) — **Serrano.Academy (Luis Serrano)** — a warm, worked-example walkthrough of the whole method.

**Interactive & visual**:
- [Speech and Language Processing — Ch. 4 (worked Naive Bayes, with figures)](https://web.stanford.edu/~jurafsky/slp3/4.pdf) — **Jurafsky & Martin** — a fully worked sentiment example with the count tables and the log-space arithmetic drawn out.
- [Naive Bayes (scikit-learn user guide, with decision-boundary examples)](https://scikit-learn.org/stable/modules/naive_bayes.html) — **scikit-learn** — variants side by side with runnable plots of how each models the data.

**Courses (free)**:
- [Speech and Language Processing — Ch. 4 (text classification)](https://web.stanford.edu/~jurafsky/slp3/4.pdf) — **Jurafsky & Martin (Stanford)** — the gold-standard free treatment of Naive Bayes for text.
- [CS229: Machine Learning — Lecture notes (Generative learning)](https://cs229.stanford.edu/main_notes.pdf) — **Stanford (Ng)** — Naive Bayes and GDA as generative classifiers, rigorously.
- [Google ML Crash Course](https://developers.google.com/machine-learning/crash-course) — **Google** — the broader classification context and evaluation, free and applied.

**Articles / blogs (free, no paywall)**:
- [Naive Bayes (scikit-learn user guide)](https://scikit-learn.org/stable/modules/naive_bayes.html) — **scikit-learn** — the practical reference: Gaussian / Multinomial / Bernoulli / Complement variants and smoothing.
- [Speech and Language Processing — full textbook hub](https://web.stanford.edu/~jurafsky/slp3/) — **Jurafsky & Martin** — the free textbook home; Ch. 4 is the Naive Bayes deep-dive.

**Key papers**:
- [On the Optimality of the Simple Bayesian Classifier under Zero-One Loss](https://gwern.net/doc/ai/1997-domingos.pdf) — **Domingos & Pazzani (1997)** — *why* Naive Bayes is accurate despite violated independence.
- [On Discriminative vs. Generative Classifiers (naive Bayes vs logistic regression)](https://ai.stanford.edu/~ang/papers/nips01-discriminativegenerative.pdf) — **Ng & Jordan (2002)** — the generative–discriminative pairing: NB converges faster, logistic regression is asymptotically better.
- [A Comparison of Event Models for Naive Bayes Text Classification](https://cdn.aaai.org/Workshops/1998/WS-98-05/WS98-05-007.pdf) — **McCallum & Nigam (1998)** — multinomial vs Bernoulli models for text.

**Books (free chapters)**:
- [Speech and Language Processing, 3rd ed. — Ch. 4 "Naive Bayes and Sentiment Classification"](https://web.stanford.edu/~jurafsky/slp3/4.pdf) — **Jurafsky & Martin** — the clearest derivation, from a text-classification angle.
- [The Elements of Statistical Learning — Ch. 6.6.3 "Naive Bayes"](https://hastie.su.domains/ElemStatLearn/) — **Hastie, Tibshirani & Friedman** — Naive Bayes among kernel / density-estimation methods (and the LDA/QDA family).
- [Information Theory, Inference, and Learning Algorithms — Ch. 3 (Bayesian inference)](https://www.inference.org.uk/itprnn/book.pdf) — **David MacKay** — the probabilistic foundations (priors, MAP) Naive Bayes rests on.

**In this platform**:
- Concept page (full explanation): [Naive Bayes](05-Naive-Bayes.md)
- Concept depth (the *why*): [AI-ML-intuition 3.04 Maximum Likelihood Estimation](../../../AI-ML-intuition/Module_3_Evaluation/3.04_Maximum_Likelihood_Estimation.md) · [3.05 Classification Metrics](../../../AI-ML-intuition/Module_3_Evaluation/3.05_Classification_Metrics_Precision_Recall_F1.md)
- Related: [Logistic Regression](02-Logistic-Regression.md) (its discriminative twin — same linear form) · [Classification Metrics](14-Classification-Metrics.md) (how to score it) · [Cross-Validation](13-Cross-Validation.md) (tune `alpha`)
- Math prerequisites: [01. Foundations](../../01.%20Foundations/concepts/README.md) — probability, Bayes' theorem, conditional independence, MAP
