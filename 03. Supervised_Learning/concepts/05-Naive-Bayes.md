---
id: "03-supervised-learning/naive-bayes"
topic: "Naive Bayes"
parent: "03-supervised-learning"
level: beginner
prereqs: ["probability", "bayes-theorem", "supervised-learning-basics"]
interview_frequency: high
updated: 2026-06-19
---

# Naive Bayes
> Apply Bayes' theorem with one bold simplifying assumption — that features are *conditionally
> independent given the class* — so the joint likelihood factorizes into a product of per-feature
> terms. Fast, needs little data, and famously strong for text classification despite its "naive" assumption.

**Why it matters:** the classic probabilistic-classifier interview question. Expect: state Bayes'
theorem and the conditional-independence assumption, explain *why* it works well even when the
assumption is violated, the difference between Gaussian / Multinomial / Bernoulli variants, why you
need **Laplace smoothing** (zero-probability problem), and why it's a strong baseline for spam and
document classification.

**⭐ Start here — suggested path:**

1. **Build intuition** — watch [StatQuest: Naive Bayes](https://www.youtube.com/watch?v=O2L2Uv9pdDA). *See the spam-filter example: how per-word probabilities multiply into a class score.*
2. **See the continuous case** — watch [StatQuest: Gaussian Naive Bayes](https://www.youtube.com/watch?v=H3EjCKtlVog). *How to handle continuous features by assuming a Gaussian per class.*
3. **Get the math** — read [SLP3 Ch. 4 "Naive Bayes and Sentiment Classification"](https://web.stanford.edu/~jurafsky/slp3/4.pdf). *The cleanest derivation: the bag-of-words model, MLE counts, and Laplace smoothing.*
4. **Understand why it works** — skim [Domingos & Pazzani: On the Optimality of the Simple Bayesian Classifier](https://gwern.net/doc/ai/1997-domingos.pdf). *Why the independence assumption can be violated wildly yet classification stays accurate.*
5. **Make it concrete** — implement with [scikit-learn Naive Bayes](https://scikit-learn.org/stable/modules/naive_bayes.html). *Build a text classifier with `MultinomialNB`, tune the smoothing `alpha`, and compare to logistic regression.*

## 🎓 Courses (free)
- [Speech and Language Processing — Ch. 4 (text classification)](https://web.stanford.edu/~jurafsky/slp3/4.pdf) — **Jurafsky & Martin (Stanford)** — the gold-standard free treatment of Naive Bayes for text.
- [CS229: Machine Learning — Lecture notes (Generative learning)](https://cs229.stanford.edu/main_notes.pdf) — **Stanford (Ng)** — Naive Bayes and GDA as generative classifiers, rigorously.
- [Google ML Crash Course](https://developers.google.com/machine-learning/crash-course) — **Google** — the broader classification context and evaluation, free and applied.

## 🎥 Videos
- [Naive Bayes, Clearly Explained!!!](https://www.youtube.com/watch?v=O2L2Uv9pdDA) — **StatQuest (Josh Starmer)** — the gentle, from-scratch intuition with the spam-filter example.
- [Gaussian Naive Bayes, Clearly Explained!!!](https://www.youtube.com/watch?v=H3EjCKtlVog) — **StatQuest (Josh Starmer)** — extending Naive Bayes to continuous features.
- [The Math Behind Bayesian Classifiers Clearly Explained!](https://www.youtube.com/watch?v=lFJbZ6LVxN8) — **Normalized Nerd** — a clean visual derivation of the posterior and the decision rule.
- [Naive Bayes Classifier: A Friendly Approach](https://www.youtube.com/watch?v=Q8l0Vip5YUw) — **Serrano.Academy (Luis Serrano)** — a warm, worked-example walkthrough of the whole method.

## 📄 Key Papers
- [On the Optimality of the Simple Bayesian Classifier under Zero-One Loss](https://gwern.net/doc/ai/1997-domingos.pdf) — **Domingos & Pazzani (1997)** — *why* Naive Bayes is accurate despite violated independence; free PDF.
- [A Comparison of Event Models for Naive Bayes Text Classification](https://cdn.aaai.org/Workshops/1998/WS-98-05/WS98-05-007.pdf) — **McCallum & Nigam (1998)** — multinomial vs Bernoulli models for text; official AAAI PDF, free.

## 📰 Articles / Blogs (free, no paywall)
- [Naive Bayes (scikit-learn user guide)](https://scikit-learn.org/stable/modules/naive_bayes.html) — **scikit-learn** — the practical reference: Gaussian / Multinomial / Bernoulli / Complement variants and smoothing.
- [MLU-Explain: Precision & Recall](https://mlu-explain.github.io/precision-recall/) — **Amazon** — how to score the classifier you just built, interactively.
- [Speech and Language Processing — Naive Bayes chapter (HTML/PDF)](https://web.stanford.edu/~jurafsky/slp3/) — **Jurafsky & Martin** — the full free textbook hub; Ch. 4 is the Naive Bayes deep-dive.

## 📚 Books (free, with chapters)
- [Speech and Language Processing, 3rd ed. — **Ch. 4 "Naive Bayes and Sentiment Classification"**](https://web.stanford.edu/~jurafsky/slp3/4.pdf) — **Jurafsky & Martin** — the clearest derivation, from a text-classification angle.
- [The Elements of Statistical Learning — **Ch. 6.6.3 "Naive Bayes"**](https://hastie.su.domains/ElemStatLearn/) — **Hastie, Tibshirani & Friedman** — Naive Bayes among the kernel / density-estimation methods.
- [Information Theory, Inference, and Learning Algorithms — **Ch. 3 (Bayesian inference)**](https://www.inference.org.uk/itprnn/book.pdf) — **David MacKay** — the probabilistic foundations Naive Bayes rests on, free PDF.

## 🔗 In this platform
- Concept depth (the *why*): [AI-ML-intuition 3.04 Maximum Likelihood Estimation](../../../AI-ML-intuition/Module_3_Evaluation/3.04_Maximum_Likelihood_Estimation.md) · [3.05 Classification Metrics](../../../AI-ML-intuition/Module_3_Evaluation/3.05_Classification_Metrics_Precision_Recall_F1.md)
- Math prerequisites (the *why*): [01. Foundations](../../01.%20Foundations/concepts/README.md) — probability, Bayes' theorem, conditional independence.
- Prior / next concepts: [02 Logistic Regression](02-Logistic-Regression.md) — its discriminative counterpart · Classification Metrics *(coming soon)*
- Related domain: [2. Data Preprocessing](../../02.%20Data_Preprocessing/README.md) — text vectorization (bag-of-words, TF-IDF) feeds Naive Bayes.
