---
id: "00-basics/overfitting-and-underfitting"
topic: "Overfitting & Underfitting (intuition)"
parent: "00-basics"
level: beginner
prereqs: ["how-models-learn"]
interview_frequency: very-high
updated: 2026-06-20
---

# Overfitting & Underfitting
> The central tension of ML. **Underfitting:** the model is too simple — it misses the real pattern
> and is wrong on both training and new data (high bias). **Overfitting:** the model is too complex —
> it memorizes the training data, including its noise, and fails on new data (high variance). The job
> is to find the sweet spot in between, where the model *generalizes*.

**Why it matters:** "What is overfitting and how do you fix it?" is one of the most-asked interview
questions, period. The bias–variance tradeoff, the train/test gap as the diagnostic, and the fixes
(more data, regularization, simpler models, early stopping) are core practitioner knowledge you'll
use on every single project.

**⭐ Start here — suggested path:**

1. **Build the picture** — watch [Overfitting and Underfitting](https://www.youtube.com/watch?v=W-0-u6XVbE4). *The too-simple vs too-complex curves that make it obvious.*
2. **Read it precisely** — [Google: Overfitting](https://developers.google.com/machine-learning/crash-course/overfitting/overfitting). *Generalization, the train/test gap, and why simpler can be better.*
3. **Understand the tradeoff** — watch [StatQuest: Bias and Variance](https://www.youtube.com/watch?v=EuBBz3bI-aA), then explore [MLU-Explain: Bias-Variance](https://mlu-explain.github.io/bias-variance/). *The framework that explains *why* over/underfitting happen.*
4. **Learn the fixes** — watch [StatQuest: Ridge (L2) Regression](https://www.youtube.com/watch?v=Q81RR3yKn30) + [StatQuest: Cross Validation](https://www.youtube.com/watch?v=fSytzGwwBVw). *Regularization and CV — your two main weapons against overfitting.*
5. **See it in code** — run [scikit-learn: Underfitting vs Overfitting](https://scikit-learn.org/stable/auto_examples/model_selection/plot_underfitting_overfitting.html). *Fit too-simple/too-complex/just-right models and watch the curves.*

## 🎓 Courses (free)
- [Google ML Crash Course — Overfitting & Generalization](https://developers.google.com/machine-learning/crash-course/overfitting/overfitting) — **Google** — free; the train/test gap, generalization, and regularization.
- [Machine Learning Specialization — Course 1 (free to audit)](https://www.coursera.org/specializations/machine-learning-introduction) — **Andrew Ng / DeepLearning.AI** — bias/variance diagnosis and regularization, taught from scratch.
- [Kaggle Learn — Intro to ML (Underfitting & Overfitting)](https://www.kaggle.com/learn/intro-to-machine-learning) — **Kaggle** — a hands-on lesson tuning model complexity to the sweet spot.

## 🎥 Videos
- [Overfitting and Underfitting in Machine Learning](https://www.youtube.com/watch?v=W-0-u6XVbE4) — **Simplilearn** — the core picture: too-simple vs too-complex vs just-right.
- [Machine Learning Fundamentals: Bias and Variance](https://www.youtube.com/watch?v=EuBBz3bI-aA) — **StatQuest (Josh Starmer)** — the tradeoff that underlies over/underfitting.
- [Overfitting, Underfitting, and Bad Data](https://www.youtube.com/watch?v=0RT2Q0qwXSA) — **IBM Technology** — what goes wrong and how to spot it, concisely.
- [Regularization Part 1: Ridge (L2) Regression](https://www.youtube.com/watch?v=Q81RR3yKn30) — **StatQuest** — the most common fix for overfitting, explained clearly.
- [Machine Learning Fundamentals: Cross Validation](https://www.youtube.com/watch?v=fSytzGwwBVw) — **StatQuest** — how to detect overfitting honestly before it bites you.

## 📄 Key Papers
- [A Few Useful Things to Know About Machine Learning](https://homes.cs.washington.edu/~pedrod/papers/cacm12.pdf) — **Pedro Domingos (2012)** — "overfitting has many faces"; the clearest framing of generalization vs memorization.
- [Dropout: A Simple Way to Prevent Neural Networks from Overfitting](https://jmlr.org/papers/volume15/srivastava14a/srivastava14a.pdf) — **Srivastava et al. (2014)** — the canonical regularization technique for deep nets.
- [Reconciling modern machine-learning practice and the bias–variance trade-off](https://arxiv.org/abs/1812.11118) — **Belkin et al. (2019)** — "double descent": where the classic picture gets surprising in deep learning.

## 📰 Articles / Blogs (free, no paywall)
- [What Is Overfitting vs. Underfitting?](https://www.ibm.com/think/topics/overfitting-vs-underfitting) — **IBM** — both failure modes side by side with detection and fixes.
- [Bias-Variance Tradeoff (interactive)](https://mlu-explain.github.io/bias-variance/) — **Amazon MLU-Explain** — a beautiful interactive visual of the core tradeoff.
- [Underfitting vs. Overfitting (scikit-learn)](https://scikit-learn.org/stable/auto_examples/model_selection/plot_underfitting_overfitting.html) — **scikit-learn docs** — runnable code that plots all three regimes.

## 📚 Books (free, with chapters)
- [An Introduction to Statistical Learning — Ch. 2.2 "Assessing Model Accuracy"](https://www.statlearning.com/) — **James et al.** — free PDF; the definitive bias–variance treatment for beginners.
- [Dive into Deep Learning — Ch. 4.4 "Model Selection, Underfitting, Overfitting"](https://d2l.ai/chapter_multilayer-perceptrons/underfit-overfit.html) — **Zhang et al.** — with runnable experiments.
- [Neural Networks and Deep Learning — Ch. 3 "Overfitting and regularization"](http://neuralnetworksanddeeplearning.com/chap3.html) — **Michael Nielsen** — free; intuition + the standard fixes.

## 🔗 In this platform
- Prev/next: [04 How Models Learn](04-How-Models-Learn.md) · [12 Your First ML Project](12-Your-First-ML-Project.md)
- Go deeper — regularization in linear models: [03. Supervised Learning](../../03.%20Supervised_Learning/concepts/README.md)
- Go deeper — dropout, weight decay, early stopping: [05. Deep Learning](../../05.%20Deep_Learning/concepts/README.md)
