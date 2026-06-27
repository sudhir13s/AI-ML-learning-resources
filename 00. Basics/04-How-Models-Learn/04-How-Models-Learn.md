---
id: "00-basics/how-models-learn"
topic: "How Models Learn (train/val/test, loss, gradient descent intuition)"
parent: "00-basics"
level: beginner
prereqs: ["what-is-ai-ml-dl", "types-of-machine-learning"]
interview_frequency: very-high
updated: 2026-06-20
---

# How Models Learn — Train/Val/Test · Loss · Gradient Descent
> Learning = **minimizing a loss**. A model makes predictions, a **loss function** scores how wrong
> they are, and **gradient descent** nudges the parameters downhill to make the loss smaller — over
> and over. You measure honestly by splitting data into **train / validation / test** so you tune on
> one set and judge on data the model has never seen.

**Why it matters:** "How does a model actually learn?" and "why do we split data into train/val/test?"
are bedrock interview questions. If you can explain loss → gradient → update and why the test set must
stay untouched, you understand the engine under every algorithm in the rest of this repo.

**⭐ Start here — suggested path:**

1. **Feel gradient descent** — watch [3Blue1Brown: Gradient descent, how networks learn](https://www.youtube.com/watch?v=IHZwWFHWa-w). *The "ball rolling downhill" picture that makes optimization click.*
2. **Get loss precisely** — read [Google: Loss](https://developers.google.com/machine-learning/crash-course/linear-regression/loss), then [Gradient descent](https://developers.google.com/machine-learning/crash-course/linear-regression/gradient-descent). *What loss measures and how the update step uses its gradient.*
3. **See it step by step** — watch [StatQuest: Gradient Descent, Step-by-Step](https://www.youtube.com/watch?v=sDv4f4s2SB8). *Real numbers through the algorithm; demystifies the math.*
4. **Learn to split data honestly** — read [Google: Dividing datasets](https://developers.google.com/machine-learning/crash-course/overfitting/dividing-datasets) + watch [StatQuest: Cross Validation](https://www.youtube.com/watch?v=fSytzGwwBVw). *Why train/val/test (and CV) prevent fooling yourself.*
5. **Watch a full learning loop in code** — [Building a neural net from scratch](https://www.youtube.com/watch?v=w8yWXqWQYmU). *Forward pass → loss → gradient → update, in plain NumPy.*

## 🎓 Courses (free)
- [Google ML Crash Course — Linear Regression (Loss & Gradient Descent)](https://developers.google.com/machine-learning/crash-course/linear-regression/loss) — **Google** — free; the cleanest loss → gradient-descent walkthrough with interactive widgets.
- [Machine Learning Specialization — Course 1 (free to audit)](https://www.coursera.org/specializations/machine-learning-introduction) — **Andrew Ng / DeepLearning.AI** — derives cost functions and gradient descent from scratch.
- [Kaggle Learn — Intro to ML (Model Validation)](https://www.kaggle.com/learn/intro-to-machine-learning) — **Kaggle** — hands-on train/validation split and measuring error.

## 🎥 Videos
- [Gradient descent, how neural networks learn](https://www.youtube.com/watch?v=IHZwWFHWa-w) — **3Blue1Brown** — the canonical visual intuition for minimizing loss.
- [Gradient Descent, Step-by-Step](https://www.youtube.com/watch?v=sDv4f4s2SB8) — **StatQuest (Josh Starmer)** — the algorithm with concrete numbers, no hand-waving.
- [Stochastic Gradient Descent, Clearly Explained](https://www.youtube.com/watch?v=vMh0zPT0tLI) — **StatQuest** — why we use mini-batches and what "stochastic" buys us.
- [Machine Learning Fundamentals: Cross Validation](https://www.youtube.com/watch?v=fSytzGwwBVw) — **StatQuest** — train/val/test and k-fold, the honest way to estimate performance.
- [Building a neural network FROM SCRATCH](https://www.youtube.com/watch?v=w8yWXqWQYmU) — **Samson Zhang** — the full learn loop (forward → loss → backprop → update) in NumPy.

## 📄 Key Papers
- [Google ML Crash Course — Reducing Loss](https://developers.google.com/machine-learning/crash-course/linear-regression/gradient-descent) — **Google** — authoritative explainer of the gradient step, learning rate, and convergence.
- [An overview of gradient descent optimization algorithms](https://www.ruder.io/optimizing-gradient-descent/) — **Sebastian Ruder (2016)** — the reference survey of SGD/momentum/Adam, free and complete.
- [A Few Useful Things to Know About Machine Learning](https://homes.cs.washington.edu/~pedrod/papers/cacm12.pdf) — **Pedro Domingos (2012)** — why generalization (test set) is the real goal, not training error.

## 📰 Articles / Blogs (free, no paywall)
- [Gradient Descent — explained](https://www.ibm.com/think/topics/gradient-descent) — **IBM** — clear text walkthrough of the optimization loop and learning rate.
- [An overview of gradient descent optimization algorithms](https://www.ruder.io/optimizing-gradient-descent/) — **Sebastian Ruder** — the deeper dive once the intuition lands.
- [Cross-validation: evaluating estimator performance](https://scikit-learn.org/stable/modules/cross_validation.html) — **scikit-learn docs** — train/val/test and k-fold with runnable code.

## 📚 Books (free, with chapters)
- [Dive into Deep Learning — Ch. 3 "Linear Neural Networks" & Ch. 4.4 "Model Selection"](https://d2l.ai/chapter_linear-regression/index.html) — **Zhang et al.** — loss, gradient descent, and train/val/test with code.
- [Neural Networks and Deep Learning — Ch. 1–2](http://neuralnetworksanddeeplearning.com/chap1.html) — **Michael Nielsen** — gradient descent and how learning works, from zero.
- [An Introduction to Statistical Learning — Ch. 5 "Resampling Methods"](https://www.statlearning.com/) — **James et al.** — free PDF; cross-validation and honest error estimation.

## 🔗 In this platform
- Next concepts: [05 Overfitting & Underfitting](../05-Overfitting-and-Underfitting/05-Overfitting-and-Underfitting.md) · [12 Your First ML Project](../12-Your-First-ML-Project/12-Your-First-ML-Project.md)
- Go deeper — the math (calculus, optimization): [01. Foundations](../../01.%20Foundations/README.md)
- Go deeper — backprop & optimizers: [05. Deep Learning](../../05.%20Deep_Learning/README.md)
