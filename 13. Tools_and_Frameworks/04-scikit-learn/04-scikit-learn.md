---
id: "13-tools-and-frameworks/scikit-learn"
topic: "scikit-learn (classical ML, pipelines, model selection)"
parent: "13-tools-and-frameworks"
level: beginner
prereqs: ["python", "numpy", "pandas"]
interview_frequency: high
updated: 2026-06-20
---

# scikit-learn — Classical ML · Pipelines · Model Selection
> The standard library for classical machine learning in Python: a consistent
> `fit`/`predict`/`transform` API across dozens of algorithms, plus preprocessing, `Pipeline`s,
> cross-validation, and hyperparameter search. The fastest path from a DataFrame to a trained,
> evaluated model.

**Why it matters:** before reaching for deep learning, most real problems are solved with
scikit-learn — and interviews lean on its concepts: train/test splits and leakage, cross-validation,
the estimator API, building leak-proof `Pipeline`s, and `GridSearchCV`/`RandomizedSearchCV`. It is
also where you internalize the *workflow* of supervised learning.

**⭐ Start here — suggested path:**

1. **See the API shape** — read [Getting Started](https://scikit-learn.org/stable/getting_started.html). *The unified `fit`/`predict`/`transform` API is the whole library in one idea.*
2. **Do a guided course** — start the [scikit-learn MOOC](https://inria.github.io/scikit-learn-mooc/) (INRIA, the maintainers). *The best free, hands-on course on the predictive-modeling workflow.*
3. **Watch the workflow live** — [Scikit-learn Crash Course](https://www.youtube.com/watch?v=0B5eIE_1vpU) (freeCodeCamp). *A concept-driven tour: preprocessing, metrics, meta-estimators.*
4. **Avoid leakage with Pipelines** — study the [User Guide](https://scikit-learn.org/stable/user_guide.html) on Pipelines, preprocessing, and cross-validation. *Pipelines + proper CV are the #1 thing interviews and real projects get wrong.*
5. **Adapt from examples** — browse the [example gallery](https://scikit-learn.org/stable/auto_examples/index.html). *Finding a worked example near your problem is the real practitioner workflow.*

## 🎓 Courses (free)
- [scikit-learn MOOC](https://inria.github.io/scikit-learn-mooc/) — **INRIA / scikit-learn core devs** — the definitive free course on the predictive-modeling workflow with the library.
- [scikit-learn User Guide](https://scikit-learn.org/stable/user_guide.html) — **scikit-learn team** — a complete, topic-by-topic course on every estimator family and workflow.

## 🎥 Videos
- [Scikit-learn Crash Course — Machine Learning Library for Python](https://www.youtube.com/watch?v=0B5eIE_1vpU) — **freeCodeCamp (Vincent Warmerdam)** — concept-first tour of the library.
- [Scikit-Learn Course — Machine Learning in Python Tutorial](https://www.youtube.com/watch?v=pqNCD_5r0IU) — **freeCodeCamp** — a structured, hands-on intro.
- [Machine Learning with Python and Scikit-Learn — Full Course](https://www.youtube.com/watch?v=hDKCxebp88A) — **freeCodeCamp** — a deep, project-based end-to-end course.
- [Scikit-Learn Tutorial — Machine Learning With Sklearn](https://www.youtube.com/watch?v=0Lt9w-BxKFQ) — **Simplilearn** — an additional guided walkthrough of the core API.

## 📄 Key Papers
- [Scikit-learn: Machine Learning in Python](https://www.jmlr.org/papers/v12/pedregosa11a.html) — **Pedregosa et al. (2011), *JMLR*** — the foundational open-access paper.
- [API design for machine learning software](https://arxiv.org/abs/1309.0238) — **Buitinck et al. (2013)** — explains the estimator/transformer/pipeline API decisions (free on arXiv).

## 📰 Articles / Blogs (free, no paywall)
- [Getting Started](https://scikit-learn.org/stable/getting_started.html) — **scikit-learn team** — the unified API and a first end-to-end example.
- [Common pitfalls and recommended practices](https://scikit-learn.org/stable/common_pitfalls.html) — **scikit-learn team** — data leakage, randomness, and how to do CV correctly.
- [Python Data Science Handbook — **Ch. 5 "Machine Learning"**](https://jakevdp.github.io/PythonDataScienceHandbook/05.00-machine-learning.html) — **Jake VanderPlas** — full chapter, scikit-learn-based, free online.

## 📚 Books (free, with chapters)
- [scikit-learn MOOC (full curriculum)](https://inria.github.io/scikit-learn-mooc/) — **INRIA** — a book-length, free, structured course with runnable notebooks.
- [Python Data Science Handbook](https://jakevdp.github.io/PythonDataScienceHandbook/) — **Jake VanderPlas** — entire book free; Ch. 5 is a thorough scikit-learn treatment.

## 🔗 In this platform
- Related domain: [03. Supervised Learning](../../03.%20Supervised_Learning/README.md) · [04. Unsupervised Learning](../../04.%20Unsupervised_Learning/README.md)
- Pairs with: [02 Pandas](../02-Pandas/02-Pandas.md) · [12 Weights & Biases](../12-Weights-and-Biases/12-Weights-and-Biases.md)
- Deeper concept (the *why*): model selection & metrics → [Deployment & MLOps](../../14.%20Deployment_and_MLOps/README.md)
