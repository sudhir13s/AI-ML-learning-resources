---
id: "02-data-preprocessing/data-leakage"
topic: "Data Leakage"
parent: "02-data-preprocessing"
level: intermediate
prereqs: ["train-test-split", "cross-validation", "pipelines"]
interview_frequency: very-high
updated: 2026-06-20
---

# Data Leakage
> When information that won't be available at prediction time sneaks into training — through preprocessing
> fit on all data, target-derived features, or train/test contamination — giving great validation scores
> that collapse in production.

**Why it matters:** *the* silent killer of applied ML, and a favorite interview question because it
separates people who've shipped models from those who haven't. Be ready to name the two big classes —
**train/test contamination** (e.g., scaling/imputing before splitting) and **target leakage** (a feature
that's a proxy for, or computed from, the label) — and the fix: split first, fit all transforms inside a
Pipeline on training folds only.

**⭐ Start here — suggested path:**

1. **The concept** — watch [What is Data Leakage in ML?](https://www.youtube.com/watch?v=n9jz7G68pVg). *The clearest short definition with examples.*
2. **See concrete cases** — watch [Examples of Data/Target Leakage](https://www.youtube.com/watch?v=NaySLPTCgDM). *Real features that secretly encode the answer.*
3. **Do the Kaggle lesson** — do [Kaggle: Data Leakage](https://www.kaggle.com/code/alexisbcook/data-leakage). *Hands-on target leakage vs train-test contamination.*
4. **The structural fix** — read [sklearn: Common pitfalls (data leakage)](https://scikit-learn.org/stable/common_pitfalls.html#data-leakage). *Why Pipeline + fit-on-train-only prevents it.*
5. **Read the deep dive** — read [Leakage in Data Mining](https://www.cs.umb.edu/~ding/history/470_670_fall_2011/papers/cs670_Tran_PreferredPaper_LeakingInDataMining.pdf). *The formal taxonomy and famous real-world examples.*

## 🎓 Courses (free)
- [Kaggle Learn — Data Leakage](https://www.kaggle.com/code/alexisbcook/data-leakage) — **Kaggle** — the single best free hands-on lesson on the topic.
- [Google ML Crash Course — Overfitting / dividing datasets](https://developers.google.com/machine-learning/crash-course/overfitting/dividing-datasets) — **Google** — the splitting discipline that prevents contamination.

## 🎥 Videos
- [What is Data Leakage In Machine Learning?](https://www.youtube.com/watch?v=n9jz7G68pVg) — **Krish Naik** — concise definition with practical examples.
- [Examples of Data or Target Leakage](https://www.youtube.com/watch?v=NaySLPTCgDM) — **Rajistics** — concrete leaky features and how they fool models.
- [What is data leakage?](https://www.youtube.com/watch?v=cApPa55X2JU) — **CodeEmporium** — intuition + how to think about train/test contamination.
- [What is data leakage in machine learning?](https://www.youtube.com/watch?v=j9X8m5KSQAc) — **Karina Data Scientist** — short, clear framing of the problem.

## 📄 Key Papers
- [Leakage in Data Mining: Formulation, Detection, and Avoidance](https://www.cs.umb.edu/~ding/history/470_670_fall_2011/papers/cs670_Tran_PreferredPaper_LeakingInDataMining.pdf) — **Kaufman, Rosset & Perlich (2011)** — the definitive treatment; the taxonomy everyone cites (free PDF).
- [The ML Reproducibility Crisis from Leakage (a survey across fields)](https://arxiv.org/abs/2207.07048) — **Kapoor & Narayanan (2022)** — documents leakage errors invalidating published ML results; free on arXiv.

## 📰 Articles / Blogs (free, no paywall)
- [Common pitfalls and recommended practices — Data leakage](https://scikit-learn.org/stable/common_pitfalls.html#data-leakage) — **scikit-learn** — the canonical "how to avoid it" reference with code.
- [Cross-validation done wrong → leakage](https://scikit-learn.org/stable/modules/cross_validation.html) — **scikit-learn** — why all preprocessing must live inside the CV loop.
- [Data leakage, explained](https://machinelearningmastery.com/data-leakage-machine-learning/) — **Jason Brownlee** — accessible overview of the failure modes (free).

## 📚 Books (free, with chapters)
- [An Introduction to Statistical Learning (Python) — **Ch. 5 "Resampling Methods"**](https://www.statlearning.com/) — **James et al.** — why resampling must respect the train/test boundary; free PDF.
- [Tidy Modeling with R — **Ch. 5 "Spending our Data"**](https://www.tmwr.org/) — **Kuhn & Silge** — disciplined data spending that structurally prevents leakage (concepts transfer); free.

## 🔗 In this platform
- Concept depth (the *why*): [AI-ML-intuition 3.07 Bias–Variance & Generalization](../../../AI-ML-intuition/Module_3_Evaluation/3.07_Bias_Variance_and_Generalization.md)
- Next concepts: [10 Train/Validation/Test Splits](../10-Train-Validation-Test-Splits/10-Train-Validation-Test-Splits.md) · [13 Data Pipelines](../13-Data-Pipelines/13-Data-Pipelines.md)
- Related domain: [12. Deployment & MLOps](../../14.%20Deployment_and_MLOps/README.md)
