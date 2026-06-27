---
id: "02-data-preprocessing/imbalanced-data"
topic: "Imbalanced Data (resampling · SMOTE · class weights)"
parent: "02-data-preprocessing"
level: intermediate
prereqs: ["classification-metrics", "train-test-split"]
interview_frequency: very-high
updated: 2026-06-20
---

# Imbalanced Data
> When one class vastly outnumbers another (fraud, disease, churn) — handled via resampling
> (under/oversampling, SMOTE), class weights, or threshold tuning — and evaluated with metrics that
> don't reward predicting the majority class.

**Why it matters:** with 99% negatives, a model that predicts "negative" always scores 99% accuracy and is
useless — so this topic tests whether you know to use precision/recall/F1/PR-AUC instead, and the three
families of fixes: **resampling** (SMOTE, random under/over), **algorithmic** (class_weight='balanced',
scale_pos_weight), and **threshold tuning**. The classic trap: SMOTE must be applied **inside CV, on
training folds only** — never before the split.

**⭐ Start here — suggested path:**

1. **Frame the problem** — watch [Balanced vs Imbalanced Datasets](https://www.youtube.com/watch?v=KR-eWajtTfo). *Why accuracy lies and what to do instead.*
2. **Resampling in practice** — watch [Handling Imbalanced Datasets](https://www.youtube.com/watch?v=JnlM4yLFNuo). *Under/oversampling with code.*
3. **SMOTE** — watch [SMOTE explained](https://www.youtube.com/watch?v=U3X98xZ4_no). *How synthetic minority samples are generated.*
4. **Weights vs resampling** — watch [Class Weights vs Resampling](https://www.youtube.com/watch?v=xotLmq8YkAw). *The often-better, simpler alternative.*
5. **Use the right metrics** — read [ROC vs PR curves (sklearn)](https://scikit-learn.org/stable/auto_examples/model_selection/plot_precision_recall.html). *Why PR-AUC beats ROC-AUC under heavy imbalance.*

## 🎓 Courses (free)
- [imbalanced-learn — User Guide](https://imbalanced-learn.org/stable/user_guide.html) — **imbalanced-learn** — the canonical library's guide to every resampling method.
- [Google ML Crash Course — Imbalanced data](https://developers.google.com/machine-learning/crash-course/overfitting/imbalanced-datasets) — **Google** — downsampling + upweighting, applied.

## 🎥 Videos
- [Balanced vs Imbalanced Datasets — how to handle them](https://www.youtube.com/watch?v=KR-eWajtTfo) — **Mahesh Huddar** — the problem framing and the menu of fixes.
- [Handling Imbalanced Datasets in ML](https://www.youtube.com/watch?v=JnlM4yLFNuo) — **codebasics** — under/oversampling with code on a real dataset.
- [SMOTE for Imbalanced Datasets](https://www.youtube.com/watch?v=U3X98xZ4_no) — **Bhavesh Bhatt** — how SMOTE synthesizes minority-class examples.
- [Class Weights vs Resampling](https://www.youtube.com/watch?v=xotLmq8YkAw) — **Professor Py** — when `class_weight='balanced'` beats resampling.
- [SMOTE, Upsampling & Downsampling in Python](https://www.youtube.com/watch?v=EqIQTN65IZs) — **Satyajit Pattnaik** — full hands-on comparison of techniques.

## 📄 Key Papers
- [SMOTE: Synthetic Minority Over-sampling Technique](https://arxiv.org/abs/1106.1813) — **Chawla et al. (2002)** — the original SMOTE paper (JAIR), free on arXiv; the method every interview references.
- [Learning from Imbalanced Data](https://www.jair.org/index.php/jair/article/view/10302) — **He & Garcia (2009)** — the definitive survey of imbalance methods and metrics; free in JAIR.

## 📰 Articles / Blogs (free, no paywall)
- [imbalanced-learn — Over-sampling](https://imbalanced-learn.org/stable/over_sampling.html) — **imbalanced-learn** — SMOTE variants (Borderline, ADASYN) with API + intuition.
- [Precision-Recall (sklearn example)](https://scikit-learn.org/stable/auto_examples/model_selection/plot_precision_recall.html) — **scikit-learn** — the right evaluation curve under imbalance.
- [imbalanced-learn — Combining over- and under-sampling](https://imbalanced-learn.org/stable/combine.html) — **imbalanced-learn** — SMOTEENN/SMOTETomek and pipeline-safe usage.

## 📚 Books (free, with chapters)
- [Feature Engineering and Selection — **Ch. 3 "Review of Key Concepts" (class imbalance, metrics)**](http://www.feat.engineering/) — **Kuhn & Johnson** — imbalance + metric pitfalls in context; free online.
- [An Introduction to Statistical Learning (Python) — **Ch. 4 "Classification"**](https://www.statlearning.com/) — **James et al.** — thresholds, confusion matrices, and why accuracy misleads; free PDF.

## 🔗 In this platform
- Concept depth (the *why*): [AI-ML-intuition 3.05 Precision, Recall, F1](../../../AI-ML-intuition/Module_3_Evaluation/3.05_Classification_Metrics_Precision_Recall_F1.md) · [3.06 ROC/AUC & PR Curves](../../../AI-ML-intuition/Module_3_Evaluation/3.06_ROC_AUC_PR_Curves.md)
- Next concepts: [10 Train/Validation/Test Splits](../10-Train-Validation-Test-Splits/10-Train-Validation-Test-Splits.md) · [11 Data Leakage](../11-Data-Leakage/11-Data-Leakage.md)
- Related domain: [03. Supervised Learning](../../03.%20Supervised_Learning/README.md)
