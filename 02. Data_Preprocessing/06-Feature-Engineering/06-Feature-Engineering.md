---
id: "02-data-preprocessing/feature-engineering"
topic: "Feature Engineering"
parent: "02-data-preprocessing"
level: intermediate
prereqs: ["pandas", "domain-knowledge", "transformations"]
interview_frequency: very-high
updated: 2026-06-20
---

# Feature Engineering
> Creating better inputs from raw data — combining columns, splitting fields, binning, log/power
> transforms, and interaction terms — so a model can learn relationships it otherwise couldn't.

**Why it matters:** "better data beats fancier algorithms" — feature engineering is where most real-world
accuracy gains come from, and the classic interview line is *"feature engineering is the part of ML that
domain knowledge buys you."* Be ready to walk through constructing a feature, why log/Box-Cox fixes skew,
when binning helps tree-free models, and how to engineer **without leaking** future or target information.

**⭐ Start here — suggested path:**

1. **Construct features** — watch [Feature Construction & Splitting](https://www.youtube.com/watch?v=ma-h30PoFms). *Combining and decomposing columns into informative features.*
2. **Fix skew** — watch [Function Transformer: log / sqrt / reciprocal](https://www.youtube.com/watch?v=cTjj3LE8E90). *Making distributions more normal so models behave.*
3. **Power transforms** — watch [Power Transformer: Box-Cox / Yeo-Johnson](https://www.youtube.com/watch?v=lV_Z4HbNAx0). *Principled, data-driven variance stabilization.*
4. **Discretize** — watch [Binning & Binarization](https://www.youtube.com/watch?v=kKWsJGKcMvo). *Turning continuous values into buckets, and when it helps.*
5. **Internalize the principle** — read [A Few Useful Things to Know about ML](https://homes.cs.washington.edu/~pedrod/papers/cacm12.pdf). *Domingos on why feature engineering is the key to applied ML.*

## 🎓 Courses (free)
- [Kaggle Learn — Feature Engineering](https://www.kaggle.com/learn/feature-engineering) — **Kaggle** — the best free hands-on course: mutual info, interactions, target encoding, clustering features.
- [Google ML Crash Course — Numerical & Categorical data](https://developers.google.com/machine-learning/crash-course/numerical-data) — **Google** — feature transforms, bucketing, and crosses, applied.

## 🎥 Videos
- [Feature Construction & Feature Splitting](https://www.youtube.com/watch?v=ma-h30PoFms) — **CampusX** — building new features by combining/decomposing existing ones.
- [Function Transformer — Log / Reciprocal / Square-Root](https://www.youtube.com/watch?v=cTjj3LE8E90) — **CampusX** — taming skewed distributions with simple transforms.
- [Power Transformer — Box-Cox & Yeo-Johnson](https://www.youtube.com/watch?v=lV_Z4HbNAx0) — **CampusX** — data-driven transforms toward normality.
- [Binning & Binarization — Discretization](https://www.youtube.com/watch?v=kKWsJGKcMvo) — **CampusX** — quantile/KMeans binning and binarization, with code.

## 📄 Key Papers
- [A Few Useful Things to Know about Machine Learning](https://homes.cs.washington.edu/~pedrod/papers/cacm12.pdf) — **Pedro Domingos (2012)** — the famous "feature engineering is the key" essay; free PDF.
- [Wide & Deep Learning for Recommender Systems](https://arxiv.org/abs/1606.07792) — **Cheng et al. (2016)** — how hand-crafted cross-features and learned embeddings combine in practice.

## 📰 Articles / Blogs (free, no paywall)
- [Preprocessing data — user guide](https://scikit-learn.org/stable/modules/preprocessing.html) — **scikit-learn** — PolynomialFeatures, KBinsDiscretizer, PowerTransformer, FunctionTransformer.
- [feature-engine documentation](https://feature-engine.trainindata.com/en/latest/) — **Train in Data** — a transformer library purpose-built for feature engineering, with clear docs.
- [Tidy Data](https://vita.had.co.nz/papers/tidy-data.pdf) — **Hadley Wickham** — the data layout that makes feature engineering clean and repeatable.

## 📚 Books (free, with chapters)
- [Feature Engineering and Selection — **Ch. 6 (numeric) & Ch. 5 (categorical)**](http://www.feat.engineering/) — **Kuhn & Johnson** — the canonical free FE text.
- [Python Data Science Handbook — **§5.4 "Feature Engineering"**](https://jakevdp.github.io/PythonDataScienceHandbook/05.04-feature-engineering.html) — **Jake VanderPlas** — derived features, polynomial features, and pipelines.

## 🔗 In this platform
- Concept depth (the *why*): [AI-ML-intuition 1.01 One-Hot Encoding](../../../AI-ML-intuition/Module_1_Representation/1.01_One-Hot_Encoding.md) · [1.17 BoW & TF-IDF](../../../AI-ML-intuition/Module_1_Representation/1.17_BoW_and_TF-IDF.md)
- Next concepts: [07 Feature Selection](../07-Feature-Selection/07-Feature-Selection.md) · [08 Date/Time & Cyclical Features](../08-Date-Time-and-Cyclical-Features/08-Date-Time-and-Cyclical-Features.md)
- Related domain: [03. Supervised Learning](../../03.%20Supervised_Learning/README.md)
