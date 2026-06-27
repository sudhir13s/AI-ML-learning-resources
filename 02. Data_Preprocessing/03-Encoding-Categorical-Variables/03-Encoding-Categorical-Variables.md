---
id: "02-data-preprocessing/encoding-categoricals"
topic: "Encoding Categorical Variables"
parent: "02-data-preprocessing"
level: beginner
prereqs: ["pandas", "one-hot-encoding"]
interview_frequency: very-high
updated: 2026-06-20
---

# Encoding Categorical Variables
> Turning categories into numbers models can use — one-hot for nominal, ordinal for ordered, and
> target/mean encoding for high-cardinality features — without inventing fake order or leaking the target.

**Why it matters:** a staple interview topic. You should know one-hot vs ordinal vs label encoding and
when each is appropriate, why label-encoding a nominal feature misleads linear models, the curse of
high-cardinality (one-hot blowup) and how target encoding addresses it, and the **leakage trap** in target
encoding (must be fit inside cross-validation / with smoothing).

**⭐ Start here — suggested path:**

1. **Ordinal vs label** — watch [Encoding Categorical Data: Ordinal & Label](https://www.youtube.com/watch?v=w2GglmYHfmM). *When order is meaningful vs not.*
2. **One-hot done right** — watch [One Hot Encoding (Day 27)](https://www.youtube.com/watch?v=U5oCv3JKWKA). *Dummy variables, drop-first, and handling unseen categories.*
3. **High cardinality** — read [Target encoding done the right way](https://maxhalford.github.io/blog/target-encoding/). *Mean encoding with smoothing + how to avoid leakage.*
4. **Reference the API** — read [sklearn: Encoding categorical features](https://scikit-learn.org/stable/modules/preprocessing.html#encoding-categorical-features). *OneHotEncoder/OrdinalEncoder/TargetEncoder, with handle_unknown.*
5. **Apply in a pipeline** — see [sklearn TargetEncoder](https://scikit-learn.org/stable/modules/generated/sklearn.preprocessing.TargetEncoder.html). *Built-in cross-fitted target encoding that prevents leakage.*

## 🎓 Courses (free)
- [Google ML Crash Course — Categorical data](https://developers.google.com/machine-learning/crash-course/categorical-data) — **Google** — free, clear treatment of vocabulary, one-hot, and embeddings.
- [Kaggle Learn — Feature Engineering (Categorical Encodings)](https://www.kaggle.com/learn/feature-engineering) — **Kaggle** — hands-on count/target encoding on real data.

## 🎥 Videos
- [Encoding Categorical Data — Ordinal & Label Encoding](https://www.youtube.com/watch?v=w2GglmYHfmM) — **CampusX** — when ordered vs unordered encoding is correct.
- [One Hot Encoding — Handling Categorical Data (Day 27)](https://www.youtube.com/watch?v=U5oCv3JKWKA) — **CampusX** — dummy variables, drop-first, unseen categories.
- [Encoding Categorical Data — ML Fundamentals](https://www.youtube.com/watch?v=6qFGxJFfmvo) — **Kody Simpson** — concise from-scratch walkthrough in Python.
- [Different Types of Feature Engineering Encoding Techniques](https://www.youtube.com/watch?v=OTPz5plKb40) — **Krish Naik** — overview of label/one-hot/target/frequency encoding and trade-offs.

## 📄 Key Papers
- [Similarity Encoding for Learning with Dirty Categorical Variables](https://arxiv.org/abs/1806.00979) — **Cerda, Varoquaux & Kégl (2018)** — handling high-cardinality and noisy categories beyond one-hot; free on arXiv.
- [Entity Embeddings of Categorical Variables](https://arxiv.org/abs/1604.06737) — **Guo & Berkhahn (2016)** — learning dense embeddings for categories (the neural alternative to one-hot).

## 📰 Articles / Blogs (free, no paywall)
- [Target encoding done the right way](https://maxhalford.github.io/blog/target-encoding/) — **Max Halford** — smoothing + leakage-safe target encoding, with math and code.
- [Encoding categorical features — user guide](https://scikit-learn.org/stable/modules/preprocessing.html#encoding-categorical-features) — **scikit-learn** — the authoritative API reference for all encoders.
- [category_encoders documentation](https://contrib.scikit-learn.org/category_encoders/) — **scikit-learn-contrib** — a whole library of encoders (target, James-Stein, CatBoost, hashing) with explanations.

## 📚 Books (free, with chapters)
- [Feature Engineering and Selection — **Ch. 5 "Encoding Categorical Predictors"**](http://www.feat.engineering/) — **Kuhn & Johnson** — one-hot, effect, and likelihood/target encodings; free online.
- [Python Data Science Handbook — **§5.4 "Feature Engineering" (Categorical Features)**](https://jakevdp.github.io/PythonDataScienceHandbook/05.04-feature-engineering.html) — **Jake VanderPlas** — practical one-hot with DictVectorizer.

## 🔗 In this platform
- Concept depth (the *why*): [AI-ML-intuition 1.01 One-Hot Encoding](../../../AI-ML-intuition/Module_1_Representation/1.01_One-Hot_Encoding.md) · [1.02 Dense Embeddings](../../../AI-ML-intuition/Module_1_Representation/1.02_Dense_Embeddings.md)
- Next concepts: [06 Feature Engineering](../06-Feature-Engineering/06-Feature-Engineering.md) · [11 Data Leakage](../11-Data-Leakage/11-Data-Leakage.md)
- Related domain: [01. Foundations](../../01.%20Foundations/README.md)
