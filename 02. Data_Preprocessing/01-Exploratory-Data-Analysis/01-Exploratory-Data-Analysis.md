---
id: "02-data-preprocessing/eda"
topic: "Exploratory Data Analysis (EDA)"
parent: "02-data-preprocessing"
level: beginner
prereqs: ["python", "pandas", "descriptive-statistics"]
interview_frequency: high
updated: 2026-06-20
---

# Exploratory Data Analysis (EDA)
> Looking at your data before you model it — distributions, missingness, outliers, correlations, and
> relationships — so every cleaning and feature decision that follows is grounded in what's actually there.

**Why it matters:** EDA is where you catch the problems that silently break models — skew, leakage hints,
data-entry errors, class imbalance, suspicious correlations. Interviewers ask "how would you approach a
new dataset?"; a structured EDA answer (univariate → bivariate → multivariate, plus quality checks) signals
real applied experience.

**⭐ Start here — suggested path:**

1. **See the workflow** — watch [EDA Univariate Analysis](https://www.youtube.com/watch?v=4HyTlbHUKSw). *Shows the first pass: one variable at a time, distributions and counts.*
2. **Add relationships** — watch [EDA Bivariate & Multivariate](https://www.youtube.com/watch?v=6D3VtEfCw7w). *How features relate to each other and to the target.*
3. **Get the mindset** — read [R4DS — Exploratory Data Analysis](https://r4ds.had.co.nz/exploratory-data-analysis.html). *The canonical "ask questions about your data" framework (language-agnostic).*
4. **Practice on real data** — do [Kaggle: Pandas](https://www.kaggle.com/learn/pandas). *Builds the muscle memory of slicing/grouping/aggregating you need for EDA.*
5. **Make plots speak** — skim the [Seaborn tutorial](https://seaborn.pydata.org/tutorial.html). *Distribution, relational, and categorical plots that reveal structure fast.*

## 🎓 Courses (free)
- [Kaggle Learn — Pandas](https://www.kaggle.com/learn/pandas) — **Kaggle** — short, hands-on; the data-wrangling skills EDA is built on.
- [Data Analysis with Python](https://www.freecodecamp.org/learn/data-analysis-with-python/) — **freeCodeCamp** — free certification covering NumPy/Pandas/visualization end to end.

## 🎥 Videos
- [EDA using Univariate Analysis](https://www.youtube.com/watch?v=4HyTlbHUKSw) — **CampusX** — the first EDA pass: distributions, counts, summary stats per column.
- [EDA using Bivariate & Multivariate Analysis](https://www.youtube.com/watch?v=6D3VtEfCw7w) — **CampusX** — feature–feature and feature–target relationships.
- [Descriptive vs Inferential Statistics](https://www.youtube.com/watch?v=tPhzDKjQBpo) — **CampusX** — the stats vocabulary that EDA summaries rely on.
- [Advance House Price — EDA Part 1](https://www.youtube.com/watch?v=ioN1jcWxbv8) — **Krish Naik** — a full applied EDA walkthrough on a real Kaggle dataset.

## 📄 Key Papers
- [The Future of Data Analysis](https://projecteuclid.org/journals/annals-of-mathematical-statistics/volume-33/issue-1/The-Future-of-Data-Analysis/10.1214/aoms/1177704711.full) — **John W. Tukey (1962)** — the manifesto that launched EDA as a discipline; free full text on Project Euclid.
- [Tidy Data](https://vita.had.co.nz/papers/tidy-data.pdf) — **Hadley Wickham (2014)** — why a consistent data layout makes EDA and modeling dramatically easier.

## 📰 Articles / Blogs (free, no paywall)
- [R for Data Science — Exploratory Data Analysis](https://r4ds.had.co.nz/exploratory-data-analysis.html) — **Wickham & Grolemund** — the definitive "questions-first" EDA framework, fully open.
- [NIST/SEMATECH e-Handbook — EDA](https://www.itl.nist.gov/div898/handbook/eda/eda.htm) — **NIST** — rigorous, free reference on EDA techniques and assumptions.
- [Seaborn tutorial](https://seaborn.pydata.org/tutorial.html) — **Seaborn docs** — the visualization grammar (distribution/relational/categorical) for EDA.
- [pandas — Getting started tutorials](https://pandas.pydata.org/docs/getting_started/intro_tutorials/index.html) — **pandas docs** — official, free, task-oriented intro to exploring data.

## 📚 Books (free, with chapters)
- [Python Data Science Handbook — **Ch. 3 (Pandas) & Ch. 4 (Matplotlib)**](https://jakevdp.github.io/PythonDataScienceHandbook/) — **Jake VanderPlas** — the free, runnable NumPy/Pandas/visualization reference.
- [R for Data Science — **Ch. 7 "Exploratory Data Analysis"**](https://r4ds.had.co.nz/exploratory-data-analysis.html) — **Wickham & Grolemund** — concepts transfer directly to Python.

## 🔗 In this platform
- Concept depth (the *why*): [AI-ML-intuition 0.02 Distributions & the Gaussian](../../../AI-ML-intuition/Module_0_Foundations/0.02_Distributions_and_the_Gaussian.md) · [0.03 Expectation, Variance, Covariance](../../../AI-ML-intuition/Module_0_Foundations/0.03_Expectation_Variance_Covariance.md)
- Next concepts: [02 Feature Scaling & Normalization](../02-Feature-Scaling-and-Normalization/02-Feature-Scaling-and-Normalization.md) · [05 Outlier Detection & Treatment](../05-Outlier-Detection-and-Treatment/05-Outlier-Detection-and-Treatment.md)
- Related domain: [04. Unsupervised Learning](../../04.%20Unsupervised_Learning/README.md)
