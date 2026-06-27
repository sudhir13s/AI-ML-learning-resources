---
id: "02-data-preprocessing/datetime-cyclical"
topic: "Handling Date/Time & Cyclical Features"
parent: "02-data-preprocessing"
level: intermediate
prereqs: ["pandas", "feature-engineering", "trigonometry"]
interview_frequency: medium
updated: 2026-06-20
---

# Handling Date/Time & Cyclical Features
> Turning timestamps into useful signals — extracting hour/day/month/weekday, computing elapsed time and
> lags, and encoding periodic features with sine/cosine so "23:00" and "00:00" are correctly *close*.

**Why it matters:** raw datetimes are useless to most models, and naively encoding hour as 0–23 tells the
model that hour 23 and hour 0 are maximally far apart — wrong. Expect to explain datetime decomposition,
the **sine/cosine cyclical encoding** trick for periodic features (hour, month, day-of-week), lag/rolling
features for time series, and the time-aware leakage trap (never use future rows; split chronologically).

**⭐ Start here — suggested path:**

1. **Decompose datetimes** — watch [Handling Date & Time Variables](https://www.youtube.com/watch?v=J73mvgG9fFs). *Extracting components and elapsed-time features.*
2. **In pandas** — watch [Extract Hour/Day/Month from DateTime](https://www.youtube.com/watch?v=ZjMTZIxgkYo). *The practical `.dt` accessor workflow.*
3. **Cyclical encoding** — watch [Cyclic Encoding for time](https://www.youtube.com/watch?v=ceh36yGRQ2w). *Why sine/cosine make periodicity continuous.*
4. **See it done right** — read [sklearn: Cyclical feature engineering](https://scikit-learn.org/stable/auto_examples/applications/plot_cyclical_feature_engineering.html). *Trigonometric vs spline encodings on a real bike-share dataset.*
5. **Time-series predictors** — read [FPP3: Useful predictors](https://otexts.com/fpp3/useful-predictors.html). *Trend, seasonal dummies, and Fourier terms for forecasting.*

## 🎓 Courses (free)
- [Kaggle Learn — Time Series](https://www.kaggle.com/learn/time-series) — **Kaggle** — trend, seasonality, lags, and time-based features, hands-on.
- [Kaggle Learn — Feature Engineering](https://www.kaggle.com/learn/feature-engineering) — **Kaggle** — where date/time features fit in a broader pipeline.

## 🎥 Videos
- [Handling Date & Time Variables (Day 34)](https://www.youtube.com/watch?v=J73mvgG9fFs) — **CampusX** — extracting components and elapsed-time features with code.
- [Extract Hour, Day, Month from Pandas DateTime](https://www.youtube.com/watch?v=ZjMTZIxgkYo) — **Mısra Turp** — the practical `.dt` accessor workflow.
- [Cyclic Encoding for Time Features](https://www.youtube.com/watch?v=ceh36yGRQ2w) — **Karnika Kapoor** — sine/cosine encoding so periodic values wrap correctly.
- [Feature Engineering for Time-Series Forecasting](https://www.youtube.com/watch?v=9QtL7m3YS9I) — **PyData (Kishan Manani)** — calendar, lag, and window features done correctly (and leakage-safe).
- [Pandas Time Series — Shifting & Lagging](https://www.youtube.com/watch?v=0lsmdNLNorY) — **codebasics** — building lag features with `shift`.

## 📄 Key Papers
- [Distributed and parallel time series feature extraction (FRESH / tsfresh)](https://arxiv.org/abs/1610.07717) — **Christ, Kempa-Liehr & Feindt (2017)** — systematic extraction + selection of temporal features; free on arXiv.
- [A Few Useful Things to Know about Machine Learning](https://homes.cs.washington.edu/~pedrod/papers/cacm12.pdf) — **Pedro Domingos (2012)** — why thoughtful feature construction (incl. temporal) drives results.

## 📰 Articles / Blogs (free, no paywall)
- [Time-related feature engineering](https://scikit-learn.org/stable/auto_examples/applications/plot_cyclical_feature_engineering.html) — **scikit-learn** — the canonical cyclical (sine/cosine) vs spline encoding example.
- [feature-engine — Datetime features](https://feature-engine.trainindata.com/en/latest/user_guide/datetime/index.html) — **Train in Data** — transformers that extract and cyclically encode datetime features.
- [pandas — Time series / date functionality](https://pandas.pydata.org/docs/user_guide/timeseries.html) — **pandas docs** — the authoritative reference for datetime handling, resampling, and offsets.

## 📚 Books (free, with chapters)
- [Forecasting: Principles and Practice (3rd ed.) — **Ch. 7 "Time series regression"** (trend, seasonal dummies, Fourier)](https://otexts.com/fpp3/useful-predictors.html) — **Hyndman & Athanasopoulos** — free, the standard forecasting text.
- [Feature Engineering and Selection — **Ch. 6 (transformations) & Ch. 9 (profile/temporal data)**](http://www.feat.engineering/) — **Kuhn & Johnson** — engineering features from structured/temporal data; free online.

## 🔗 In this platform
- Concept depth (the *why*): [AI-ML-intuition 1.03 Positional Encoding](../../../AI-ML-intuition/Module_1_Representation/1.03_Positional_Encoding.md) — the same sine/cosine idea that encodes sequence position.
- Next concepts: [06 Feature Engineering](../06-Feature-Engineering/06-Feature-Engineering.md) · [11 Data Leakage](../11-Data-Leakage/11-Data-Leakage.md)
- Related domain: [03. Supervised Learning](../../03.%20Supervised_Learning/README.md)
