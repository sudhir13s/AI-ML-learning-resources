---
id: "13-tools-and-frameworks/pandas"
topic: "Pandas (DataFrames, ETL, time series)"
parent: "13-tools-and-frameworks"
level: beginner
prereqs: ["python", "numpy"]
interview_frequency: high
updated: 2026-06-20
---

# Pandas — DataFrames · ETL · Time Series
> The workhorse for tabular data in Python: the labeled `Series`/`DataFrame`, plus the loading,
> cleaning, joining, grouping, reshaping, and time-series tooling that turn raw files into
> model-ready datasets. If NumPy is the array, Pandas is the spreadsheet with superpowers.

**Why it matters:** the daily reality of ML work is data wrangling, and interviews probe it —
`groupby` semantics, merge/join types, `apply` vs vectorization, handling missing data, and
reshaping (pivot/melt). Fluency here is what makes the modeling part possible.

**⭐ Start here — suggested path:**

1. **See the shape of it** — read [10 minutes to pandas](https://pandas.pydata.org/docs/user_guide/10min.html). *A fast, official tour of Series/DataFrame, selection, and the core verbs.*
2. **Build it by hand** — watch [Complete Pandas Data Science Tutorial](https://www.youtube.com/watch?v=vmEHCJofslg) (Keith Galli). *Reading CSVs, filtering, and groupby typed out makes the API stick.*
3. **Work the official intro tutorials** — do the [Getting Started tutorials](https://pandas.pydata.org/docs/getting_started/intro_tutorials/index.html). *Short, task-focused lessons (read data, select, plot, combine) from the maintainers.*
4. **Master groupby & reshaping** — study the [User Guide](https://pandas.pydata.org/docs/user_guide/index.html) sections on Group By, Merge/Join, and Reshaping. *These are the highest-leverage, most-interviewed operations.*
5. **Read it from the source** — work through the free [Python for Data Analysis](https://wesmckinney.com/book/) by Pandas' creator. *The definitive, idiomatic treatment of the library.*

## 🎓 Courses (free)
- [Getting started with pandas](https://pandas.pydata.org/docs/getting_started/index.html) — **pandas team** — the official on-ramp with intro tutorials and comparisons to SQL/Excel/R.
- [pandas User Guide](https://pandas.pydata.org/docs/user_guide/index.html) — **pandas team** — a topic-by-topic course (indexing, groupby, merge, time series, categoricals).

## 🎥 Videos
- [Complete Python Pandas Data Science Tutorial](https://www.youtube.com/watch?v=vmEHCJofslg) — **Keith Galli** — reading files, sorting, filtering, and groupby end to end.
- [Python Pandas Tutorial (Part 1): Getting Started](https://www.youtube.com/watch?v=ZyhVh-qRZPA) — **Corey Schafer** — the start of the gold-standard multi-part Pandas series.
- [Data Analysis with Python — Full Course](https://www.youtube.com/watch?v=r-uOLxNrNk8) — **freeCodeCamp** — Pandas within the full NumPy/Matplotlib/Seaborn workflow.
- [Python NumPy Tutorial for Beginners](https://www.youtube.com/watch?v=QUT1VHiLmmI) — **freeCodeCamp** — the NumPy foundation Pandas is built on (do this first if arrays are new).

## 📄 Key Papers
- [Data Structures for Statistical Computing in Python](https://proceedings.scipy.org/articles/Majora-92bf1922-00a) — **Wes McKinney (SciPy 2010)** — the original open-access paper introducing pandas' design.
- [pandas User Guide](https://pandas.pydata.org/docs/user_guide/index.html) — **pandas team** — the authoritative reference for semantics of every core operation.

## 📰 Articles / Blogs (free, no paywall)
- [10 minutes to pandas](https://pandas.pydata.org/docs/user_guide/10min.html) — **pandas team** — the canonical quickstart.
- [Getting Started tutorials](https://pandas.pydata.org/docs/getting_started/intro_tutorials/index.html) — **pandas team** — bite-sized, task-oriented official lessons.
- [Python Data Science Handbook — **Ch. 3 "Data Manipulation with Pandas"**](https://jakevdp.github.io/PythonDataScienceHandbook/03.00-introduction-to-pandas.html) — **Jake VanderPlas** — full chapter free online.

## 📚 Books (free, with chapters)
- [Python for Data Analysis, 3rd ed.](https://wesmckinney.com/book/) — **Wes McKinney** — the full text, free online, written by Pandas' creator.
- [Python Data Science Handbook](https://jakevdp.github.io/PythonDataScienceHandbook/) — **Jake VanderPlas** — entire book free; Ch. 3 is the Pandas reference.

## 🔗 In this platform
- Related domain: [02. Data_Preprocessing](../../02.%20Data_Preprocessing/) · [03. Supervised Learning](../../03.%20Supervised_Learning/README.md)
- Pairs with: [01 NumPy](../01-NumPy/01-NumPy.md) · [03 Data Visualization](../03-Data-Visualization/03-Data-Visualization.md) · [04 scikit-learn](../04-scikit-learn/04-scikit-learn.md)
