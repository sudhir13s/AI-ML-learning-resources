---
id: "13-tools-and-frameworks/numpy"
topic: "NumPy (arrays, broadcasting, vectorization)"
parent: "13-tools-and-frameworks"
level: beginner
prereqs: ["python"]
interview_frequency: high
updated: 2026-06-20
---

# NumPy — Arrays · Broadcasting · Vectorization
> The foundation of the entire Python scientific stack: a fast, contiguous N-dimensional array
> (`ndarray`) plus the vectorized math, broadcasting rules, and indexing that let you replace slow
> Python loops with C-speed operations. Pandas, scikit-learn, PyTorch, and JAX all stand on it.

**Why it matters:** almost every data/ML interview assumes NumPy fluency — broadcasting, views vs
copies, axis semantics, and "why is this loop slow?" Vectorized thinking is the mental model you carry
into every tensor library you'll ever use.

**⭐ Start here — suggested path:**

1. **Get the mental model** — read the official [Absolute Beginner's Guide](https://numpy.org/doc/stable/user/absolute_beginners.html). *The clearest from-zero intro to arrays, shapes, and indexing — written by the maintainers.*
2. **See it built live** — watch [Complete NumPy Tutorial](https://www.youtube.com/watch?v=GB9ByFAIAH4) (Keith Galli). *Watching arrays, slicing, and math typed out cements the syntax fast.*
3. **Internalize broadcasting & vectorization** — work the [Quickstart](https://numpy.org/doc/stable/user/quickstart.html), focusing on broadcasting and axis arguments. *This is the single idea that makes NumPy fast and the most-tested concept.*
4. **Go deeper, properly** — follow the [SciPy NumPy tutorial](https://www.youtube.com/watch?v=ZB7BZMhfPgk) (Enthought). *A rigorous treatment of memory layout, views, and dtype that explains the "why" behind performance.*
5. **Keep it as a reference** — bookmark the [API reference](https://numpy.org/doc/stable/reference/index.html) and the [official tutorials](https://numpy.org/numpy-tutorials/). *You'll return to these constantly while building.*

## 🎓 Courses (free)
- [NumPy: learn](https://numpy.org/learn/) — **NumPy team** — the curated official hub of tutorials, talks, and books, all free.
- [NumPy Tutorials (applied)](https://numpy.org/numpy-tutorials/) — **NumPy team** — runnable, real-world notebooks (e.g. linear regression, image processing) from the maintainers.

## 🎥 Videos
- [Complete Python NumPy Tutorial (Arrays, Indexing, Math, Reshaping)](https://www.youtube.com/watch?v=GB9ByFAIAH4) — **Keith Galli** — a focused, hands-on NumPy walkthrough from scratch.
- [Python NumPy Tutorial for Beginners](https://www.youtube.com/watch?v=QUT1VHiLmmI) — **freeCodeCamp** — full beginner course covering the core API end to end.
- [Introduction to Numerical Computing with NumPy (SciPy 2019)](https://www.youtube.com/watch?v=ZB7BZMhfPgk) — **Enthought** — the rigorous tutorial on memory, views, dtypes, and broadcasting.
- [Data Analysis with Python — Full Course (NumPy, Pandas, Matplotlib, Seaborn)](https://www.youtube.com/watch?v=r-uOLxNrNk8) — **freeCodeCamp** — NumPy in the context of the full data stack.

## 📄 Key Papers
- [Array programming with NumPy](https://www.nature.com/articles/s41586-020-2649-2) — **Harris et al. (2020), *Nature*** — the authoritative open-access paper on NumPy's design and role in science.
- [NumPy API reference](https://numpy.org/doc/stable/reference/index.html) — **NumPy team** — the canonical specification of every function and its semantics.

## 📰 Articles / Blogs (free, no paywall)
- [Python NumPy Tutorial (CS231n)](https://cs231n.github.io/python-numpy-tutorial/) — **Stanford CS231n** — the classic, concise NumPy primer used by a top deep-learning course.
- [NumPy: the absolute basics for beginners](https://numpy.org/doc/stable/user/absolute_beginners.html) — **NumPy team** — the official gentle on-ramp.
- [NumPy quickstart](https://numpy.org/doc/stable/user/quickstart.html) — **NumPy team** — broadcasting, indexing, and shape manipulation in one tight page.

## 📚 Books (free, with chapters)
- [NumPy Tutorials (notebook collection)](https://numpy.org/numpy-tutorials/) — **NumPy team** — free, chapter-like runnable notebooks maintained by the project.
- [Python Data Science Handbook — **Ch. 2 "Introduction to NumPy"**](https://jakevdp.github.io/PythonDataScienceHandbook/02.00-introduction-to-numpy.html) — **Jake VanderPlas** — full text free online; the best book chapter on NumPy.

## 🔗 In this platform
- Related domain: [02. Data_Preprocessing](../../02.%20Data_Preprocessing/) · [05. Deep Learning](../../05.%20Deep_Learning/concepts/README.md)
- Next tools: [02 Pandas](02-Pandas.md) · [05 PyTorch](05-PyTorch.md)
