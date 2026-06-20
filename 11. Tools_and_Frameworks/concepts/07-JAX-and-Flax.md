---
id: "11-tools-and-frameworks/jax-flax"
topic: "JAX (+ Flax) (functional autodiff, jit, vmap, pmap)"
parent: "11-tools-and-frameworks"
level: advanced
prereqs: ["python", "numpy", "neural-networks"]
interview_frequency: medium
updated: 2026-06-20
---

# JAX (+ Flax) — Functional Autodiff · jit · vmap · pmap
> A NumPy-compatible array library with **composable function transformations**: `grad` (autodiff),
> `jit` (XLA compilation to fast fused kernels), `vmap` (auto-vectorization/batching), and `pmap`
> (multi-device parallelism). **Flax** layers a neural-network library on top. The functional,
> pure-function style makes high-performance research code reproducible and easy to parallelize.

**Why it matters:** JAX is the framework of choice for a lot of cutting-edge research (and TPUs).
It tests a different mental model than PyTorch — pure functions and immutable state, `grad` as a
function transform, why `jit` needs static shapes/tracing, and `vmap`/`pmap` for batching and
scaling. Understanding it sharpens your grasp of autodiff and compilation in general.

**⭐ Start here — suggested path:**

1. **See the four transforms** — read the [JAX Quickstart](https://docs.jax.dev/en/latest/quickstart.html). *`grad`, `jit`, `vmap` in a few cells — the whole idea in one page.*
2. **Shift your mental model** — work [Thinking in JAX](https://docs.jax.dev/en/latest/notebooks/thinking_in_jax.html). *Pure functions, tracing, and why JAX differs from NumPy/PyTorch — the key to not getting surprised.*
3. **Watch the intro** — [Intro to JAX: Accelerating ML research](https://www.youtube.com/watch?v=WdTeDXsOSj4) (Google). *A concise official overview of the design and why it's fast.*
4. **Go hero-level on video** — [Machine Learning with JAX — From Zero to Hero](https://www.youtube.com/watch?v=SstuvS-tVc0) (Aleksa Gordić). *Builds real intuition for PyTrees, stateless models, and transforms.*
5. **Add neural nets with Flax** — follow [Flax NNX basics](https://flax.readthedocs.io/en/latest/nnx_basics.html). *Flax gives you Modules and training state on top of pure JAX.*

## 🎓 Courses (free)
- [JAX tutorials](https://docs.jax.dev/en/latest/tutorials.html) — **JAX team** — the official, structured learning path through the transforms.
- [UvA Deep Learning — Intro to JAX + Flax](https://uvadlc-notebooks.readthedocs.io/en/latest/tutorial_notebooks/JAX/tutorial2/Introduction_to_JAX.html) — **University of Amsterdam** — a rigorous, free university tutorial with notebooks.
- [Flax documentation](https://flax.readthedocs.io/en/latest/) — **Flax team (Google)** — the official guide to building/training neural nets in JAX.

## 🎥 Videos
- [Intro to JAX: Accelerating Machine Learning research](https://www.youtube.com/watch?v=WdTeDXsOSj4) — **Google / TensorFlow** — the concise official overview.
- [JAX Crash Course — Accelerating Machine Learning code](https://www.youtube.com/watch?v=juo5G3t4qAo) — **AssemblyAI** — a fast, practical first look at `grad`/`jit`/`vmap`.
- [Machine Learning with JAX — From Zero to Hero (Tutorial #1)](https://www.youtube.com/watch?v=SstuvS-tVc0) — **Aleksa Gordić (The AI Epiphany)** — the best deep, intuition-building JAX series.
- [PyTorch for Deep Learning — Full Course](https://www.youtube.com/watch?v=V_xro1bcAuA) — **freeCodeCamp** — useful PyTorch contrast for the imperative-vs-functional mental model.

## 📄 Key Papers
- [Compiling machine learning programs via high-level tracing (JAX/XLA)](https://mlsys.org/Conferences/doc/2018/146.pdf) — **Frostig, Johnson & Leary (2018)** — the foundational JAX/Autograd-to-XLA paper.
- [JAX documentation](https://docs.jax.dev/en/latest/index.html) — **JAX team** — the canonical reference for transforms and semantics.

## 📰 Articles / Blogs (free, no paywall)
- [JAX Quickstart](https://docs.jax.dev/en/latest/quickstart.html) — **JAX team** — the four transforms in one page.
- [Thinking in JAX](https://docs.jax.dev/en/latest/notebooks/thinking_in_jax.html) — **JAX team** — the functional/tracing mental model.
- [Flax NNX basics](https://flax.readthedocs.io/en/latest/nnx_basics.html) — **Flax team** — Modules and training state on top of JAX.

## 📚 Books (free, with chapters)
- [Autodidax: JAX core from scratch](https://docs.jax.dev/en/latest/autodidax.html) — **JAX team** — a book-length walkthrough that builds JAX's autodiff/`jit` internals from scratch.
- [UvA DL Notebooks — JAX track](https://uvadlc-notebooks.readthedocs.io/en/latest/tutorial_notebooks/JAX/tutorial2/Introduction_to_JAX.html) — **University of Amsterdam** — free, chapter-like notebooks.

## 🔗 In this platform
- Related domain: [05. Deep Learning](../../05.%20Deep_Learning/concepts/README.md) · [14. Advanced Research Mathematics](../../14.%20Advanced_Research_Mathematics/)
- Compare with: [05 PyTorch](05-PyTorch.md) · [06 TensorFlow & Keras](06-TensorFlow-and-Keras.md)
