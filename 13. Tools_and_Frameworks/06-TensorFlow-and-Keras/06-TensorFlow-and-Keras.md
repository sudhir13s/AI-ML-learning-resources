---
id: "13-tools-and-frameworks/tensorflow-keras"
topic: "TensorFlow & Keras (graphs, layers, fit/serve)"
parent: "13-tools-and-frameworks"
level: intermediate
prereqs: ["python", "numpy", "neural-networks"]
interview_frequency: high
updated: 2026-06-20
---

# TensorFlow & Keras — Graphs · Layers · fit / serve
> Google's production-grade DL platform (**TensorFlow**) with **Keras** as its high-level API:
> stack layers, call `model.compile()` / `model.fit()`, and you have a trained model. TensorFlow
> adds graph execution (`tf.function`), `tf.data` pipelines, and a mature serving/mobile/edge stack
> (TF Serving, TFLite, TF.js).

**Why it matters:** Keras is the friendliest on-ramp to deep learning, and TensorFlow still powers
huge amounts of production ML. Interviews probe the Sequential vs Functional API, eager vs graph
execution (`@tf.function`), `tf.data` input pipelines, and how Keras `fit`/callbacks structure
training — plus the trade-offs vs PyTorch.

**⭐ Start here — suggested path:**

1. **Build a model in minutes** — read [Keras: Getting started](https://keras.io/getting_started/) and the [TF beginner quickstart](https://www.tensorflow.org/tutorials). *Keras' `Sequential` + `fit` gives a working classifier almost immediately.*
2. **Take the full course** — [TensorFlow 2.0 Complete Course](https://www.youtube.com/watch?v=tPYj3fFJGjk) (freeCodeCamp / Tim Ruscica). *A 7h end-to-end tour of TF + Keras for beginners.*
3. **Go deep on Keras** — [Keras with TensorFlow Course](https://www.youtube.com/watch?v=qFJeN9V1ZsI) (freeCodeCamp). *Data prep, building/training nets, CNNs, transfer learning.*
4. **Learn the APIs that matter** — study the [Keras developer guides](https://keras.io/guides/) (Functional API, custom training, `tf.data`). *The Functional API and custom loops are the common interview/extension topics.*
5. **Keep references handy** — bookmark [TF tutorials](https://www.tensorflow.org/tutorials) and [Keras examples](https://keras.io/examples/). *Adapting a near example is the real workflow.*

## 🎓 Courses (free)
- [TensorFlow tutorials](https://www.tensorflow.org/tutorials) — **TensorFlow team** — the official, layered set from beginner quickstart to advanced.
- [Keras developer guides](https://keras.io/guides/) — **Keras team** — a structured course on the Sequential/Functional APIs, training, and customization.
- [TensorFlow: learn](https://www.tensorflow.org/learn) — **TensorFlow team** — the curated hub of courses, books, and learning paths.

## 🎥 Videos
- [TensorFlow 2.0 Complete Course — Neural Networks for Beginners](https://www.youtube.com/watch?v=tPYj3fFJGjk) — **freeCodeCamp (Tim Ruscica)** — the full TF + Keras beginner course.
- [Keras with TensorFlow Course — Deep Learning for Beginners](https://www.youtube.com/watch?v=qFJeN9V1ZsI) — **freeCodeCamp (deeplizard)** — Keras-focused, including CNNs and transfer learning.
- [Intro to JAX: Accelerating ML research](https://www.youtube.com/watch?v=WdTeDXsOSj4) — **TensorFlow** — useful context on where graph compilation/XLA fits (shared with JAX).
- [Google TensorFlow and Convolutional Neural Networks](https://www.youtube.com/watch?v=EoysuTMmmMc) — **Darshan Patel** — an additional hands-on CNN-in-TF walkthrough.

## 📄 Key Papers
- [TensorFlow: A System for Large-Scale Machine Learning](https://arxiv.org/abs/1603.04467) — **Abadi et al. (2016), OSDI** — the foundational paper on TensorFlow's design.
- [TensorFlow API reference](https://www.tensorflow.org/api_docs) — **TensorFlow team** — the canonical specification of the framework.

## 📰 Articles / Blogs (free, no paywall)
- [Keras: Getting started](https://keras.io/getting_started/) — **Keras team** — install + first model in one page.
- [TensorFlow 2 quickstart for beginners](https://www.tensorflow.org/tutorials) — **TensorFlow team** — a working classifier in a few cells.
- [Keras code examples](https://keras.io/examples/) — **Keras team** — a large, free, copy-adaptable example gallery across domains.

## 📚 Books (free, with chapters)
- [Keras developer guides (full set)](https://keras.io/guides/) — **Keras team** — a book-length, free progression through the API.
- [TensorFlow: learn (curated books & paths)](https://www.tensorflow.org/learn) — **TensorFlow team** — the official directory of free learning materials.

## 🔗 In this platform
- Related domain: [05. Deep Learning](../../05.%20Deep_Learning/README.md) · [07. Computer Vision](../../07.%20Computer%20Vision/README.md)
- Compare with: [05 PyTorch](../05-PyTorch/05-PyTorch.md) · [07 JAX (+ Flax)](../07-JAX-and-Flax/07-JAX-and-Flax.md)
- Deeper concept (the *why*): serving & deployment → [Deployment & MLOps](../../14.%20Deployment_and_MLOps/README.md)
