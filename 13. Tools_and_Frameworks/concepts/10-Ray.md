---
id: "13-tools-and-frameworks/ray"
topic: "Ray (distributed training, tuning, serving)"
parent: "13-tools-and-frameworks"
level: advanced
prereqs: ["python", "pytorch"]
interview_frequency: medium
updated: 2026-06-20
---

# Ray — Distributed Training · Tuning · Serving
> A general-purpose framework for scaling Python from a laptop to a cluster with minimal code
> change: `@ray.remote` turns functions into parallel **tasks** and classes into stateful
> **actors**. On top of that core sit ML libraries — **Ray Train** (distributed training), **Ray
> Tune** (hyperparameter search), **Ray Serve** (model serving), **Ray Data**, and **RLlib**.

**Why it matters:** scaling out is unavoidable for large models and large sweeps, and Ray is the
most common Python-native way to do it. Interviews and real work touch on the tasks-vs-actors model,
why `ray.get` is blocking, how Ray Tune parallelizes hyperparameter search, and how Ray Train wraps
distributed data-parallel training without the usual boilerplate.

**⭐ Start here — suggested path:**

1. **Run it locally** — read [Ray Getting Started](https://docs.ray.io/en/latest/ray-overview/getting-started.html). *Install, then parallelize a function with `@ray.remote` — the core idea in minutes.*
2. **See the big picture** — [Ray: A Framework for Scaling Python & ML](https://www.youtube.com/watch?v=uzt-CwohQC8) (Anyscale). *Where tasks, actors, and the ML libraries fit together.*
3. **Distribute training** — work the [Ray Train docs](https://docs.ray.io/en/latest/train/train.html) + [Ray Train video](https://www.youtube.com/watch?v=GmtyTb0eARo). *Distributed data-parallel training with little boilerplate is the highest-value use.*
4. **Parallelize tuning** — study [Ray Tune](https://docs.ray.io/en/latest/tune/index.html) and watch [Scalable model training with Ray Tune](https://www.youtube.com/watch?v=eAWUZJe571Y). *A distributed hyperparameter sweep in <10 lines is Ray's signature win.*
5. **Understand the architecture** — read the [Ray paper](https://arxiv.org/abs/1712.05889). *The tasks/actors execution model is what every interview probes.*

## 🎓 Courses (free)
- [Ray documentation](https://docs.ray.io/en/latest/index.html) — **Anyscale / Ray team** — the authoritative guides for Core, Train, Tune, Serve, Data, and RLlib.
- [Ray Getting Started](https://docs.ray.io/en/latest/ray-overview/getting-started.html) — **Ray team** — the official on-ramp with runnable quickstarts per library.

## 🎥 Videos
- [Ray: A Framework for Scaling and Distributing Python & ML](https://www.youtube.com/watch?v=uzt-CwohQC8) — **Anyscale** — the overview of the whole stack.
- [Ray Train: A Production-Ready Library for Distributed Deep Learning](https://www.youtube.com/watch?v=GmtyTb0eARo) — **Anyscale** — distributed training in depth.
- [Tutorial: Scalable model training with Ray Tune](https://www.youtube.com/watch?v=eAWUZJe571Y) — **Software Underground** — hands-on hyperparameter tuning at scale.
- [Ray Tune: Distributed Hyperparameter Optimization Made Simple](https://www.youtube.com/watch?v=KgYZtlbFYXE) — **SF Python (Xiaowei Jiang)** — the tuning library explained.

## 📄 Key Papers
- [Ray: A Distributed Framework for Emerging AI Applications](https://arxiv.org/abs/1712.05889) — **Moritz et al. (2018), OSDI** — the foundational paper on Ray's tasks/actors model.
- [Ray documentation](https://docs.ray.io/en/latest/index.html) — **Ray team** — the canonical reference for the API and libraries.

## 📰 Articles / Blogs (free, no paywall)
- [Ray Getting Started](https://docs.ray.io/en/latest/ray-overview/getting-started.html) — **Ray team** — `@ray.remote` tasks/actors in a few cells.
- [Ray Train: Scalable Model Training](https://docs.ray.io/en/latest/train/train.html) — **Ray team** — distributed training patterns and APIs.
- [Ray Tune](https://docs.ray.io/en/latest/tune/index.html) — **Ray team** — experiment execution and hyperparameter tuning at scale.

## 📚 Books (free, with chapters)
- [Ray documentation (full)](https://docs.ray.io/en/latest/index.html) — **Ray team** — a book-length, free, library-by-library guide.
- [ray-project/ray (examples & docs source)](https://github.com/ray-project/ray) — **Anyscale** — runnable examples and the source, free on GitHub.

## 🔗 In this platform
- Related domain: [12. Deployment & MLOps](../../14.%20Deployment_and_MLOps/concepts/README.md) · [10. Reinforcement Learning](../../08.%20Reinforcement_Learning/concepts/README.md) *(RLlib)*
- Pairs with: [05 PyTorch](05-PyTorch.md) · [12 Weights & Biases](12-Weights-and-Biases.md)
- Deeper concept (the *why*): distributed training & serving → [Deployment & MLOps](../../14.%20Deployment_and_MLOps/concepts/README.md)
