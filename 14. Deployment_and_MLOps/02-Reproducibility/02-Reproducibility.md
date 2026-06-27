---
id: "14-mlops/reproducibility"
topic: "Reproducibility (seeds, environments, lineage)"
parent: "14-mlops-and-deployment"
level: intermediate
prereqs: ["software-engineering", "ml-lifecycle"]
interview_frequency: medium
updated: 2026-06-20
---

# Reproducibility — Seeds, Environments & Lineage
> Being able to re-run a result and get the same model: pin the code, data, config, environment, and
> randomness so a run is a deterministic function of versioned inputs. The bedrock that experiment
> tracking, versioning, and CI/CD all stand on.

**Why it matters:** "your model trained great last month but you can't recreate it — what went wrong?"
Interviewers probe the full surface: random seeds and non-determinism (cuDNN, data shuffling), pinned
environments (Docker, lockfiles), data + code + config versioning, and lineage (which data + commit +
hyperparameters produced this artifact). The discipline that separates a demo from a system.

**⭐ Start here — suggested path:**

1. **See the problem** — watch [Joelle Pineau: Reproducible, Reusable, Robust ML](https://www.youtube.com/watch?v=wVkViYY_fwA). *The reproducibility crisis framing that motivates every practice below.*
2. **Get the checklist** — read [CMU ML Blog: Reproducibility](https://blog.ml.cmu.edu/2020/08/31/5-reproducibility/) and the [ML Reproducibility Checklist (paper)](https://arxiv.org/abs/2003.12206). *Concrete things to pin: code, data, environment, seeds, compute.*
3. **Separate the layers** — read [The Gradient: Independently Reproducible ML](https://thegradient.pub/independently-reproducible-machine-learning/). *Distinguishes "re-run my code" from "someone reproduces my result" — interviewers love this distinction.*
4. **Pin the environment** — read [Made With ML: Versioning](https://madewithml.com/courses/mlops/versioning/) and pin deps + data + model. *Turns the principles into code + DVC + a reproducible run.*
5. **Containerize it** — make the run a deterministic function of a pinned image; see [08 Model Packaging & Containerization](../08-Model-Packaging-and-Containerization/08-Model-Packaging-and-Containerization.md). *Environment parity is the last reproducibility gap.*

## 🎓 Courses (free)
- [Made With ML — Versioning & Reproducibility](https://madewithml.com/courses/mlops/versioning/) — **Goku Mohandas** — pin code, data, and models so a run is reproducible end-to-end.
- [Full Stack Deep Learning — Course 2022](https://fullstackdeeplearning.com/course/2022/) — **FSDL** — experiment management and reproducible training as part of shipping ML.

## 🎥 Videos
- [Reproducible, Reusable, and Robust Reinforcement Learning](https://www.youtube.com/watch?v=wVkViYY_fwA) — **Joelle Pineau (NeurIPS)** — the keynote that defined the ML reproducibility agenda.
- [2022 Toronto Workshop on Reproducibility](https://www.youtube.com/watch?v=e9CujtFbmmQ) — **Joelle Pineau** — practical guidelines: checklists, code release, variance reporting.
- [MLOps Zoomcamp 1.1 — Introduction](https://www.youtube.com/watch?v=s0uaFZSzwfI) — **DataTalks.Club** — frames why reproducibility + tracking + versioning are the MLOps foundation.
- [Versioning Data with DVC (Hands-On)](https://www.youtube.com/watch?v=kLKBcPonMYw) — **DVCorg** — pinning data + models to a Git commit, the practical core of lineage.

## 📄 Key Papers
- [Improving Reproducibility in ML Research (NeurIPS 2019 Reproducibility Program)](https://arxiv.org/abs/2003.12206) — **Pineau et al. (2020)** — the reproducibility checklist and what conferences now require.
- [Hidden Technical Debt in Machine Learning Systems](https://papers.nips.cc/paper/2015/file/86df7dcfd896fcaf2674f757a2463eba-Paper.pdf) — **Sculley et al. (2015)** — config debt and undeclared consumers — direct enemies of reproducibility.
- [Challenges in Deploying ML: A Survey of Case Studies](https://arxiv.org/abs/2011.09926) — **Paleyes et al. (2020)** — real failures where lost lineage broke production systems.

## 📰 Articles / Blogs (free, no paywall)
- [5 — Reproducibility](https://blog.ml.cmu.edu/2020/08/31/5-reproducibility/) — **CMU ML Blog** — clear breakdown of method/data/code reproducibility and how to achieve each.
- [Independently Reproducible Machine Learning](https://thegradient.pub/independently-reproducible-machine-learning/) — **The Gradient** — the spectrum from "re-runs" to "independently reproducible."
- [The reproducibility crisis in ML-based science (Princeton)](https://reproducible.cs.princeton.edu/) — **Kapoor & Narayanan** — leakage and pitfalls that silently break reproducibility.

## 📚 Books (free, with chapters)
- [Designing Machine Learning Systems — **Ch. 6 "Model Development & Offline Evaluation"** (experiment tracking, versioning, reproducibility)](https://huyenchip.com/mlops/) — **Chip Huyen** — author's notes/talks free.
- [Machine Learning Engineering — **Ch. 5–6** (experiment management & reproducible training)](http://www.mlebook.com/wiki/doku.php) — **Andriy Burkov** — read-first chapters free online.

## 🔗 In this platform
- Builds on: [01 ML Lifecycle & MLOps Maturity](../01-ML-Lifecycle-and-MLOps-Maturity/01-ML-Lifecycle-and-MLOps-Maturity.md)
- Next concepts: [03 Experiment Tracking](../03-Experiment-Tracking/03-Experiment-Tracking.md) · [04 Data & Model Versioning](../04-Data-and-Model-Versioning/04-Data-and-Model-Versioning.md) · [08 Model Packaging & Containerization](../08-Model-Packaging-and-Containerization/08-Model-Packaging-and-Containerization.md)
