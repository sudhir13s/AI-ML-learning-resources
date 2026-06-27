---
id: "13-tools-and-frameworks/weights-and-biases"
topic: "Weights & Biases (experiment tracking & sweeps)"
parent: "13-tools-and-frameworks"
level: intermediate
prereqs: ["python", "pytorch"]
interview_frequency: medium
updated: 2026-06-20
---

# Weights & Biases — Experiment Tracking & Sweeps
> The de-facto experiment-tracking tool: add `wandb.init()` and `wandb.log({...})` to your training
> script and every run's metrics, hyperparameters, system stats, gradients, and artifacts are logged
> to an auto-generated dashboard you can compare and share. **Sweeps** automate hyperparameter
> search; **Artifacts** version datasets and models; **Reports** turn results into shareable docs.

**Why it matters:** reproducibility and experiment comparison are central to real ML work, and "how
do you track experiments?" is a common MLOps interview question. Knowing the `init`/`log`/`config`
loop, how Sweeps parallelize hyperparameter search, and how Artifacts version data/models is
practical fluency every team expects.

**⭐ Start here — suggested path:**

1. **Log your first run** — follow the [W&B Quickstart](https://docs.wandb.ai/quickstart/). *`wandb.init()` + `wandb.log()` gives an instant dashboard — the core idea in 5 minutes.*
2. **See the full lifecycle** — [Weights & Biases End-to-End Demo](https://www.youtube.com/watch?v=tHAFujRhZLA) (W&B). *How tracking fits into a real train → evaluate → compare workflow.*
3. **Track experiments properly** — read [Experiments / track](https://docs.wandb.ai/models/track) and log `config` for hyperparameters. *Logging config is what makes runs comparable and reproducible.*
4. **Automate tuning with Sweeps** — study the [Sweeps guide](https://docs.wandb.ai/guides/sweeps/). *Declarative hyperparameter search is W&B's highest-leverage feature.*
5. **Go beyond tracking** — [Using W&B beyond experiment tracking](https://www.youtube.com/watch?v=_2DNYGv3jiM) (W&B). *Artifacts, Tables, and Reports for data versioning and collaboration.*

## 🎓 Courses (free)
- [W&B documentation](https://docs.wandb.ai/) — **Weights & Biases** — the authoritative guides for tracking, sweeps, artifacts, and reports.
- [W&B Quickstart](https://docs.wandb.ai/quickstart/) — **Weights & Biases** — the official hands-on first run.

## 🎥 Videos
- [Weights & Biases End-to-End Demo](https://www.youtube.com/watch?v=tHAFujRhZLA) — **Weights & Biases** — the full model-lifecycle walkthrough.
- [Using W&B beyond experiment tracking](https://www.youtube.com/watch?v=_2DNYGv3jiM) — **Weights & Biases** — Sweeps, Tables, and Artifacts.
- [PyTorch for Deep Learning — Full Course](https://www.youtube.com/watch?v=V_xro1bcAuA) — **freeCodeCamp** — the training loop you'd instrument with W&B.
- [Tutorial: Scalable model training with Ray Tune](https://www.youtube.com/watch?v=eAWUZJe571Y) — **Software Underground** — hyperparameter tuning context (complements W&B Sweeps).

## 📄 Key Papers
- [Experiment tracking with Weights & Biases](https://wandb.ai/site/experiment-tracking/) — **Weights & Biases** — the authoritative description of the tracking model.
- [W&B documentation](https://docs.wandb.ai/) — **Weights & Biases** — the canonical reference for the platform's APIs.

## 📰 Articles / Blogs (free, no paywall)
- [W&B Quickstart](https://docs.wandb.ai/quickstart/) — **Weights & Biases** — `init`/`log`/`config` in one page.
- [Sweeps (hyperparameter search)](https://docs.wandb.ai/guides/sweeps/) — **Weights & Biases** — declarative, parallel HPO.
- [Experiment tracking overview](https://wandb.ai/site/experiment-tracking/) — **Weights & Biases** — what to track and why.

## 📚 Books (free, with chapters)
- [W&B documentation (full)](https://docs.wandb.ai/) — **Weights & Biases** — a book-length, free guide across all features.
- [wandb (open-source client & examples)](https://github.com/wandb/wandb) — **Weights & Biases** — the SDK and runnable examples, free on GitHub.

## 🔗 In this platform
- Related domain: [12. Deployment & MLOps](../../14.%20Deployment_and_MLOps/README.md) · [05. Deep Learning](../../05.%20Deep_Learning/README.md)
- Pairs with: [05 PyTorch](../05-PyTorch/05-PyTorch.md) · [04 scikit-learn](../04-scikit-learn/04-scikit-learn.md) · [10 Ray](../10-Ray/10-Ray.md)
- Deeper concept (the *why*): experiment tracking & MLOps discipline → [Deployment & MLOps](../../14.%20Deployment_and_MLOps/README.md)
