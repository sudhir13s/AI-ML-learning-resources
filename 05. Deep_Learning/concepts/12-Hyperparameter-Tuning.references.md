---
id: "05-deep-learning/hyperparameter-tuning/references"
topic: "Hyperparameter Tuning — References"
parent: "05-deep-learning/hyperparameter-tuning"
type: references
updated: 2026-06-22
---

# Hyperparameter Tuning — references and further reading

> Companion link library for **[Hyperparameter Tuning](12-Hyperparameter-Tuning.md)** (the concept page). This file holds the curated links — external sources *and* internal links to related pages on this platform — kept separate so it can be reused as a standalone reference list. Grouped by type, best-first. Every entry is from a primary author or a recognized deep explainer — chosen for depth on *this* topic, not popularity.

**Start here — suggested path**:
1. **Build intuition** — watch [Tuning Process (C2W3L01)](https://www.youtube.com/watch?v=AXDByU3D1hA) (**Andrew Ng**). *Which knobs matter most and the order to tune them — learning rate first.*
2. **See the search habits** — watch [Hyperparameter Tuning in Practice (C2W3L03)](https://www.youtube.com/watch?v=wKkcBPp3F1Y) (**Andrew Ng**). *Coarse-to-fine, log scales, and the panda-vs-caviar workflow.*
3. **Get the evidence** — read [Random Search for Hyper-Parameter Optimization](https://www.jmlr.org/papers/v13/bergstra12a.html) (**Bergstra & Bengio, 2012**). *The classic result: random beats grid when only a few dimensions matter.*
4. **Go Bayesian** — read [Exploring Bayesian Optimization](https://distill.pub/2020/bayesian-optimization/) (**Distill**). *Interactive: how a surrogate + acquisition picks the next trial.*
5. **Make it concrete** — run a TPE/Hyperband sweep with [Optuna](https://optuna.org/) following [d2l Ch. 19](https://d2l.ai/chapter_hyperparameter-optimization/index.html). *Watching the search converge makes the trade-offs real.*

**Videos**:
- [Tuning Process (C2W3L01)](https://www.youtube.com/watch?v=AXDByU3D1hA) — **DeepLearningAI (Andrew Ng)** — prioritizing which hyperparameters to tune first.
- [Hyperparameter Tuning in Practice (C2W3L03)](https://www.youtube.com/watch?v=wKkcBPp3F1Y) — **DeepLearningAI (Andrew Ng)** — coarse-to-fine search, log-scale sampling, pandas vs caviar.
- [Using an Appropriate Scale (C2W3L02)](https://www.youtube.com/watch?v=cSoK_6Rkbfg) — **DeepLearningAI (Andrew Ng)** — exactly *why* you sample the learning rate on a log scale.
- [Gaussian Processes and Bayesian Optimization](https://www.youtube.com/watch?v=4vGiHC35j9s) — **ritvikmath** — the GP surrogate + acquisition idea, built up visually from scratch.

**Interactive & visual**:
- [Exploring Bayesian Optimization](https://distill.pub/2020/bayesian-optimization/) — **Distill (Agnihotri & Batra)** — interactive surrogate + acquisition; drag points and watch EI move.
- [A Visual Exploration of Gaussian Processes](https://distill.pub/2019/visual-exploration-gaussian-processes/) — **Distill** — the surrogate model behind Bayesian optimization, made visual.

**Courses (free)**:
- [Dive into Deep Learning — Ch. 19 Hyperparameter Optimization](https://d2l.ai/chapter_hyperparameter-optimization/index.html) — **Zhang, Lipton, Li & Smola** — random search, Bayesian opt, and Hyperband with runnable code.
- [Stanford CS231n — Setting up the data and the loss / babysitting](https://cs231n.github.io/neural-networks-3/) — **Stanford (Karpathy / Li / Johnson)** — hyperparameter search ranges, random vs grid, and monitoring.

**Articles / blogs (free, no paywall)**:
- [A Recipe for Training Neural Networks](https://karpathy.github.io/2019/04/25/recipe/) — **Andrej Karpathy** — the disciplined coarse-to-fine tuning workflow that prevents wasted trials.
- [Optuna: A Next-generation Hyperparameter Optimization Framework (docs)](https://optuna.readthedocs.io/en/stable/) — **Optuna team (Preferred Networks)** — define-by-run API, TPE sampler, and pruners, with examples.
- [Massively Parallel Hyperparameter Optimization (ASHA)](https://blog.ml.cmu.edu/2018/12/12/massively-parallel-hyperparameter-optimization/) — **Determined AI / CMU (Liam Li)** — asynchronous successive halving for clusters, from a Hyperband author.
- [How (Not) to Tune Your Model With Hyperopt](https://www.databricks.com/blog/2021/04/15/how-not-to-tune-your-model-with-hyperopt.html) — **Databricks** — practical TPE/Bayesian tuning pitfalls and search-space design.
- [The Deep Learning Tuning Playbook](https://github.com/google-research/tuning_playbook) — **Google Research (Godbole, Dahl, et al.)** — a rigorous, opinionated guide to tuning deep nets; the modern practitioner's bible.

**Key papers**:
- [Random Search for Hyper-Parameter Optimization](https://www.jmlr.org/papers/v13/bergstra12a.html) — **Bergstra & Bengio (2012)** — why random search dominates grid in practice (the low-effective-dimensionality argument).
- [Algorithms for Hyper-Parameter Optimization (TPE)](https://papers.nips.cc/paper/2011/hash/86e8f7ab32cfd12577bc2619bc635690-Abstract.html) — **Bergstra, Bardenet, Bengio & Kégl (2011)** — the Tree-structured Parzen Estimator behind Hyperopt/Optuna.
- [Practical Bayesian Optimization of Machine Learning Algorithms](https://arxiv.org/abs/1206.2944) — **Snoek, Larochelle & Adams (2012)** — GP-based Bayesian optimization with Expected Improvement for ML.
- [Hyperband: A Novel Bandit-Based Approach to Hyperparameter Optimization](https://arxiv.org/abs/1603.06560) — **Li, Jamieson, DeSalvo, Rostamizadeh & Talwalkar (2017)** — successive halving + the bracket hedge.
- [BOHB: Robust and Efficient Hyperparameter Optimization at Scale](https://arxiv.org/abs/1807.01774) — **Falkner, Klein & Hutter (2018)** — Bayesian optimization married to Hyperband.
- [Population Based Training of Neural Networks](https://arxiv.org/abs/1711.09846) — **Jaderberg et al. (2017, DeepMind)** — evolving hyperparameter *schedules* during training.
- [Practical Recommendations for Gradient-Based Training of Deep Architectures](https://arxiv.org/abs/1206.5533) — **Yoshua Bengio (2012)** — the canonical guide to setting LR, batch size, and regularizers.
- [Optuna: A Next-generation Hyperparameter Optimization Framework](https://arxiv.org/abs/1907.10902) — **Akiba et al. (2019)** — the define-by-run framework, its TPE sampler, and pruning.
- [DARTS: Differentiable Architecture Search](https://arxiv.org/abs/1806.09055) — **Liu, Simonyan & Yang (2018)** — making architecture search gradient-based; the NAS frontier.
- [A Taxonomy of Global Optimization Methods Based on Response Surfaces (EGO)](https://link.springer.com/article/10.1023/A:1012771025575) — **Jones (2001)** — the response-surface / Expected-Improvement lineage Bayesian opt descends from.

**Books (free chapters)**:
- [Deep Learning — §11.4 Selecting Hyperparameters](https://www.deeplearningbook.org/contents/guidelines.html) — **Goodfellow, Bengio & Courville** — manual vs automatic search and what each knob does.
- [Automated Machine Learning: Methods, Systems, Challenges](https://www.automl.org/book/) — **Hutter, Kotthoff & Vanschoren (eds.)** — the open-access AutoML book; full chapters on HPO, Bayesian opt, and NAS.

**In this platform**:
- Concept page (full explanation): [Hyperparameter Tuning](12-Hyperparameter-Tuning.md)
- Prerequisites — the knobs you're tuning: [Optimizers](07-Optimizers.md) · [Learning-Rate Schedules & Warmup](08-Learning-Rate-Schedules-and-Warmup.md) · [Regularization](09-Regularization.md)
- The validation protocol (leak-free model selection): [Cross-Validation](../../03.%20Supervised_Learning/concepts/13-Cross-Validation.md)
- Concept depth (the *why*): [AI-ML-intuition 2.09 Learning Rate Schedules](../../../AI-ML-intuition/Module_2_Optimization/2.09_Learning_Rate_Schedules.md) · [3.07 Bias–Variance & Generalization](../../../AI-ML-intuition/Module_3_Evaluation/3.07_Bias_Variance_and_Generalization.md)
- Field overview: [Deep Learning](../README.md) · Related: [13. Tools & Frameworks](../../13.%20Tools_and_Frameworks/concepts/README.md) (Optuna / Ray Tune in practice)
