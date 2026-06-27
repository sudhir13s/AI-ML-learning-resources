---
id: "14-mlops/ab-testing-shadow-and-canary-deployment"
topic: "A/B Testing · Shadow & Canary Deployment"
parent: "14-mlops-and-deployment"
level: advanced
prereqs: ["model-serving", "model-registry-and-governance"]
interview_frequency: high
updated: 2026-06-20
---

# A/B Testing · Shadow & Canary Deployment
> Strategies for releasing a new model safely, because **offline metrics don't guarantee online wins**.
> **Shadow**: run the new model in parallel, serve nothing — measure risk-free. **Canary**: route a small
> % of live traffic to it, ramp up if healthy. **A/B test**: split traffic and measure the business metric
> that decides the winner. Test in production, without betting the whole house.

**Why it matters:** the "your offline AUC went up — how do you know it's actually better in prod, and how
do you roll it out without risk?" question. Interviewers want the shadow → canary → full-rollout ladder,
when to use each, and the A/B basics (split, primary metric, why offline ≠ online). The deployment
counterpart to monitoring. (The *statistics* of A/B tests live in 01. Foundations — link below.)

**⭐ Start here — suggested path:**

1. **Get the ladder** — read [CD4ML: progressive delivery section](https://martinfowler.com/articles/cd4ml.html) for how models are rolled out gradually. *Frames shadow/canary/A-B as one safety continuum.*
2. **Shadow first** — watch [Shadow Deployment Explained](https://www.youtube.com/watch?v=1NUshG4ahoU) and read [SageMaker Shadow Tests](https://docs.aws.amazon.com/sagemaker/latest/dg/shadow-tests.html). *Test under real traffic with zero user risk.*
3. **Then canary** — read [Canary Release](https://martinfowler.com/bliki/CanaryRelease.html). *The 1% → 20% → 100% ramp and what to watch during it.*
4. **Then A/B** — watch [A/B Testing & Canary Deployments for ML Models](https://www.youtube.com/watch?v=57HEs4l6_44). *Splitting traffic and choosing the decision metric.*
5. **Get the stats right** — for hypothesis tests, power, and CUPED, go to [01. Foundations](../../01.%20Foundations/README.md). *MLOps owns the deployment mechanics; Foundations owns the experiment statistics.*

## 🎓 Courses (free)
- [Made With ML — MLOps Course (deployment strategies)](https://madewithml.com/courses/mlops/) — **Goku Mohandas** — safe rollout within the production workflow.
- [Google — Rules of Machine Learning](https://developers.google.com/machine-learning/guides/rules-of-ml) — **Google** — "measure the metric that matters" and launch discipline.

## 🎥 Videos
- [A/B Testing & Canary Deployments for ML Models: Safe Rollouts](https://www.youtube.com/watch?v=57HEs4l6_44) — **Uplatz** — the rollout strategies compared.
- [Shadow Deployment in Machine Learning Explained](https://www.youtube.com/watch?v=1NUshG4ahoU) — **community** — what shadow mode is and why it's the safe first step.
- [Deploying AI Models: How Does Shadow Deployment Work?](https://www.youtube.com/watch?v=gCnp8QWtCMc) — **community** — the mechanics of mirroring production traffic.
- [How to Master Shadow Deployment for Model Testing](https://www.youtube.com/watch?v=_kokPCOkvec) — **community** — deeper walkthrough of shadow testing in practice.

## 📄 Key Papers
- [Challenges in Deploying ML: A Survey of Case Studies](https://arxiv.org/abs/2011.09926) — **Paleyes et al. (2020)** — why offline metrics mislead and rollout safety matters.
- [The ML Test Score](https://research.google/pubs/the-ml-test-score-a-rubric-for-ml-production-readiness-and-technical-debt-reduction/) — **Breck et al. (Google, 2017)** — "test in production" as part of production readiness.

## 📰 Articles / Blogs (free, no paywall)
- [Canary Release](https://martinfowler.com/bliki/CanaryRelease.html) — **Martin Fowler** — the canonical definition of progressive rollout.
- [Continuous Delivery for ML (CD4ML)](https://martinfowler.com/articles/cd4ml.html) — **martinfowler.com** — model rollout inside the CD pipeline.
- [Shadow Tests (SageMaker)](https://docs.aws.amazon.com/sagemaker/latest/dg/shadow-tests.html) — **AWS** — production shadow-testing mechanics and metrics.

## 📚 Books (free, with chapters)
- [Designing Machine Learning Systems — **Ch. 9 "Continual Learning & Test in Production"** (shadow, canary, A/B)](https://huyenchip.com/mlops/) — **Chip Huyen** — the canonical chapter on test-in-prod (author notes free).
- [Machine Learning Engineering — **Ch. 8 "Model Deployment"** (deployment strategies)](http://www.mlebook.com/wiki/doku.php) — **Andriy Burkov** — read-first chapters free.

## 🔗 In this platform
- Builds on: [09 Model Serving](../09-Model-Serving/09-Model-Serving.md) · [13 Model Registry & Governance](../13-Model-Registry-and-Governance/13-Model-Registry-and-Governance.md)
- Next concepts: [12 Data & Concept Drift Detection](../12-Data-and-Concept-Drift-Detection/12-Data-and-Concept-Drift-Detection.md) · [07 CI/CD for ML & CT](../07-CICD-for-ML-and-Continuous-Training/07-CICD-for-ML-and-Continuous-Training.md)
- Related concept (covered elsewhere): A/B-test statistics (hypothesis tests, power, CUPED) → [01. Foundations](../../01.%20Foundations/README.md)
