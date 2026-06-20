---
id: "00-basics/ml-workflow-and-lifecycle"
topic: "The ML Workflow & Lifecycle"
parent: "00-basics"
level: beginner
prereqs: ["what-is-ai-ml-dl", "types-of-machine-learning"]
interview_frequency: high
updated: 2026-06-20
---

# The ML Workflow & Lifecycle
> Every ML project follows the same arc: **frame the problem → collect & explore data → prepare
> features → train a model → evaluate → deploy → monitor → iterate.** It's a *loop*, not a line —
> you cycle back constantly. The famous rule of thumb: 50–80% of the work is data, not modeling.

**Why it matters:** interviewers ask "walk me through how you'd approach an ML project end to end"
to see if you think like a practitioner, not just an algorithm-memorizer. Knowing the lifecycle
(and that data prep dominates, and that deployment/monitoring are where projects actually fail) is
what separates someone who's shipped ML from someone who's only done tutorials.

**⭐ Start here — suggested path:**

1. **See the whole loop** — watch [The 7 Steps of Machine Learning](https://www.youtube.com/watch?v=nKW8Ndu7Mjw). *Google's clean, beginner overview of the end-to-end arc.*
2. **Learn the industry-standard process** — watch [Introduction to CRISP-DM](https://www.youtube.com/watch?v=q_okDS2RtzY), then read [What is CRISP-DM?](https://www.datascience-pm.com/crisp-dm-2/). *The 6-phase methodology that names every stage you'll repeat on real projects.*
3. **Internalize why data dominates** — watch [Andrew Ng on data-centric AI](https://www.youtube.com/watch?v=06-AZXmwHjo). *Why improving data beats tweaking models — the most important lifecycle lesson.*
4. **Read the hard-won rules** — skim [Google: Rules of ML](https://developers.google.com/machine-learning/guides/rules-of-ml). *Battle-tested guidance for each lifecycle stage from Google engineers.*
5. **Watch one go end to end** — work through part of [ML for Everybody – Full Course](https://www.youtube.com/watch?v=i_LwzRVP7bg). *See data → train → evaluate happen in code, start to finish.*

## 🎓 Courses (free)
- [Google ML Crash Course — Production ML Systems](https://developers.google.com/machine-learning/crash-course/production-ml-systems) — **Google** — free; the lifecycle from data to deployment and monitoring.
- [Kaggle Learn — Intro to Machine Learning](https://www.kaggle.com/learn/intro-to-machine-learning) — **Kaggle** — free, hands-on; runs you through the build-train-evaluate loop in notebooks.
- [Made With ML — MLOps lessons](https://madewithml.com/) — **Goku Mohandas** — free, open course covering the full design → develop → deploy lifecycle.

## 🎥 Videos
- [The 7 Steps of Machine Learning](https://www.youtube.com/watch?v=nKW8Ndu7Mjw) — **Google Cloud Tech** — the clearest beginner overview of the end-to-end arc.
- [Introduction to the CRISP-DM Methodology](https://www.youtube.com/watch?v=q_okDS2RtzY) — **Data Science / Analytics** — the standard 6-phase project process.
- [6 Phases in CRISP-DM Methodology](https://www.youtube.com/watch?v=NinRBxDVdnM) — **Data Mining** — each phase explained with what you actually do in it.
- [A Chat with Andrew Ng on MLOps: Model-centric to Data-centric AI](https://www.youtube.com/watch?v=06-AZXmwHjo) — **DeepLearning.AI** — why the data stage dominates project success.
- [Machine Learning for Everybody – Full Course](https://www.youtube.com/watch?v=i_LwzRVP7bg) — **freeCodeCamp** — the lifecycle demonstrated end to end in code.

## 📄 Key Papers
- [Hidden Technical Debt in Machine Learning Systems](https://papers.nips.cc/paper/2015/file/86df7dcfd896fcaf2674f757a2463eba-Paper.pdf) — **Sculley et al. (Google, 2015)** — why the model is the small part; the lifecycle around it is where debt accumulates.
- [Rules of Machine Learning: Best Practices for ML Engineering](https://developers.google.com/machine-learning/guides/rules-of-ml) — **Martin Zinkevich (Google)** — the canonical practitioner's guide to each lifecycle stage.
- [CRISP-DM 1.0: Step-by-step data mining guide](https://inseaddataanalytics.github.io/INSEADAnalytics/CRISP_DM.pdf) — **Chapman et al. (2000)** — the original document defining the standard lifecycle.

## 📰 Articles / Blogs (free, no paywall)
- [What is CRISP-DM?](https://www.datascience-pm.com/crisp-dm-2/) — **Data Science PM** — the six phases explained clearly, with how iteration works.
- [End-to-End Machine Learning Course Project](https://www.freecodecamp.org/news/end-to-end-machine-learning-course-project/) — **freeCodeCamp** — a written walkthrough of every lifecycle stage on one dataset.
- [The end-to-end ML workflow](https://ml-ops.org/content/end-to-end-ml-workflow) — **ml-ops.org** — vendor-neutral reference for the production ML lifecycle.

## 📚 Books (free, with chapters)
- [Dive into Deep Learning — Ch. 1 "Introduction"](https://d2l.ai/chapter_introduction/index.html) — **Zhang et al.** — frames the data → model → evaluate loop with runnable code.
- [An Introduction to Statistical Learning — Ch. 2 "Statistical Learning"](https://www.statlearning.com/) — **James et al.** — free PDF; how to assess models within the workflow.
- [Approaching (Almost) Any Machine Learning Problem](https://github.com/abhishekkrthakur/approachingalmost) — **Abhishek Thakur** — free PDF; a practical, project-shaped walk through the workflow.

## 🔗 In this platform
- Next concepts: [04 How Models Learn](04-How-Models-Learn.md) · [12 Your First ML Project](12-Your-First-ML-Project.md)
- Go deeper — data stages: [02. Data Preprocessing](../../02.%20Data_Preprocessing/README.md)
- Go deeper — deployment & monitoring: [12. Deployment & MLOps](../../12.%20Deployment_and_MLOps/concepts/README.md)
