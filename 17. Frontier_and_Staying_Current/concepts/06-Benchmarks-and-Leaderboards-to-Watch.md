---
id: "17-frontier/benchmarks-and-leaderboards"
topic: "Benchmarks & Leaderboards to Watch"
parent: "17-frontier"
level: intermediate
prereqs: ["model-evaluation-basics"]
interview_frequency: medium
updated: 2026-06-20
---

# Benchmarks & Leaderboards to Watch
> The standardized tasks and public rankings that let you compare models objectively — and the
> skill of reading them critically (what a benchmark measures, where it saturates, and how it gets
> gamed). Leaderboards are how the field tracks progress; knowing which to trust is half the battle.

**Why it matters:** "how would you evaluate / which model is best for X?" is a core interview and
on-the-job question. You need to know the major benchmarks (MMLU, HumanEval, Chatbot Arena, HELM,
SWE-bench), what each actually tests, and the failure modes — contamination, saturation, and
overfitting to the leaderboard — so you don't mistake a high score for real capability.

**⭐ Start here — suggested path:**

1. **Learn what benchmarks measure** — read ⭐ [30 LLM benchmarks and how they work](https://www.evidentlyai.com/llm-guide/llm-benchmarks). *Each benchmark tests a narrow slice (knowledge, code, reasoning); knowing the slice is how you read a score correctly.*
2. **Watch the live leaderboards** — bookmark [Chatbot Arena / LMArena](https://lmarena.ai/) (human-preference Elo) and [Papers with Code SOTA](https://paperswithcode.com/sota) (task-level). *Human-preference and task-level rankings answer different questions; use both.*
3. **Understand rigorous, multi-metric eval** — skim [Stanford HELM](https://crfm.stanford.edu/helm/). *HELM evaluates many models on many scenarios with many metrics — the antidote to single-number hype.*
4. **Know the failure modes** — read about contamination/saturation in the Chatbot Arena and benchmark-survey papers. *A benchmark stops being useful once it's saturated or leaks into training data; spotting this is the key skill.*
5. **Run an eval yourself** — try the [lm-evaluation-harness](https://github.com/EleutherAI/lm-evaluation-harness) on a small model. *Reproducing a benchmark number teaches you how brittle and configuration-dependent these scores are.*

## 🎓 Courses (free)
- [Stanford HELM](https://crfm.stanford.edu/helm/) — **Stanford CRFM** — a living, transparent evaluation suite; the docs teach what holistic evaluation should look like.
- [Hugging Face — Open LLM Leaderboard](https://huggingface.co/spaces/open-llm-leaderboard/open_llm_leaderboard) — **Hugging Face** — an open, reproducible leaderboard whose methodology page is a mini-course in fair eval.

## 🎥 Videos
- [How to Stay Up to Date on ML Research](https://www.youtube.com/watch?v=b1iz0l5KYig) — **Super Data Science (Jon Krohn)** — using leaderboards and SOTA tables as a tracking signal.
- [How To Read AI Research Papers Effectively](https://www.youtube.com/watch?v=K6Wui3mn-uI) — **DeepLearning.AI** — read the eval section critically: baselines, metrics, ablations.
- [Dr. Joelle Pineau — Reproducible, Reusable, Robust RL (NeurIPS 2018)](https://www.youtube.com/watch?v=Kee4ch3miVA) — **NeurIPS** — why headline benchmark numbers mislead without variance and proper baselines.
- [PyTorch Paper Replicating (Vision Transformer)](https://www.youtube.com/watch?v=tjpW_BY8y3g) — **Daniel Bourke** — replicating reported numbers shows how config-sensitive benchmark results are.

## 📄 Key Papers
- [Measuring Massive Multitask Language Understanding (MMLU)](https://arxiv.org/abs/2009.03300) — **Hendrycks et al. (2021)** — the 57-subject knowledge benchmark; the canonical "broad knowledge" score.
- [Evaluating Large Language Models Trained on Code (HumanEval)](https://arxiv.org/abs/2107.03374) — **Chen et al. (2021)** — the pass@k code benchmark behind "coding ability" claims.
- [Chatbot Arena: An Open Platform for Evaluating LLMs by Human Preference](https://arxiv.org/abs/2403.04132) — **Chiang et al. (2024)** — the Elo, human-preference leaderboard methodology.
- [Holistic Evaluation of Language Models (HELM)](https://arxiv.org/abs/2211.09110) — **Liang et al. (2022)** — multi-metric, multi-scenario evaluation; the case against single-number rankings.

## 📰 Articles / Blogs (free, no paywall)
- [30 LLM evaluation benchmarks and how they work](https://www.evidentlyai.com/llm-guide/llm-benchmarks) — **Evidently AI** — a clear catalogue of what each major benchmark measures.
- [Chatbot Arena — Benchmarking LLMs in the Wild with Elo](https://www.lmsys.org/blog/2023-05-03-arena/) — **LMSYS** — how human-preference ranking works and why it complements static tests.
- [Papers with Code — SOTA leaderboards](https://paperswithcode.com/sota) — **Papers with Code** — per-task rankings, each linked to a paper and code.
- [SWE-bench](https://www.swebench.com/) — **SWE-bench** — a real-world coding-agent benchmark (resolve GitHub issues); a current frontier yardstick.

## 📚 Books (free, with chapters)
- [Dive into Deep Learning](https://d2l.ai/) — **Zhang et al.** — its evaluation chapters ground the metrics that benchmarks aggregate.
- [Speech and Language Processing, 3rd ed. — Ch. on evaluation](https://web.stanford.edu/~jurafsky/slp3/) — **Jurafsky & Martin** — the reference on NLP metrics (BLEU, F1, perplexity) that underlie many leaderboards.

## 🔗 In this platform
- Per-concept index: [Frontier & Staying Current — concepts](README.md)
- Pair with: [02 arXiv & Papers with Code](02-arXiv-and-Papers-with-Code.md) · [11 Evaluating Hype vs Substance](11-Evaluating-Hype-vs-Substance.md)
- Evaluation depth lives in: [09. LLMs](../../09.%20LLMs/README.md) · [12. Deployment & MLOps](../../14.%20Deployment_and_MLOps/concepts/README.md)
