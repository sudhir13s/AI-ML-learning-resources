---
id: "14-mlops/llmops"
topic: "LLMOps (eval · guardrails · prompt versioning · cost/latency)"
parent: "14-mlops-and-deployment"
level: advanced
prereqs: ["model-serving", "model-monitoring-and-observability", "llms"]
interview_frequency: very-high
updated: 2026-06-20
---

# LLMOps — Eval · Guardrails · Prompt Versioning · Cost/Latency
> MLOps adapted to LLM applications, where the "model" is often a frozen API and the system you operate is
> prompts + retrieval + tools. What changes: evaluation is open-ended (LLM-as-judge, rubrics, no single
> accuracy number), **prompts are versioned artifacts**, guardrails defend inputs/outputs (prompt injection,
> PII, toxicity, hallucination), and cost/latency per token become first-class operational metrics.

**Why it matters:** the fastest-growing interview area. Expect "how is LLMOps different from MLOps?" —
the answers: evaluation without ground truth (offline evals + LLM-as-judge + online feedback), prompt
versioning and regression testing, guardrails as the production safety layer, and managing token cost +
latency. Builds on LLM internals (covered in 09. LLMs) but focuses on *operating* LLM systems.

**⭐ Start here — suggested path:**

1. **Get the landscape** — read [Building LLM Applications for Production](https://huyenchip.com/2023/04/11/llm-engineering.html) and watch [the companion talk](https://www.youtube.com/watch?v=spamOhG7BOA). *The canonical map of what LLMOps must handle.*
2. **Define LLMOps** — read [What is LLMOps?](https://www.databricks.com/glossary/llmops). *Where it overlaps with and diverges from MLOps.*
3. **Evaluate LLMs** — browse [OpenAI Evals](https://github.com/openai/evals) and watch the [Google Cloud LLMOps course intro](https://www.youtube.com/watch?v=tabmG21y290). *Eval harnesses, LLM-as-judge, regression testing for prompts.*
4. **Add guardrails** — watch [Combining Guardrails and LLM Evaluations](https://www.youtube.com/watch?v=gXdBwgVZ_Ho). *Input/output filters: prompt injection, PII, toxicity, format checks.*
5. **Trace & monitor in prod** — read [LangSmith docs](https://docs.smith.langchain.com/) and watch [LLM Evals & Guardrails in Production](https://www.youtube.com/watch?v=041gk0N8gPA). *Tracing, online evals, and feedback loops.*

## 🎓 Courses (free)
- [What is LLMOps? (LLM Operations Guide)](https://www.databricks.com/glossary/llmops) — **Databricks** — the discipline, components, and how it differs from MLOps.
- [LangSmith — Documentation](https://docs.smith.langchain.com/) — **LangChain** — tracing, evaluation, and monitoring for LLM apps.

## 🎥 Videos
- [Building LLM Applications for Production // Chip Huyen](https://www.youtube.com/watch?v=spamOhG7BOA) — **Chip Huyen (LLMs in Prod)** — the foundational LLMOps talk: eval, cost, latency, reliability.
- [New Course with Google Cloud: LLMOps](https://www.youtube.com/watch?v=tabmG21y290) — **DeepLearning.AI** — the LLMOps lifecycle and tooling overview.
- [AI with Assurance: Combining Guardrails and LLM Evaluations](https://www.youtube.com/watch?v=gXdBwgVZ_Ho) — **community** — guardrails + evals as the production safety layer.
- [LLM Evals & Guardrails in Production](https://www.youtube.com/watch?v=041gk0N8gPA) — **TMLS** — practitioners on evaluating and guarding LLMs live.

## 📄 Key Papers
- [Challenges in Deploying ML: A Survey of Case Studies](https://arxiv.org/abs/2011.09926) — **Paleyes et al. (2020)** — deployment challenges that compound for LLM systems.
- [Hidden Technical Debt in Machine Learning Systems](https://papers.nips.cc/paper/2015/file/86df7dcfd896fcaf2674f757a2463eba-Paper.pdf) — **Sculley et al. (2015)** — the debt patterns that prompts/chains reintroduce at scale.

## 📰 Articles / Blogs (free, no paywall)
- [Building LLM Applications for Production](https://huyenchip.com/2023/04/11/llm-engineering.html) — **Chip Huyen** — the definitive free LLMOps overview (eval, prompt versioning, cost/latency).
- [What is LLMOps?](https://www.databricks.com/glossary/llmops) — **Databricks** — clear definition and component breakdown.
- [OpenAI Evals](https://github.com/openai/evals) — **OpenAI** — open framework for evaluating LLM behavior and regressions.

## 📚 Books (free, with chapters)
- [Designing Machine Learning Systems — **Ch. 8–9** (monitoring, distribution shift, test-in-prod) applied to LLM apps](https://huyenchip.com/mlops/) — **Chip Huyen** — author notes/talks free.
- [Machine Learning Engineering — **Ch. 9 "Monitoring & Maintenance"** (operating models in prod)](http://www.mlebook.com/wiki/doku.php) — **Andriy Burkov** — read-first chapters free.

## 🔗 In this platform
- Builds on: [09 Model Serving](09-Model-Serving.md) · [11 Model Monitoring & Observability](11-Model-Monitoring-and-Observability.md)
- Next concepts: [16 Cost Optimization](16-Cost-Optimization.md)
- Related concepts (covered elsewhere): LLM evaluation & benchmarks → [09. LLMs — LLM Evaluation](../../09.%20LLMs/concepts/19-LLM-Evaluation-and-Benchmarks.md) · prompting & in-context learning → [09. LLMs — Prompting](../../09.%20LLMs/concepts/16-Prompting-and-In-Context-Learning.md) · LLM inference cost/latency internals → [09. LLMs — Inference Optimization & Serving](../../09.%20LLMs/concepts/09-Inference-Optimization-and-Serving.md)
