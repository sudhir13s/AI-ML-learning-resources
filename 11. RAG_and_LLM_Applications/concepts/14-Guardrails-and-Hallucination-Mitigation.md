---
id: "11-rag-and-llm-apps/guardrails-hallucination-mitigation"
topic: "Guardrails & Hallucination Mitigation"
parent: "11-rag-and-llm-apps"
level: intermediate
prereqs: ["rag-fundamentals", "rag-evaluation", "citations-and-attribution"]
interview_frequency: high
updated: 2026-06-20
---

# Guardrails & Hallucination Mitigation
> Even with retrieval, an LLM can ignore the context, leak PII, go off-topic, or be jailbroken.
> **Guardrails** are programmable checks around the model — **input rails** (block prompt injection,
> off-topic, unsafe queries), **output rails** (fact-check against context, detect hallucination,
> redact PII, enforce format), and **dialog/retrieval rails**. Combined with grounding, citations,
> and abstention ("I don't know"), they make RAG safe to ship.

**Why it matters:** the "how do you keep an LLM app safe and truthful in production?" question. You'll
explain *why* RAG reduces but doesn't eliminate hallucination, the rail taxonomy (input/output/dialog/
retrieval/execution), self-check methods (SelfCheckGPT, groundedness checks), prompt-injection
defense, and when to make the model refuse rather than guess.

**⭐ Start here — suggested path:**

1. **Get the concept** — watch [What are guardrails for LLMs?](https://www.youtube.com/watch?v=FLOXGvqdwbM). *Input/output rails and why you wrap the model with checks.*
2. **See the categories** — read [IBM: What Are AI Guardrails?](https://www.ibm.com/think/topics/ai-guardrails). *The taxonomy (safety, topic, fact-check, PII) and where each rail sits.*
3. **Build output rails** — watch [Guardrails for LLM Applications (Guardrails AI)](https://www.youtube.com/watch?v=7V1w5gnZ-kw). *Validating/structuring outputs and retrying on failure, in code.*
4. **Guardrail a RAG app** — watch [How to implement LLM guardrails for RAG applications](https://www.youtube.com/watch?v=l5K4r_TJz_8). *Fact-checking retrieved context + blocking unsafe outputs in a RAG pipeline.*
5. **Read the sources** — skim [NeMo Guardrails](https://arxiv.org/abs/2310.10501) and [SelfCheckGPT](https://arxiv.org/abs/2303.08896). *Programmable rails, and a zero-resource hallucination check.*

## 🎓 Courses (free)
- [Building & Evaluating Advanced RAG](https://www.deeplearning.ai/short-courses/building-evaluating-advanced-rag/) — **DeepLearning.AI × TruLens** — groundedness checks are the core hallucination guardrail; this teaches measuring them.
- [Guardrails AI — documentation & quickstart](https://www.guardrailsai.com/docs) — **Guardrails AI** — free, hands-on guide to input/output validators and structured-output enforcement.

## 🎥 Videos
- [What are guardrails for LLMs?](https://www.youtube.com/watch?v=FLOXGvqdwbM) — **Red Hat** — clear conceptual overview of input/output rails.
- [Guardrails for LLM Applications (with Guardrails AI)](https://www.youtube.com/watch?v=7V1w5gnZ-kw) — **Sunny Savita** — building validators and retries with the Guardrails AI library.
- [NeMo Guardrails — Tame your LLM without Prompt Engineering](https://www.youtube.com/watch?v=3DfV6URqrZA) — **Coding Crash Courses** — programmable Colang rails for topic/safety control.
- [How to implement LLM guardrails for RAG applications](https://www.youtube.com/watch?v=l5K4r_TJz_8) — **IBM Developer** — fact-checking and output moderation inside a RAG pipeline.

## 📄 Key Papers
- [NeMo Guardrails: A Toolkit for Controllable and Safe LLM Applications](https://arxiv.org/abs/2310.10501) — **Rebedea et al. (2023)** — programmable rails (input/dialog/retrieval/execution/output).
- [SelfCheckGPT: Zero-Resource Black-Box Hallucination Detection](https://arxiv.org/abs/2303.08896) — **Manakul et al. (2023)** — self-consistency sampling to flag likely hallucinations.
- [Survey of Hallucination in Natural Language Generation](https://arxiv.org/abs/2202.03629) — **Ji et al. (2022)** — the reference taxonomy of hallucination types and mitigations.

## 📰 Articles / Blogs (free, no paywall)
- [What Are AI Guardrails?](https://www.ibm.com/think/topics/ai-guardrails) — **IBM** — the categories of guardrails and how they reduce risk.
- [NVIDIA NeMo-Guardrails (GitHub)](https://github.com/NVIDIA-NeMo/Guardrails) — **NVIDIA** — the open-source toolkit, with examples to read and run.
- [guardrails-ai/guardrails (GitHub)](https://github.com/guardrails-ai/guardrails) — **Guardrails AI** — validators for structure, safety, and hallucination, open source.
- [Adversarial Prompting / Risks](https://www.promptingguide.ai/risks/adversarial) — **DAIR.AI** — prompt injection & jailbreaks that input rails must defend against.

## 📚 Books (free, with chapters)
- [Speech and Language Processing, 3rd ed. — **Ch. 14 (QA & RAG — faithfulness/abstention)**](https://web.stanford.edu/~jurafsky/slp3/14.pdf) — **Jurafsky & Martin** — grounding and answer-faithfulness foundations behind output guardrails, free PDF.

## 🔗 In this platform
- Concept depth (the *why*): [AI-ML-intuition 8.02 Retrieval-Augmented Generation](../../../AI-ML-intuition/Module_8_LLMs_and_Agentic_Systems/8.02_Retrieval_Augmented_Generation.md) · [8.01 In-Context Learning & Prompting](../../../AI-ML-intuition/Module_8_LLMs_and_Agentic_Systems/8.01_In-Context_Learning_and_Prompting.md)
- Prereqs: [11 RAG Evaluation](11-RAG-Evaluation.md) · [13 Citations & Attribution](13-Citations-and-Attribution.md) · Next: [15 LLM App Orchestration](15-LLM-App-Orchestration.md)
- Related domain: [09. LLMs — Hallucination & Alignment Basics](../../09.%20LLMs/concepts/20-Hallucination-and-Alignment-Basics.md)
