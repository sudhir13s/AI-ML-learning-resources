---
id: "11-rag-and-llm-apps/guardrails-hallucination-mitigation/references"
topic: "Guardrails & Hallucination Mitigation — References"
parent: "11-rag-and-llm-apps/guardrails-hallucination-mitigation"
type: references
updated: 2026-07-02
---

# Guardrails & Hallucination Mitigation — references and further reading

> Companion link library for **[Guardrails & Hallucination Mitigation](14-Guardrails-and-Hallucination-Mitigation.md)**
> (the concept page). External sources *and* internal cross-links, kept separate so it can be reused as
> a standalone list. Grouped by type, best-first. Every entry is free/open (no paywall) and chosen for
> depth on *this* topic — input/output rails, prompt-injection defense, grounding-based abstention, and
> the false-refuse/false-allow tradeoff. Every formula/mechanism cited on the concept page (indirect
> prompt injection, Llama Guard, NeMo Guardrails, selective prediction / risk-coverage) appears here as
> a primary source.

**Start here — suggested path**:
1. **Get the concept** — watch [What are guardrails for LLMs?](https://www.youtube.com/watch?v=FLOXGvqdwbM) (**Red Hat**). *Input/output rails and why you wrap the model with checks.*
2. **See the categories** — read [What Are AI Guardrails?](https://www.ibm.com/think/topics/ai-guardrails) (**IBM**). *The taxonomy (safety, topic, fact-check, PII) and where each rail sits.*
3. **Know the attack** — skim [Indirect Prompt Injection (Greshake et al. 2023)](https://arxiv.org/abs/2302.12173). *Why a retrieved document can hijack your LLM — the threat input rails defend against.*
4. **Build output rails** — watch [How to implement LLM guardrails for RAG applications](https://www.youtube.com/watch?v=l5K4r_TJz_8) (**IBM Developer**). *Fact-checking retrieved context + blocking unsafe outputs in a RAG pipeline.*
5. **Read the sources** — skim [NeMo Guardrails (Rebedea et al. 2023)](https://arxiv.org/abs/2310.10501) and [Llama Guard (Inan et al. 2023)](https://arxiv.org/abs/2312.06674). *Programmable rails, and the trained input/output safety classifier.*

**Videos**:
- [What are guardrails for LLMs?](https://www.youtube.com/watch?v=FLOXGvqdwbM) — **Red Hat** — a clear conceptual overview of input/output rails and why they exist.
- [How to implement LLM guardrails for RAG applications](https://www.youtube.com/watch?v=l5K4r_TJz_8) — **IBM Developer** — fact-checking and output moderation inside a RAG pipeline.
- [Guardrails for LLM Applications (with Guardrails AI)](https://www.youtube.com/watch?v=7V1w5gnZ-kw) — **Sunny Savita** — building input/output validators and retries with the Guardrails AI library, in code.
- [NeMo Guardrails — Tame your LLM without Prompt Engineering](https://www.youtube.com/watch?v=3DfV6URqrZA) — **Coding Crash Courses** — programmable Colang rails for topic/safety control.

**Interactive & visual**:
- [Guardrails AI — docs & quickstart](https://www.guardrailsai.com/docs) — **Guardrails AI** — a runnable, free walkthrough of `Guard` objects and input/output validators.
- [Azure AI Content Safety — Prompt Shields quickstart](https://learn.microsoft.com/en-us/azure/ai-services/content-safety/quickstart-jailbreak) — **Microsoft** — detect direct and indirect prompt-injection attacks against your app, hands-on.
- [Adversarial Prompting / Risks](https://www.promptingguide.ai/risks/adversarial) — **DAIR.AI** — a catalogue of prompt injection & jailbreak techniques that input rails must defend against.

**Courses (free)**:
- [Red Teaming LLM Applications](https://www.deeplearning.ai/short-courses/red-teaming-llm-applications/) — **DeepLearning.AI × Giskard** — how to probe an app for injection/jailbreak/leakage — the attacks guardrails must stop.
- [Building & Evaluating Advanced RAG](https://www.deeplearning.ai/short-courses/building-evaluating-advanced-rag/) — **DeepLearning.AI × TruLens** — groundedness checks are the core hallucination guardrail; this teaches measuring them.

**Articles / blogs (free, no paywall)**:
- [What Are AI Guardrails?](https://www.ibm.com/think/topics/ai-guardrails) — **IBM** — the categories of guardrails and how they reduce risk, clearly organized.
- [NVIDIA NeMo-Guardrails (GitHub)](https://github.com/NVIDIA-NeMo/Guardrails) — **NVIDIA** — the open-source toolkit with runnable examples of input/dialog/output rails.
- [guardrails-ai/guardrails (GitHub)](https://github.com/guardrails-ai/guardrails) — **Guardrails AI** — validators for structure, PII, toxicity, and hallucination, open source.
- [Prompt Shields in Azure AI Content Safety](https://learn.microsoft.com/en-us/azure/ai-services/content-safety/concepts/jailbreak-detection) — **Microsoft** — how the production input rail detects direct + indirect injection, with the API.
- [Groundedness detection (Azure AI Content Safety)](https://learn.microsoft.com/en-us/azure/ai-services/content-safety/concepts/groundedness) — **Microsoft** — the production output rail: detect (and correct) ungrounded generations against provided sources.

**Key papers**:
- [Not what you've signed up for: Compromising Real-World LLM-Integrated Applications with Indirect Prompt Injection](https://arxiv.org/abs/2302.12173) — **Greshake et al. (2023)** — the threat model the input rail defends against: instructions planted in retrieved content that the LLM then follows.
- [Llama Guard: LLM-based Input-Output Safeguard for Human-AI Conversations](https://arxiv.org/abs/2312.06674) — **Inan et al. (2023)** — the trained input/output safety classifier over a risk taxonomy; the generalizing answer to regex bypass.
- [NeMo Guardrails: A Toolkit for Controllable and Safe LLM Applications with Programmable Rails](https://arxiv.org/abs/2310.10501) — **Rebedea et al. (2023, EMNLP demo)** — the five programmable rail categories (input/dialog/retrieval/execution/output) and the Colang language.
- [Selective Classification for Deep Neural Networks](https://arxiv.org/abs/1705.08500) — **Geifman & El-Yaniv (2017, NeurIPS)** — the reject-option / risk-coverage framework behind grounding-based abstention and the false-refuse/false-allow tradeoff.
- [SelfCheckGPT: Zero-Resource Black-Box Hallucination Detection for Generative Large Language Models](https://arxiv.org/abs/2303.08896) — **Manakul, Liusie & Gales (2023, EMNLP)** — a sampling-based output rail that flags non-self-consistent (likely hallucinated) claims with no gold context.
- [Survey of Hallucination in Natural Language Generation](https://arxiv.org/abs/2202.03629) — **Ji et al. (2022)** — the reference taxonomy of hallucination types and mitigations the output rail targets.
- [Retrieval-Augmented Generation for Knowledge-Intensive NLP Tasks](https://arxiv.org/abs/2005.11401) — **Lewis et al. (2020)** — the RAG pipeline these guardrails wrap.

**Books (free chapters)**:
- [Speech and Language Processing, 3rd ed. — Ch. 14 "Question Answering & Information Retrieval"](https://web.stanford.edu/~jurafsky/slp3/14.pdf) — **Jurafsky & Martin** — grounding and answer-faithfulness/abstention foundations behind output guardrails, free PDF.

**In this platform**:
- Concept page (full explanation): [Guardrails & Hallucination Mitigation](14-Guardrails-and-Hallucination-Mitigation.md)
- Concept depth (the *why*): [AI-ML-intuition 8.02 Retrieval-Augmented Generation](../../../AI-ML-intuition/Module_8_LLMs_and_Agentic_Systems/8.02_Retrieval_Augmented_Generation.md) · [8.01 In-Context Learning & Prompting](../../../AI-ML-intuition/Module_8_LLMs_and_Agentic_Systems/8.01_In-Context_Learning_and_Prompting.md)
- Machinery reused here: [05 Hybrid Search (the DenseRetriever)](../05-Hybrid-Search-BM25-and-Dense/05-Hybrid-Search-BM25-and-Dense.md) · [11 RAG Evaluation (the faithfulness/grounding proxy)](../11-RAG-Evaluation/11-RAG-Evaluation.md) · [13 Citations & Attribution (the grounding cosine + cosine≠entailment caveat)](../13-Citations-and-Attribution/13-Citations-and-Attribution.md)
- Foundations: [01 RAG Fundamentals](../01-RAG-Fundamentals/01-RAG-Fundamentals.md) · [08 Advanced RAG (Self-RAG support check)](../08-Advanced-RAG-Parent-Doc-Fusion-Self-RAG/08-Advanced-RAG-Parent-Doc-Fusion-Self-RAG.md)
- Next / related: [15 LLM App Orchestration](../15-LLM-App-Orchestration/15-LLM-App-Orchestration.md)
- Related domain: [09. LLMs — Hallucination & Alignment Basics](../../09.%20LLMs/20-Hallucination-and-Alignment-Basics/20-Hallucination-and-Alignment-Basics.md)
