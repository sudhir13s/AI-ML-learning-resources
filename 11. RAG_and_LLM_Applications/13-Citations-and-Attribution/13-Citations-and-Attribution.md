---
id: "11-rag-and-llm-apps/citations-and-attribution"
topic: "Citations & Attribution"
parent: "11-rag-and-llm-apps"
level: intermediate
prereqs: ["rag-fundamentals", "rag-evaluation"]
interview_frequency: medium
updated: 2026-06-20
---

# Citations & Attribution
> A trustworthy RAG answer doesn't just *be* grounded — it *shows* its sources, citing the exact
> passages each claim came from so users (and auditors) can verify it. Attribution turns "trust me"
> into "check for yourself," cuts hallucination, and is mandatory in regulated/enterprise settings.
> Techniques range from prompt-instructed citations to tool-calling spans to model-native citation
> APIs.

**Why it matters:** the "how do you make RAG verifiable / production-trustworthy?" question. You'll
discuss attribution methods (prompt the model to cite chunk IDs, return supporting snippets,
post-process to match claims→sources), the difference between *grounded* and *correctly cited*, and
how to evaluate citation quality (precision/recall of citations, citation faithfulness).

**⭐ Start here — suggested path:**

1. **See the methods** — read [LangChain: How to get a model to cite sources](https://python.langchain.com/docs/how_to/qa_citations/). *The five ways to add citations (tool-calling IDs, snippets, prompting, retrieval/generation post-processing).*
2. **Build a citing pipeline** — watch [How to Get LLM Answers With Sources](https://www.youtube.com/watch?v=69gUQ4XHg0o). *Return source documents alongside answers using LCEL.*
3. **Make citations verifiable** — watch [RAG — but with Verified Citations!](https://www.youtube.com/watch?v=-wGzSnhQKPM). *Forcing the model to cite spans that actually support each claim.*
4. **Use model-native citations** — read [Anthropic: Introducing Citations](https://www.anthropic.com/news/introducing-citations-api) + [docs](https://docs.anthropic.com/en/docs/build-with-claude/citations). *Sentence-level grounding built into the API; outperforms many custom setups.*
5. **Read the source** — skim [Enabling LLMs to Generate Text with Citations (ALCE)](https://arxiv.org/abs/2305.14627). *How to benchmark citation quality (fluency, correctness, citation precision/recall).*

## 🎓 Courses (free)
- [RAG from Scratch — Generation](https://github.com/langchain-ai/rag-from-scratch) — **LangChain (Lance Martin)** — generation notebooks where returning sources/citations fits naturally.
- [Building & Evaluating Advanced RAG](https://www.deeplearning.ai/short-courses/building-evaluating-advanced-rag/) — **DeepLearning.AI × TruLens** — groundedness eval, the metric attribution must satisfy.

## 🎥 Videos
- [How to Get LLM Answers With Sources — Advanced RAG](https://www.youtube.com/watch?v=69gUQ4XHg0o) — **M&M Tech** — returning source documents with answers via LangChain LCEL.
- [RAG — but with Verified Citations!](https://www.youtube.com/watch?v=-wGzSnhQKPM) — **Trelis Research** — making citations point to spans that genuinely support each claim.
- [Building a RAG System with In-line Citations Using Workflows](https://www.youtube.com/watch?v=P4xHWojIB-M) — **LlamaIndex** — inline citations wired into a retrieval workflow.
- [Unlocking Advanced RAG: Citations and Attributions](https://www.youtube.com/watch?v=RnCuOL-LBAw) — **Zilliz** — attribution patterns and why they reduce hallucination.

## 📄 Key Papers
- [Enabling Large Language Models to Generate Text with Citations (ALCE)](https://arxiv.org/abs/2305.14627) — **Gao et al. (2023)** — the benchmark + metrics for citation quality (correctness, precision, recall).
- [Attributed Question Answering (AQA)](https://arxiv.org/abs/2212.08037) — **Bohnet et al. (2022)** — formalizes attribution and AIS (attributable to identified sources) evaluation.
- [Learning Fine-Grained Grounded Citations for Attributed LLMs](https://arxiv.org/abs/2408.04568) — **Huang et al. (2024)** — training models to cite at sub-sentence granularity.

## 📰 Articles / Blogs (free, no paywall)
- [How to get a model to cite sources](https://python.langchain.com/docs/how_to/qa_citations/) — **LangChain** — the canonical menu of citation techniques with code.
- [Introducing Citations](https://www.anthropic.com/news/introducing-citations-api) — **Anthropic** — model-native, sentence-level source grounding.
- [Citations (Claude docs)](https://docs.anthropic.com/en/docs/build-with-claude/citations) — **Anthropic** — how the citation API chunks sources and attaches references.

## 📚 Books (free, with chapters)
- [Speech and Language Processing, 3rd ed. — **Ch. 14 (QA & RAG — answer grounding)**](https://web.stanford.edu/~jurafsky/slp3/14.pdf) — **Jurafsky & Martin** — the QA-grounding foundations behind attribution, free PDF.

## 🔗 In this platform
- Concept depth (the *why*): [AI-ML-intuition 8.02 Retrieval-Augmented Generation](../../../AI-ML-intuition/Module_8_LLMs_and_Agentic_Systems/8.02_Retrieval_Augmented_Generation.md)
- Prereqs: [01 RAG Fundamentals](../01-RAG-Fundamentals/01-RAG-Fundamentals.md) · [11 RAG Evaluation](../11-RAG-Evaluation/11-RAG-Evaluation.md) · Next: [14 Guardrails & Hallucination Mitigation](../14-Guardrails-and-Hallucination-Mitigation/14-Guardrails-and-Hallucination-Mitigation.md)
- Related domain: [09. LLMs — Hallucination & Alignment Basics](../../09.%20LLMs/20-Hallucination-and-Alignment-Basics/20-Hallucination-and-Alignment-Basics.md)
