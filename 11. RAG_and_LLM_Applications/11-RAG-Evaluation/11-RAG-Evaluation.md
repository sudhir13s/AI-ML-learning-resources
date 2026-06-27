---
id: "11-rag-and-llm-apps/rag-evaluation"
topic: "RAG Evaluation (RAGAS · faithfulness · groundedness)"
parent: "11-rag-and-llm-apps"
level: intermediate
prereqs: ["rag-fundamentals", "embedding-models-for-retrieval"]
interview_frequency: high
updated: 2026-06-20
---

# RAG Evaluation — RAGAS · Faithfulness · Groundedness
> You can't improve what you can't measure. RAG has *two* things to evaluate: **retrieval** (did we
> fetch the right context? — context precision/recall) and **generation** (is the answer grounded in
> that context and on-topic? — faithfulness/groundedness + answer relevance). Frameworks like
> **RAGAS**, **TruLens** (the "RAG triad"), and **DeepEval** automate this, usually with an
> **LLM-as-judge**.

**Why it matters:** the "how do you know your RAG is actually good?" question that separates demos from
products. You'll define context precision/recall vs faithfulness vs answer relevance, explain
reference-free LLM-judge scoring (and its biases), how to build a golden eval set, and how to
attribute a bad answer to retrieval vs generation.

**⭐ Start here — suggested path:**

1. **Learn the two halves** — watch [RAG Evaluation: Precision, Recall, Faithfulness, RAGAS](https://www.youtube.com/watch?v=7_LTU0LA374). *Separate retrieval metrics from generation metrics.*
2. **Master the RAG triad** — read [TruLens: The RAG Triad](https://www.trulens.org/getting_started/core_concepts/rag_triad/). *Context relevance + groundedness + answer relevance = no-hallucination confidence.*
3. **Know the metrics precisely** — read [RAGAS: Available Metrics](https://docs.ragas.io/en/stable/concepts/metrics/available_metrics/). *Exactly what faithfulness, context precision, and context recall compute.*
4. **See it scored** — watch [RAG Evaluation Metrics Explained](https://www.youtube.com/watch?v=wOoYP55eYF0). *Worked examples of each metric on real query/context/answer triples.*
5. **Read the source** — skim [RAGAS paper](https://arxiv.org/abs/2309.15217). *Reference-free LLM-judged evaluation of RAG pipelines.*

## 🎓 Courses (free)
- [Building & Evaluating Advanced RAG](https://www.deeplearning.ai/short-courses/building-evaluating-advanced-rag/) — **DeepLearning.AI × TruLens** — the RAG triad and how to run it; the canonical free eval course.
- [RAGAS docs — Get Started: Evaluate a RAG system](https://docs.ragas.io/en/stable/getstarted/rag_eval/) — **Exploding Gradients** — hands-on, free walkthrough of evaluating a pipeline end to end.

## 🎥 Videos
- [RAG Evaluation: Precision, Recall, Faithfulness, RAGAS](https://www.youtube.com/watch?v=7_LTU0LA374) — **Logical Lenses** — clear split of retrieval vs generation metrics.
- [RAG Evaluation Metrics Explained](https://www.youtube.com/watch?v=wOoYP55eYF0) — **Shrijayan** — context precision/recall, relevancy, and faithfulness with examples.
- [What is RAGAS? Explained in 60 Seconds](https://www.youtube.com/watch?v=KNlD8hwmUdM) — **CodeCraft Academy** — a fast orientation to the RAGAS metric set.
- [Evaluate RAG with LLM Evals and Benchmarking](https://www.youtube.com/watch?v=LrMguHcbpO8) — **Arize AI** — LLM-as-judge evals and how to benchmark a RAG system.

## 📄 Key Papers
- [RAGAS: Automated Evaluation of Retrieval-Augmented Generation](https://arxiv.org/abs/2309.15217) — **Es et al. (2023)** — the reference-free, LLM-judged RAG eval framework.
- [ARES: An Automated Evaluation Framework for RAG](https://arxiv.org/abs/2311.09476) — **Saad-Falcon et al. (2023)** — trains lightweight judges for context relevance, faithfulness, answer relevance.
- [Evaluating Retrieval Quality in RAG (eRAG)](https://arxiv.org/abs/2404.13781) — **Salemi & Zamani (2024)** — how to evaluate the retriever by its downstream effect on generation.

## 📰 Articles / Blogs (free, no paywall)
- [The RAG Triad](https://www.trulens.org/getting_started/core_concepts/rag_triad/) — **TruLens** — context relevance, groundedness, answer relevance, clearly defined.
- [RAGAS: Available Metrics](https://docs.ragas.io/en/stable/concepts/metrics/available_metrics/) — **Exploding Gradients** — the authoritative metric definitions.
- [explodinggradients/ragas (GitHub)](https://github.com/explodinggradients/ragas) — **Exploding Gradients** — the open-source library to read and run.
- [LLM-as-a-Judge](https://huggingface.co/learn/cookbook/en/llm_judge) — **Hugging Face** — how (and how *not*) to use an LLM as the evaluator behind these metrics.

## 📚 Books (free, with chapters)
- [Speech and Language Processing, 3rd ed. — **Ch. 14 (QA, IR & RAG — evaluation)**](https://web.stanford.edu/~jurafsky/slp3/14.pdf) — **Jurafsky & Martin** — retrieval/answer evaluation foundations (precision/recall, EM/F1), free PDF.

## 🔗 In this platform
- Concept depth (the *why*): [AI-ML-intuition 8.02 Retrieval-Augmented Generation](../../../AI-ML-intuition/Module_8_LLMs_and_Agentic_Systems/8.02_Retrieval_Augmented_Generation.md)
- Prereqs: [01 RAG Fundamentals](../01-RAG-Fundamentals/01-RAG-Fundamentals.md) · Next: [12 Long-Context vs RAG](../12-Long-Context-vs-RAG/12-Long-Context-vs-RAG.md)
- Related domain: [09. LLMs — LLM Evaluation & Benchmarks](../../09.%20LLMs/19-LLM-Evaluation-and-Benchmarks/19-LLM-Evaluation-and-Benchmarks.md) · [06. NLP — NLP Evaluation Metrics](../../06.%20NLP/18-NLP-Evaluation-Metrics/18-NLP-Evaluation-Metrics.md)
