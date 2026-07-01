---
id: "11-rag-and-llm-apps/rag-evaluation/references"
topic: "RAG Evaluation (RAGAS · faithfulness · groundedness) — References"
parent: "11-rag-and-llm-apps/rag-evaluation"
type: references
updated: 2026-07-02
---

# RAG Evaluation — references and further reading

> Companion link library for **[RAG Evaluation](11-RAG-Evaluation.md)** (the concept page). External
> sources *and* internal cross-links, kept separate so it can be reused as a standalone list. Grouped
> by type, best-first. Every entry is free/open (no paywall) and chosen for depth on *this* topic —
> measuring RAG at both stages: retrieval (context precision/recall) and generation (the RAGAS/TruLens
> triad: faithfulness, answer relevance, context relevance), plus the LLM-as-judge that powers them.
> Every formula/mechanism cited on the concept page (RAGAS metrics, MT-Bench judge biases, ARES, the
> triad) appears here as a primary source.

**Start here — suggested path**:
1. **Learn the two halves** — watch [RAG Evaluation: Precision, Recall, Faithfulness, RAGAS](https://www.youtube.com/watch?v=7_LTU0LA374) (**Logical Lenses**). *Separate retrieval metrics from generation metrics.*
2. **Master the RAG triad** — read [TruLens: The RAG Triad](https://www.trulens.org/getting_started/core_concepts/rag_triad/). *Context relevance + faithfulness + answer relevance = no-hallucination confidence.*
3. **Know the metrics precisely** — read [RAGAS: Available Metrics](https://docs.ragas.io/en/stable/concepts/metrics/available_metrics/). *Exactly what faithfulness, context precision, and context recall compute.*
4. **See it scored** — watch [RAG Evaluation Metrics Explained](https://www.youtube.com/watch?v=wOoYP55eYF0) (**Shrijayan**). *Worked examples of each metric on real query/context/answer triples.*
5. **Read the source** — skim the [RAGAS paper](https://arxiv.org/abs/2309.15217) (**Es et al. 2023**). *Reference-free, LLM-judged evaluation of RAG pipelines — the framework this chapter builds toward.*

**Videos**:
- [RAG Evaluation: Precision, Recall, Faithfulness, RAGAS Explained Clearly](https://www.youtube.com/watch?v=7_LTU0LA374) — **Logical Lenses** — a clear split of retrieval vs generation metrics.
- [RAG Evaluation Metrics Explained](https://www.youtube.com/watch?v=wOoYP55eYF0) — **Shrijayan** — context precision/recall, relevancy, and faithfulness with worked examples.
- [What is RAGAS? Explained in 60 Seconds](https://www.youtube.com/watch?v=KNlD8hwmUdM) — **CodeCraft Academy** — a fast orientation to the RAGAS metric set.
- [RAG Time! Evaluate RAG with LLM Evals and Benchmarking](https://www.youtube.com/watch?v=LrMguHcbpO8) — **Arize AI** — LLM-as-judge evals and how to benchmark a RAG system.

**Interactive & visual**:
- [RAGAS — Get Started: Evaluate a RAG system](https://docs.ragas.io/en/stable/getstarted/rag_eval/) — **Exploding Gradients** — a runnable, free walkthrough of evaluating a pipeline end to end.
- [explodinggradients/ragas (GitHub)](https://github.com/explodinggradients/ragas) — **Exploding Gradients** — the open-source library to read and run; the metric implementations behind the docs.
- [The RAG Triad (TruLens)](https://www.trulens.org/getting_started/core_concepts/rag_triad/) — **TruLens** — the three feedback functions, defined and demoed on a live app.

**Courses (free)**:
- [Building & Evaluating Advanced RAG](https://www.deeplearning.ai/short-courses/building-evaluating-advanced-rag/) — **DeepLearning.AI × TruLens** — the RAG triad and how to run it; the canonical free eval course.
- [RAGAS docs — Get Started: Evaluate a RAG system](https://docs.ragas.io/en/stable/getstarted/rag_eval/) — **Exploding Gradients** — hands-on, free walkthrough of evaluating a pipeline end to end.

**Articles / blogs (free, no paywall)**:
- [The RAG Triad](https://www.trulens.org/getting_started/core_concepts/rag_triad/) — **TruLens** — context relevance, groundedness (faithfulness), answer relevance, clearly defined — the concept page's triad source.
- [RAGAS: Available Metrics](https://docs.ragas.io/en/stable/concepts/metrics/available_metrics/) — **Exploding Gradients** — the authoritative metric definitions the page's faithfulness / context-precision math is derived from.
- [explodinggradients/ragas (GitHub)](https://github.com/explodinggradients/ragas) — **Exploding Gradients** — the open-source library to read and run.
- [LLM-as-a-Judge](https://huggingface.co/learn/cookbook/en/llm_judge) — **Hugging Face** — how (and how *not*) to use an LLM as the evaluator behind these metrics; practical remedies for judge bias.

**Key papers**:
- [RAGAS: Automated Evaluation of Retrieval-Augmented Generation](https://arxiv.org/abs/2309.15217) — **Es, James, Espinosa-Anke & Schockaert (2023)** — the primary source: reference-free, LLM-judged faithfulness (supported-claims / total-claims), answer relevance (questions regenerated from the answer), and context relevance/precision — the metrics this chapter derives.
- [Judging LLM-as-a-Judge with MT-Bench and Chatbot Arena](https://arxiv.org/abs/2306.05685) — **Zheng et al. (2023)** — the LLM-as-judge biases (position, verbosity, self-enhancement) and the position-swap fix behind the pitfalls section; the reason to calibrate a judge.
- [ARES: An Automated Evaluation Framework for RAG](https://arxiv.org/abs/2311.09476) — **Saad-Falcon, Khattab, Potts & Zaharia (2023)** — trains lightweight judges for context relevance, faithfulness, and answer relevance — cheaper than a big-LLM judge per example.
- [Evaluating Retrieval Quality in RAG (eRAG)](https://arxiv.org/abs/2404.13781) — **Salemi & Zamani (2024)** — evaluate the retriever by its downstream effect on generation, connecting the retrieval and generation surfaces.
- [Retrieval-Augmented Generation for Knowledge-Intensive NLP Tasks](https://arxiv.org/abs/2005.11401) — **Lewis et al. (2020)** — the original RAG whose two stages (retrieve, generate) are the two surfaces this chapter evaluates.

**Books (free chapters)**:
- [Speech and Language Processing, 3rd ed. — Ch. 14 "Question Answering & Information Retrieval"](https://web.stanford.edu/~jurafsky/slp3/14.pdf) — **Jurafsky & Martin** — retrieval/answer evaluation foundations (precision/recall, EM/F1) the modern metrics build on, free PDF.

**In this platform**:
- Concept page (full explanation): [RAG Evaluation](11-RAG-Evaluation.md)
- Concept depth (the *why*): [AI-ML-intuition 8.02 Retrieval-Augmented Generation](../../../AI-ML-intuition/Module_8_LLMs_and_Agentic_Systems/8.02_Retrieval_Augmented_Generation.md)
- Metrics reused here: [06 Re-ranking (nDCG@k, MRR — the ranking metrics this chapter imports)](../06-Re-ranking-Cross-Encoders/06-Re-ranking-Cross-Encoders.md) · [08 Advanced RAG (the encoder-cosine support check reused as the faithfulness proxy)](../08-Advanced-RAG-Parent-Doc-Fusion-Self-RAG/08-Advanced-RAG-Parent-Doc-Fusion-Self-RAG.md)
- Foundations: [01 RAG Fundamentals](../01-RAG-Fundamentals/01-RAG-Fundamentals.md) · [03 Embedding Models](../03-Embedding-Models-for-Retrieval/03-Embedding-Models-for-Retrieval.md) · [05 Hybrid Search (the DenseRetriever this reuses)](../05-Hybrid-Search-BM25-and-Dense/05-Hybrid-Search-BM25-and-Dense.md)
- Next / related: [12 Long-Context vs RAG](../12-Long-Context-vs-RAG/12-Long-Context-vs-RAG.md) · [14 Guardrails & Hallucination Mitigation](../14-Guardrails-and-Hallucination-Mitigation/14-Guardrails-and-Hallucination-Mitigation.md)
- Related domain: [09. LLMs — LLM Evaluation & Benchmarks](../../09.%20LLMs/19-LLM-Evaluation-and-Benchmarks/19-LLM-Evaluation-and-Benchmarks.md) · [06. NLP — NLP Evaluation Metrics](../../06.%20NLP/18-NLP-Evaluation-Metrics/18-NLP-Evaluation-Metrics.md)
