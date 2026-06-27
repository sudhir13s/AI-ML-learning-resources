---
id: "06-nlp/question-answering/references"
topic: "Question Answering — References"
parent: "06-nlp/question-answering"
type: references
updated: 2026-06-27
---

# Question Answering — references and further reading

> Companion link library for **[Question Answering](11-Question-Answering.md)** (the concept page). This file holds the curated links — external sources *and* internal links to related pages on this platform — kept separate so it can be reused as a standalone reference list. Grouped by type, best-first. Every entry is from a primary author or a recognized deep explainer — chosen for depth on *this* topic, not popularity.

**Start here — suggested path**:
1. **Frame the task** — read [SLP3 Ch. 11: Information Retrieval and RAG](https://web.stanford.edu/~jurafsky/slp3/11.pdf) (**Jurafsky & Martin**). *Extractive vs open-domain QA before any model.*
2. **See extractive QA** — watch [Applying BERT to Question Answering (SQuAD v1.1)](https://www.youtube.com/watch?v=l8ZYCvgGu0o) (**Chris McCormick**), then read ⭐ [Question Answering with a Fine-Tuned BERT](https://mccormickml.com/2020/03/10/question-answering-with-a-fine-tuned-BERT/). *Exactly how start/end span prediction works.*
3. **Get open-domain QA** — read [How to Build an Open-Domain QA System](https://lilianweng.github.io/posts/2020-10-29-odqa/) (**Lilian Weng**). *The retriever–reader pipeline, rigorously.*
4. **Read the sources** — [SQuAD](https://arxiv.org/abs/1606.05250) → [BERT](https://arxiv.org/abs/1810.04805) → [DPR](https://arxiv.org/abs/2004.04906) → [RAG](https://arxiv.org/abs/2005.11401). *Benchmark, model, dense retriever, retrieval-augmented generation.*
5. **Make it concrete** — code it with the [HF Question Answering chapter](https://huggingface.co/learn/llm-course/chapter7/7). *Fine-tune a span-QA model on SQuAD.*

**Videos**:
- [Applying BERT to Question Answering (SQuAD v1.1)](https://www.youtube.com/watch?v=l8ZYCvgGu0o) — **Chris McCormick** — start/end span prediction explained in detail, the clearest walkthrough of the span head.
- [Stanford CS224N: Question Answering](https://www.youtube.com/watch?v=yIdF-17HwSk) — **Stanford (Chris Manning / Danqi Chen)** — reading comprehension and open-domain QA from the people who built DrQA/SQuAD.
- [Stanford CS224N: Natural Language Generation](https://www.youtube.com/watch?v=1uMo8olr5ng) — **Stanford** — why EM/F1 break on generated answers and how generation is evaluated; the evaluation half of QA.
- [Text Extraction From a Corpus Using BERT (QA)](https://www.youtube.com/watch?v=XaQ0CBlQ4cY) — **Abhishek Thakur** — implement extractive QA hands-on, end to end.
- [RAG: Retrieval-Augmented Generation, explained](https://www.youtube.com/watch?v=T-D1OfcDW1M) — **IBM Technology** — the retrieve-then-generate idea and why it grounds answers, concisely.
- [Building Production-Ready RAG Applications](https://www.youtube.com/watch?v=TRjq7t2Ms5I) — **Jerry Liu (LlamaIndex)** — open-domain QA as it is actually built today: retrieval quality, chunking, and the failure modes.

**Courses (free)**:
- [Hugging Face LLM Course — Ch. 7: Question Answering](https://huggingface.co/learn/llm-course/chapter7/7) — **Hugging Face** — fine-tune extractive QA on SQuAD, code-first, with the start/end head.
- [Stanford CS224N: NLP with Deep Learning](https://web.stanford.edu/class/cs224n/) — **Stanford (Manning)** — the QA + reading-comprehension lecture, with slides and notes.

**Articles / blogs (free, no paywall)**:
- [Question Answering with a Fine-Tuned BERT](https://mccormickml.com/2020/03/10/question-answering-with-a-fine-tuned-BERT/) — **Chris McCormick** — span prediction step by step (blog + Colab); the best single explainer of the span head.
- [How to Build an Open-Domain Question Answering System](https://lilianweng.github.io/posts/2020-10-29-odqa/) — **Lilian Weng (OpenAI)** — the definitive free survey of open-domain QA: retriever, reader, closed-book, generative.
- [Retrieval Augmented Generation (RAG)](https://huggingface.co/docs/transformers/model_doc/rag) — **Hugging Face** — the original RAG model, end to end, with code.
- [SQuAD Explorer + leaderboard](https://rajpurkar.github.io/SQuAD-explorer/) — **Stanford** — browse the benchmark, the official EM/F1 scorer behavior, and SQuAD 2.0 unanswerables.

**Key papers**:
- [SQuAD: 100,000+ Questions for Machine Comprehension of Text](https://arxiv.org/abs/1606.05250) — **Rajpurkar et al. (2016)** — the benchmark that defined extractive QA and the span-prediction setup.
- [Know What You Don't Know: Unanswerable Questions for SQuAD (2.0)](https://arxiv.org/abs/1806.03822) — **Rajpurkar et al. (2018)** — unanswerable questions; the model must abstain.
- [BERT: Pre-training of Deep Bidirectional Transformers](https://arxiv.org/abs/1810.04805) — **Devlin et al. (2018)** — the encoder + start/end head that solved span QA.
- [Reading Wikipedia to Answer Open-Domain Questions (DrQA)](https://arxiv.org/abs/1704.00051) — **Chen et al. (2017)** — the retrieve-then-read template (TF-IDF retriever + neural reader).
- [Dense Passage Retrieval for Open-Domain QA (DPR)](https://arxiv.org/abs/2004.04906) — **Karpukhin et al. (2020)** — the dense bi-encoder retriever that supplanted lexical retrieval.
- [Retrieval-Augmented Generation for Knowledge-Intensive NLP (RAG)](https://arxiv.org/abs/2005.11401) — **Lewis et al. (2020)** — retrieve passages, condition a generator; the dominant modern QA pattern.
- [HotpotQA: A Dataset for Diverse, Explainable Multi-hop QA](https://arxiv.org/abs/1809.09600) — **Yang et al. (2018)** — multi-hop reasoning over multiple documents with supporting-fact supervision.
- [CoQA: A Conversational Question Answering Challenge](https://arxiv.org/abs/1808.07042) — **Reddy et al. (2019)** — follow-up questions, coreference, and dialogue context.
- [How Much Knowledge Can You Pack Into the Parameters of a Language Model?](https://arxiv.org/abs/2002.08910) — **Roberts et al. (2020)** — closed-book QA with T5; knowledge in the weights.
- [Lost in the Middle: How Language Models Use Long Contexts](https://arxiv.org/abs/2307.03172) — **Liu et al. (2023)** — the U-shaped accuracy curve that haunts long-context "answer from the doc" QA.

**Books (free chapters)**:
- [Speech and Language Processing, 3rd ed. — Ch. 11 "Information Retrieval and Retrieval-Augmented Generation"](https://web.stanford.edu/~jurafsky/slp3/11.pdf) — **Jurafsky & Martin** — IR + QA + RAG in the standard text, free PDF.
- [Speech and Language Processing, 3rd ed. — Ch. 14 "Question Answering, Information Retrieval, and RAG" (older draft)](https://web.stanford.edu/~jurafsky/slp3/14.pdf) — **Jurafsky & Martin** — extended QA coverage, reading comprehension and open-domain.

**In this platform**:
- Concept page (full explanation): [Question Answering](11-Question-Answering.md)
- Runnable code (the seeded source of truth): [teaching notebook](code/11-Question-Answering.ipynb) · [demo script](code/question_answering.py) · [figure generator](code/make_figures_11.py)
- The encoder QA fine-tunes on: [Contextual Embeddings (ELMo/BERT)](../06-Contextual-Embeddings-ELMo-BERT/06-Contextual-Embeddings-ELMo-BERT.md)
- The retriever half of open-domain QA: [Information Retrieval & Semantic Search](../16-Information-Retrieval-and-Semantic-Search/16-Information-Retrieval-and-Semantic-Search.md) · embeddings behind dense retrieval: [Sentence & Document Embeddings](../07-Sentence-and-Document-Embeddings/07-Sentence-and-Document-Embeddings.md)
- The generator backbone (RAG reader): [Seq2Seq & Encoder–Decoder](../08-Sequence-to-Sequence-and-Encoder-Decoder/08-Sequence-to-Sequence-and-Encoder-Decoder.md)
- How answers are scored (EM/F1): [NLP Evaluation Metrics](../18-NLP-Evaluation-Metrics/18-NLP-Evaluation-Metrics.md)
- Resolving conversational follow-ups: [Coreference Resolution](../14-Coreference-Resolution/14-Coreference-Resolution.md)
- The *why* behind RAG, in depth: [AI-ML-intuition 8.02 Retrieval-Augmented Generation](../../../AI-ML-intuition/Module_8_LLMs_and_Agentic_Systems/8.02_Retrieval_Augmented_Generation.md)
