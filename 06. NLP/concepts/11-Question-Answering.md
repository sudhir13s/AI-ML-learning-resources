---
id: "06-nlp/question-answering"
topic: "Question Answering (extractive & generative)"
parent: "06-nlp"
level: intermediate
prereqs: ["contextual-embeddings", "seq2seq-encoder-decoder"]
interview_frequency: high
updated: 2026-06-19
---

# Question Answering — extractive & generative
> Answering a question from text. **Extractive QA** predicts a span in a passage (BERT + start/end
> pointers on SQuAD); **generative/open-domain QA** retrieves passages then *generates* an answer
> (the retriever-reader / RAG pattern).

**Why it matters:** QA is the bridge from representations to useful applications and a favorite
interview design question. Be ready to explain the **SQuAD** span-prediction setup (start/end logits,
`[SEP]` segments), **SQuAD 2.0** unanswerable questions, **extractive vs generative** trade-offs,
the **retriever–reader** pipeline, and how modern **RAG** grounds LLM answers in retrieved evidence.

**⭐ Start here — suggested path:**

1. **Build intuition** — read [SLP3 Ch. 11](https://web.stanford.edu/~jurafsky/slp3/11.pdf) on IR + QA. *Frame extractive vs open-domain before any model.*
2. **See extractive QA** — watch [Applying BERT to Question Answering (SQuAD v1.1)](https://www.youtube.com/watch?v=l8ZYCvgGu0o) (**InnerWorkingsAI**), then read ⭐ [QA with a Fine-Tuned BERT](https://mccormickml.com/2020/03/10/question-answering-with-a-fine-tuned-BERT/) (**Chris McCormick**). *Exactly how start/end span prediction works.*
3. **Get open-domain QA** — read [How to Build an Open-Domain QA System](https://lilianweng.github.io/posts/2020-10-29-odqa/) (**Lilian Weng**). *The retriever–reader pipeline, rigorously.*
4. **Read the sources** — [SQuAD](https://arxiv.org/abs/1606.05250) → [BERT](https://arxiv.org/abs/1810.04805). *The benchmark and the model that solved span QA.*
5. **Make it concrete** — code it with the [HF Question Answering guide](https://huggingface.co/learn/llm-course/chapter7/7). *Fine-tune a span-QA model on SQuAD.*

## 🎓 Courses (free)
- [Hugging Face LLM Course — Ch. 7.7: Question Answering](https://huggingface.co/learn/llm-course/chapter7/7) — **Hugging Face** — fine-tune extractive QA on SQuAD, code-first.
- [Stanford CS224N: NLP with Deep Learning](https://web.stanford.edu/class/cs224n/) — **Stanford (Manning)** — the QA + reading-comprehension lecture.

## 🎥 Videos
- [Applying BERT to Question Answering (SQuAD v1.1)](https://www.youtube.com/watch?v=l8ZYCvgGu0o) — **InnerWorkingsAI** — start/end span prediction explained in detail.
- [Text Extraction From a Corpus Using BERT (QA)](https://www.youtube.com/watch?v=XaQ0CBlQ4cY) — **Abhishek Thakur** — implement extractive QA hands-on.
- [Question Answering with HuggingFace Transformers](https://www.youtube.com/watch?v=J76T73cpu8Q) — **Bhavesh Bhatt** — QA in a few lines with the pipeline API.
- [BERT Neural Network — EXPLAINED!](https://www.youtube.com/watch?v=xI0HHN5XKDo) — **CodeEmporium** — the encoder QA fine-tunes on top of.

## 📄 Key Papers
- [SQuAD: 100,000+ Questions for Machine Comprehension of Text](https://arxiv.org/abs/1606.05250) — **Rajpurkar et al. (2016)** — the benchmark that defined extractive QA.
- [BERT: Pre-training of Deep Bidirectional Transformers](https://arxiv.org/abs/1810.04805) — **Devlin et al. (2018)** — the model that solved span QA.
- [Reading Wikipedia to Answer Open-Domain Questions (DrQA)](https://arxiv.org/abs/1704.00051) — **Chen et al. (2017)** — the retriever–reader pipeline.

## 📰 Articles / Blogs (free, no paywall)
- [Question Answering with a Fine-Tuned BERT](https://mccormickml.com/2020/03/10/question-answering-with-a-fine-tuned-BERT/) — **Chris McCormick** — span prediction, step by step (blog + Colab).
- [How to Build an Open-Domain Question Answering System](https://lilianweng.github.io/posts/2020-10-29-odqa/) — **Lilian Weng** — the definitive free survey of open-domain QA.
- [SQuAD Explorer](https://rajpurkar.github.io/SQuAD-explorer/) — **Stanford** — browse the benchmark and leaderboard.

## 📚 Books (free, with chapters)
- [Speech and Language Processing, 3rd ed. — **Ch. 11 "Information Retrieval and Retrieval-Augmented Generation"**](https://web.stanford.edu/~jurafsky/slp3/11.pdf) — **Jurafsky & Martin** — IR + QA + RAG in the standard text.
- [Speech and Language Processing, 3rd ed. — **Ch. 14 "Question Answering, Information Retrieval, and RAG" (older draft)**](https://web.stanford.edu/~jurafsky/slp3/14.pdf) — **Jurafsky & Martin** — extended QA coverage.

## 🔗 In this platform
- Prior step: [06 Contextual Embeddings (BERT)](06-Contextual-Embeddings-ELMo-BERT.md) — the encoder QA fine-tunes.
- Concept depth (the *why*): [AI-ML-intuition 8.02 Retrieval-Augmented Generation](../../../AI-ML-intuition/Module_8_LLMs_and_Agentic_Systems/8.02_Retrieval_Augmented_Generation.md)
- Related: [16 Information Retrieval & Semantic Search](16-Information-Retrieval-and-Semantic-Search.md) — the retriever half of open-domain QA.
