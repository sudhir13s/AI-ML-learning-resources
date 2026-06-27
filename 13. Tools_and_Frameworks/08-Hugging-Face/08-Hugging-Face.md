---
id: "13-tools-and-frameworks/hugging-face"
topic: "Hugging Face (Transformers · Datasets · Hub)"
parent: "13-tools-and-frameworks"
level: intermediate
prereqs: ["python", "pytorch", "transformers"]
interview_frequency: very-high
updated: 2026-06-20
---

# Hugging Face — Transformers · Datasets · Hub
> The center of gravity for modern NLP/LLM work: **Transformers** (load and run thousands of
> pretrained models with a uniform API), **Datasets** (efficient, streaming data loading),
> **Tokenizers**, and the **Hub** (the GitHub of models and datasets). The `pipeline()` one-liner
> takes you from "I want sentiment analysis" to a working model in three lines.

**Why it matters:** almost every applied LLM/NLP role expects Hugging Face fluency — `pipeline`,
`AutoModel`/`AutoTokenizer`, fine-tuning with `Trainer`, loading/streaming datasets, and pushing to
the Hub. It is the practical layer on top of the transformer architecture and the default toolkit
for RAG, fine-tuning, and inference.

**⭐ Start here — suggested path:**

1. **Run a model in 3 lines** — [Transformers quick tour](https://huggingface.co/docs/transformers/quicktour). *The `pipeline()` API gives an instant win and the right mental model.*
2. **See it on video** — [Getting Started With Hugging Face in 15 Minutes](https://www.youtube.com/watch?v=QEaBAZQCtwE) (AssemblyAI). *Pipelines, tokenizers, and models in one quick pass.*
3. **Take the official course** — start the free [LLM/NLP Course](https://huggingface.co/learn/llm-course/chapter1/1). *The authoritative path through Transformers, Datasets, Tokenizers, and the Hub.*
4. **Learn fine-tuning & Datasets** — work the course chapters on `Trainer` and the [Datasets docs](https://huggingface.co/docs/datasets/index). *Fine-tuning and efficient data loading are the most common real tasks.*
5. **Publish to the Hub** — follow the [Hub docs](https://huggingface.co/docs/hub/index) to push a model/dataset. *Understanding the Hub (repos, model cards, Spaces) is part of the workflow.*

## 🎓 Courses (free)
- [Hugging Face — learn](https://huggingface.co/learn) — **Hugging Face** — the hub of free courses (LLM/NLP, Deep RL, Diffusion, Agents).
- [LLM / NLP Course](https://huggingface.co/learn/llm-course/chapter1/1) — **Hugging Face** — the definitive, free, ecosystem-wide course.

## 🎥 Videos
- [Getting Started With Hugging Face in 15 Minutes](https://www.youtube.com/watch?v=QEaBAZQCtwE) — **AssemblyAI** — pipelines, tokenizers, models, with PyTorch & TF.
- [Welcome to the Hugging Face course](https://www.youtube.com/watch?v=00GKzGyWFEs) — **Hugging Face** — the official course intro from the team.
- [Basics of Hugging Face — Tutorial for Beginners](https://www.youtube.com/watch?v=J3tMzGigqww) — **Skilled Engg** — a hands-on first walkthrough of the platform.
- [PyTorch for Deep Learning — Full Course](https://www.youtube.com/watch?v=V_xro1bcAuA) — **freeCodeCamp** — the PyTorch foundation most HF models run on.

## 📄 Key Papers
- [HuggingFace's Transformers: State-of-the-art Natural Language Processing](https://arxiv.org/abs/1910.03771) — **Wolf et al. (2020), EMNLP** — the foundational paper on the Transformers library.
- [Transformers documentation](https://huggingface.co/docs/transformers/index) — **Hugging Face** — the canonical API reference.

## 📰 Articles / Blogs (free, no paywall)
- [Transformers quick tour](https://huggingface.co/docs/transformers/quicktour) — **Hugging Face** — `pipeline`, `AutoModel`, `AutoTokenizer` in one page.
- [Datasets documentation](https://huggingface.co/docs/datasets/index) — **Hugging Face** — efficient loading, streaming, and processing.
- [Hugging Face Hub documentation](https://huggingface.co/docs/hub/index) — **Hugging Face** — repos, model cards, and Spaces.

## 📚 Books (free, with chapters)
- [Hugging Face LLM/NLP Course (full curriculum)](https://huggingface.co/learn/llm-course/chapter1/1) — **Hugging Face** — a free, book-length, chapter-structured guide.
- [Hugging Face documentation hub](https://huggingface.co/docs) — **Hugging Face** — the complete free reference across all libraries.

## 🔗 In this platform
- Related domain: [09. LLMs](../../09.%20LLMs/README.md) · [06. NLP](../../06.%20NLP/README.md) · [10. GenAI](../../10.%20GenAI/README.md)
- Pairs with: [05 PyTorch](../05-PyTorch/05-PyTorch.md) · [13 Gradio & Streamlit](../13-Gradio-and-Streamlit/13-Gradio-and-Streamlit.md)
- Deeper concept (the *why*): transformers & LLM internals → [Deep Learning](../../05.%20Deep_Learning/README.md); RAG pipelines → [RAG & LLM Applications](../../11.%20RAG_and_LLM_Applications/README.md)
