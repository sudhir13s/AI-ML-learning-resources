---
id: "17-frontier/model-hubs-and-demos"
topic: "Open Model Hubs & Demos (Hugging Face / Spaces)"
parent: "17-frontier"
level: beginner
prereqs: []
interview_frequency: low
updated: 2026-06-20
---

# Open Model Hubs & Demos (Hugging Face / Spaces)
> Where open models, datasets, and runnable demos live — and how to use them to *try* a new method in
> minutes instead of reading about it for hours. The Hugging Face Hub (models + datasets + Spaces),
> plus local runners like Ollama, are how you go from "I heard about this model" to "I ran it."

**Why it matters:** the fastest way to understand a new model is to use it. Knowing how to pull a
model from the Hub, run a Space demo, or load it locally turns the frontier from headlines into
hands-on experience — and "have you actually used X?" is a question that separates readers from
practitioners in interviews.

**⭐ Start here — suggested path:**

1. **Tour the Hub** — browse ⭐ [Hugging Face Models](https://huggingface.co/models) and [Datasets](https://huggingface.co/datasets); sort by trending/downloads. *The Hub is the field's app store for open models; trending tells you what people actually use.*
2. **Try a demo with zero setup** — open a few [Spaces](https://huggingface.co/spaces) and interact with live models in the browser. *Spaces let you experience a model's behavior before writing a line of code.*
3. **Run a model in code** — follow the [Transformers quickstart](https://huggingface.co/docs/transformers/index) to load a model in a few lines. *Loading a pipeline is the "hello world" of using the frontier hands-on.*
4. **Run one locally** — use [Ollama](https://ollama.com/) (or [GGUF via the Hub](https://huggingface.co/docs/hub/ollama)) to run an open model on your machine. *Local inference removes API/cost friction and deepens your intuition for size/latency trade-offs.*
5. **Read model cards critically** — check the model card for training data, license, eval, and limitations before trusting it. *The model card is where capability claims meet caveats — a direct tie-in to "hype vs substance."*

## 🎓 Courses (free)
- [Hugging Face — LLM / NLP Course](https://huggingface.co/learn/nlp-course/chapter1/1) — **Hugging Face** — the official free course on using the Hub, Transformers, and Spaces.
- [Hugging Face Hub docs](https://huggingface.co/docs/hub/index) — **Hugging Face** — the reference for models, datasets, Spaces, and model cards.

## 🎥 Videos
- [What is Hugging Face? — Models, Datasets & Spaces](https://www.youtube.com/watch?v=qP9mbY3wuWk) — **Hugging Face** — the official orientation to the Hub's three pillars.
- [How to Create a Hugging Face Space: A Beginner's Guide](https://www.youtube.com/watch?v=xqdTFyRdtjQ) — **Marqo** — build and share a runnable model demo.
- [PyTorch Paper Replicating (Vision Transformer)](https://www.youtube.com/watch?v=tjpW_BY8y3g) — **Daniel Bourke** — once you can pull models, replicating one cements understanding.
- [PyTorch Transformers from Scratch](https://www.youtube.com/watch?v=U0s0f995w14) — **Aladdin Persson** — the architecture behind most Hub models, implemented from the paper.

## 📄 Key Papers
- [Transformers: State-of-the-Art Natural Language Processing](https://arxiv.org/abs/1910.03771) — **Wolf et al. (2020)** — the paper behind the 🤗 Transformers library that the Hub is built around.
- [Datasets: A Community Library for NLP](https://arxiv.org/abs/2109.02846) — **Lhoest et al. (2021)** — the design of the Hub's datasets library.
- [Model Cards for Model Reporting](https://arxiv.org/abs/1810.03993) — **Mitchell et al. (2019)** — why every Hub model ships a card, and what to read in it.

## 📰 Articles / Blogs (free, no paywall)
- [Hugging Face Spaces overview](https://huggingface.co/docs/hub/en/spaces-overview) — **Hugging Face** — how browser-based model demos work (Gradio/Streamlit/Docker).
- [Use Ollama with any GGUF model on the Hub](https://huggingface.co/docs/hub/ollama) — **Hugging Face** — run open Hub models locally in one command.
- [Hugging Face blog](https://huggingface.co/blog) — **Hugging Face** — release notes and walkthroughs for new open models as they land.
- [Hugging Face Daily Papers](https://huggingface.co/papers) — **Hugging Face** — many top papers ship with a model/demo you can try the same day.

## 📚 Books (free, with chapters)
- [Natural Language Processing with Transformers — free notebooks](https://github.com/nlp-with-transformers/notebooks) — **Tunstall, von Werra & Wolf** — the Hugging Face team's book companion code (runnable on free Colab) for using the Hub and Transformers.
- [Dive into Deep Learning](https://d2l.ai/) — **Zhang et al.** — runnable code that pairs with models you pull from the Hub.

## 🔗 In this platform
- Per-concept index: [Frontier & Staying Current — concepts](../README.md)
- Pair with: [02 arXiv & Papers with Code](../02-arXiv-and-Papers-with-Code/02-arXiv-and-Papers-with-Code.md) · [07 Reproducing Papers (code)](../07-Reproducing-Papers-Code/07-Reproducing-Papers-Code.md) · [11 Evaluating Hype vs Substance](../11-Evaluating-Hype-vs-Substance/11-Evaluating-Hype-vs-Substance.md)
- Build on these models: [09. LLMs](../../09.%20LLMs/README.md) · [10. GenAI](../../10.%20GenAI/README.md) · [16. RAG & LLM Applications](../../11.%20RAG_and_LLM_Applications/README.md)
