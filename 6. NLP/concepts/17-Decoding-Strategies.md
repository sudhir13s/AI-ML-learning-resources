---
id: "06-nlp/decoding-strategies"
topic: "Decoding Strategies (greedy · beam · top-k · top-p/nucleus · temperature)"
parent: "06-nlp"
level: intermediate
prereqs: ["seq2seq-encoder-decoder", "softmax", "autoregressive-generation"]
interview_frequency: high
updated: 2026-06-19
---

# Decoding Strategies — greedy · beam · top-k · top-p/nucleus · temperature
> How an autoregressive model turns next-token probabilities into actual text. The choice —
> **greedy**, **beam search**, **top-k**, **top-p (nucleus)**, plus **temperature** — trades off
> quality, diversity, and the dreaded repetition/degeneration, and is independent of the model itself.

**Why it matters:** decoding is the high-frequency LLM interview topic because it's the knob you turn
in production without retraining. Be ready to explain **greedy vs beam** (and why beam search can be
*worse* for open-ended generation), how **temperature** rescales the softmax, the difference between
**top-k** (fixed count) and **top-p/nucleus** (dynamic mass), why pure sampling degenerates, and how
nucleus sampling fixes it.

**⭐ Start here — suggested path:**

1. **Build intuition** — read ⭐ [How to generate text](https://huggingface.co/blog/how-to-generate) (**Hugging Face / von Platen**). *The definitive, code-backed tour of every method.*
2. **See sampling knobs** — watch [Random Sampling (Temperature, top-p, top-k) for LLMs](https://www.youtube.com/watch?v=mti5XUm22Og) (**Minsuk Heo**). *How each parameter reshapes the distribution.*
3. **Read the source** — [The Curious Case of Neural Text Degeneration](https://arxiv.org/abs/1904.09751) (**Holtzman et al., 2020**). *Why beam/greedy degenerate and nucleus sampling fixes it.*
4. **Get the reference** — [SLP3 Ch. 7 §"Sampling for LLM Generation"](https://web.stanford.edu/~jurafsky/slp3/7.pdf). *Greedy, beam, temperature, top-k, top-p in the standard text.*
5. **Make it concrete** — code it with [HF Text generation strategies](https://huggingface.co/docs/transformers/generation_strategies). *Flip `num_beams`, `top_k`, `top_p`, `temperature` and compare.*

## 🎓 Courses (free)
- [Hugging Face LLM Course — Ch. 1: Transformer Models](https://huggingface.co/learn/llm-course/chapter1/1) — **Hugging Face** — where generation/decoding fits in the pipeline, code-first.
- [Stanford CS224N: NLP with Deep Learning](https://web.stanford.edu/class/cs224n/) — **Stanford (Manning)** — the natural-language-generation lecture covers decoding.

## 🎥 Videos
- [Random Sampling (Temperature, top-p, top-k) for LLMs](https://www.youtube.com/watch?v=mti5XUm22Og) — **Minsuk Heo** — clear, visual tour of the sampling parameters.
- [The Secret Controls for your LLM: Temperature, Top-K, Top-P](https://www.youtube.com/watch?v=MkaazQttbpc) — **Gary Explains** — practical effect of each knob on output.
- [Text generation algorithms (NLP video 14)](https://www.youtube.com/watch?v=3oEb_fFmPnY) — **Rachel Thomas (fast.ai)** — greedy/beam/sampling in a course context.
- [AI Language Models & Transformers](https://www.youtube.com/watch?v=rURRYI66E54) — **Computerphile (Rob Miles)** — why generation samples from a distribution at all.

## 📄 Key Papers
- [The Curious Case of Neural Text Degeneration (Nucleus Sampling)](https://arxiv.org/abs/1904.09751) — **Holtzman et al. (2020)** — introduces top-p; explains degeneration.
- [Hierarchical Neural Story Generation (top-k sampling)](https://arxiv.org/abs/1805.04833) — **Fan, Lewis & Dauphin (2018)** — popularizes top-k sampling.

## 📰 Articles / Blogs (free, no paywall)
- [How to generate text: different decoding methods](https://huggingface.co/blog/how-to-generate) — **Hugging Face** — the canonical free explainer with runnable code.
- [Text generation strategies (docs)](https://huggingface.co/docs/transformers/generation_strategies) — **Hugging Face** — every strategy and its parameters.

## 📚 Books (free, with chapters)
- [Speech and Language Processing, 3rd ed. — **Ch. 7 "Large Language Models" (sampling/decoding)**](https://web.stanford.edu/~jurafsky/slp3/7.pdf) — **Jurafsky & Martin** — greedy, beam, temperature, top-k, top-p.
- [Speech and Language Processing, 3rd ed. — **Ch. 12 "Machine Translation" (beam search)**](https://web.stanford.edu/~jurafsky/slp3/12.pdf) — **Jurafsky & Martin** — beam search in the decoding context.

## 🔗 In this platform
- Concept depth (the *why*): [AI-ML-intuition 5.05 Autoregressive Generation & Sampling](../../../AI-ML-intuition/Module_5_Generation/5.05_Autoregressive_Generation_Sampling.md) · [5.01 Entropy & KL (temperature)](../../../AI-ML-intuition/Module_5_Generation/5.01_Information_Theory_Entropy_KL_Divergence.md)
- Prior step: [08 Sequence-to-Sequence & Encoder–Decoder](08-Sequence-to-Sequence-and-Encoder-Decoder.md) — the models being decoded.
- Evaluation: [18 NLP Evaluation Metrics](18-NLP-Evaluation-Metrics.md) — how decoded text is scored.
