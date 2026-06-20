---
id: "09-llms/decoding-and-sampling"
topic: "Decoding & Sampling for LLMs (temperature · top-k · top-p)"
parent: "09-llms"
level: intermediate
prereqs: ["language-modeling-objectives", "softmax", "decoder-only-architecture"]
interview_frequency: very-high
updated: 2026-06-20
---

# Decoding & Sampling for LLMs
> How the model turns next-token probabilities into actual text. **Greedy** and **beam search**
> maximize likelihood (great for translation, bland/repetitive for open-ended text); **sampling** with
> **temperature**, **top-k**, and **top-p (nucleus)** trades likelihood for diversity. Plus repetition
> penalties, min-p, and speculative decoding. The knobs that control creativity vs reliability.

**Why it matters:** asked in both ML and applied interviews. Be ready to explain how temperature
reshapes the softmax, the difference between top-k (fixed count) and top-p (dynamic mass), why pure
likelihood maximization degenerates ("neural text degeneration"), and when beam search is/ isn't right.

**⭐ Start here — suggested path:**

1. **Get the overview** — watch [How LLMs Actually Pick Words — Decoding Strategies Explained](https://www.youtube.com/watch?v=o-_SZ_itxeA). *Greedy, beam, top-k/p, min-p in one clear video.*
2. **Read with code** — [HF: How to generate text](https://huggingface.co/blog/how-to-generate). *Every decoding method with runnable examples.*
3. **See the failure mode** — [Nucleus Sampling: Neural Text Degeneration walkthrough](https://www.youtube.com/watch?v=dCORspO2yVY). *Why max-likelihood decoding degenerates and top-p fixes it.*
4. **Read the source** — [The Curious Case of Neural Text Degeneration](https://arxiv.org/abs/1904.09751). *The paper that introduced nucleus (top-p) sampling.*
5. **Use it in practice** — [HF generation strategies docs](https://huggingface.co/docs/transformers/en/generation_strategies). *The parameters and defaults you'll actually tune.*

## 🎓 Courses (free)
- [Hugging Face — Text generation strategies](https://huggingface.co/docs/transformers/en/generation_strategies) — **Hugging Face** — greedy/beam/sampling, with parameters and code.
- [Stanford CS336 — Decoding & inference](https://stanford-cs336.github.io/spring2025/) — **Stanford** — decoding within the inference stack.

## 🎥 Videos
- [Greedy? Min-p? Beam Search? How LLMs Actually Pick Words](https://www.youtube.com/watch?v=o-_SZ_itxeA) — **AI Coffee Break (Letitia)** — the clearest survey of decoding strategies.
- [What is beam search? Decoding strategy explained](https://www.youtube.com/watch?v=vCcXs5nxmbI) — **The AI Loop** — beam search intuition.
- [Nucleus Sampling: The Curious Case of Neural Text Degeneration](https://www.youtube.com/watch?v=dCORspO2yVY) — **TechViz** — the top-p paper, walked through.
- [Let's reproduce GPT-2 (124M)](https://www.youtube.com/watch?v=l8pRSuU81PU) — **Andrej Karpathy** — sampling implemented in the generation loop.

## 📄 Key Papers
- [The Curious Case of Neural Text Degeneration (Nucleus Sampling)](https://arxiv.org/abs/1904.09751) — **Holtzman et al. (2019)** — introduces top-p; explains why likelihood-max decoding degenerates.
- [Hierarchical Neural Story Generation (top-k sampling)](https://arxiv.org/abs/1805.04833) — **Fan et al. (2018)** — popularized top-k sampling for open-ended generation.
- [Language Models are Few-Shot Learners (GPT-3)](https://arxiv.org/abs/2005.14165) — **Brown et al. (2020)** — temperature/sampling settings used at scale.

## 📰 Articles / Blogs (free, no paywall)
- [How to generate text: decoding methods with Transformers](https://huggingface.co/blog/how-to-generate) — **Hugging Face** — the canonical, code-first decoding explainer.
- [How do temperature, top-k, and top-p sampling differ?](https://sebastianraschka.com/faq/docs/temperature-topk-topp-sampling.html) — **Sebastian Raschka** — a crisp side-by-side. *(If blocked, see the HF article above.)*
- [Controllable Neural Text Generation](https://lilianweng.github.io/posts/2021-01-02-controllable-text-generation/) — **Lilian Weng** — decoding and steering generation.

## 📚 Books (free, with chapters)
- [Speech and Language Processing, 3rd ed. — **Ch. 10 "Large Language Models"**](https://web.stanford.edu/~jurafsky/slp3/10.pdf) — **Jurafsky & Martin** — sampling, temperature, top-k/p decoding.
- [Dive into Deep Learning — **Ch. 10 "Beam Search"**](https://d2l.ai/chapter_natural-language-processing-pretraining/index.html) — **Zhang et al.** — greedy vs beam search with code.

## 🔗 In this platform
- Concept depth (the *why*): [Module 5.05 Autoregressive Generation & Sampling](../../../AI-ML-intuition/Module_5_Generation/5.05_Autoregressive_Generation_Sampling.md)
- Related (covered elsewhere): [Decoding Strategies (NLP card)](../../06.%20NLP/concepts/17-Decoding-Strategies.md)
- Related concepts: [Language Modeling Objectives](01-Language-Modeling-Objectives.md) · [Chain-of-Thought Reasoning](17-Chain-of-Thought-Reasoning.md) · [Inference Optimization & Serving](09-Inference-Optimization-and-Serving.md)
