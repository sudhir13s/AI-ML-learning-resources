---
id: "09-llms/decoding-and-sampling/references"
topic: "Decoding & Sampling — References"
parent: "09-llms/decoding-and-sampling"
type: references
updated: 2026-06-27
---

# Decoding & Sampling — references and further reading

> Companion link library for **[Decoding & Sampling](18-Decoding-and-Sampling.md)** (the concept page). This file holds the curated links — external sources *and* internal links to related pages on this platform — kept separate so it can be reused as a standalone reference list. Grouped by type, best-first. Every entry is from a primary author or a recognized deep explainer — chosen for depth on *this* topic (how to turn a next-token distribution into text), not popularity.

**Start here — suggested path**:
1. **Get the overview** — watch [Greedy? Min-p? Beam Search? How LLMs Actually Pick Words](https://www.youtube.com/watch?v=o-_SZ_itxeA) (**AI Coffee Break with Letitia**). *Greedy, beam, top-k/p, min-p in one clear survey — the cleanest map of the whole space.*
2. **Read with runnable code** — [How to generate text: decoding methods with Transformers](https://huggingface.co/blog/how-to-generate) (**Hugging Face**). *Every method with copy-pasteable examples — the canonical code-first explainer.*
3. **See the failure mode** — read [The Curious Case of Neural Text Degeneration](https://arxiv.org/abs/1904.09751) (**Holtzman et al. 2019**). *Why likelihood-maximizing decoders degenerate, and the nucleus (top-p) fix — the conceptual heart of the topic.*
4. **Compare the knobs side-by-side** — read [How do temperature, top-k, and top-p sampling differ?](https://magazine.sebastianraschka.com/p/llm-sampling) (**Sebastian Raschka**). *A crisp, correct contrast of the three core sampling controls.*
5. **Use it in practice** — [Generation strategies](https://huggingface.co/docs/transformers/en/generation_strategies) (**Hugging Face**). *The actual parameters and defaults you'll tune in real systems.*

**Videos**:
- [Greedy? Min-p? Beam Search? How LLMs Actually Pick Words](https://www.youtube.com/watch?v=o-_SZ_itxeA) — **AI Coffee Break with Letitia** — the clearest survey of decoding strategies, greedy through min-p.
- [Let's build GPT: from scratch, in code, spelled out](https://www.youtube.com/watch?v=kCc8FmEb1nY) — **Andrej Karpathy** — the generation loop where temperature and sampling are implemented line by line.
- [Beam Search — decoding strategy explained](https://www.youtube.com/watch?v=vCcXs5nxmbI) — **The AI Loop** — beam-search intuition and the width/length-penalty tradeoffs.
- [Nucleus Sampling: The Curious Case of Neural Text Degeneration](https://www.youtube.com/watch?v=dCORspO2yVY) — **TechViz** — the top-p paper walked through, with the degeneration plots.

**Interactive & visual**:
- [LLM Visualizer (3D)](https://bbycroft.net/llm) — **Brendan Bycroft** — walk a token through a small GPT's full forward pass and *see* the logits the decoder then samples from.
- [Generation strategies playground](https://huggingface.co/docs/transformers/en/generation_strategies) — **Hugging Face** — change `temperature`, `top_k`, `top_p` and watch the output shift.

**Courses (free)**:
- [Stanford CS336 — Language Modeling from Scratch (inference & decoding)](https://stanford-cs336.github.io/spring2025/) — **Stanford** — decoding within the full LLM inference stack.
- [Hugging Face — Text generation strategies](https://huggingface.co/docs/transformers/en/generation_strategies) — **Hugging Face** — greedy/beam/sampling with parameters and code, end to end.

**Articles / blogs (free, no paywall)**:
- [How to generate text: decoding methods with Transformers](https://huggingface.co/blog/how-to-generate) — **Hugging Face (Patrick von Platen)** — the canonical, code-first decoding explainer (greedy, beam, top-k, top-p side by side).
- [Understanding the Three Most Common LLM Sampling Strategies](https://magazine.sebastianraschka.com/p/llm-sampling) — **Sebastian Raschka** — a crisp, correct contrast of temperature, top-k, and top-p.
- [Controllable Neural Text Generation](https://lilianweng.github.io/posts/2021-01-02-controllable-text-generation/) — **Lilian Weng (OpenAI)** — decoding and steering generation, including degeneration and sampling.
- [Speeding up the GPT — KV cache](https://www.dipkumar.dev/becoming-the-unbeatable/posts/gpt-kvcache/) — **Dipkumar Patel** — the generation loop the decoder runs inside (where the distribution is produced each step).

**Key papers**:
- [A Contrastive Framework for Neural Text Generation (Contrastive Search)](https://arxiv.org/abs/2202.06417) — **Su, Lan, Wang, Yogatama, Kong & Collier (2022)** — picks tokens that are high-probability *and* representation-space-dissimilar to the context, attacking degeneration at its source.
- [Distilling the Knowledge in a Neural Network](https://arxiv.org/abs/1503.02531) — **Hinton, Vinyals & Dean (2015)** — origin of temperature-scaled softmax ($p_i = \text{softmax}(z_i/T)$), §2; the decoding temperature knob is the same transform.
- [Hierarchical Neural Story Generation (top-k sampling)](https://arxiv.org/abs/1805.04833) — **Fan, Lewis & Dauphin (2018)** — introduces and popularizes top-k truncated sampling for open-ended generation.
- [Language Models are Few-Shot Learners (GPT-3)](https://arxiv.org/abs/2005.14165) — **Brown et al. (2020)** — temperature/sampling settings used at scale; the practitioner reference for decoding in production.
- [Google's Neural Machine Translation System (GNMT)](https://arxiv.org/abs/1609.08144) — **Wu et al. (2016)** — §7 gives the length-normalized beam-search score $\frac{1}{L^\alpha}\sum_t \log p$ that corrects beam's bias toward short sequences.
- [Locally Typical Sampling](https://arxiv.org/abs/2202.00666) — **Meister, Pimentel, Wiher & Cotterell (2022)** — keep tokens whose information content is *typical* (near the distribution's entropy), an information-theoretic alternative to top-p.
- [The Curious Case of Neural Text Degeneration (Nucleus Sampling)](https://arxiv.org/abs/1904.09751) — **Holtzman, Buys, Du, Forbes & Choi (2019)** — introduces top-p (nucleus) sampling and the degeneration analysis; the single most important paper for this topic.

**Books (free, with chapters)**:
- [Dive into Deep Learning — Ch. 10 "Beam Search"](https://d2l.ai/chapter_recurrent-modern/beam-search.html) — **Zhang, Lipton, Li & Smola** — greedy as the $b=1$ special case, beam search, and length-normalized scoring with runnable code.
- [Speech and Language Processing, 3rd ed. — Ch. 10 "Large Language Models"](https://web.stanford.edu/~jurafsky/slp3/10.pdf) — **Jurafsky & Martin** — autoregressive generation, greedy vs sampling, temperature, and top-k/top-p decoding.

**In this platform**:
- Concept page (full explanation): [Decoding & Sampling](18-Decoding-and-Sampling.md)
- Related (NLP card, seq2seq framing): [Decoding Strategies](../../06.%20NLP/17-Decoding-Strategies/17-Decoding-Strategies.md)
- Foundations: [Loss Functions (softmax & cross-entropy)](../../05.%20Deep_Learning/concepts/04-Loss-Functions.md) · [Language Modeling Objectives](../01-Language-Modeling-Objectives/01-Language-Modeling-Objectives.md) · [Decoder-only Architecture](../04-Decoder-only-Architecture/04-Decoder-only-Architecture.md)
- Makes it fast (a *speed* technique, not a strategy): [Inference Optimization & Serving — speculative decoding](../09-Inference-Optimization-and-Serving/09-Inference-Optimization-and-Serving.md)
- Uses it: [Chain-of-Thought Reasoning](../17-Chain-of-Thought-Reasoning/17-Chain-of-Thought-Reasoning.md) · [LLM Evaluation & Benchmarks](../19-LLM-Evaluation-and-Benchmarks/19-LLM-Evaluation-and-Benchmarks.md)
```
