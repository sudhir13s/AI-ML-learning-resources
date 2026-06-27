---
id: "06-nlp/decoding-strategies/references"
topic: "Decoding Strategies — References"
parent: "06-nlp/decoding-strategies"
type: references
updated: 2026-06-27
---

# Decoding Strategies — references and further reading

> Companion link library for **[Decoding Strategies](17-Decoding-Strategies.md)** (the concept page). This file holds the curated links — external sources *and* internal links to related pages on this platform — kept separate so it can be reused as a standalone reference list. Grouped by type, best-first. Every entry is from a primary author or a recognized deep explainer — chosen for depth on *this* topic, not popularity.

**Start here — suggested path**:
1. **Build intuition** — read [How to generate text: different decoding methods](https://huggingface.co/blog/how-to-generate) (**Patrick von Platen, Hugging Face**). *The definitive, code-backed tour of greedy, beam, top-k, and top-p — run it as you read.*
2. **See the sampling knobs move** — watch [Random Sampling (Temperature, top-p, top-k) for LLMs](https://www.youtube.com/watch?v=mti5XUm22Og) (**Minsuk Heo**). *Each parameter visibly reshaping the distribution.*
3. **Read the source result** — [The Curious Case of Neural Text Degeneration](https://arxiv.org/abs/1904.09751) (**Holtzman et al., 2020**). *Why greedy/beam degenerate and nucleus sampling fixes it — the paper that introduced top-p.*
4. **Get the textbook reference** — [SLP3 — Large Language Models (sampling & decoding)](https://web.stanford.edu/~jurafsky/slp3/10.pdf) (**Jurafsky & Martin**). *Greedy, beam, temperature, top-k, top-p in the standard text.*
5. **Make it concrete** — code it with [HF Text generation strategies](https://huggingface.co/docs/transformers/generation_strategies) (**Hugging Face**). *Flip `num_beams`, `top_k`, `top_p`, `temperature`, `repetition_penalty` and compare outputs.*

**Videos**:
- [Random Sampling (Temperature, top-p, top-k) for LLMs](https://www.youtube.com/watch?v=mti5XUm22Og) — **Minsuk Heo** — clear, visual tour of the sampling parameters and how each reshapes the distribution.
- [The Secret Controls for your LLM: Temperature, Top-K, Top-P](https://www.youtube.com/watch?v=MkaazQttbpc) — **Gary Explains** — practical effect of each knob on real output.
- [AI Language Models & Transformers](https://www.youtube.com/watch?v=rURRYI66E54) — **Computerphile (Rob Miles)** — why generation samples from a distribution at all; the conceptual ground floor.
- [Speculative Decoding: When Two LLMs are Faster than One](https://www.youtube.com/watch?v=S-8yr_RibJ4) — **Efficient NLP** — the draft-and-verify speedup explained clearly with the rejection-sampling intuition.

**Courses (free)**:
- [Stanford CS224N — Natural Language Generation](https://web.stanford.edu/class/cs224n/) — **Stanford (Manning)** — the NLG lecture covers greedy, beam, sampling, and degeneration in a course context.
- [Hugging Face LLM Course — Transformer Models](https://huggingface.co/learn/llm-course/chapter1/1) — **Hugging Face** — where generation/decoding fits in the pipeline, code-first.

**Articles / blogs (free, no paywall)**:
- [How to generate text: different decoding methods](https://huggingface.co/blog/how-to-generate) — **Patrick von Platen (Hugging Face)** — the canonical free explainer with runnable code for every method.
- [Text generation strategies (docs)](https://huggingface.co/docs/transformers/generation_strategies) — **Hugging Face** — every strategy and its exact parameters, with code.
- [Generating Human-level Text with Contrastive Search](https://huggingface.co/blog/introducing-csearch) — **Tian Lan & Hugging Face** — contrastive search explained and demonstrated against sampling.
- [Assisted Generation: a new direction toward low-latency text generation](https://huggingface.co/blog/assisted-generation) — **Joao Gante (Hugging Face)** — speculative/assisted decoding in practice, with measured speedups.
- [Mirostat and the modern samplers](https://gist.github.com/kalomaze/4473f3f975ff5e5fade06e632498f73e) — **kalomaze** — a practitioner's tour of min-p and the newer truncation samplers.

**Papers**:
- [The Curious Case of Neural Text Degeneration (Nucleus Sampling)](https://arxiv.org/abs/1904.09751) — **Holtzman et al. (2020)** — introduces top-p; shows human text isn't the most probable and why maximization degenerates.
- [Hierarchical Neural Story Generation (top-k sampling)](https://arxiv.org/abs/1805.04833) — **Fan, Lewis & Dauphin (2018)** — popularizes top-k sampling for open-ended generation.
- [Google's Neural Machine Translation System](https://arxiv.org/abs/1609.08144) — **Wu et al. (2016)** — the length-normalization / coverage-penalty form used in production beam search.
- [Six Challenges for Neural Machine Translation](https://arxiv.org/abs/1706.03872) — **Koehn & Knowles (2017)** — documents the "beam search curse": larger beams can hurt quality.
- [Diverse Beam Search](https://arxiv.org/abs/1610.02424) — **Vijayakumar et al. (2018)** — decoding diverse solutions by grouping beams with a dissimilarity penalty.
- [A Contrastive Framework for Neural Text Generation](https://arxiv.org/abs/2202.06417) — **Su et al. (2022)** — contrastive search and the degeneration penalty for coherent, non-repetitive deterministic text.
- [Locally Typical Sampling](https://arxiv.org/abs/2202.00666) — **Meister et al. (2023)** — sampling tokens whose surprise is close to the distribution's entropy.
- [Truncation Sampling as Language Model Desmoothing (epsilon/eta)](https://arxiv.org/abs/2210.15191) — **Hewitt et al. (2022)** — principled absolute/entropy thresholds for truncating the tail.
- [CTRL: A Conditional Transformer Language Model](https://arxiv.org/abs/1909.05858) — **Keskar et al. (2019)** — origin of the repetition penalty used across modern decoders.
- [Turning Up the Heat: Min-p Sampling for Creative and Coherent LLM Outputs](https://arxiv.org/abs/2407.01082) — **Nguyen et al. (2024)** — the relative-floor (min-p) truncation that behaves more gracefully than top-p at high temperature.
- [Distilling the Knowledge in a Neural Network](https://arxiv.org/abs/1503.02531) — **Hinton, Vinyals & Dean (2015)** — §2 defines temperature-scaled softmax $\text{softmax}(z/T)$, the exact operation temperature decoding uses.
- [Fast Inference from Transformers via Speculative Decoding](https://arxiv.org/abs/2211.17192) — **Leviathan et al. (2023)** — the provably-lossless draft-and-verify speedup.
- [Accelerating Large Language Model Decoding with Speculative Sampling](https://arxiv.org/abs/2302.01318) — **Chen et al. (2023)** — concurrent speculative-decoding formulation from DeepMind.
- [A Mathematical Theory of Communication](https://people.math.harvard.edu/~ctm/home/text/others/shannon/entropy/entropy.pdf) — **Shannon (1948)** — defines entropy $H=-\sum p\log p$, the "peakiness" measure temperature controls and the basis for typical sampling.

**Books (free chapters)**:
- [Speech and Language Processing, 3rd ed. — Ch. 10 "Large Language Models" (sampling & decoding)](https://web.stanford.edu/~jurafsky/slp3/10.pdf) — **Jurafsky & Martin** — greedy, beam, temperature, top-k, top-p in the standard reference.
- [Speech and Language Processing, 3rd ed. — Ch. 13 "Machine Translation" (beam search)](https://web.stanford.edu/~jurafsky/slp3/13.pdf) — **Jurafsky & Martin** — beam search in its original decoding setting.

**Runnable code (in this chapter)**:
- [Step-by-step teaching notebook](code/17-Decoding-Strategies.ipynb) — every number on the page, computed and asserted, one idea per cell (greedy/beam, length norm, temperature, top-k/top-p, degeneration, GPT-2).
- [Canonical demo script](code/decoding_strategies.py) — the single seeded source of truth; run `python decoding_strategies.py` (every claim asserted before it prints).
- [Figure generator](code/make_figures_17.py) — regenerates all eight `decode_*` figures from the same functions.

**In this platform**:
- Concept page (full explanation): [Decoding Strategies](17-Decoding-Strategies.md)
- Prior step (the models being decoded): [08 Sequence-to-Sequence & Encoder–Decoder](../08-Sequence-to-Sequence-and-Encoder-Decoder/08-Sequence-to-Sequence-and-Encoder-Decoder.md)
- Puts decoding to work: [12 Machine Translation](../12-Machine-Translation/12-Machine-Translation.md) (beam search) · [13 Text Summarization](../13-Text-Summarization/13-Text-Summarization.md) (beam + no-repeat n-gram)
- How decoded text is scored: [18 NLP Evaluation Metrics](../18-NLP-Evaluation-Metrics/18-NLP-Evaluation-Metrics.md)
- The LLM-systems view (KV-cache interaction, prefill/decode phases, throughput): [09 LLMs · 18 Decoding & Sampling](../../09.%20LLMs/18-Decoding-and-Sampling/18-Decoding-and-Sampling.md) — the LLM-serving counterpart to this general sequence-generation page (no overlap; cross-linked from the speculative-decoding section)
- The speedup decode relies on: [09 LLMs · KV Cache](../../09.%20LLMs/05-KV-Cache/05-KV-Cache.md) (why decode is memory-bound; speculative decoding)
- The *why* behind the math: [AI-ML-intuition 5.05 Autoregressive Generation & Sampling](../../../AI-ML-intuition/Module_5_Generation/5.05_Autoregressive_Generation_Sampling.md) · [5.01 Entropy & KL (temperature)](../../../AI-ML-intuition/Module_5_Generation/5.01_Information_Theory_Entropy_KL_Divergence.md)
