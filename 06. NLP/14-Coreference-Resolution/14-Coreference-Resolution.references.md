---
id: "06-nlp/coreference-resolution/references"
topic: "Coreference Resolution — References"
parent: "06-nlp/coreference-resolution"
type: references
updated: 2026-06-27
---

# Coreference Resolution — references and further reading

> Companion link library for **[Coreference Resolution](14-Coreference-Resolution.md)** (the concept page). External sources *and* internal cross-links, kept separate so it can be reused as a standalone reference list. Grouped by type, best-first; every entry is a primary author or recognized deep explainer — chosen for depth on coreference, not popularity. Links verified.

**Start here — suggested path**:
1. **Frame the task** — read [SLP3 Ch. 23 "Coreference Resolution and Entity Linking"](https://web.stanford.edu/~jurafsky/slp3/23.pdf) (**Jurafsky & Martin**). *Mentions, chains, anaphora, and all three metrics — the definitive write-up.*
2. **See the phenomena** — watch [Anaphora and Coreference Resolution](https://www.youtube.com/watch?v=jaN8kJ7JeY0) (**IIT Madras**). *Why pronouns and definite descriptions are hard.*
3. **Get the modern model** — watch [Stanford CS224N Lec 13 — Coreference Resolution](https://www.youtube.com/watch?v=rpwEWLaueRk) (**Stanford, Manning**). *Mention-pair → mention-ranking → end-to-end neural coref.*
4. **Read the source** — [End-to-end Neural Coreference Resolution](https://arxiv.org/abs/1707.07045) (**Lee et al., 2017**). *Span enumeration + ranking, no pipeline — derive the loss.*
5. **Make it concrete** — try [fastcoref](https://github.com/shon-otmazgin/fastcoref) (**Otmazgin et al.**) or prompt an LLM. *Resolve coreference clusters on real text.*

**Videos**:
- [Stanford CS224N Lec 13 — Coreference Resolution](https://www.youtube.com/watch?v=rpwEWLaueRk) — **Stanford (Christopher Manning)** — the canonical lecture: mention-ranking → end-to-end neural span-ranking, by one of the field's leaders.
- [Anaphora and Coreference Resolution, Discourse Connectives](https://www.youtube.com/watch?v=jaN8kJ7JeY0) — **IIT Madras B.S. Programme** — the linguistic foundations: anaphora, cataphora, salience.
- [Reference Resolution — Anaphora & Coreference](https://www.youtube.com/watch?v=v73Xc7GaR60) — **Dhana DataSciEngg Lectures** — worked examples of resolving references step by step.
- [Coreference Resolution in NLP](https://www.youtube.com/watch?v=Zz5UveXquLY) — **TecHno RayZ** — concise concept-level overview for a first pass.

**Courses (free)**:
- [Stanford CS224N: NLP with Deep Learning](https://web.stanford.edu/class/cs224n/) — **Stanford (Manning)** — the course housing the dedicated coreference lecture and notes.
- [Hugging Face LLM Course — Ch. 1: Transformer Models](https://huggingface.co/learn/llm-course/chapter1/1) — **Hugging Face** — the contextual encoders (BERT/SpanBERT) that modern coref builds on.

**Articles / blogs (free, no paywall)**:
- [Coreference Resolution chapter (SLP3 draft, Ch. 23)](https://web.stanford.edu/~jurafsky/slp3/23.pdf) — **Jurafsky & Martin** — the free, definitive treatment of mentions, mention-ranking, neural coref, and the MUC/B³/CEAF metrics.
- [fastcoref — fast & accurate neural coreference (GitHub)](https://github.com/shon-otmazgin/fastcoref) — **Shon Otmazgin et al.** — a maintained, practical coref library with usage examples; the modern successor to neuralcoref.
- [NeuralCoref (GitHub)](https://github.com/huggingface/neuralcoref) — **Hugging Face** — the classic spaCy coref pipeline, with an explanation of the neural mention-ranking model (note: pinned to older versions).
- [Maverick: efficient and accurate coreference resolution (GitHub)](https://github.com/SapienzaNLP/maverick-coref) — **Sapienza NLP** — a current, lightweight SpanBERT-style coref model with strong CoNLL scores.

**Key papers**:
- [End-to-end Neural Coreference Resolution](https://arxiv.org/abs/1707.07045) — **Lee, He, Lewis & Zettlemoyer (2017)** — the span-ranking model that removed the mention-detection pipeline; the page's core derivation.
- [Higher-order Coreference Resolution with Coarse-to-fine Inference](https://arxiv.org/abs/1804.05392) — **Lee, He & Zettlemoyer (2018)** — coarse-to-fine pruning + higher-order span refinement on top of the 2017 model.
- [SpanBERT: Improving Pre-training by Representing and Predicting Spans](https://arxiv.org/abs/1907.10529) — **Joshi, Chen, Liu, Weld, Zettlemoyer & Levy (2020)** — span-boundary pretraining; the encoder that pushed coref past 79 CoNLL F1.
- [A Machine Learning Approach to Coreference Resolution of Noun Phrases](https://aclanthology.org/J01-4004/) — **Soon, Ng & Lim (2001)** — the canonical mention-pair model with hand-engineered features.
- [Deep Reinforcement Learning for Mention-Ranking Coreference Models](https://arxiv.org/abs/1609.08667) — **Clark & Manning (2016)** — neural mention-ranking and entity-level coref with RL.
- [Resolving Pronoun References (the Hobbs algorithm)](https://doi.org/10.1016/0024-3841(78)90006-2) — **Jerry Hobbs (1978)** — the syntactic-tree-search baseline; "Hobbs distance" survives as a feature.
- [A Model-Theoretic Coreference Scoring Scheme (MUC)](https://aclanthology.org/M95-1005/) — **Vilain, Burger, Aberdeen, Connolly & Hirschman (1995)** — the link-based MUC metric.
- [Algorithms for Scoring Coreference Chains (B³)](https://www.aaai.org/Papers/Symposia/Spring/1998/SS-98-01/SS98-01-013.pdf) — **Bagga & Baldwin (1998)** — the mention-based B³ metric computed by hand in the page.
- [On Coreference Resolution Performance Metrics (CEAF)](https://aclanthology.org/H05-1004/) — **Xiaoqiang Luo (2005)** — the entity-alignment CEAF metric (φ4).
- [The Winograd Schema Challenge](https://cdn.aaai.org/ocs/4492/4492-21843-1-PB.pdf) — **Levesque, Davis & Morgenstern (2012)** — commonsense pronoun resolution as an alternative to the Turing Test.
- [CoNLL-2012 Shared Task: Modeling Multilingual Unrestricted Coreference in OntoNotes](https://aclanthology.org/W12-4501/) — **Pradhan et al. (2012)** — the benchmark and the MUC/B³/CEAF average that defines reported scores.
- [Gender Bias in Coreference Resolution (WinoBias)](https://arxiv.org/abs/1804.06876) — **Zhao, Wang, Yatskar, Ordonez & Chang (2018)** — measuring and mitigating gender bias in coref systems.
- [Word-Level Coreference Resolution](https://arxiv.org/abs/2109.04127) — **Dobrovolskii (2021)** — ranking single head words instead of all spans for an O(T) candidate set.

**Books (free chapters)**:
- [Speech and Language Processing, 3rd ed. — Ch. 23 "Coreference Resolution and Entity Linking"](https://web.stanford.edu/~jurafsky/slp3/23.pdf) — **Jurafsky & Martin** — mentions, chains, mention-ranking, neural coref, and the metrics.
- [Speech and Language Processing, 3rd ed. — Ch. 24 "Discourse Coherence"](https://web.stanford.edu/~jurafsky/slp3/24.pdf) — **Jurafsky & Martin** — the discourse context (Centering Theory, salience) that coreference lives in.

**In this platform**:
- Concept page (full explanation): [Coreference Resolution](14-Coreference-Resolution.md)
- Runnable code: [teaching notebook](code/14-Coreference-Resolution.ipynb) · [seeded source module `coreference.py`](code/coreference.py) · [figure generator `make_figures_14.py`](code/make_figures_14.py) — the mention-ranking softmax, transitive closure, and all three metrics from scratch, with every page figure regenerated from the same functions.
- Prior step / mention detection: [Sequence Labeling — POS & NER](../09-Sequence-Labeling-POS-and-NER/09-Sequence-Labeling-POS-and-NER.md) — NER feeds candidate mentions.
- Encoder foundations: [Contextual Embeddings — ELMo, BERT](../06-Contextual-Embeddings-ELMo-BERT/06-Contextual-Embeddings-ELMo-BERT.md) — the SpanBERT-style representations span-ranking coref runs on · [Attention Mechanism](../../05.%20Deep_Learning/15-Attention-Mechanism/15-Attention-Mechanism.md) — the span head-attention pooling.
- Downstream uses: [Question Answering](../11-Question-Answering/11-Question-Answering.md) · [Text Summarization](../13-Text-Summarization/13-Text-Summarization.md) · [Machine Translation](../12-Machine-Translation/12-Machine-Translation.md) — all need resolved entities.
- Metrics neighbor: [NLP Evaluation Metrics](../18-NLP-Evaluation-Metrics/18-NLP-Evaluation-Metrics.md) — where MUC/B³/CEAF sit among task metrics.
