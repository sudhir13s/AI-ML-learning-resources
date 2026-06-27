---
id: "09-llms/prompting-and-in-context-learning/references"
topic: "Prompting & In-Context Learning — References"
parent: "09-llms/prompting-and-in-context-learning"
type: references
updated: 2026-06-27
---

# Prompting & In-Context Learning — references and further reading

> Companion link library for **[Prompting & In-Context Learning](16-Prompting-and-In-Context-Learning.md)** (the concept page). This file holds the curated links — external sources *and* internal links to related pages on this platform — kept separate so it can be reused as a standalone reference list. Grouped by type, best-first. Every entry is from a primary author or a recognized deep explainer — chosen for depth on *this* topic, not popularity, and free / no paywall.

**Start here — suggested path**:
1. **Get the mental model** — watch [Intro to Large Language Models](https://www.youtube.com/watch?v=zjkBMFhNj_g) (**Andrej Karpathy**). *Frames prompting as conditioning a next-token predictor — the foundation for everything below.*
2. **Read the source** — skim [Language Models are Few-Shot Learners (GPT-3)](https://arxiv.org/abs/2005.14165) (**Brown et al. 2020**), §2–3. *Where zero/one/few-shot ICL was demonstrated at scale.*
3. **See the mechanism** — read [In-context Learning and Induction Heads](https://transformer-circuits.pub/2022/in-context-learning-and-induction-heads/index.html) (**Olsson et al., Anthropic**). *The induction-head circuit behind copying-from-context — the page's Experiment 1, in real models.*
4. **Understand the brittleness** — read [Calibrate Before Use](https://arxiv.org/abs/2102.09690) (**Zhao et al. 2021**). *Majority-label/recency bias and the content-free calibration fix — the page's Experiment 3, plus the cure.*
5. **Learn the practical toolkit** — work through the [Prompt Engineering Guide](https://www.promptingguide.ai/) (**DAIR.AI**). *Zero/few-shot, roles, delimiters, structured outputs — the hands-on patterns.*

**Videos**:
- [Intro to Large Language Models](https://www.youtube.com/watch?v=zjkBMFhNj_g) — **Andrej Karpathy** — prompting as conditioning a next-token predictor; the foundational mental model for ICL.
- [Deep Dive into LLMs like ChatGPT](https://www.youtube.com/watch?v=7xTGNNLPyMI) — **Andrej Karpathy** — how prompting interacts with the rest of the stack (instruction tuning, RLHF); where ICL sits.
- [Prompt Engineering Tutorial — Master ChatGPT and LLM Responses](https://www.youtube.com/watch?v=_ZvnD73m40o) — **freeCodeCamp** — a full, structured walkthrough of effective prompting patterns.
- [Visualizing Attention, a Transformer's Heart](https://www.youtube.com/watch?v=eMlx5fFNoYc) — **3Blue1Brown** — the most visual explanation of Q/K/V; watch it to *see* the attention that induction heads exploit.

**Interactive & visual**:
- [LLM Visualizer (3D)](https://bbycroft.net/llm) — **Brendan Bycroft** — walk a token through a small GPT's full forward pass and see where attention reads from earlier positions (the substrate of induction).
- [Prompt Engineering Guide — techniques playground](https://www.promptingguide.ai/techniques) — **DAIR.AI** — zero-shot, few-shot, and CoT prompt patterns with runnable examples.

**Courses (free)**:
- [Stanford CS224N — Natural Language Processing with Deep Learning](https://web.stanford.edu/class/cs224n/) — **Stanford** — the LLM/prompting lectures situate ICL within language modeling.
- [Hugging Face LLM Course — Chapter 1](https://huggingface.co/learn/llm-course/chapter1/1) — **Hugging Face** — prompting and zero/few-shot use of LLMs, hands-on.

**Articles / blogs (free, no paywall)**:
- [Prompt Engineering](https://lilianweng.github.io/posts/2023-03-15-prompt-engineering/) — **Lilian Weng (OpenAI)** — a rigorous survey of ICL, demonstration selection, calibration, and CoT, with the key papers organised.
- [Prompt Engineering Guide](https://www.promptingguide.ai/) — **DAIR.AI** — the definitive free, open prompting reference: zero/few-shot, roles, delimiters, structured outputs, and technique library.
- [Understanding Large Language Models](https://magazine.sebastianraschka.com/p/understanding-large-language-models) — **Sebastian Raschka** — where in-context learning sits among LLM capabilities and post-training methods.
- [How does in-context learning work? A framework for understanding the differences from training](https://ai.stanford.edu/blog/understanding-incontext/) — **Stanford AI Lab (Xie & Min)** — the Bayesian-task-inference view, explained for a general audience by the authors.

**Key papers**:
- [Calibrate Before Use: Improving Few-Shot Performance of Language Models](https://arxiv.org/abs/2102.09690) — **Zhao et al. (2021)** — majority-label, recency, and common-token biases in ICL, and the content-free affine calibration that fixes them.
- [Fantastically Ordered Prompts and Where to Find Them](https://arxiv.org/abs/2104.08786) — **Lu et al. (2021)** — demonstration *order* swings few-shot accuracy from near-chance to near-SOTA; good orderings are hard to find a priori.
- [In-context Learning and Induction Heads](https://transformer-circuits.pub/2022/in-context-learning-and-induction-heads/index.html) — **Olsson et al. (2022, Anthropic / Transformer Circuits)** — the induction-head mechanism and its phase-change co-occurrence with ICL ability; the core of the page's mechanism section.
- [Language Models are Few-Shot Learners (GPT-3)](https://arxiv.org/abs/2005.14165) — **Brown et al. (2020)** — defines zero/one/few-shot and shows in-context learning emerges at scale.
- [An Explanation of In-context Learning as Implicit Bayesian Inference](https://arxiv.org/abs/2111.02080) — **Xie et al. (2021)** — ICL as inferring a latent task/concept from the prompt; the statistical account (Hypothesis 2).
- [Rethinking the Role of Demonstrations: What Makes In-Context Learning Work?](https://arxiv.org/abs/2202.12837) — **Min et al. (2022)** — ICL works even with many *wrong* demonstration labels; the format, input distribution, and label space matter more than label correctness.
- [Transformers Learn In-Context by Gradient Descent](https://arxiv.org/abs/2212.07677) — **von Oswald et al. (2022)** — the construction showing attention can implement gradient-descent steps; the learning-dynamics account (Hypothesis 3).
- [A Survey on In-context Learning](https://arxiv.org/abs/2301.00234) — **Dong et al. (2023)** — comprehensive survey of ICL methods, demonstration design, and analyses; a map of the whole area.
- [Chain-of-Thought Prompting Elicits Reasoning in Large Language Models](https://arxiv.org/abs/2201.11903) — **Wei et al. (2022)** — the specific prompting *technique* covered in the next chapter; included here as the canonical pointer.

**Books (free chapters)**:
- [Speech and Language Processing, 3rd ed. — Ch. 12 "Model Alignment, Prompting, and In-Context Learning"](https://web.stanford.edu/~jurafsky/slp3/) — **Jurafsky & Martin** — the dedicated, free textbook chapter on prompting and ICL.
- [A Survey of Large Language Models](https://arxiv.org/abs/2303.18223) — **Zhao et al. (2023)** — §6 covers prompting and in-context learning as a reference.

**In this platform**:
- Concept page (full explanation): [Prompting & In-Context Learning](16-Prompting-and-In-Context-Learning.md)
- Concept depth (the intuition): [Module 8.01 In-Context Learning & Prompting](../../../AI-ML-intuition/Module_8_LLMs_and_Agentic_Systems/8.01_In-Context_Learning_and_Prompting.md)
- The next step (a specific prompting technique): [Chain-of-Thought Reasoning](../17-Chain-of-Thought-Reasoning/17-Chain-of-Thought-Reasoning.md)
- Contrast — methods that change weights: [Supervised Fine-Tuning](../13-Supervised-Fine-Tuning/13-Supervised-Fine-Tuning.md) · [Instruction Tuning](../14-Instruction-Tuning/14-Instruction-Tuning.md) · [RLHF & DPO](../15-RLHF-and-DPO/15-RLHF-and-DPO.md)
- Foundations (the attention that induction heads exploit): [Attention Mechanism](../../05.%20Deep_Learning/concepts/15-Attention-Mechanism.md) · [Decoder-only Architecture](../04-Decoder-only-Architecture/04-Decoder-only-Architecture.md)
- Where ICL is put to work: [RAG & LLM Applications](../../11.%20RAG_and_LLM_Applications/README.md) · the [KV cache](../05-KV-Cache/05-KV-Cache.md) and [long-context methods](../08-Long-Context-Methods/08-Long-Context-Methods.md) that make long few-shot prompts affordable
- Decoding that turns the ICL-conditioned distribution into text: [Decoding & Sampling](../18-Decoding-and-Sampling/18-Decoding-and-Sampling.md)
