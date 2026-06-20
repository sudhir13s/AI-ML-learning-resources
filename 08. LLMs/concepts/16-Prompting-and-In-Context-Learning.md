---
id: "08-llms/prompting-and-in-context-learning"
topic: "Prompting & In-Context Learning"
parent: "08-llms"
level: intermediate
prereqs: ["language-modeling-objectives", "decoder-only-architecture"]
interview_frequency: very-high
updated: 2026-06-20
---

# Prompting & In-Context Learning
> The surprising emergent ability of large LMs to learn a task **from examples in the prompt** — no
> weight updates. Zero-shot (instruction only), few-shot (a handful of demonstrations), and the prompt
> patterns (role, delimiters, output format) that steer behavior. The cheapest, fastest way to adapt
> a model — and a frequent first stop before fine-tuning.

**Why it matters:** the "what is in-context learning and why does it work?" question. Be ready to
define zero/one/few-shot, explain that ICL does *no* gradient updates (it's conditioning), discuss the
leading hypotheses (induction heads, implicit gradient descent), and contrast prompting vs fine-tuning.

**⭐ Start here — suggested path:**

1. **Build the mental model** — watch [Karpathy: Intro to LLMs](https://www.youtube.com/watch?v=zjkBMFhNj_g). *Frames prompting as conditioning a next-token predictor.*
2. **Learn the patterns** — [Prompt Engineering Guide](https://www.promptingguide.ai/). *Zero/few-shot, roles, delimiters, structured outputs — the practical toolkit.*
3. **Take a short course** — [Prompt Engineering Tutorial](https://www.youtube.com/watch?v=_ZvnD73m40o) (freeCodeCamp). *A structured walkthrough of effective prompting.*
4. **Read the source** — [GPT-3: Language Models are Few-Shot Learners](https://arxiv.org/abs/2005.14165). *Where in-context learning was first demonstrated at scale.*
5. **Understand mechanism** — [Module 8.01 intuition](../../../AI-ML-intuition/Module_8_LLMs_and_Agentic_Systems/8.01_In-Context_Learning_and_Prompting.md). *Why ICL emerges and how it relates to attention.*

## 🎓 Courses (free)
- [Prompt Engineering Guide](https://www.promptingguide.ai/) — **DAIR.AI** — the definitive free, open prompting reference + techniques.
- [Hugging Face LLM Course — Ch. 1](https://huggingface.co/learn/llm-course/chapter1/1) — **Hugging Face** — prompting and zero/few-shot use of LLMs.

## 🎥 Videos
- [Intro to Large Language Models](https://www.youtube.com/watch?v=zjkBMFhNj_g) — **Andrej Karpathy** — prompting as conditioning; the foundational mental model.
- [Prompt Engineering Tutorial — Master ChatGPT and LLM Responses](https://www.youtube.com/watch?v=_ZvnD73m40o) — **freeCodeCamp** — a full, structured prompting course.
- [Chain-of-thought prompting — Explained!](https://www.youtube.com/watch?v=AFE6x81AP4k) — **CodeEmporium** — the most important prompting pattern for reasoning.
- [Deep Dive into LLMs like ChatGPT](https://www.youtube.com/watch?v=7xTGNNLPyMI) — **Andrej Karpathy** — how prompting interacts with instruction tuning and RLHF.

## 📄 Key Papers
- [Language Models are Few-Shot Learners (GPT-3)](https://arxiv.org/abs/2005.14165) — **Brown et al. (2020)** — in-context learning emerges at scale.
- [Chain-of-Thought Prompting Elicits Reasoning](https://arxiv.org/abs/2201.11903) — **Wei et al. (2022)** — the prompting pattern that unlocks multi-step reasoning.
- [Toolformer: Language Models Can Teach Themselves to Use Tools](https://arxiv.org/abs/2302.04761) — **Schick et al. (2023)** — prompting extended to tool/API use.

## 📰 Articles / Blogs (free, no paywall)
- [Prompt Engineering](https://lilianweng.github.io/posts/2021-01-02-controllable-text-generation/) — **Lilian Weng** — controllable generation and prompting techniques.
- [Prompt Engineering Guide (techniques)](https://www.promptingguide.ai/) — **DAIR.AI** — comprehensive, free patterns library.
- [Understanding Large Language Models](https://magazine.sebastianraschka.com/p/understanding-large-language-models) — **Sebastian Raschka** — where ICL sits among LLM capabilities.

## 📚 Books (free, with chapters)
- [Speech and Language Processing, 3rd ed. — **Ch. 12 "Model Alignment, Prompting & In-Context Learning"**](https://web.stanford.edu/~jurafsky/slp3/) — **Jurafsky & Martin** — the dedicated prompting/ICL chapter.
- [A Survey of Large Language Models](https://arxiv.org/abs/2303.18223) — **Zhao et al. (2023)** — §6 prompting and in-context learning, free reference.

## 🔗 In this platform
- Concept depth (the *why*): [Module 8.01 In-Context Learning & Prompting](../../../AI-ML-intuition/Module_8_LLMs_and_Agentic_Systems/8.01_In-Context_Learning_and_Prompting.md)
- Applications (covered elsewhere): [Retrieval-Augmented Generation → RAG & LLM Applications](../../16.%20RAG_and_LLM_Applications/concepts/README.md) · [Agents & Tool Use (intuition 8.03)](../../../AI-ML-intuition/Module_8_LLMs_and_Agentic_Systems/8.03_Agents_and_Tool_Use.md)
- Related concepts: [Chain-of-Thought Reasoning](17-Chain-of-Thought-Reasoning.md) · [Instruction Tuning](14-Instruction-Tuning.md) · [Decoding & Sampling](18-Decoding-and-Sampling.md)
