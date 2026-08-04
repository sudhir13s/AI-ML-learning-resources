---
id: "05-deep-learning"
topic: "Deep Learning"
level: intermediate
built_from: ["linear-algebra", "calculus", "python", "machine-learning-basics"]
updated: 2026-06-27
---

# Deep Learning
> Neural networks that learn hierarchical representations from data — the engine behind modern
> vision, language, and generative AI. This is the curated shortlist of the *best free* resources;
> for the math intuition behind each idea, see the platform links at the bottom.

**⭐ Start here:** [Neural Networks — 3Blue1Brown](https://www.youtube.com/playlist?list=PLZHQObOWTQDNU6R1_67000Dx_ZCJB-3pi) for intuition, then [Neural Networks: Zero to Hero — Karpathy](https://karpathy.ai/zero-to-hero.html) to build one from scratch.

## 📑 Concept Index
Every chapter is a self-contained folder (`NN-Concept/NN-Concept.md`) with its page and a curated
`.references.md` resource card (free, open courses · videos · papers · articles · books · cross-links).
> **✅ ready.** New to deep learning? Start with the field overview below, then work top to bottom.

### Foundations of neural nets
1. ✅ [Perceptron & MLP (Feedforward Networks)](neural-network-foundations/perceptron-and-mlp/perceptron-and-mlp.md)
2. ✅ [Backpropagation & Computational Graphs](neural-network-foundations/backpropagation-and-computational-graphs/backpropagation-and-computational-graphs.md)
3. ✅ [Activation Functions (ReLU · GELU · sigmoid · tanh · softmax)](stabilization-and-architectural-blocks/activation-functions/activation-functions.md)
4. ✅ [Loss Functions (MSE · cross-entropy)](optimization-and-training/loss-functions/loss-functions.md)
5. ✅ [Weight Initialization (Xavier/Glorot · He)](optimization-and-training/weight-initialization/weight-initialization.md)
6. ✅ [Vanishing / Exploding Gradients & Gradient Clipping](optimization-and-training/vanishing-exploding-gradients/vanishing-exploding-gradients.md)

### Training & optimization
7. ✅ [Optimizers (SGD · Momentum · Adam · AdamW · RMSprop)](optimization-and-training/optimizers/optimizers.md)
8. ✅ [Learning-Rate Schedules & Warmup](optimization-and-training/learning-rate-schedules-and-warmup/learning-rate-schedules-and-warmup.md)
9. ✅ [Regularization (L1/L2 · weight decay · early stopping)](optimization-and-training/regularization/regularization.md)
10. ✅ [Dropout](stabilization-and-architectural-blocks/dropout/dropout.md)
11. ✅ [Normalization (Batch · Layer · Group)](stabilization-and-architectural-blocks/normalization/normalization.md)
12. ✅ [Hyperparameter Tuning](optimization-and-training/hyperparameter-tuning/hyperparameter-tuning.md)

### Architectures
13. ✅ [CNNs & Convolution](neural-architectures/cnns-and-convolution/cnns-and-convolution.md)
14. ✅ [RNN / LSTM / GRU](neural-architectures/rnn-lstm-gru/rnn-lstm-gru.md)
15. ✅ [Attention Mechanism](attention-and-transformers/attention-mechanism/attention-mechanism.md)
16. ✅ [Transformer Architecture](attention-and-transformers/transformer-architecture/transformer-architecture.md)
17. ✅ [Positional Encoding](attention-and-transformers/positional-encoding/positional-encoding.md)
18. ✅ [Residual / Skip Connections](stabilization-and-architectural-blocks/residual-skip-connections/residual-skip-connections.md)
19. ✅ [Autoencoders](neural-architectures/autoencoders/autoencoders.md)

### Related concepts (canonical home is another section)
> These topics are used across many areas, so they're kept in one place to avoid repetition.
- **Word / sentence embeddings** — Word2Vec · GloVe · contextual embeddings → [NLP](../modalities-and-generative-models/natural-language-processing/README.md)
- **Vision architectures in depth** — ResNet/Inception, detection, segmentation → [Computer Vision](../modalities-and-generative-models/computer-vision/README.md)
- **Pretraining & LLM-scale models** — BERT · GPT · scaling laws · RLHF → [LLMs](../llms-applications-and-agents/README.md)
- **Pure math** — PCA/SVD · probability · optimization theory → [Foundations · Maths for AI-ML](../foundations/mathematical-foundations/README.md)

## 🎓 Courses (free)
- [Neural Networks: Zero to Hero](https://karpathy.ai/zero-to-hero.html) — **Andrej Karpathy** — builds backprop → GPT from scratch in plain Python; the best hands-on course in existence.
- [Practical Deep Learning for Coders](https://course.fast.ai/) — **fast.ai (Jeremy Howard)** — top-down, code-first, get models working fast; the best "learn by doing" path.
- [MIT 6.S191: Intro to Deep Learning](http://introtodeeplearning.com/) — **MIT (Amini et al.)** — concise, current, free lectures + labs (refreshed yearly).

## 🎥 Videos / Lectures
- [Neural Networks series](https://www.youtube.com/playlist?list=PLZHQObOWTQDNU6R1_67000Dx_ZCJB-3pi) — **3Blue1Brown** — the definitive visual intuition for what a network *is* and how backprop works.
- [Deep Learning Specialization](https://www.coursera.org/specializations/deep-learning) — **Andrew Ng / DeepLearning.AI** — free to audit; the canonical structured walkthrough of the whole field.

## 📄 Key Papers
- [Deep Learning](https://www.nature.com/articles/nature14539) — **LeCun, Bengio & Hinton (Nature, 2015)** — the field's authoritative review by its founders.
- [Deep Residual Learning (ResNet)](https://arxiv.org/abs/1512.03385) — **He et al. (2015)** — the idea that made networks *deep*; one of the most-cited papers in ML.

## 📰 Articles / Blogs
- [colah.github.io](https://colah.github.io/) — **Chris Olah** — the gold standard for explaining backprop, LSTMs, and representations visually.
- [Distill.pub](https://distill.pub/) — **Distill** — interactive, peer-reviewed deep-learning explainers (archival but timeless).

## 📚 Books (free)
- [Dive into Deep Learning (d2l.ai)](https://d2l.ai/) — **Zhang, Lipton, Li & Smola** — free, interactive, runnable code in PyTorch/JAX; the best modern textbook.
- [Neural Networks and Deep Learning](http://neuralnetworksanddeeplearning.com/) — **Michael Nielsen** — free, the clearest from-first-principles introduction.
- [Deep Learning](https://www.deeplearningbook.org/) — **Goodfellow, Bengio & Courville** — free online; the rigorous reference text.

## 🔗 In this platform
- **Understand the math:** [AI-ML-intuition — Module 2 (Optimization)](../../AI-ML-intuition/learning-and-optimization/) · [Module 4 (Stabilization)](../../AI-ML-intuition/training-stability/)
- **Build it:** [AI-ML-problemsets](../../AI-ML-problemsets/)
- **Prereq math:** [Maths for AI-ML curriculum](../foundations/mathematical-foundations/maths-for-ai-ml/README.md)
