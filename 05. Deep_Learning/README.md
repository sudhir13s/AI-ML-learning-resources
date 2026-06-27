---
id: "05-deep-learning"
topic: "Deep Learning"
level: intermediate
prereqs: ["linear-algebra", "calculus", "python", "machine-learning-basics"]
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
1. ✅ [Perceptron & MLP (Feedforward Networks)](01-Perceptron-and-MLP/01-Perceptron-and-MLP.md)
2. ✅ [Backpropagation & Computational Graphs](02-Backpropagation-and-Computational-Graphs/02-Backpropagation-and-Computational-Graphs.md)
3. ✅ [Activation Functions (ReLU · GELU · sigmoid · tanh · softmax)](03-Activation-Functions/03-Activation-Functions.md)
4. ✅ [Loss Functions (MSE · cross-entropy)](04-Loss-Functions/04-Loss-Functions.md)
5. ✅ [Weight Initialization (Xavier/Glorot · He)](05-Weight-Initialization/05-Weight-Initialization.md)
6. ✅ [Vanishing / Exploding Gradients & Gradient Clipping](06-Vanishing-Exploding-Gradients/06-Vanishing-Exploding-Gradients.md)

### Training & optimization
7. ✅ [Optimizers (SGD · Momentum · Adam · AdamW · RMSprop)](07-Optimizers/07-Optimizers.md)
8. ✅ [Learning-Rate Schedules & Warmup](08-Learning-Rate-Schedules-and-Warmup/08-Learning-Rate-Schedules-and-Warmup.md)
9. ✅ [Regularization (L1/L2 · weight decay · early stopping)](09-Regularization/09-Regularization.md)
10. ✅ [Dropout](10-Dropout/10-Dropout.md)
11. ✅ [Normalization (Batch · Layer · Group)](11-Normalization/11-Normalization.md)
12. ✅ [Hyperparameter Tuning](12-Hyperparameter-Tuning/12-Hyperparameter-Tuning.md)

### Architectures
13. ✅ [CNNs & Convolution](13-CNNs-and-Convolution/13-CNNs-and-Convolution.md)
14. ✅ [RNN / LSTM / GRU](14-RNN-LSTM-GRU/14-RNN-LSTM-GRU.md)
15. ✅ [Attention Mechanism](15-Attention-Mechanism/15-Attention-Mechanism.md)
16. ✅ [Transformer Architecture](16-Transformer-Architecture/16-Transformer-Architecture.md)
17. ✅ [Positional Encoding](17-Positional-Encoding/17-Positional-Encoding.md)
18. ✅ [Residual / Skip Connections](18-Residual-Skip-Connections/18-Residual-Skip-Connections.md)
19. ✅ [Autoencoders](19-Autoencoders/19-Autoencoders.md)

### Related concepts (canonical home is another section)
> These topics are used across many areas, so they're kept in one place to avoid repetition.
- **Word / sentence embeddings** — Word2Vec · GloVe · contextual embeddings → [NLP](../06.%20NLP/README.md)
- **Vision architectures in depth** — ResNet/Inception, detection, segmentation → [Computer Vision](../07.%20Computer%20Vision/README.md)
- **Pretraining & LLM-scale models** — BERT · GPT · scaling laws · RLHF → [LLMs](../09.%20LLMs/README.md)
- **Pure math** — PCA/SVD · probability · optimization theory → [Foundations · Maths for AI-ML](../01.%20Foundations/README.md)

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
- **Understand the math:** [AI-ML-intuition — Module 2 (Optimization)](../../AI-ML-intuition/Module_2_Optimization/) · [Module 4 (Stabilization)](../../AI-ML-intuition/Module_4_Stabilization/)
- **Build it:** [AI-ML-problemsets](../../AI-ML-problemsets/)
- **Prereq math:** [Maths for AI-ML curriculum](../01.%20Foundations/Maths%20for%20AI-ML/README.md)
