---
id: "01-foundations/cross-entropy-and-kl-divergence"
topic: "Cross-Entropy & KL Divergence"
parent: "01-foundations"
level: intermediate
prereqs: ["01-foundations/entropy"]
interview_frequency: very-high
updated: 2026-06-20
---

# Cross-Entropy & KL Divergence
> Cross-entropy `H(p,q) = −Σ p log q` is the cost of coding data from `p` using a model `q`; KL
> divergence `D(p‖q) = H(p,q) − H(p)` is the *extra* cost — a measure of how far `q` is from `p`.
> Minimizing cross-entropy is exactly maximum-likelihood training, which is why it's the default
> loss for classification and language modeling.

**Why it matters:** this is the single most important information-theory topic for ML. Interviewers
ask why cross-entropy is the classification loss, how it relates to MLE and to KL, why KL is
asymmetric and non-negative (Gibbs' inequality), and where forward vs reverse KL show up (VAEs,
distillation, RLHF).

**⭐ Start here — suggested path:**

1. **The bridge from entropy** — watch [Aurélien Géron: Entropy, Cross-Entropy and KL-Divergence](https://www.youtube.com/watch?v=ErfnhcEV1O8). *Defines all three and how they relate, fast.*
2. **Cross-entropy as a loss** — watch [StatQuest: Neural Networks Part 6 — Cross Entropy](https://www.youtube.com/watch?v=6ArSys5qHAU). *Exactly how it's used to train classifiers.*
3. **KL intuition** — watch [Adian Liusie: Intuitively Understanding the KL Divergence](https://www.youtube.com/watch?v=SxGYPqCgJWM). *Why KL measures distributional distance (and why it's asymmetric).*
4. **The math + MLE link** — read [colah: Visual Information Theory](https://colah.github.io/posts/2015-09-Visual-Information/) and [MML Ch. 8 (cross-entropy ↔ MLE)](https://mml-book.github.io/book/mml-book.pdf). *Minimizing cross-entropy = maximizing likelihood.*
5. **Connect to ML** — read [AI-ML-intuition 3.03 Categorical Cross-Entropy / NLL](../../../AI-ML-intuition/Module_3_Evaluation/3.03_Categorical_Cross-Entropy_NLL.md) and [5.01 Entropy & KL](../../../AI-ML-intuition/Module_5_Generation/5.01_Information_Theory_Entropy_KL_Divergence.md). *The platform's loss-function deep dives.*

## 🎓 Courses (free)
- [Stanford CS231n — Softmax & cross-entropy loss notes](https://cs231n.github.io/linear-classify/) — **Stanford** — cross-entropy derived as the classification loss.
- [Khan Academy — Journey into Information Theory](https://www.khanacademy.org/computing/computer-science/informationtheory) — **Khan Academy** — the entropy foundations cross-entropy/KL build on.

## 🎥 Videos
- [A Short Introduction to Entropy, Cross-Entropy and KL-Divergence](https://www.youtube.com/watch?v=ErfnhcEV1O8) — **Aurélien Géron** — all three concepts and their relationships.
- [Neural Networks Part 6: Cross Entropy](https://www.youtube.com/watch?v=6ArSys5qHAU) — **StatQuest (Josh Starmer)** — cross-entropy as a training loss.
- [Intuitively Understanding the KL Divergence](https://www.youtube.com/watch?v=SxGYPqCgJWM) — **Adian Liusie** — the meaning and asymmetry of KL.
- [Entropy (for data science) Clearly Explained](https://www.youtube.com/watch?v=YtebGVx-Fxw) — **StatQuest (Josh Starmer)** — the entropy baseline KL is measured against.

## 📄 Key Papers
- [On Information and Sufficiency (KL divergence)](https://projecteuclid.org/journals/annals-of-mathematical-statistics/volume-22/issue-1/On-Information-and-Sufficiency/10.1214/aoms/1177729694.full) — **Kullback & Leibler (1951)** — the original definition of relative entropy (open access on Project Euclid).
- [MacKay — Information Theory, Inference & Learning Algorithms (Ch. 2)](https://www.inference.org.uk/itprnn/book.pdf) — **David MacKay** — relative entropy (KL) and Gibbs' inequality; free.

## 📰 Articles / Blogs (free, no paywall)
- [Visual Information Theory](https://colah.github.io/posts/2015-09-Visual-Information/) — **Christopher Olah** — the definitive free visual treatment of cross-entropy and KL.
- [Kullback-Leibler Divergence Explained](https://www.countbayesie.com/blog/2017/5/9/kullback-leibler-divergence-explained) — **Will Kurt (Count Bayesie)** — an intuitive, free walkthrough from information loss to VAEs.

## 📚 Books (free, with chapters)
- [Information Theory, Inference, and Learning Algorithms — **Ch. 2 (Relative Entropy)**](https://www.inference.org.uk/itprnn/book.pdf) — **David MacKay** — KL and cross-entropy in the free classic.
- [Elements of Information Theory — **Ch. 2 (Relative Entropy)**](http://www.cs.columbia.edu/~vh/courses/LexicalSemantics/Association/Cover&Thomas-Ch2.pdf) — **Cover & Thomas** — the standard reference (free chapter PDF).

## 🔗 In this platform
- Concept depth (the *why*): [AI-ML-intuition 3.03 Categorical Cross-Entropy / NLL](../../../AI-ML-intuition/Module_3_Evaluation/3.03_Categorical_Cross-Entropy_NLL.md) · [5.01 Entropy & KL](../../../AI-ML-intuition/Module_5_Generation/5.01_Information_Theory_Entropy_KL_Divergence.md)
- Curriculum context: [Maths for AI-ML — Phase 3 (Information Theory, row 3.5)](../Maths%20for%20AI-ML/README.md)
- Prereq: [22 Entropy](22-Entropy.md) · Related: [19 Maximum Likelihood Estimation](19-Maximum-Likelihood-Estimation.md) (cross-entropy = MLE) · [24 Mutual Information](24-Mutual-Information.md)
</content>
