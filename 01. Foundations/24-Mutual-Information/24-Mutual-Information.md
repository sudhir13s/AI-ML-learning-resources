---
id: "01-foundations/mutual-information"
topic: "Mutual Information"
parent: "01-foundations"
level: advanced
prereqs: ["01-foundations/entropy", "01-foundations/cross-entropy-and-kl-divergence"]
interview_frequency: medium
updated: 2026-06-20
---

# Mutual Information
> Mutual information `I(X;Y) = H(X) − H(X|Y)` measures how much knowing one variable reduces
> uncertainty about another — the information they *share*. It's symmetric, non-negative, and zero
> exactly when the variables are independent, which makes it the principled measure of dependence
> behind feature selection, representation learning (InfoNCE), and decision-tree splits.

**Why it matters:** MI generalizes correlation to *any* (including nonlinear) dependence, so it
appears in feature selection, the information-bottleneck view of deep nets, and contrastive
objectives (InfoNCE lower-bounds MI). Interviewers ask how MI relates to entropy and KL, why it's
zero under independence, and where it shows up in modern representation learning.

**⭐ Start here — suggested path:**

1. **Definition & intuition** — watch [Ben Lambert: An introduction to mutual information](https://www.youtube.com/watch?v=U9h1xkNELvY). *MI as the reduction in uncertainty, with the entropy diagram.*
2. **From entropy/KL** — watch [Aurélien Géron: Entropy, Cross-Entropy and KL](https://www.youtube.com/watch?v=ErfnhcEV1O8) and note `I(X;Y) = D(p(x,y)‖p(x)p(y))`. *MI is the KL between joint and product-of-marginals.*
3. **Formalize** — read [Cover & Thomas Ch. 2 (Entropy, Relative Entropy & Mutual Information)](http://www.cs.columbia.edu/~vh/courses/LexicalSemantics/Association/Cover&Thomas-Ch2.pdf). *The definitions, chain rules, and the data-processing inequality.*
4. **Modern ML use** — read [Representation Learning with Contrastive Predictive Coding (InfoNCE)](https://arxiv.org/abs/1807.03748). *How MI maximization drives self-supervised representations.*
5. **Connect to ML** — read [AI-ML-intuition 5.01 Information Theory: Entropy & KL](../../../AI-ML-intuition/Module_5_Generation/5.01_Information_Theory_Entropy_KL_Divergence.md) and [1.13 Contrastive Learning (SimCLR/InfoNCE)](../../../AI-ML-intuition/Module_1_Representation/1.13_Representation_Contrastive_Learning_SimCLR_InfoNCE.md). *Where MI becomes a training objective.*

## 🎓 Courses (free)
- [Stanford EE376A — Information Theory](https://web.stanford.edu/class/ee376a/) — **Stanford** — mutual information, channel capacity, and the data-processing inequality; free notes.
- [Khan Academy — Journey into Information Theory](https://www.khanacademy.org/computing/computer-science/informationtheory) — **Khan Academy** — the entropy groundwork MI builds on.

## 🎥 Videos
- [An introduction to mutual information](https://www.youtube.com/watch?v=U9h1xkNELvY) — **Ben Lambert** — the clearest short MI explainer.
- [A Short Introduction to Entropy, Cross-Entropy and KL-Divergence](https://www.youtube.com/watch?v=ErfnhcEV1O8) — **Aurélien Géron** — the entropy/KL pieces MI is assembled from.
- [Entropy (for data science) Clearly Explained](https://www.youtube.com/watch?v=YtebGVx-Fxw) — **StatQuest (Josh Starmer)** — entropy and conditional entropy, the ingredients of MI.
- [Measuring information | Journey into information theory](https://www.youtube.com/watch?v=PtmzfpV6CDE) — **Khan Academy** — information content underlying MI.

## 📄 Key Papers
- [Cover & Thomas — Ch. 2 (Entropy, Relative Entropy & Mutual Information)](http://www.cs.columbia.edu/~vh/courses/LexicalSemantics/Association/Cover&Thomas-Ch2.pdf) — **Cover & Thomas** — the canonical definition and properties of MI (free chapter PDF).
- [Representation Learning with Contrastive Predictive Coding (InfoNCE)](https://arxiv.org/abs/1807.03748) — **van den Oord, Li & Vinyals (2018)** — MI maximization as a self-supervised objective.

## 📰 Articles / Blogs (free, no paywall)
- [Visual Information Theory](https://colah.github.io/posts/2015-09-Visual-Information/) — **Christopher Olah** — entropy, KL, and the joint/marginal picture MI rests on.
- [Scholarpedia — Mutual Information](http://www.scholarpedia.org/article/Mutual_information) — **Latham & Roudi** — a rigorous, free encyclopedic treatment.

## 📚 Books (free, with chapters)
- [Information Theory, Inference, and Learning Algorithms — **Ch. 8 (Dependent Random Variables / MI)**](https://www.inference.org.uk/itprnn/book.pdf) — **David MacKay** — mutual information in the free classic.
- [Elements of Information Theory — **Ch. 2 (Mutual Information)**](http://www.cs.columbia.edu/~vh/courses/LexicalSemantics/Association/Cover&Thomas-Ch2.pdf) — **Cover & Thomas** — the standard reference (free chapter PDF).

## 🔗 In this platform
- Concept depth (the *why*): [AI-ML-intuition 5.01 Information Theory: Entropy & KL](../../../AI-ML-intuition/Module_5_Generation/5.01_Information_Theory_Entropy_KL_Divergence.md) · [1.13 Contrastive Learning (SimCLR/InfoNCE)](../../../AI-ML-intuition/Module_1_Representation/1.13_Representation_Contrastive_Learning_SimCLR_InfoNCE.md)
- Curriculum context: [Maths for AI-ML — Phase 3 (Information Theory, row 3.5)](../Maths%20for%20AI-ML/README.md)
- Prereqs: [22 Entropy](../22-Entropy/22-Entropy.md) · [23 Cross-Entropy & KL Divergence](../23-Cross-Entropy-and-KL-Divergence/23-Cross-Entropy-and-KL-Divergence.md)
</content>
