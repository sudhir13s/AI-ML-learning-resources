---
id: "05-deep-learning/loss-functions"
topic: "Loss Functions (MSE · cross-entropy)"
parent: "05-deep-learning"
level: beginner
prereqs: ["feedforward-networks", "probability"]
interview_frequency: high
updated: 2026-06-19
---

# Loss Functions (MSE · cross-entropy)
> The objective the network minimizes — it scores how wrong a prediction is and supplies the gradient
> that drives learning. **MSE** (mean squared error) fits regression; **cross-entropy** fits
> classification and pairs naturally with softmax. Choosing the right loss for the task (and its
> probabilistic meaning) is what makes training well-posed.

**Why it matters:** a high-frequency fundamentals question — explain why MSE suits regression and
cross-entropy suits classification, derive cross-entropy from **maximum likelihood / negative
log-likelihood**, show why the softmax+cross-entropy gradient simplifies to `(ŷ − y)`, and reason
about when to use MAE vs MSE (outlier sensitivity) and binary vs categorical cross-entropy.

**⭐ Start here — suggested path:**

1. **Build intuition** — watch [Cross Entropy](https://www.youtube.com/watch?v=6ArSys5qHAU) (**StatQuest**). *Why we penalize confident-and-wrong predictions hard.*
2. **Connect to information theory** — watch ⭐ [Entropy, Cross-Entropy and KL-Divergence](https://www.youtube.com/watch?v=ErfnhcEV1O8) (**Aurélien Géron**). *Where the cross-entropy formula comes from.*
3. **Get the math** — read [Cross-entropy loss, explained](https://gombru.github.io/2018/05/23/cross_entropy_loss/) (**Raúl Gómez**). *Logits → softmax → cross-entropy, with the clean gradient.*
4. **See the MLE view** — read [d2l §3.1 (loss) and §4.1 (softmax + cross-entropy)](https://d2l.ai/chapter_multilayer-perceptrons/index.html). *Cross-entropy as negative log-likelihood, with code.*
5. **Make it concrete** — implement MSE and cross-entropy by hand and check gradients. *Deriving `(ŷ − y)` once makes the softmax+CE pairing permanent.*

## 🎓 Courses (free)
- [Stanford CS231n — Loss Functions and Optimization](https://cs231n.github.io/optimization-1/) — **Stanford (Karpathy / Li / Johnson)** — softmax vs SVM loss, with worked gradients.
- [Dive into Deep Learning — Linear & Softmax Regression](https://d2l.ai/chapter_linear-classification/index.html) — **Zhang et al.** — MSE and cross-entropy from first principles, with runnable code.

## 🎥 Videos
- [Neural Networks Part 6: Cross Entropy](https://www.youtube.com/watch?v=6ArSys5qHAU) — **StatQuest (Josh Starmer)** — the cleanest from-scratch intuition for cross-entropy.
- [A Short Introduction to Entropy, Cross-Entropy and KL-Divergence](https://www.youtube.com/watch?v=ErfnhcEV1O8) — **Aurélien Géron** — the information-theory grounding for the loss.
- [Intuitively Understanding the Cross Entropy Loss](https://www.youtube.com/watch?v=Pwgpl9mKars) — **Adian Liusie** — builds the loss up from likelihood, very clear.
- [Logistic Regression Cost Function](https://www.youtube.com/watch?v=SHEPb1JHw5o) — **DeepLearningAI (Andrew Ng)** — binary cross-entropy derived as a likelihood.

## 📄 Key Papers
- [Deep Learning (Nature review)](https://www.nature.com/articles/nature14539) — **LeCun, Bengio & Hinton (2015)** — frames the loss/objective inside the training picture.
- [Focal Loss for Dense Object Detection](https://arxiv.org/abs/1708.02002) — **Lin et al. (2017)** — a class-imbalance reshaping of cross-entropy; a common interview follow-up.

## 📰 Articles / Blogs (free, no paywall)
- [Understanding Categorical Cross-Entropy Loss](https://gombru.github.io/2018/05/23/cross_entropy_loss/) — **Raúl Gómez** — softmax/sigmoid + cross-entropy with the gradient, clearly laid out.
- [Cross-Entropy Loss](https://www.pinecone.io/learn/cross-entropy-loss/) — **Pinecone** — concise, no-paywall walk-through with intuition and formulas.
- [The Softmax function and its derivative](https://e2eml.school/softmax.html) — **Brandon Rohrer** — why softmax+cross-entropy gives the clean `(ŷ − y)` gradient.

## 📚 Books (free, with chapters)
- [Dive into Deep Learning — **Ch. 4 "Linear Neural Networks for Classification"** (Softmax & Cross-Entropy)](https://d2l.ai/chapter_linear-classification/index.html) — **Zhang et al.** — the loss derived as negative log-likelihood, with code.
- [Deep Learning — **§5.5 "Maximum Likelihood Estimation"** + **§6.2 (output units & loss)**](https://www.deeplearningbook.org/contents/ml.html) — **Goodfellow, Bengio & Courville** — losses derived as MLE, rigorously.
- [Neural Networks and Deep Learning — **Ch. 3 (cross-entropy cost)**](http://neuralnetworksanddeeplearning.com/chap3.html) — **Michael Nielsen** — why cross-entropy speeds learning vs quadratic loss.

## 🔗 In this platform
- Concept depth (the *why*): [AI-ML-intuition 3.01 MSE / L2 Loss](../../../AI-ML-intuition/Module_3_Evaluation/3.01_Mean_Squared_Error_MSE_L2_Loss.md) · [3.03 Cross-Entropy / NLL](../../../AI-ML-intuition/Module_3_Evaluation/3.03_Categorical_Cross-Entropy_NLL.md) · [3.04 Maximum Likelihood](../../../AI-ML-intuition/Module_3_Evaluation/3.04_Maximum_Likelihood_Estimation.md)
- Prerequisite: [03 Activation Functions](03-Activation-Functions.md) (softmax pairs with cross-entropy)
- Next concept: [07 Optimizers](07-Optimizers.md) (what minimizes the loss)
- Field overview: [Deep Learning](../README.md)
