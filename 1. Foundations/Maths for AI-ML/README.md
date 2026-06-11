# 📚 Mathematics for AI/ML — Curriculum & Resource Index

> **Role of this page:** the *learning path* for ML math — what to study, in what order, and
> with which courses/videos. **Deep understanding lives in the
> [`AIML-intuition`](../../../AIML-intuition/) repo** (intuition, equations, walkthroughs,
> diagrams, code); this page tells you *when* to read each of its pages and *what to watch
> first*. Absorbed from the retired `math-for-AIML-Q5` syllabus.

## Phase map

| Phase | Discipline | Primarily unlocks (AIML-intuition) | Status |
| :-- | :--- | :--- | :-- |
| **1** | **Linear Algebra** | [Module 1 — Representation](../../../AIML-intuition/Module_1_Representation/) | ✅ below |
| **2** | Calculus & Matrix Calculus | [Module 2 — Optimization](../../../AIML-intuition/Module_2_Optimization/) (2A) | ✅ below |
| **3** | Probability & Information Theory | [Module 5 — Generation](../../../AIML-intuition/Module_5_Generation/), parts of 3 | ✅ below |
| **4** | Statistics & Statistical Learning | [Module 3 — Evaluation](../../../AIML-intuition/Module_3_Evaluation/) | ✅ below |
| **5** | Optimization for ML/DL | [Module 2 — Optimization](../../../AIML-intuition/Module_2_Optimization/) (2B) | ✅ below |
| **6** | Applied Math for Modern DL | [Modules 4](../../../AIML-intuition/Module_4_Stabilization/) & [6](../../../AIML-intuition/Module_6_Reinforcement_Learning/) | ✅ below |
| — | **Specializations** | [Computer Vision](../../7.%20Computer%20Vision/README.md) · [Neuroscience & Brain-Inspired AI](../../13.%20Neuroscience_and_Brain_Inspired_AI/README.md) · [Advanced Research Math](../../14.%20Advanced_Research_Mathematics/README.md) | ✅ separate pages |

> **Mapping note:** discipline → module is *primary, not exclusive* — one discipline feeds
> several modules (probability underlies both Evaluation and Generation). The two indexes cut
> the same content two ways: by **discipline** (here) and by **ML-pipeline stage**
> ([Master Pattern Map](../../../AIML-intuition/0_Master_Pattern_Map.md)).

---

## Phase 1 — Linear Algebra → Module 1 (Representation)

**Goal:** be fluent with vectors, matrices, subspaces, projections, decompositions, eigen
concepts, and tensors — the language of embeddings, attention, and dimensionality reduction.

### Core resource backbone
- **MIT OCW 18.06** Linear Algebra — Gilbert Strang — [course](https://ocw.mit.edu/courses/18-06-linear-algebra-spring-2010/) · [video lectures](https://ocw.mit.edu/courses/18-06-linear-algebra-spring-2010/video_galleries/video-lectures/)
- **3Blue1Brown** — [Essence of Linear Algebra](https://www.youtube.com/playlist?list=PLZHQObOWTQDPD3MizzM2xVFitgF8hE_ab) (visual intuition)
- **Khan Academy** — [Linear Algebra](https://www.khanacademy.org/math/linear-algebra)
- **Stanford CS229** — [Linear Algebra review (PDF)](https://cs229.stanford.edu/section/cs229-linalg.pdf) (applied)

### Study order & what each sub-topic unlocks

| Study (in order) | Key sub-topics | → Read in AIML-intuition |
| :--- | :--- | :--- |
| **1.1 Vector spaces & geometry** | scalars/vectors, span, independence, basis, **norms & distance**, **inner products / orthogonality**, **projections** | [1.01 One-Hot](../../../AIML-intuition/Module_1_Representation/1.01_One-Hot_Encoding.md), [1.06 Dot Product](../../../AIML-intuition/Module_1_Representation/1.06_Vector_Similarities_The_Scaled_Dot-Product.md), [1.07-1.08 Cosine/Euclidean](../../../AIML-intuition/Module_1_Representation/1.07-1.08_Similarities_Distances_Euclidean_vs_Cosine.md), [1.09 Manhattan](../../../AIML-intuition/Module_1_Representation/1.09_Manhattan_L1_Distance.md) |
| **1.2 Matrices as linear maps** | transformations, multiplication, **rank / low-rank**, determinant, inverse | [1.02 Dense Embeddings](../../../AIML-intuition/Module_1_Representation/1.02_Dense_Embeddings.md), [1.04 Graphs](../../../AIML-intuition/Module_1_Representation/1.04_Graph_Representations.md) |
| **1.3 Linear systems & least squares** | row reduction, four fundamental subspaces, **least squares = projection**, pseudoinverse | foundations for 1.05 |
| **1.4 Eigen & spectral thinking** | eigenvalues/vectors, diagonalization, **spectral theorem**, Rayleigh quotient | [1.04 Graph Laplacian](../../../AIML-intuition/Module_1_Representation/1.04_Graph_Representations.md), [1.05 PCA/SVD](../../../AIML-intuition/Module_1_Representation/1.05_Spectral_Methods_PCA_SVD.md) |
| **1.5 Matrix decompositions** | LU, QR, **Cholesky** (covariance), **SVD** | [1.05 PCA/SVD](../../../AIML-intuition/Module_1_Representation/1.05_Spectral_Methods_PCA_SVD.md), [1.10 Mahalanobis](../../../AIML-intuition/Module_1_Representation/1.10_Mahalanobis_Distance.md) |
| **1.6 PCA & spectral methods** | covariance, **PCA**, whitening | [1.05](../../../AIML-intuition/Module_1_Representation/1.05_Spectral_Methods_PCA_SVD.md), [1.11-1.12 t-SNE/UMAP](../../../AIML-intuition/Module_1_Representation/1.11-1.12_Dimensionality_Reduction_for_Representation_t-SNE_UMAP.md) |
| **1.7 Tensor algebra** | tensors, tensor ops, Kronecker product | tensor intuition across all of DL |

### Suggested first pass
1. Watch 3B1B *Essence of Linear Algebra* end-to-end for intuition.
2. Work MIT 18.06 lectures for rows 1.1 → 1.5 with the CS229 review as the applied cheat-sheet.
3. As you finish each row, read the linked AIML-intuition pages for the ML payoff.

**Completion target:** explain why cosine similarity normalizes magnitude, what the
Laplacian's spectrum encodes, and how SVD yields PCA.

---

## Phase 2 — Calculus & Matrix Calculus → Module 2A (Derivatives & Graphs)

**Goal:** own the machinery of differentiation that powers training — from a single partial
derivative up to the Jacobians flowing through autograd.

### Core resource backbone
- **3Blue1Brown** — [Essence of Calculus](https://www.youtube.com/playlist?list=PLZHQObOWTQDMsr9K-rj53DwVRMYO3t5Yr) · [Backpropagation, intuitively](https://www.youtube.com/watch?v=Ilg3gGewQ5U)
- **MIT OCW 18.01/18.02** Single & Multivariable Calculus — [18.02 multivariable](https://ocw.mit.edu/courses/18-02-multivariable-calculus-fall-2007/)
- **Matrix Calculus for Deep Learning** (Parr & Howard) — [explained.ai/matrix-calculus](https://explained.ai/matrix-calculus/)
- **Andrej Karpathy** — [micrograd: spelled-out intro to backpropagation](https://www.youtube.com/watch?v=VMj-3S1tku0) (build autograd from scratch)

### Study order & what each sub-topic unlocks

| Study (in order) | Key sub-topics | → Read in AIML-intuition |
| :--- | :--- | :--- |
| **2.1 Derivatives & sensitivity** | derivative as rate of change, **partial derivatives**, gradient vector | [2.01 Partial Derivatives & Gradient](../../../AIML-intuition/Module_2_Optimization/2.01_Partial_Derivatives_and_the_Gradient.md) |
| **2.2 Chain rule** | composition, **multivariable chain rule**, credit assignment | [2.02 Backpropagation](../../../AIML-intuition/Module_2_Optimization/2.02_Backpropagation_The_Chain_Rule.md) |
| **2.3 Matrix calculus** | **Jacobian**, Hessian & curvature, vector/matrix derivative rules | [2.03 Jacobian & Hessian](../../../AIML-intuition/Module_2_Optimization/2.03_Jacobian_and_Hessian.md) |
| **2.4 Autodiff mechanics** | **computational graphs**, forward vs reverse mode | [2.04 Computational Graphs](../../../AIML-intuition/Module_2_Optimization/2.04_Computational_Graphs_Autograd.md) |

---

## Phase 5 — Optimization for ML/DL → Module 2B (Optimizers)

**Goal:** understand how models actually descend the loss landscape — and why modern
optimizers exist.

### Core resource backbone
- **Stanford CS231n** — [Optimization notes](https://cs231n.github.io/optimization-1/) · [update rules](https://cs231n.github.io/neural-networks-3/)
- **An overview of gradient descent optimization algorithms** (Sebastian Ruder) — [ruder.io/optimizing-gradient-descent](https://www.ruder.io/optimizing-gradient-descent/)
- **3Blue1Brown** — [Gradient descent, how neural networks learn](https://www.youtube.com/watch?v=IHZwWFHWa-w)
- **StatQuest** — [Gradient Descent, step-by-step](https://www.youtube.com/watch?v=sDv4f4s2SB8)

### Study order & what each sub-topic unlocks

| Study (in order) | Key sub-topics | → Read in AIML-intuition |
| :--- | :--- | :--- |
| **5.1 Gradient descent** | batch vs stochastic vs mini-batch, step size, convergence | [2.05 Gradient Descent & SGD](../../../AIML-intuition/Module_2_Optimization/2.05_Gradient_Descent_and_SGD.md) |
| **5.2 Momentum methods** | exponential moving averages, **momentum**, Nesterov | [2.06 SGD with Momentum](../../../AIML-intuition/Module_2_Optimization/2.06_SGD_with_Momentum.md) |
| **5.3 Adaptive optimizers** | per-parameter rates, **Adam**, bias correction | [2.07 Adam](../../../AIML-intuition/Module_2_Optimization/2.07_Adam_Optimizer.md) |
| **5.4 Weight decay done right** | L2 vs decoupled decay, **AdamW** | [2.08 AdamW](../../../AIML-intuition/Module_2_Optimization/2.08_AdamW_Decoupled_Weight_Decay.md) |
| **5.5 Schedules** | warmup, step/cosine decay, cyclical | [2.09 Learning-Rate Schedules](../../../AIML-intuition/Module_2_Optimization/2.09_Learning_Rate_Schedules.md) |
| **5.6 Objective shaping** | L1/L2 penalties, sparsity vs shrinkage, weight decay | [2.10 Regularization (L1/L2)](../../../AIML-intuition/Module_2_Optimization/2.10_Regularization_L1_L2.md) |

**Completion target:** explain why momentum damps oscillation, what Adam's two moments track,
and why AdamW decouples weight decay from the gradient.

---

## Phase 3 — Probability & Information Theory → Module 5 (Generation), parts of Module 3

**Goal:** the language of uncertainty — probability spaces, distributions, expectation,
Bayes, entropy/KL — underpinning probabilistic modeling, generative models, and
information-theoretic losses.

### Core resource backbone
- **Harvard Stat 110** — Joe Blitzstein — [lectures](https://projects.iq.harvard.edu/stat110/home) (the gold-standard probability course)
- **Khan Academy** — [Probability & Statistics](https://www.khanacademy.org/math/statistics-probability)
- **3Blue1Brown** — [Bayes theorem](https://www.youtube.com/watch?v=HZGCoVF3YvM) · [Binomial distributions](https://www.youtube.com/watch?v=8idr1WZ1A7Q)
- **StatQuest** — distributions & information-theory intuition playlists
- **Stanford CS229** — [Probability review notes](https://cs229.stanford.edu/section/cs229-prob.pdf)

### Study order & what each sub-topic unlocks
| Study (in order) | Key sub-topics | → Read in AIML-intuition |
| :--- | :--- | :--- |
| **3.1 Probability foundations** | conditional probability, independence, **Bayes' rule** | groundwork for everything below |
| **3.2 Random variables & distributions** | PMF/PDF/CDF, Gaussian, **multivariate Gaussian** | [1.10 Mahalanobis](../../../AIML-intuition/Module_1_Representation/1.10_Mahalanobis_Distance.md), [3.04 MLE](../../../AIML-intuition/Module_3_Evaluation/3.04_Maximum_Likelihood_Estimation.md) |
| **3.3 Expectation & moments** | expectation, variance/covariance, **Jensen's inequality** | ELBO's derivation in [5.02](../../../AIML-intuition/Module_5_Generation/5.02_Latent_Variable_Models_ELBO_VAEs.md) |
| **3.4 Limit theorems** | law of large numbers, **CLT** | why mini-batch gradients work ([2.05](../../../AIML-intuition/Module_2_Optimization/2.05_Gradient_Descent_and_SGD.md)) |
| **3.5 Information theory** | **entropy, cross-entropy, KL**, mutual information | [5.01 Entropy & KL](../../../AIML-intuition/Module_5_Generation/5.01_Information_Theory_Entropy_KL_Divergence.md), [3.03 Cross-Entropy](../../../AIML-intuition/Module_3_Evaluation/3.03_Categorical_Cross-Entropy_NLL.md) |
| **3.6 Bayesian inference** | priors/posteriors, **MAP vs MLE**, latent variables | [3.04 MLE](../../../AIML-intuition/Module_3_Evaluation/3.04_Maximum_Likelihood_Estimation.md), [5.02 ELBO/VAEs](../../../AIML-intuition/Module_5_Generation/5.02_Latent_Variable_Models_ELBO_VAEs.md) |
| 3.7 Graphical models | Bayes nets, MRFs (overview) | future probabilistic-modeling pages |

---

## Phase 4 — Statistics & Statistical Learning → Module 3 (Evaluation)

**Goal:** how models are fit from data and whether conclusions are reliable — estimation,
inference, regression theory, generalization, and ML evaluation methodology.

### Core resource backbone
- **StatQuest** — [Statistics fundamentals playlist](https://www.youtube.com/playlist?list=PLblh5JKOoLUK0FLuzwntyYI10UQFUhsY9)
- **ISLR** — [An Introduction to Statistical Learning](https://www.statlearning.com/) (free book)
- **Stanford CS229** — [main notes](https://cs229.stanford.edu/main_notes.pdf) (bias-variance, learning theory)
- **MIT OCW 18.650** — Statistics for Applications

### Study order & what each sub-topic unlocks
| Study (in order) | Key sub-topics | → Read in AIML-intuition |
| :--- | :--- | :--- |
| **4.1 Estimation** | bias/variance/MSE decomposition, **MLE**, MAP | [3.01 MSE](../../../AIML-intuition/Module_3_Evaluation/3.01_Mean_Squared_Error_MSE_L2_Loss.md), [3.04 MLE](../../../AIML-intuition/Module_3_Evaluation/3.04_Maximum_Likelihood_Estimation.md) |
| **4.2 Inference** | confidence intervals, hypothesis tests, bootstrap | evaluating model comparisons honestly |
| **4.3 Regression theory** | linear/logistic regression, **regularized regression** | [2.10 Regularization](../../../AIML-intuition/Module_2_Optimization/2.10_Regularization_L1_L2.md) |
| **4.4 Learning theory** | empirical vs population risk, **overfitting/generalization**, VC/PAC (overview) | the *why* behind [2.10](../../../AIML-intuition/Module_2_Optimization/2.10_Regularization_L1_L2.md) and Module 4's recipes |
| **4.5 Evaluation methodology** | train/val/test, **cross-validation**, distribution shift, calibration | experimental hygiene for everything |
| **4.6 Metrics** | regression & classification metrics, **ROC/PR**, proper scoring rules | future Module 3 metric pages |

---

## Phase 6 — Applied Math for Modern DL → Modules 4 & 6 (+ ties everything together)

**Goal:** the capstone — see the earlier phases' math reassembled inside real architectures:
networks, sequence models, transformers, generative models, and scaling behavior.

### Core resource backbone
- **Stanford CS231n** — [CNNs for Visual Recognition](https://cs231n.github.io/)
- **Stanford CS224n** — [NLP with Deep Learning](https://web.stanford.edu/class/cs224n/) (attention, transformers)
- **3Blue1Brown** — [Neural networks playlist](https://www.youtube.com/playlist?list=PLZHQObOWTQDNU6R1_67000Dx_ZCJB-3pi) (incl. the 2024 attention chapters)
- **Karpathy** — [Neural Networks: Zero to Hero](https://www.youtube.com/playlist?list=PLAqhIrjkxbuWI23v9cThsA9GvCAUhRvKZ)
- **Distill.pub** — visual explainers

### Study order & what each sub-topic unlocks
| Study (in order) | Key sub-topics | → Read in AIML-intuition |
| :--- | :--- | :--- |
| **6.1 Network math** | affine maps + nonlinearity, universal approximation, **regularization** | [4.12 Init](../../../AIML-intuition/Module_4_Stabilization/4C_Training_Stability/4.12_Weight_Initialization_Xavier_He.md), [2.10](../../../AIML-intuition/Module_2_Optimization/2.10_Regularization_L1_L2.md) |
| **6.2 Representation learning** | embeddings, latent geometry, **contrastive objectives** | Module 1 ([1.02](../../../AIML-intuition/Module_1_Representation/1.02_Dense_Embeddings.md), [1.13](../../../AIML-intuition/Module_1_Representation/1.13_Representation_Contrastive_Learning_SimCLR_InfoNCE.md), [1.14](../../../AIML-intuition/Module_1_Representation/1.14_Triplet_Loss.md)) |
| **6.3 Sequence math** | autoregressive factorization, RNN dynamics, BPTT, **gating** | [4.07 Gating](../../../AIML-intuition/Module_4_Stabilization/4B_Architectural_Motifs/4.07_Gating_Mechanisms_LSTM_GRU.md) |
| **6.4 Transformer math** | QKV projections, **scaled attention**, multi-head subspaces, positional encoding, residual+norm blocks | [1.06](../../../AIML-intuition/Module_1_Representation/1.06_Vector_Similarities_The_Scaled_Dot-Product.md), [1.03](../../../AIML-intuition/Module_1_Representation/1.03_Positional_Encoding.md), [4.08](../../../AIML-intuition/Module_4_Stabilization/4B_Architectural_Motifs/4.08_Multi-Head_Attention_Routing.md), [4.06](../../../AIML-intuition/Module_4_Stabilization/4B_Architectural_Motifs/4.06_Residual_Skip_Connections.md), [4.02](../../../AIML-intuition/Module_4_Stabilization/4A_Normalization/4.02_Layer_Normalization.md)/[4.05](../../../AIML-intuition/Module_4_Stabilization/4A_Normalization/4.05_RMSNorm.md) |
| **6.5 Generative math** | likelihood models, **ELBO/VAE, diffusion**, flows (overview) | [5.02](../../../AIML-intuition/Module_5_Generation/5.02_Latent_Variable_Models_ELBO_VAEs.md), [5.03](../../../AIML-intuition/Module_5_Generation/5.03_Diffusion_Models.md) |
| **6.6 Scaling behavior** | overparameterization, **scaling laws**, emergence (overview) | future Training-Dynamics module |

---

## Specializations (elective deep-dive indexes)

Retired Q5 specialization syllabi, kept as topic indexes; they get the full treatment if/when
those tracks are pursued:

- **Computer Vision math** — images as signals, convolution & Fourier, projective geometry &
  camera models, SIFT-era features, CNNs→ViTs, segmentation/detection losses, multi-view/3D,
  generative vision. *Backbone: CS231n + Multiple View Geometry (Hartley & Zisserman).*
- **Neuroscience & brain-inspired AI** — dynamical systems, integrate-and-fire &
  Hodgkin-Huxley models, Hebbian/STDP plasticity, neural coding & Bayesian brain, attractor
  networks, spiking/neuromorphic computing, predictive coding vs backprop.
- **Research math** — measure-theoretic probability, concentration inequalities, EM/variational
  inference/MCMC, information geometry, optimal transport/Wasserstein, RKHS & kernels, NTK,
  spectral graph theory & GNNs, implicit bias, score matching & SDE-based generation, causality,
  Bellman operators. *Backbone: Boyd & Vandenberghe; advanced course notes.*

## Other supporting areas
**Graph Theory** (feeds [1.04](../../../AIML-intuition/Module_1_Representation/1.04_Graph_Representations.md)) ·
**Numerical Methods** (floating point → [4.11 FP16 Loss Scaling](../../../AIML-intuition/Module_4_Stabilization/4C_Training_Stability/4.11_Loss_Scaling_FP16.md)) ·
**Fourier transforms** (CV specialization).
