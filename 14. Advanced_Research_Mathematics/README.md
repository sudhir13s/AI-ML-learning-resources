# 🔬 Advanced Research Mathematics — Curriculum (Specialization)

> Elective deep-dive track for research-grade ML math, absorbed and expanded from the retired
> `math-for-AIML-Q5` research specialization. This is the *third tier*: study after the
> [main curriculum's](../1.%20Foundations/Maths%20for%20AI-ML/README.md) phases and the corresponding
> [AI-ML-intuition](../../AI-ML-intuition/) modules. Each row names the payoff that justifies it.

### Core resource backbone
- **Convex Optimization** (Boyd & Vandenberghe) — [free book + lectures](https://web.stanford.edu/~boyd/cvxbook/)
- **All of Statistics** (Wasserman) — the compact rigorous bridge
- **Mathematics for Machine Learning** (Deisenroth) — [free](https://mml-book.github.io/) — the on-ramp to everything below
- **Francis Bach's blog** + **Lil'Log** — research-level explainers worth their weight in lectures

## The twelve tracks

| Track | Key sub-topics | Best resources | Why it's worth it (payoff) |
| :--- | :--- | :--- | :--- |
| **R1. Advanced linear algebra** | block/structured matrices, **Kronecker & tensor decompositions**, spectral perturbation, random matrix intuition | Strang *Linear Algebra and Learning from Data*; Kolda & Bader tensor survey | second-order optimizers (K-FAC), model compression beyond [LoRA](../../AI-ML-intuition/Module_7_Scaling_and_Adaptation/7.02_LoRA_Low_Rank_Adaptation.md) |
| **R2. Advanced probability** | measure-theoretic fluency, **concentration inequalities** (Hoeffding/McDiarmid), martingales, **Gaussian processes** | Wasserman ch. 1–5; Vershynin *High-Dimensional Probability* | generalization bounds; GP-based Bayesian optimization; why [0.04](../../AI-ML-intuition/Module_0_Foundations/0.04_Law_of_Large_Numbers_and_CLT.md) has teeth |
| **R3. Advanced inference** | **EM at depth**, variational inference, **MCMC**, modern approximate posteriors | Bishop ch. 9–11; Blei's VI review | the full family behind [5.06 EM](../../AI-ML-intuition/Module_5_Generation/5.06_GMMs_and_EM.md) and [5.02 ELBO](../../AI-ML-intuition/Module_5_Generation/5.02_Latent_Variable_Models_ELBO_VAEs.md) |
| **R4. Information theory at depth** | entropy rates, MI estimation, rate-distortion, **information bottleneck**, MDL, PAC-Bayes | Cover & Thomas; Tishby's IB lectures | representation-learning theory; compression views of generalization |
| **R5. Manifolds & geometry** | manifolds, tangent spaces, Riemannian metrics, **information geometry**, optimization on manifolds | Deisenroth ch. 7+; Absil *Optimization on Matrix Manifolds* | natural gradients; why [t-SNE/UMAP](../../AI-ML-intuition/Module_1_Representation/1.11-1.12_Dimensionality_Reduction_for_Representation_t-SNE_UMAP.md) talk about manifolds |
| **R6. Optimal transport** | Wasserstein distance, Kantorovich duality, **Sinkhorn**, OT in generative modeling | Peyré & Cuturi *Computational OT* (free) | the math under [WGAN](../../AI-ML-intuition/Module_5_Generation/5.04_GANs_and_WGAN.md) and flow matching |
| **R7. Kernels & function spaces** | Hilbert spaces, **RKHS**, kernel regression/GPs, **neural tangent kernel** | Schölkopf & Smola; the NTK paper | infinite-width theory; the deep extension of [1.16](../../AI-ML-intuition/Module_1_Representation/1.16_The_Kernel_Trick.md) |
| **R8. Spectral graphs & geometric DL** | graph Laplacian spectra, graph signal processing, **message passing/GNNs**, equivariance & symmetry | Bronstein's *Geometric Deep Learning* (free proto-book + lectures) | GNNs; the symmetry lens that unifies [convolution](../../AI-ML-intuition/Module_4_Stabilization/4B_Architectural_Motifs/4.13_Convolution.md), attention, and graphs |
| **R9. Advanced optimization** | convex analysis, proximal methods, mirror descent, min-max/saddle problems, **implicit bias of GD** | Boyd & Vandenberghe; Bach's blog | why SGD finds *generalizing* minima — the field's deepest open question |
| **R10. Transformer theory** | attention as kernel similarity, low-rank attention views, expressivity limits, scaling-law theory | *Transformer Circuits* (Anthropic); Tay et al. efficiency survey | mechanistic interpretability; principled architecture work beyond [4.15](../../AI-ML-intuition/Module_4_Stabilization/4D_Nonlinearities/4.15_The_Transformer_Block.md) |
| **R11. Generative-model theory** | **score matching, SDE/probability-flow views of diffusion**, flow matching, energy-based models | Song's score-SDE paper + blog; Lipman's flow-matching | where [5.03 diffusion](../../AI-ML-intuition/Module_5_Generation/5.03_Diffusion_Models.md) research actually lives |
| **R12. Causality & decision theory** | structural causal models, counterfactuals, **bandit theory**, Bellman operators, control links | Pearl *Causality* / *Book of Why*; Lattimore & Szepesvári *Bandit Algorithms* (free) | causal ML; the rigorous backbone under [Module 6](../../AI-ML-intuition/Module_6_Reinforcement_Learning/) |

### How to use this track
- **Don't read it linearly.** Pick the track your current work touches (building GNNs → R8;
  diffusion research → R11; theory-flavored interviews → R2 + R9).
- Each track is *months*, not weeks — treat a track like a graduate seminar: anchor text +
  3–4 key papers + one implementation.
- The highest-leverage single track for most ML engineers: **R2 (concentration)** — it turns
  "the model seems to generalize" into statements with error bars.
