---
id: "07-computer-vision"
topic: "Computer Vision"
level: intermediate
prereqs: ["deep-learning", "linear-algebra"]
updated: 2026-06-27
---

# 🖼️ Computer Vision
> The mathematical and deep-learning spine of vision — images as signals, convolution and
> frequency thinking, projective geometry and cameras, deep vision architectures (CNNs · ViTs),
> detection & segmentation, and generative/self-supervised vision.

## 📑 Concept Index
Every chapter is a self-contained folder (`NN-Concept/NN-Concept.md`) with its page.
> **✅ ready · ⬜ coming soon.** New to vision? Start with the field overview below, then work top to bottom.

### Image processing & classical features
1. ✅ [Image Representation & Filtering (edges · HOG · SIFT)](01-Image-Representation-and-Filtering/01-Image-Representation-and-Filtering.md)
2. ✅ [Pooling & Receptive Fields](02-Pooling-and-Receptive-Fields/02-Pooling-and-Receptive-Fields.md)

### Classification & backbones
3. ✅ [Classic CNN Architectures (LeNet · AlexNet · VGG · ResNet · Inception · EfficientNet)](03-Classic-CNN-Architectures/03-Classic-CNN-Architectures.md)
4. ✅ [Image Classification](04-Image-Classification/04-Image-Classification.md)
5. ✅ [Transfer Learning for Vision](05-Transfer-Learning-for-Vision/05-Transfer-Learning-for-Vision.md)
6. ✅ [Data Augmentation](06-Data-Augmentation/06-Data-Augmentation.md)

### Detection & segmentation
7. ✅ [Object Detection (R-CNN family · YOLO · SSD)](07-Object-Detection/07-Object-Detection.md)
8. ✅ [Semantic Segmentation (FCN · U-Net · DeepLab)](08-Semantic-Segmentation/08-Semantic-Segmentation.md)
9. ✅ [Instance Segmentation (Mask R-CNN)](09-Instance-Segmentation/09-Instance-Segmentation.md)
10. ✅ [Detection & Segmentation Metrics (IoU · mAP)](10-Detection-and-Segmentation-Metrics/10-Detection-and-Segmentation-Metrics.md)

### Modern & specialized vision
11. ✅ [Vision Transformers (ViT)](11-Vision-Transformers/11-Vision-Transformers.md)
12. ✅ [Self-Supervised Vision (SimCLR · MAE · DINO)](12-Self-Supervised-Vision/12-Self-Supervised-Vision.md)
13. ✅ [Pose Estimation](13-Pose-Estimation/13-Pose-Estimation.md)
14. ✅ [Optical Flow & Video Understanding](14-Optical-Flow-and-Video/14-Optical-Flow-and-Video.md)
15. ✅ [Optical Character Recognition (OCR)](15-OCR/15-OCR.md)
16. ✅ [3D & Depth Estimation](16-3D-and-Depth-Estimation/16-3D-and-Depth-Estimation.md)

### Related concepts (covered in another section)
> These topics are foundational or generative and live in their canonical home to avoid repetition.
- **CNNs & Convolution** (the operation, kernels, stride/padding math) → [Deep Learning › CNNs & Convolution](../05.%20Deep_Learning/13-CNNs-and-Convolution/13-CNNs-and-Convolution.md)
- **Residual / skip connections · Batch & Group Norm** (the motifs deep vision backbones rely on) → [Deep Learning](../05.%20Deep_Learning/README.md)
- **Image generation** — GANs · Diffusion · text-to-image → [GenAI](../10.%20GenAI/README.md)
- **Contrastive / embedding math** (InfoNCE, triplet loss reused by self-supervised vision) → [NLP](../06.%20NLP/README.md) · [Deep Learning](../05.%20Deep_Learning/README.md)

## 📐 Mathematics curriculum (specialization)

> Elective deep-dive track, absorbed and expanded from the retired `math-for-AIML-Q5`
> CV specialization. Same format as the [main math curriculum](../01.%20Foundations/Maths%20for%20AI-ML/README.md):
> what to study → why → best resources → which AI-ML-intuition pages it unlocks.

**Goal:** the mathematical spine of vision — images as signals, convolution and frequency
thinking, projective geometry and cameras, deep vision architectures, and generative vision.

### Core resource backbone
- **Stanford CS231n** — [CNNs for Visual Recognition](https://cs231n.github.io/) (the anchor course)
- **Multiple View Geometry** (Hartley & Zisserman) — the geometric-vision bible
- **First Principles of Computer Vision** (Shree Nayar, Columbia) — [YouTube channel](https://www.youtube.com/@firstprinciplesofcomputerv3258) — exceptional visual lectures on classical CV
- **Szeliski** — [Computer Vision: Algorithms and Applications](https://szeliski.org/Book/) (free)

### Study order & what each module unlocks

| Module | Key sub-topics | Best resources | → AI-ML-intuition |
| :--- | :--- | :--- | :--- |
| **V1. Images as signals** | images as functions, sampling & aliasing, noise models, color spaces | Nayar (First Principles): Image Formation playlist; Szeliski ch. 2 | [0.02 Distributions](../../AI-ML-intuition/Module_0_Foundations/0.02_Distributions_and_the_Gaussian.md) (noise) |
| **V2. Filtering & convolution** | convolution as local aggregation, edge detectors, stride/padding/pooling, **Fourier intuition**, multi-scale/wavelets | Nayar: Image Processing; 3B1B [Fourier](https://www.youtube.com/watch?v=spUNpyF58BY) + [convolution](https://www.youtube.com/watch?v=KuXjwB4LzSA) | **[4.13 Convolution](../../AI-ML-intuition/Module_4_Stabilization/4B_Architectural_Motifs/4.13_Convolution.md)** |
| **V3. Geometric vision** | homogeneous coordinates, projective geometry, **camera model**, homographies | Hartley & Zisserman ch. 2–4; Nayar: Imaging playlist | linear maps ([1.05](../../AI-ML-intuition/Module_1_Representation/1.05_Spectral_Methods_PCA_SVD.md) intuitions) |
| **V4. Classical features** | corners/interest points, SIFT/ORB descriptors, optical flow | Nayar: Features playlist; Szeliski ch. 7 | [1.07-1.08 distances](../../AI-ML-intuition/Module_1_Representation/1.07-1.08_Similarities_Distances_Euclidean_vs_Cosine.md) (descriptor matching) |
| **V5. Deep vision** | CNNs as feature hierarchies, classic architectures (AlexNet→ResNet), norms & residuals in vision, transfer learning | CS231n lectures 5–9 | [4.13](../../AI-ML-intuition/Module_4_Stabilization/4B_Architectural_Motifs/4.13_Convolution.md), [4.06 Residuals](../../AI-ML-intuition/Module_4_Stabilization/4B_Architectural_Motifs/4.06_Residual_Skip_Connections.md), [4.03 GroupNorm](../../AI-ML-intuition/Module_4_Stabilization/4A_Normalization/4.03_Group_Normalization.md), [7.03 Transfer](../../AI-ML-intuition/Module_7_Scaling_and_Adaptation/7.03_Transfer_Learning_and_Fine_Tuning.md) |
| **V6. Vision Transformers** | images as patch tokens, self-attention for vision, CNN-ViT hybrids | [ViT paper](https://arxiv.org/abs/2010.11929); CS231n ViT lecture | [4.15 Transformer Block](../../AI-ML-intuition/Module_4_Stabilization/4D_Nonlinearities/4.15_The_Transformer_Block.md), [4.08 MHA](../../AI-ML-intuition/Module_4_Stabilization/4B_Architectural_Motifs/4.08_Multi-Head_Attention_Routing.md) |
| **V7. Detection & segmentation** | bounding-box geometry, IoU/mAP, semantic vs instance segmentation, structured losses | CS231n lecture 11; [Mask R-CNN](https://arxiv.org/abs/1703.06870) | [3.05/3.06 metrics](../../AI-ML-intuition/Module_3_Evaluation/3.06_ROC_AUC_PR_Curves.md) (mAP = AP per class) |
| **V8. 3D & multi-view** | stereo & depth, epipolar geometry, structure-from-motion, (NeRF/Gaussian-splatting overview) | Hartley & Zisserman ch. 9–12; [NeRF](https://arxiv.org/abs/2003.08934) | — (geometry track) |
| **V9. Generative & self-supervised vision** | autoencoders, contrastive pretraining (SimCLR/CLIP), diffusion for images | [CLIP](https://arxiv.org/abs/2103.00020); [SimCLR](https://arxiv.org/abs/2002.05709) | [1.13 Contrastive](../../AI-ML-intuition/Module_1_Representation/1.13_Representation_Contrastive_Learning_SimCLR_InfoNCE.md), [5.02 VAEs](../../AI-ML-intuition/Module_5_Generation/5.02_Latent_Variable_Models_ELBO_VAEs.md), [5.03 Diffusion](../../AI-ML-intuition/Module_5_Generation/5.03_Diffusion_Models.md) |

### Suggested first pass
1. V2 + V5 first if you're deep-learning-bound (convolution → CNNs → ViTs is the modern spine).
2. V1, V3, V4 for the classical foundation (essential for robotics/AR/3D roles; skippable for pure DL roles).
3. V9 last — it reuses half of AI-ML-intuition's Modules 1 and 5.

**Completion target:** explain convolution's two priors and parameter math, walk a pinhole
camera model, justify ViT patch tokenization, and read a diffusion-vision paper without fear.
