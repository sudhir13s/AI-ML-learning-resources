---
id: "07-computer-vision/3d-and-depth-estimation"
topic: "3D & Depth Estimation"
parent: "07-computer-vision"
level: advanced
prereqs: ["cnns", "image-representation-and-filtering", "linear-algebra"]
interview_frequency: medium
updated: 2026-06-20
---

# 3D & Depth Estimation
> Recover the third dimension from images. **Stereo** triangulates depth from two calibrated views via
> **epipolar geometry** and disparity; **monocular depth** estimation predicts a depth map from a single
> image with a CNN (MiDaS, Depth Anything); **structure-from-motion / multi-view** reconstructs 3D from
> many images; and **NeRF** represents a scene as a neural field for photorealistic novel-view synthesis.

**Why it matters:** the 3D-vision question for robotics, AR/VR, and autonomous-driving roles — the
pinhole camera model, why stereo gives metric depth but monocular is scale-ambiguous, how disparity
relates to depth (`depth ∝ 1/disparity`), epipolar geometry, and what NeRF's volumetric rendering does.
It tests whether you can reason geometrically, not just run a CNN.

**⭐ Start here — suggested path:**

1. **Monocular intuition** — watch ⭐ [How Neural Nets estimate depth from 2D images](https://www.youtube.com/watch?v=sz30TDttIBA). *Why a single image can predict depth and where it's ambiguous.*
2. **The 2D vs 3D framing** — read [Depth map](https://en.wikipedia.org/wiki/Depth_map) + [Neural Radiance Field overview](https://en.wikipedia.org/wiki/Neural_radiance_field). *Depth maps, disparity, and the NeRF idea.*
3. **NeRF in depth** — watch [NeRF Explained (Yannic Kilcher)](https://www.youtube.com/watch?v=CRlN-cYFxTk), then [NeRF Paper Explained (Aladdin Persson)](https://www.youtube.com/watch?v=WSfEfZ0ilw4). *Volumetric rendering and novel-view synthesis.*
4. **Read the sources** — [Depth Map Prediction (Eigen)](https://arxiv.org/abs/1406.2283) → [MiDaS](https://arxiv.org/abs/1907.01341) (robust monocular depth) → ⭐ [NeRF](https://arxiv.org/abs/2003.08934). *Deep monocular depth, then neural scene representation.*
5. **Make it concrete** — explore the [NeRF project page](https://www.matthewtancik.com/nerf). *See the results and the rendering pipeline, with code.*

## 🎓 Courses (free)
- [First Principles of Computer Vision — Imaging & Stereo](https://fpcv.cs.columbia.edu/) — **Shree Nayar (Columbia)** — the camera model, epipolar geometry, and stereo, free.
- [Stanford CS231n](https://cs231n.github.io/) — **Stanford** — deep monocular depth and 3D representations appear in the later lectures.

## 🎥 Videos
- [How Neural Nets estimate depth from 2D images](https://www.youtube.com/watch?v=sz30TDttIBA) — **Neural Breakdown with AVB** — monocular depth (MiDaS, Depth Anything), clearly explained.
- [NeRF: Neural Radiance Fields (Paper Explained)](https://www.youtube.com/watch?v=CRlN-cYFxTk) — **Yannic Kilcher** — volumetric rendering and novel-view synthesis from sparse views.
- [NeRFs: Neural Radiance Fields — Paper Explained](https://www.youtube.com/watch?v=WSfEfZ0ilw4) — **Aladdin Persson** — a second, complementary read of the NeRF paper.
- [Overview — Optical Flow](https://www.youtube.com/watch?v=lnXFcmLB7sM) — **First Principles of CV (Shree Nayar)** — motion/geometry foundations underpinning multi-view 3D.

## 📄 Key Papers
- [Depth Map Prediction from a Single Image (Eigen et al.)](https://arxiv.org/abs/1406.2283) — **Eigen et al. (2014)** — the first deep monocular depth network.
- [Towards Robust Monocular Depth Estimation (MiDaS)](https://arxiv.org/abs/1907.01341) — **Ranftl et al. (2019)** — cross-dataset training for robust, transferable depth.
- [NeRF: Representing Scenes as Neural Radiance Fields](https://arxiv.org/abs/2003.08934) — **Mildenhall et al. (2020)** — neural volumetric scene representation; a landmark.

## 📰 Articles / Blogs (free, no paywall)
- [Depth map](https://en.wikipedia.org/wiki/Depth_map) — **Wikipedia** — depth maps, disparity, and the depth–disparity relationship.
- [Neural radiance field](https://en.wikipedia.org/wiki/Neural_radiance_field) — **Wikipedia** — the NeRF representation and volumetric rendering, free.
- [NeRF project page](https://www.matthewtancik.com/nerf) — **Mildenhall et al.** — results, method, and code, fully open.

## 📚 Books (free, with chapters)
- [Computer Vision: Algorithms and Applications, 2nd ed. — **Ch. 11 (Stereo)** + **Ch. 12 (3D reconstruction)**](https://szeliski.org/Book/) — **Richard Szeliski** — the geometric-vision backbone, free.
- [Dive into Deep Learning — **Ch. 14 (Computer Vision)**](https://d2l.ai/chapter_computer-vision/index.html) — **Zhang et al.** — the CNN backbone deep depth models build on, with code.

## 🔗 In this platform
- Concept depth (the *why*): [AI-ML-intuition 1.05 Spectral Methods (PCA / SVD)](../../../AI-ML-intuition/Module_1_Representation/1.05_Spectral_Methods_PCA_SVD.md) (the linear-algebra under projective geometry)
- Foundation: [Image Representation & Filtering](01-Image-Representation-and-Filtering.md) · [Deep Learning › CNNs & Convolution](../../05.%20Deep_Learning/concepts/13-CNNs-and-Convolution.md)
- Related: [Optical Flow & Video Understanding](14-Optical-Flow-and-Video.md) (motion-based geometry) · image generation → [GenAI](../../10.%20GenAI/concepts/README.md)
