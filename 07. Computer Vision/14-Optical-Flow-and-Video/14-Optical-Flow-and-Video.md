---
id: "07-computer-vision/optical-flow-and-video"
topic: "Optical Flow & Video Understanding"
parent: "07-computer-vision"
level: advanced
prereqs: ["image-representation-and-filtering", "cnns"]
interview_frequency: medium
updated: 2026-06-20
---

# Optical Flow & Video Understanding
> **Optical flow** is the per-pixel motion field between two frames — how each pixel moved — estimated
> classically (Lucas–Kanade, Horn–Schunck) under the brightness-constancy assumption, or with deep nets
> (FlowNet, RAFT). **Video understanding** adds the time axis: action recognition with two-stream
> (appearance + flow) or 3D convolutional networks (I3D) that learn spatiotemporal features.

**Why it matters:** the temporal-vision question — state the brightness-constancy + small-motion
assumptions and the aperture problem, contrast sparse (Lucas–Kanade) vs dense flow, explain how flow
feeds video models (two-stream, I3D), and why modeling time is fundamentally harder than single images.
Relevant for video, robotics, AR, and autonomous-driving roles.

**⭐ Start here — suggested path:**

1. **What is optical flow** — watch ⭐ [Overview — Optical Flow (Nayar)](https://www.youtube.com/watch?v=lnXFcmLB7sM). *Motion fields and the brightness-constancy idea.*
2. **The classical method** — watch [Lucas–Kanade Method (Nayar)](https://www.youtube.com/watch?v=6wMoHgpVUn8). *The sparse-flow workhorse and the aperture problem.*
3. **Go coarse-to-fine** — watch [Hierarchical Lucas–Kanade](https://www.youtube.com/watch?v=i03K_tOwtZ8). *Pyramids handle large motion — the practical trick.*
4. **Read the sources** — [FlowNet](https://arxiv.org/abs/1504.06852) → ⭐ [RAFT](https://arxiv.org/abs/2003.12039) (deep flow), then [Two-Stream](https://arxiv.org/abs/1406.2199) → [I3D](https://arxiv.org/abs/1705.07750) (video action recognition). *Deep flow, then how flow feeds video models.*
5. **Make it concrete** — run the [OpenCV optical-flow tutorial](https://docs.opencv.org/4.x/d4/dee/tutorial_optical_flow.html) or [torchvision RAFT example](https://pytorch.org/vision/main/auto_examples/others/plot_optical_flow.html). *Compute and visualize flow on real video.*

## 🎓 Courses (free)
- [First Principles of Computer Vision — Optical Flow](https://fpcv.cs.columbia.edu/) — **Shree Nayar (Columbia)** — the definitive free lecture series on flow (overview, Lucas–Kanade, coarse-to-fine).
- [Stanford CS231n](https://cs231n.github.io/) — **Stanford** — video understanding and spatiotemporal models appear in the later lectures.

## 🎥 Videos
- [Overview — Optical Flow](https://www.youtube.com/watch?v=lnXFcmLB7sM) — **First Principles of CV (Shree Nayar)** — motion fields and brightness constancy, clearly.
- [Lucas–Kanade Method](https://www.youtube.com/watch?v=6wMoHgpVUn8) — **First Principles of CV (Shree Nayar)** — the classic sparse-flow algorithm and the aperture problem.
- [Hierarchical Lucas–Kanade Optical Flow](https://www.youtube.com/watch?v=i03K_tOwtZ8) — **Gorthi Subrahmanyam** — coarse-to-fine pyramids for large motion.
- [Detection & Segmentation (CS231n Lec 11)](https://www.youtube.com/watch?v=nDPWywWRIRo) — **Stanford** — the CNN foundations deep flow and video models build on.

## 📄 Key Papers
- [FlowNet: Learning Optical Flow with CNNs](https://arxiv.org/abs/1504.06852) — **Dosovitskiy et al. (2015)** — the first end-to-end deep optical-flow network.
- [RAFT: Recurrent All-Pairs Field Transforms](https://arxiv.org/abs/2003.12039) — **Teed & Deng (2020)** — the modern, accurate deep flow architecture.
- [Two-Stream Convolutional Networks for Action Recognition](https://arxiv.org/abs/1406.2199) — **Simonyan & Zisserman (2014)** — appearance + motion streams for video.
- [Quo Vadis, Action Recognition? (I3D)](https://arxiv.org/abs/1705.07750) — **Carreira & Zisserman (2017)** — inflated 3D convolutions for spatiotemporal features.

## 📰 Articles / Blogs (free, no paywall)
- [Optical flow](https://en.wikipedia.org/wiki/Optical_flow) — **Wikipedia** — brightness constancy, the aperture problem, and methods in one page.
- [Optical Flow tutorial](https://docs.opencv.org/4.x/d4/dee/tutorial_optical_flow.html) — **OpenCV** — sparse and dense flow with runnable code.
- [Optical Flow with RAFT (torchvision)](https://pytorch.org/vision/main/auto_examples/others/plot_optical_flow.html) — **PyTorch** — compute and visualize deep flow on video, free.

## 📚 Books (free, with chapters)
- [Computer Vision: Algorithms and Applications, 2nd ed. — **Ch. 9 (Motion estimation)**](https://szeliski.org/Book/) — **Richard Szeliski** — optical flow and video motion, free.
- [Dive into Deep Learning — **Ch. 14 (Computer Vision)**](https://d2l.ai/chapter_computer-vision/index.html) — **Zhang et al.** — the CNN backbone video models extend, with code.

## 🔗 In this platform
- Foundation: [Image Representation & Filtering](01-Image-Representation-and-Filtering.md) (flow uses image gradients) · [Deep Learning › CNNs & Convolution](../../05.%20Deep_Learning/concepts/13-CNNs-and-Convolution.md)
- Related: [Pose Estimation](13-Pose-Estimation.md) (pose tracking over frames) · [3D & Depth Estimation](16-3D-and-Depth-Estimation.md)
