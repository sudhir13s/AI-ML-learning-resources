---
id: "07-computer-vision/pose-estimation"
topic: "Pose Estimation"
parent: "07-computer-vision"
level: advanced
prereqs: ["cnns", "object-detection", "semantic-segmentation"]
interview_frequency: medium
updated: 2026-06-20
---

# Pose Estimation
> Localize a set of **keypoints** (joints: wrists, elbows, knees…) for each person in an image, then
> connect them into a skeleton. The dominant approach predicts per-joint **heatmaps** with a CNN;
> multi-person methods are **top-down** (detect people → estimate each one's pose) or **bottom-up**
> (detect all joints → group them, e.g. OpenPose's Part Affinity Fields). 3D pose lifts this to depth.

**Why it matters:** a focused vision question for AR/robotics/sports/health roles — heatmap regression vs
direct coordinate regression, top-down vs bottom-up trade-offs (accuracy vs speed/scaling with crowd
size), how Part Affinity Fields associate joints to people, and how 2D pose is lifted to 3D. It cleanly
combines detection, dense prediction, and structured output.

**⭐ Start here — suggested path:**

1. **Build intuition** — watch ⭐ [Human Pose Estimation Explained (2D & 3D)](https://www.youtube.com/watch?v=_sobpAW16c0). *The task, keypoints, and the 2D→3D distinction.*
2. **See it run** — watch [OpenPose Body Keypoint Detection](https://www.youtube.com/watch?v=kZbptWIXCos). *A real multi-person bottom-up system in action.*
3. **Hands-on** — watch [Human Pose Estimation with Deep Learning (LearnOpenCV)](https://www.youtube.com/watch?v=UCoR-mF3KI8). *Run a heatmap-based estimator end to end.*
4. **Read the sources** — [DeepPose](https://arxiv.org/abs/1312.4659) → [OpenPose (Part Affinity Fields)](https://arxiv.org/abs/1611.08050) → ⭐ [Simple Baselines for Pose Estimation](https://arxiv.org/abs/1804.06208). *Regression → bottom-up association → the strong heatmap baseline.*
5. **Make it concrete** — work through [MediaPipe Pose Landmarker](https://ai.google.dev/edge/mediapipe/solutions/vision/pose_landmarker). *Production-grade real-time pose with code.*

## 🎓 Courses (free)
- [Stanford CS231n](https://cs231n.github.io/) — **Stanford** — pose/keypoint estimation appears within the detection & dense-prediction lectures.
- [MediaPipe Pose Landmarker guide](https://ai.google.dev/edge/mediapipe/solutions/vision/pose_landmarker) — **Google** — a free, hands-on guide to running real-time pose estimation.

## 🎥 Videos
- [Human Pose Estimation Explained (2D & 3D)](https://www.youtube.com/watch?v=_sobpAW16c0) — **What's AI (Louis-François Bouchard)** — the clearest conceptual overview.
- [OpenPose Body Keypoint Detection](https://www.youtube.com/watch?v=kZbptWIXCos) — **David Fombella** — a real bottom-up multi-person system demonstrated.
- [Human Pose Estimation using Deep Learning](https://www.youtube.com/watch?v=UCoR-mF3KI8) — **LearnOpenCV** — a hands-on heatmap-based walkthrough.
- [Detection & Segmentation (CS231n Lec 11)](https://www.youtube.com/watch?v=nDPWywWRIRo) — **Stanford** — the dense-prediction foundations heatmap pose builds on.

## 📄 Key Papers
- [DeepPose](https://arxiv.org/abs/1312.4659) — **Toshev & Szegedy (2014)** — first deep pose method (direct joint-coordinate regression).
- [Realtime Multi-Person 2D Pose Estimation (OpenPose)](https://arxiv.org/abs/1611.08050) — **Cao et al. (2016)** — bottom-up Part Affinity Fields for joint association.
- [Simple Baselines for Human Pose Estimation and Tracking](https://arxiv.org/abs/1804.06208) — **Xiao et al. (2018)** — the strong, minimal heatmap baseline.

## 📰 Articles / Blogs (free, no paywall)
- [OpenPose (official repo + docs)](https://github.com/CMU-Perceptual-Computing-Lab/openpose) — **CMU Perceptual Computing Lab** — reference implementation and explanation, free.
- [MediaPipe Pose Landmarker](https://ai.google.dev/edge/mediapipe/solutions/vision/pose_landmarker) — **Google** — model card, landmarks, and runnable examples.
- [Human Pose Estimation with Deep Learning](https://learnopencv.com/deep-learning-based-human-pose-estimation-using-opencv-cpp-python/) — **LearnOpenCV** — heatmaps and architectures, free.

## 📚 Books (free, with chapters)
- [Computer Vision: Algorithms and Applications, 2nd ed. — **Ch. 6.4 (Pose & keypoint estimation)**](https://szeliski.org/Book/) — **Richard Szeliski** — pose estimation in the recognition landscape, free.
- [Dive into Deep Learning — **Ch. 14 (Computer Vision)**](https://d2l.ai/chapter_computer-vision/index.html) — **Zhang et al.** — the detection/dense-prediction machinery pose builds on, with code.

## 🔗 In this platform
- Foundation: [Object Detection](../07-Object-Detection/07-Object-Detection.md) (top-down pose needs a detector) · [Semantic Segmentation](../08-Semantic-Segmentation/08-Semantic-Segmentation.md) (heatmaps are dense prediction)
- Foundation: [Deep Learning › CNNs & Convolution](../../05.%20Deep_Learning/13-CNNs-and-Convolution/13-CNNs-and-Convolution.md)
- Related: [Optical Flow & Video Understanding](../14-Optical-Flow-and-Video/14-Optical-Flow-and-Video.md) (pose tracking over time)
