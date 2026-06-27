---
id: "07-computer-vision/image-representation-and-filtering"
topic: "Image Representation & Filtering (edges, HOG, SIFT)"
parent: "07-computer-vision"
level: beginner
prereqs: ["linear-algebra", "convolution", "gradients"]
interview_frequency: medium
updated: 2026-06-20
---

# Image Representation & Filtering — Edges · HOG · SIFT
> An image is a grid of pixel intensities (a function `I(x, y)`); **filtering** convolves it with a small
> kernel to smooth, sharpen, or measure gradients, and **hand-crafted features** (edges, HOG, SIFT)
> summarize those gradients into descriptors that are robust to lighting, scale, and small shifts.
> This is the pre-deep-learning vision pipeline — and the intuition every CNN feature map inherits.

**Why it matters:** before you can explain *why* a CNN's first layer learns edge detectors, you need to
know what Sobel/Canny edges, the HOG descriptor, and SIFT keypoints actually compute. Interviewers use
these to test whether you understand convolution, image gradients, scale/rotation invariance, and the
classical baselines (HOG + SVM for detection, SIFT for matching) that deep learning replaced.

**⭐ Start here — suggested path:**

1. **See an image as a function** — watch [Computerphile: Finding the Edges (Sobel)](https://www.youtube.com/watch?v=uihBwtPIBxM). *Grounds the whole topic: gradients = differences between neighboring pixels.*
2. **Learn filtering & convolution** — read [CS231n: Image Classification notes](https://cs231n.github.io/) intro + watch [Image Filtering, Convolution, Edge Detection (Shah)](https://www.youtube.com/watch?v=Q7aGsfUxXL4). *Blur, sharpen, and gradient kernels are all the same operation.*
3. **Get edges right** — watch [Canny Edge Detector (Nayar)](https://www.youtube.com/watch?v=hUC1uoigH6s). *The canonical edge pipeline: gradient → non-max suppression → hysteresis.*
4. **Hand-crafted descriptors** — watch [HOG (UCF)](https://www.youtube.com/watch?v=0Zib1YEE4LU) then [SIFT (UCF)](https://www.youtube.com/watch?v=NPcMS49V5hg), and read the ⭐ [SIFT paper](https://www.cs.ubc.ca/~lowe/papers/ijcv04.pdf). *How gradients become scale/rotation-invariant features for detection and matching.*
5. **Make it concrete** — run the [OpenCV smoothing/filtering tutorial](https://docs.opencv.org/4.x/d4/d13/tutorial_py_filtering.html). *Implementing blur → Sobel → Canny on a real image cements it.*

## 🎓 Courses (free)
- [Stanford CS231n](https://cs231n.github.io/) — **Stanford (Li, Karpathy, Johnson)** — opens with the classical image-classification pipeline (pixels, features) before CNNs; the anchor course.
- [First Principles of Computer Vision](https://fpcv.cs.columbia.edu/) — **Shree Nayar (Columbia)** — exceptional, fully free lecture series on image formation, filtering, edges, and features.

## 🎥 Videos
- [Finding the Edges (Sobel Operator)](https://www.youtube.com/watch?v=uihBwtPIBxM) — **Computerphile (Mike Pound)** — the clearest intro to image gradients and edge kernels.
- [Image Filtering, Convolution, Edge Detection](https://www.youtube.com/watch?v=Q7aGsfUxXL4) — **Mubarak Shah (UCF)** — university lecture tying filtering and convolution to edges.
- [Canny Edge Detector](https://www.youtube.com/watch?v=hUC1uoigH6s) — **First Principles of CV (Shree Nayar)** — the full Canny pipeline, beautifully visualized.
- [Histograms of Oriented Gradients (HOG)](https://www.youtube.com/watch?v=0Zib1YEE4LU) — **UCF CRCV** — how gradient histograms become a detection descriptor.
- [Scale-Invariant Feature Transform (SIFT)](https://www.youtube.com/watch?v=NPcMS49V5hg) — **UCF CRCV** — keypoints, scale-space, and descriptors for matching.

## 📄 Key Papers
- [Distinctive Image Features from Scale-Invariant Keypoints (SIFT)](https://www.cs.ubc.ca/~lowe/papers/ijcv04.pdf) — **David Lowe (2004)** — the landmark scale/rotation-invariant feature; still the reference.
- [A Performance Evaluation of Local Descriptors](https://www.robots.ox.ac.uk/~vgg/research/affine/det_eval_files/mikolajczyk_pami2004.pdf) — **Mikolajczyk & Schmid (2005)** — the benchmark that compared SIFT against alternatives.

## 📰 Articles / Blogs (free, no paywall)
- [Canny Edge Detector](https://en.wikipedia.org/wiki/Canny_edge_detector) — **Wikipedia** — concise, correct walkthrough of the multi-stage edge algorithm.
- [Scale-Invariant Feature Transform](https://en.wikipedia.org/wiki/Scale-invariant_feature_transform) — **Wikipedia** — full SIFT pipeline (DoG, keypoints, orientation, descriptor) in one page.
- [Sobel Edge Detector](https://homepages.inf.ed.ac.uk/rbf/HIPR2/sobel.htm) — **Univ. of Edinburgh (HIPR2)** — classic teaching page with the kernels and worked examples.
- [Smoothing & Filtering Images](https://docs.opencv.org/4.x/d4/d13/tutorial_py_filtering.html) — **OpenCV** — hands-on blur/sharpen/gradient filters in code.

## 📚 Books (free, with chapters)
- [Computer Vision: Algorithms and Applications, 2nd ed. — **Ch. 3 (Image processing)** + **Ch. 7 (Feature detection)**](https://szeliski.org/Book/) — **Richard Szeliski** — the field's free standard reference; filtering, edges, and SIFT in depth.

## 🔗 In this platform
- Concept depth (the *why*): [AI-ML-intuition 4.13 Convolution](../../../AI-ML-intuition/Module_4_Stabilization/4B_Architectural_Motifs/4.13_Convolution.md) — filtering *is* convolution; CNNs learn these kernels.
- Foundation: [Deep Learning › CNNs & Convolution](../../05.%20Deep_Learning/13-CNNs-and-Convolution/13-CNNs-and-Convolution.md)
- Next concepts: [02 Pooling & Receptive Fields](../02-Pooling-and-Receptive-Fields/02-Pooling-and-Receptive-Fields.md) · [03 Classic CNN Architectures](../03-Classic-CNN-Architectures/03-Classic-CNN-Architectures.md)
