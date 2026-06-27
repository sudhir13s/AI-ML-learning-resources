---
id: "05-deep-learning/cnns/references"
topic: "CNNs & Convolution — References"
parent: "05-deep-learning/cnns"
type: references
updated: 2026-06-22
---

# CNNs & Convolution — references and further reading

> Companion link library for **[CNNs & Convolution](13-CNNs-and-Convolution.md)** (the concept page). External sources *and* internal links to related pages on this platform, kept separate so it can be reused as a standalone reference list. Grouped by type, best-first. Every entry is from a primary author or a recognized deep explainer — chosen for depth on *this* topic.

**Start here — suggested path**:
1. **Build intuition** — watch [But what is a convolution?](https://www.youtube.com/watch?v=KuXjwB4LzSA) (**3Blue1Brown**), then play with [CNN Explainer](https://poloclub.github.io/cnn-explainer/). *See the operation, then watch real feature maps light up interactively.*
2. **See why it works** — [How Convolutional Neural Networks work](https://www.youtube.com/watch?v=FmpDIaiMIeA) (**Brandon Rohrer**). *Local patterns, filters, and pooling from the ground up.*
3. **Get the math** — [CS231n notes: Convolutional Networks](https://cs231n.github.io/convolutional-networks/), then the [guide to convolution arithmetic](https://arxiv.org/abs/1603.07285) for stride/pad/dilation/transposed shapes. *Output-size arithmetic, parameter counts, and every variant's geometry.*
4. **Read the sources** — [LeNet](http://yann.lecun.com/exdb/publis/pdf/lecun-01a.pdf) → [AlexNet](https://papers.nips.cc/paper/4824-imagenet-classification-with-deep-convolutional-neural-networks.pdf) → [VGG](https://arxiv.org/abs/1409.1556) → [ResNet](https://arxiv.org/abs/1512.03385). *The progression that made CNNs deep and dominant.*
5. **Make it concrete** — work through [d2l Ch. 7 (CNNs)](https://d2l.ai/chapter_convolutional-neural-networks/index.html). *Implementing conv/pool layers cements the dimensions and gradients.*

**Videos**:
- [But what is a convolution?](https://www.youtube.com/watch?v=KuXjwB4LzSA) — **3Blue1Brown** — the cleanest visual definition of the convolution operation itself.
- [How Convolutional Neural Networks work](https://www.youtube.com/watch?v=FmpDIaiMIeA) — **Brandon Rohrer** — filters, feature maps, and pooling from first principles.
- [CNN: Convolutional Neural Networks Explained](https://www.youtube.com/watch?v=py5byOOHZM8) — **Computerphile** — concise, intuitive overview of why CNNs suit images.
- [Convolutional Neural Networks (CNNs) explained](https://www.youtube.com/watch?v=YRhxdVk_sIs) — **deeplizard** — layer-by-layer walkthrough with clear visuals.

**Interactive & visual**:
- [CNN Explainer](https://poloclub.github.io/cnn-explainer/) — **Georgia Tech (Polo Club)** — an interactive in-browser CNN; hover any neuron to watch the exact patch×kernel sum that produced it, layer by layer.
- [Computing Receptive Fields of CNNs](https://distill.pub/2019/computing-receptive-fields/) — **Distill** — rigorous, visual treatment of receptive fields (a favorite interview follow-up), with the jump/RF recurrence.
- [Deconvolution and Checkerboard Artifacts](https://distill.pub/2016/deconv-checkerboard/) — **Distill (Odena et al.)** — why transposed convolutions produce checkerboard patterns, and the resize-then-convolve fix.

**Courses (free)**:
- [Stanford CS231n — Convolutional Neural Networks for Visual Recognition](https://cs231n.stanford.edu/) — **Stanford (Li / Karpathy / Johnson)** — the definitive CNN course; notes + assignments are the standard reference.
- [Practical Deep Learning for Coders](https://www.fast.ai/) — **fast.ai (Jeremy Howard)** — code-first path to training real CNNs quickly.

**Articles / blogs (free, no paywall)**:
- [CS231n — Convolutional Networks](https://cs231n.github.io/convolutional-networks/) — **Stanford CS231n** — the canonical written reference on conv/pool layer mechanics and sizing.
- [Conv Nets: A Modular Perspective](https://colah.github.io/posts/2014-07-Conv-Nets-Modular/) — **Christopher Olah** — convolution as composable modules; the clearest "why this structure" essay.
- [An Intuitive Guide to Convolutional Neural Networks](https://www.freecodecamp.org/news/an-intuitive-guide-to-convolutional-neural-networks-260c2de0a050/) — **freeCodeCamp (Daphne Cornelisse)** — filters, strides, and pooling explained from scratch with worked numbers.

**Key papers**:
- [Gradient-Based Learning Applied to Document Recognition (LeNet-5)](http://yann.lecun.com/exdb/publis/pdf/lecun-01a.pdf) — **LeCun, Bottou, Bengio & Haffner (1998)** — the original convolutional network and the conv→pool→FC template.
- [ImageNet Classification with Deep CNNs (AlexNet)](https://papers.nips.cc/paper/4824-imagenet-classification-with-deep-convolutional-neural-networks.pdf) — **Krizhevsky, Sutskever & Hinton (2012)** — ReLU + dropout + GPUs; the result that launched the deep-learning era.
- [Very Deep Convolutional Networks (VGG)](https://arxiv.org/abs/1409.1556) — **Simonyan & Zisserman (2014)** — small 3×3 filters stacked deep; a clean, influential design.
- [Going Deeper with Convolutions (GoogLeNet/Inception)](https://arxiv.org/abs/1409.4842) — **Szegedy et al. (2014)** — multi-scale Inception modules, 1×1 bottlenecks, and global average pooling.
- [Deep Residual Learning (ResNet)](https://arxiv.org/abs/1512.03385) — **He et al. (2015)** — skip connections that made *very* deep CNNs trainable.
- [MobileNets: Efficient CNNs for Mobile Vision](https://arxiv.org/abs/1704.04861) — **Howard et al. (2017)** — depthwise-separable convolutions and the $1/C_{out}+1/K^2$ cost reduction.
- [A Guide to Convolution Arithmetic for Deep Learning](https://arxiv.org/abs/1603.07285) — **Dumoulin & Visin (2016)** — the definitive shape reference for stride, padding, dilation, and transposed convolutions (with animations).
- [Network In Network](https://arxiv.org/abs/1312.4400) — **Lin, Chen & Yan (2013)** — origin of the 1×1 convolution and global average pooling.
- [Making Convolutional Networks Shift-Invariant Again](https://arxiv.org/abs/1904.11486) — **Zhang (2019)** — why naive downsampling breaks shift-invariance, and anti-aliased pooling as the fix.
- [An Image is Worth 16×16 Words (ViT)](https://arxiv.org/abs/2010.11929) — **Dosovitskiy et al. (2020)** — Vision Transformers; the data-vs-prior trade against CNNs.

**Books (free chapters)**:
- [Dive into Deep Learning — Ch. 7 "Convolutional Neural Networks" + Ch. 8 (Modern CNNs)](https://d2l.ai/chapter_convolutional-neural-networks/index.html) — **Zhang et al.** — convolution, padding/stride, pooling, and LeNet→ResNet with runnable code.
- [Deep Learning — Ch. 9 "Convolutional Networks"](https://www.deeplearningbook.org/contents/convnets.html) — **Goodfellow, Bengio & Courville** — the rigorous treatment of convolution, pooling, and the priors CNNs encode.

**In this platform**:
- Concept page (full explanation): [CNNs & Convolution](13-CNNs-and-Convolution.md)
- Concept depth (the *why*): [AI-ML-intuition 4.13 Convolution](../../../AI-ML-intuition/Module_4_Stabilization/4B_Architectural_Motifs/4.13_Convolution.md)
- Prerequisite: [Backpropagation & Computational Graphs](../02-Backpropagation-and-Computational-Graphs/02-Backpropagation-and-Computational-Graphs.md)
- Related: [Normalization](../11-Normalization/11-Normalization.md) (BatchNorm lives in CNNs) · [Residual / Skip Connections](../18-Residual-Skip-Connections/18-Residual-Skip-Connections.md) (what made CNNs go deep) · [Vanishing & Exploding Gradients](../06-Vanishing-Exploding-Gradients/06-Vanishing-Exploding-Gradients.md) (the problem residuals solve) · [Attention Mechanism](../15-Attention-Mechanism/15-Attention-Mechanism.md) (the CNN-vs-Transformer trade) · [Computer Vision](../../07.%20Computer%20Vision/concepts/README.md) (vision architectures, detection, segmentation in depth)
- Field overview: [Deep Learning](../README.md)
