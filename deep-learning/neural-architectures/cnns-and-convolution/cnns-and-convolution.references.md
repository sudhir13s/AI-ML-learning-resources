---
id: "05-deep-learning/cnns/references"
topic: "CNNs & Convolution — References"
parent: "05-deep-learning/cnns"
type: references
updated: 2026-07-03
---

# CNNs & Convolution — references and further reading

> Companion link library for **[CNNs & Convolution](cnns-and-convolution.md)** (the concept page). External sources *and* internal links to related pages on this platform, kept separate so it can be reused as a standalone reference list. Grouped by type, best-first. Every entry is free / open (no paywall) and from a primary author or a recognized deep explainer — chosen for depth on *this* topic, not popularity.

**Start here — suggested path**:
1. **Build intuition** — watch [But what is a convolution?](https://www.youtube.com/watch?v=KuXjwB4LzSA) (**3Blue1Brown**), then play with [CNN Explainer](https://poloclub.github.io/cnn-explainer/). *See the operation, then watch real feature maps light up interactively.*
2. **See it step by step** — [Image Classification with Convolutional Neural Networks](https://www.youtube.com/watch?v=HGwBXDKFk9I) (**StatQuest, Josh Starmer**). *Filters and pooling, one gentle step at a time.*
3. **Get the math** — [CS231n notes: Convolutional Networks](https://cs231n.github.io/convolutional-networks/), then the [guide to convolution arithmetic](https://arxiv.org/abs/1603.07285) for stride/pad/dilation/transposed shapes. *Output-size arithmetic, parameter counts, and every variant's geometry.*
4. **Read the sources** — [LeNet](http://yann.lecun.com/exdb/publis/pdf/lecun-01a.pdf) → [AlexNet](https://papers.nips.cc/paper/4824-imagenet-classification-with-deep-convolutional-neural-networks.pdf) → [VGG](https://arxiv.org/abs/1409.1556) → [ResNet](https://arxiv.org/abs/1512.03385). *The progression that made CNNs deep and dominant.*
5. **Make it concrete** — run this chapter's [notebook](code/cnns-and-convolution.ipynb) (from-scratch conv forward *and* backward, verified), then work through [d2l Ch. 7 (CNNs)](https://d2l.ai/chapter_convolutional-neural-networks/index.html). *Implementing conv/pool layers cements the dimensions and gradients.*

**Videos**:
- [But what is a convolution?](https://www.youtube.com/watch?v=KuXjwB4LzSA) — **3Blue1Brown** — the cleanest visual definition of the convolution operation itself.
- [Image Classification with Convolutional Neural Networks (CNNs)](https://www.youtube.com/watch?v=HGwBXDKFk9I) — **StatQuest (Josh Starmer)** — filters, feature maps, and pooling walked through one step at a time.
- [CS231n (Spring 2017) — Lecture 5: Convolutional Neural Networks](https://www.youtube.com/playlist?list=PL3FW7Lu3i5JvHM8ljYj-zLfQRF3EO8sYv) — **Stanford (Li / Johnson / Yeung, course co-created by Karpathy)** — the definitive lecture on conv/pool mechanics, sizing, and history.
- [Neural Networks: Zero to Hero — building backprop from scratch](https://www.youtube.com/watch?v=VMj-3S1tku0) — **Andrej Karpathy** — the reverse-mode autodiff a conv layer's backward pass plugs into, built by hand (the foundation for "backprop through a convolution").
- [How Convolutional Neural Networks work](https://www.youtube.com/watch?v=FmpDIaiMIeA) — **Brandon Rohrer** — filters, feature maps, and pooling from first principles.
- [CNN: Convolutional Neural Networks Explained](https://www.youtube.com/watch?v=py5byOOHZM8) — **Computerphile** — concise, intuitive overview of why CNNs suit images.

**Interactive & visual**:
- [CNN Explainer](https://poloclub.github.io/cnn-explainer/) — **Georgia Tech (Polo Club)** — an interactive in-browser CNN; hover any neuron to watch the exact patch×kernel sum that produced it, layer by layer.
- [Computing Receptive Fields of CNNs](https://distill.pub/2019/computing-receptive-fields/) — **Distill** — rigorous, visual treatment of receptive fields (a favorite interview follow-up), with the jump/RF recurrence.
- [Deconvolution and Checkerboard Artifacts](https://distill.pub/2016/deconv-checkerboard/) — **Distill (Odena et al.)** — why transposed convolutions produce checkerboard patterns, and the resize-then-convolve fix.

**Courses (free)**:
- [Stanford CS231n — Deep Learning for Computer Vision](https://cs231n.github.io/) — **Stanford (Li / Karpathy / Johnson)** — the definitive CNN course; notes + assignments are the standard reference.
- [Practical Deep Learning for Coders](https://www.fast.ai/) — **fast.ai (Jeremy Howard)** — code-first path to training real CNNs quickly.

**Articles / blogs (free, no paywall)**:
- [CS231n — Convolutional Networks](https://cs231n.github.io/convolutional-networks/) — **Stanford CS231n** — the canonical written reference on conv/pool layer mechanics and sizing.
- [Conv Nets: A Modular Perspective](https://colah.github.io/posts/2014-07-Conv-Nets-Modular/) — **Christopher Olah** — convolution as composable modules; the clearest "why this structure" essay.
- [An Intuitive Guide to Convolutional Neural Networks](https://www.freecodecamp.org/news/an-intuitive-guide-to-convolutional-neural-networks-260c2de0a050/) — **freeCodeCamp (Daphne Cornelisse)** — filters, strides, and pooling explained from scratch with worked numbers.

**Key papers**:
- [Backpropagation Applied to Handwritten Zip Code Recognition](http://yann.lecun.com/exdb/publis/pdf/lecun-89e.pdf) — **LeCun, Boser, Denker et al. (1989)** — the first end-to-end trained convolutional network (weight sharing + backprop on digit images), the direct ancestor of every CNN.
- [Gradient-Based Learning Applied to Document Recognition (LeNet-5)](http://yann.lecun.com/exdb/publis/pdf/lecun-01a.pdf) — **LeCun, Bottou, Bengio & Haffner (1998)** — the definitive LeNet paper and the conv→pool→FC template.
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
- [Deep Learning — Ch. 9 "Convolutional Networks"](https://www.deeplearningbook.org/contents/convnets.html) — **Goodfellow, Bengio & Courville** — the rigorous treatment of convolution, pooling, and the priors CNNs encode (the source for this page's derivations).

**In this platform**:
- Concept page (full explanation): [CNNs & Convolution](cnns-and-convolution.md)
- Runnable code: [the from-scratch convolution module `cnn.py`](code/cnn.py) · [the step-by-step notebook](code/cnns-and-convolution.ipynb) — the module implements conv forward *and* backward, gradient-checks and autograd-checks them, cross-checks against `scipy`/`torch`, and trains a CNN against a larger MLP on real digits; the notebook runs it one measurement at a time.
- Concept depth (the *why*): [AI-ML-intuition 4.13 Convolution](../../../../AI-ML-intuition/architectural-mechanisms/locality-and-weight-sharing/convolution-intuition.md)
- The backward pass a conv plugs into: [02 Backpropagation & Computational Graphs](../../neural-network-foundations/backpropagation-and-computational-graphs/backpropagation-and-computational-graphs.md) — a conv layer's backward pass *is* backprop; all three of its gradients are themselves convolutions (the matmul VJP, specialized to a weight-shared sparse matrix)
- The dense layer a conv replaces: [01 Perceptron & MLP](../../neural-network-foundations/perceptron-and-mlp/perceptron-and-mlp.md)
- Normalization in conv blocks: [11 Normalization](../../stabilization-and-architectural-blocks/normalization/normalization.md) (BatchNorm normalizes per channel over batch+space — natural for weight-shared kernels)
- What made CNNs go deep: [18 Residual / Skip Connections](../../stabilization-and-architectural-blocks/residual-skip-connections/residual-skip-connections.md) (ResNet) · the problem they solve: [06 Vanishing & Exploding Gradients](../../optimization-and-training/vanishing-exploding-gradients/vanishing-exploding-gradients.md)
- The CNN-vs-Transformer trade: [15 Attention Mechanism](../../attention-and-transformers/attention-mechanism/attention-mechanism.md) · forward to [09 LLMs](../../../09.%20LLMs/README.md) (ViT and multimodal models)
- Where CNNs go next: [Computer Vision](../../../modalities-and-generative-models/computer-vision/README.md) (detection, segmentation, modern backbones) · biological inspiration: [Neuroscience & Brain-Inspired AI](../../../specialized-studies/neuroscience-and-brain-inspired-ai/README.md) (Hubel & Wiesel's edge-selective cells → CNN first-layer filters)
- Field overview: [Deep Learning](../../README.md)
