# 🌐 AI-ML Learning Resources

> A **curated library of the best free resources on the internet** for learning AI / ML / DL —
> not a link dump. For each topic, a short shortlist of *the* best courses, videos, papers,
> articles, and books, chosen for **authority and clarity** (top institutions, researchers, and
> the people who explained it best), and **free / open** wherever possible.

## The curation bar
- **Authoritative** — Stanford/MIT/DeepMind/Anthropic, seminal-paper authors, 3Blue1Brown, Karpathy, Olah, StatQuest… the best explainer, not the top SEO result.
- **Free / open** preferred (audit links, free books, open courses).
- **~2 per type** — a couple of courses, videos, papers, articles, books. Curated, not exhaustive.
- Every entry says **who made it and why it's the best**.

## Structure (one pattern, everywhere)
The tree is the chartered one: **section → sub-area → topic package**, all kebab-case, with
`course.yaml` declaring the sections and a `metadata.yaml` (`page-order:`) fixing the reading
order inside every sub-area. A topic package holds its teaching page, its
`*.references.md` companion, and its own `images/`, `code/` and `notebooks/` assets. Sub-areas
keep a single **`README.md`** — the curated resource index, with YAML frontmatter
(`id`, `topic`, `level`, `prereqs`) and sections for Courses / Videos / Papers / Articles /
Books. *This consistent, parseable format lets the library double as a dataset for the
interview-prep project.*

## 🗺️ Sections

### [Foundations](foundations/README.md)
| Sub-area | Level |
| :--- | :--- |
| [AI/ML Orientation](foundations/README.md) | beginner |
| [Programming and Data Foundations](foundations/README.md) | beginner |
| [Mathematical Foundations](foundations/mathematical-foundations/README.md) — incl. the [full math curriculum](foundations/mathematical-foundations/maths-for-ai-ml/README.md) | beginner |
| [Data Preparation](foundations/data-preparation/README.md) | beginner |
| [Tools and Frameworks](foundations/tools-and-frameworks/README.md) | beginner |
| [Research Literacy](foundations/research-literacy/README.md) | advanced |

### Core Machine Learning
| Sub-area | Level |
| :--- | :--- |
| [Supervised Learning](core-machine-learning/supervised-learning/README.md) | intermediate |
| [Unsupervised Learning](core-machine-learning/unsupervised-learning/README.md) | intermediate |
| [Reinforcement Learning](core-machine-learning/reinforcement-learning/README.md) | advanced |
| Model Selection and Evaluation | intermediate |

### [Deep Learning](deep-learning/README.md)
Neural-network foundations · optimization and training · stabilization and architectural
blocks · neural architectures · attention and transformers · self-supervised learning.

### Modalities and Generative Models
| Sub-area | Level |
| :--- | :--- |
| [Natural Language Processing](modalities-and-generative-models/natural-language-processing/README.md) | intermediate |
| [Computer Vision](modalities-and-generative-models/computer-vision/README.md) | intermediate |
| [Generative Models](modalities-and-generative-models/generative-models/README.md) | advanced |
| [Diffusion Models](modalities-and-generative-models/diffusion-models/README.md) | advanced |
| [Multimodal Learning](modalities-and-generative-models/multimodal-learning/README.md) | advanced |
| [Video Understanding](modalities-and-generative-models/video-understanding/README.md) | advanced |
| [Audio and Speech](modalities-and-generative-models/audio-and-speech/README.md) | advanced |

### LLMs, Applications and Agents
| Sub-area | Level |
| :--- | :--- |
| [RAG and Knowledge Systems](llms-applications-and-agents/rag-and-knowledge-systems/overview.md) | advanced |
| [Agentic AI](llms-applications-and-agents/agentic-ai/overview.md) | advanced |
| [Large Language Models](09.%20LLMs/README.md) — still on its legacy folder name; re-homed by the LLM wave | advanced |

### [Deployment and MLOps](deployment-and-mlops/README.md)
Lifecycle and reproducibility · data and training platforms · packaging and serving ·
release and deployment · monitoring and reliability · governance and economics.

### Specialized Studies
| Sub-area | Level |
| :--- | :--- |
| [Advanced Mathematics for AI Research](specialized-studies/advanced-mathematics-for-ai-research/README.md) | advanced |
| [Neuroscience and Brain-Inspired AI](specialized-studies/neuroscience-and-brain-inspired-ai/README.md) | advanced |

### Specializations (deep-dive curricula)
- [Computer Vision math](modalities-and-generative-models/computer-vision/README.md) · [Neuroscience & Brain-Inspired AI](specialized-studies/neuroscience-and-brain-inspired-ai/README.md) · [Advanced Research Math](specialized-studies/advanced-mathematics-for-ai-research/README.md) — full what/why/resources curricula
- [LLM Systems Engineering curriculum](llm_systems_curriculum.md) — 14-chapter inference-stack syllabus

## 🔗 Sibling projects
- [AI-ML-intuition](../AI-ML-intuition/) — deep concept pages (the *why*)
- [AI-ML-problemsets](../AI-ML-problemsets/) — problems, labs & build projects (the *practice*)
- [PyTorch-fundamental-notes](../PyTorch-fundamental-notes/) — PyTorch mechanics
