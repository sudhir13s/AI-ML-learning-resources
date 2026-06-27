---
id: "13-tools-and-frameworks/gradio-streamlit"
topic: "Gradio & Streamlit (interactive ML demos & apps)"
parent: "13-tools-and-frameworks"
level: beginner
prereqs: ["python"]
interview_frequency: low
updated: 2026-06-20
---

# Gradio & Streamlit — Interactive ML Demos & Apps
> Two Python libraries that turn a model or script into a shareable web app with no front-end code.
> **Gradio** wraps any function in an input/output UI in a few lines (and powers Hugging Face
> Spaces) — perfect for model demos. **Streamlit** turns a Python script into a reactive data app
> with widgets, charts, and layout — perfect for dashboards and data tools.

**Why it matters:** being able to ship a demo is how ML work gets seen and used. Gradio is the
fastest path from model to an interactive, shareable demo (and the standard for HF Spaces);
Streamlit is the go-to for data dashboards. Both are common in portfolios and increasingly expected
for communicating results — knowing when to reach for each is the practical skill.

**⭐ Start here — suggested path:**

1. **Ship a demo in 5 lines** — read the [Gradio Quickstart](https://www.gradio.app/guides/quickstart). *Wrap a function in `gr.Interface` and you have a live UI immediately.*
2. **Watch it built** — [Gradio Crash Course](https://www.youtube.com/watch?v=eE7CamOE-PA) (AssemblyAI). *Components, inputs/outputs, and sharing in one sitting.*
3. **Build a data app** — work the [Streamlit Get Started](https://docs.streamlit.io/get-started). *The script-reruns-on-interaction model is Streamlit's whole mental model.*
4. **See Streamlit end to end** — [Streamlit Crash Course: From Zero to Data App](https://www.youtube.com/watch?v=d7fnzDQ5qM8) (Streamlit). *Widgets, layout, caching, and deploy.*
5. **Choose the right tool** — Gradio for model demos / HF Spaces, Streamlit for dashboards; browse the [Gradio guides](https://www.gradio.app/guides) and [Streamlit gallery](https://streamlit.io/gallery). *Seeing real examples makes the choice obvious.*

## 🎓 Courses (free)
- [Gradio guides & docs](https://www.gradio.app/guides) — **Gradio (Hugging Face)** — the authoritative, structured guides from quickstart to advanced.
- [Streamlit documentation](https://docs.streamlit.io/) — **Streamlit (Snowflake)** — the official guide: get started, API reference, and deployment.

## 🎥 Videos
- [Gradio Crash Course — Fastest way to build & share ML apps](https://www.youtube.com/watch?v=eE7CamOE-PA) — **AssemblyAI** — the complete Gradio intro.
- [Gradio in Python Crash Course](https://www.youtube.com/watch?v=X4R1KIjcCQk) — **Prof. Reza** — an additional hands-on Gradio walkthrough.
- [Streamlit Crash Course: From Zero to Data App](https://www.youtube.com/watch?v=d7fnzDQ5qM8) — **Streamlit** — the official end-to-end crash course.
- [How to Build a Streamlit App (Beginner tutorial)](https://www.youtube.com/watch?v=-IM3531b1XU) — **Mısra Turp** — a friendly first Streamlit app.

## 📄 Key Papers
- [Gradio: Hassle-Free Sharing and Testing of ML Models in the Wild](https://arxiv.org/abs/1906.02569) — **Abid et al. (2019)** — the foundational paper introducing Gradio (free on arXiv).
- [Streamlit documentation](https://docs.streamlit.io/) — **Streamlit** — the canonical reference for the app framework.

## 📰 Articles / Blogs (free, no paywall)
- [Gradio Quickstart](https://www.gradio.app/guides/quickstart) — **Gradio** — a live UI in a few lines.
- [Streamlit Get Started](https://docs.streamlit.io/get-started) — **Streamlit** — the reactive-script model and first app.
- [Streamlit API reference](https://docs.streamlit.io/develop/api-reference) — **Streamlit** — every widget and layout primitive.

## 📚 Books (free, with chapters)
- [Gradio guides (full set)](https://www.gradio.app/guides) — **Gradio** — a free, chapter-structured progression.
- [Streamlit documentation (full)](https://docs.streamlit.io/) — **Streamlit** — a book-length, free guide including the gallery of [examples](https://streamlit.io/gallery).

## 🔗 In this platform
- Related domain: [12. Deployment & MLOps](../../14.%20Deployment_and_MLOps/concepts/README.md) · [10. GenAI](../../10.%20GenAI/concepts/README.md)
- Pairs with: [08 Hugging Face](08-Hugging-Face.md) *(Gradio powers HF Spaces)*
- Deeper concept (the *why*): demos, serving & deployment → [Deployment & MLOps](../../14.%20Deployment_and_MLOps/concepts/README.md); LLM apps → [RAG & LLM Applications](../../11.%20RAG_and_LLM_Applications/README.md)
