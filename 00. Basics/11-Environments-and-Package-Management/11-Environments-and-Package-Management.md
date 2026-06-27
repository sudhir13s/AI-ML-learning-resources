---
id: "00-basics/environments-and-package-management"
topic: "Environments & Package Management (uv · conda · pip)"
parent: "00-basics"
level: beginner
prereqs: ["python-for-ml"]
interview_frequency: low
updated: 2026-06-20
---

# Environments & Package Management — uv · conda · pip
> Every ML project depends on specific versions of NumPy, Pandas, PyTorch, and dozens more. A
> **virtual environment** is an isolated sandbox so each project gets its own dependencies without
> clashing. **pip** installs packages, **venv** isolates them, **conda** does both (plus non-Python
> deps), and **uv** is the fast, modern all-in-one. Get this right early and you avoid the #1 source
> of "it works on my machine" pain.

**Why it matters:** "how do you manage dependencies / reproduce an environment?" comes up in any
practical or MLOps-flavored interview, and broken environments are the most common reason a beginner's
project won't run. Reproducibility (a lockfile or `requirements.txt`) is a real engineering skill.

**⭐ Start here — suggested path:**

1. **Understand *why* environments exist** — watch [Python Virtual Environments — Full Tutorial](https://www.youtube.com/watch?v=Y21OR1OPC9A). *Isolation and dependency clashes, made concrete.*
2. **Learn the standard pip + venv flow** — read [Installing packages with pip and venv](https://packaging.python.org/en/latest/guides/installing-using-pip-and-virtual-environments/). *The official, always-works baseline every Python dev knows.*
3. **See it hands-on** — watch [Corey Schafer: venv (Mac & Linux)](https://www.youtube.com/watch?v=Kg1Yvry_Ydk). *Create, activate, freeze, and remove an environment.*
4. **Meet the modern tool** — watch [UV: a faster all-in-one package manager](https://www.youtube.com/watch?v=AMdG7IjgSPM) + read [uv docs](https://docs.astral.sh/uv/). *The fast, lockfile-first workflow that's becoming the default.*
5. **Know when to use conda** — skim [conda getting started](https://docs.conda.io/projects/conda/en/stable/user-guide/getting-started.html). *Why data science often reaches for conda (binary/GPU deps).*

## 🎓 Courses (free)
- [Python Packaging User Guide — Installing packages](https://packaging.python.org/en/latest/guides/installing-using-pip-and-virtual-environments/) — **PyPA** — free, official; the canonical pip + venv workflow.
- [uv — Getting started](https://docs.astral.sh/uv/getting-started/) — **Astral** — free, official; the modern fast-installer + project workflow.
- [conda — Getting started](https://docs.conda.io/projects/conda/en/stable/user-guide/getting-started.html) — **Anaconda** — free, official; environments and packages for data science.

## 🎥 Videos
- [Python Virtual Environments — Full Tutorial for Beginners](https://www.youtube.com/watch?v=Y21OR1OPC9A) — **freeCodeCamp** — the *why* and *how* of isolation, thoroughly.
- [Python Tutorial: VENV (Mac & Linux)](https://www.youtube.com/watch?v=Kg1Yvry_Ydk) — **Corey Schafer** — the clean, canonical venv walkthrough.
- [UV — A Faster, All-in-One Package Manager](https://www.youtube.com/watch?v=AMdG7IjgSPM) — **ArjanCodes** — the modern uv workflow (replaces pip + venv).
- [Watch this before installing Python](https://www.youtube.com/watch?v=28eLP22SMTA) — **NeuralNine** — common environment mistakes and how to avoid them.

## 📄 Key Papers
- [uv documentation (official)](https://docs.astral.sh/uv/) — **Astral** — the authoritative reference for the fast Rust-based package/project manager.
- [PEP 405 — Python Virtual Environments](https://peps.python.org/pep-0405/) — **Python core** — the design spec that introduced `venv` into the standard library.
- [Reproducibility in machine learning research](https://arxiv.org/abs/2003.12206) — **Pineau et al. (2021)** — why pinned environments and reproducible setups matter scientifically.

## 📰 Articles / Blogs (free, no paywall)
- [Installing packages using pip and virtual environments](https://packaging.python.org/en/latest/guides/installing-using-pip-and-virtual-environments/) — **PyPA** — the official baseline workflow, step by step.
- [venv — Creation of virtual environments](https://docs.python.org/3/library/venv.html) — **Python docs** — the standard-library tool, authoritatively.
- [uv — Features overview](https://docs.astral.sh/uv/getting-started/features/) — **Astral** — what uv replaces (pip, venv, pip-tools, pyenv) in one tool.

## 📚 Books (free, with chapters)
- [The Hitchhiker's Guide to Python — Virtual Environments](https://docs.python-guide.org/dev/virtualenvs/) — **Kenneth Reitz et al.** — free; the community-standard guide to environments.
- [Python Packaging User Guide (full)](https://packaging.python.org/en/latest/) — **PyPA** — free, official; everything about installing, building, and distributing.
- [The Good Research Code Handbook — Environments](https://goodresearch.dev/) — **Patrick Mineault** — free online book; environments and dependencies as part of clean, reproducible code.

## 🔗 In this platform
- Prev/next: [10 Jupyter & Google Colab](10-Jupyter-and-Google-Colab.md) · [12 Your First ML Project](12-Your-First-ML-Project.md)
- Go deeper — reproducibility, containers, MLOps: [12. Deployment & MLOps](../../14.%20Deployment_and_MLOps/concepts/README.md)
- Go deeper — frameworks & tooling: [11. Tools & Frameworks](../../13.%20Tools_and_Frameworks/concepts/README.md)
