"""Shared import bootstrap for the 09. LLMs figure/animation generators that live in tools/.

The figure/animation generators were moved out of each chapter's reader-facing ``code/`` dir into
this shared ``tools/`` dir (so the app's code viewer shows only the teaching demo + notebook, not
the build tooling). Many generators still import their chapter's demo module (e.g.
``decoding_sampling``, ``rlhf_dpo``) and their sibling generator (``make_figures_NN``). Importing
this module first puts every ``09. LLMs/*/code`` dir — and ``tools/`` itself — on ``sys.path`` so
those imports resolve from the new location:

    import _pathsetup  # noqa: F401  (sys.path bootstrap; must precede chapter-module imports)
    from _pathsetup import LLM_IMAGES

It also exposes the figure output dirs so a moved generator computes ``OUT_DIR`` correctly despite
its new location:
  * ``LLM_IMAGES``                  — the shared ``09. LLMs/images`` dir (most chapters),
  * ``chapter_images("10-Quantization")`` — a chapter-local ``images`` dir (quant, long-context).
"""

from __future__ import annotations

import sys
from pathlib import Path

_ROOT = Path(__file__).resolve().parent.parent  # AI-ML-learning-resources/
_LLMS = _ROOT / "09. LLMs"

#: The shared chapter image dir most generators write to (``../images`` from a chapter's code/).
LLM_IMAGES = _LLMS / "images"

# Put every chapter code/ dir (for demo-module imports) and tools/ (for sibling-generator
# imports) on sys.path, so a generator run from tools/ resolves the same names it used to.
for _code in sorted(_LLMS.glob("*/code")):
    _p = str(_code)
    if _p not in sys.path:
        sys.path.insert(0, _p)
_TOOLS = str(Path(__file__).resolve().parent)
if _TOOLS not in sys.path:
    sys.path.insert(0, _TOOLS)


def chapter_images(chapter: str) -> Path:
    """Return a chapter-local ``images`` dir, e.g. ``chapter_images("10-Quantization")``."""
    return _LLMS / chapter / "images"
