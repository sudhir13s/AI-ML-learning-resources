"""Shared import bootstrap for the LLM figure/animation generators that live in tools/.

The figure/animation generators were moved out of each topic's reader-facing ``code/`` dir into
this shared ``tools/`` dir (so the app's code viewer shows only the teaching demo + notebook, not
the build tooling). Many generators still import their topic's demo module (e.g.
``decoding_sampling``, ``rlhf_dpo``) and their sibling generator (``make_figures_NN``). Importing
this module first puts every LLM topic ``code/`` dir — and ``tools/`` itself — on ``sys.path`` so
those imports resolve from the new location:

    import _pathsetup  # noqa: F401  (sys.path bootstrap; must precede topic-module imports)
    from _pathsetup import topic_images

Since the chartered restructure every topic owns its figures, so there is no shared image dir
any more. ``topic_images("llms-applications-and-agents/inference-and-runtime/kv-cache")`` returns
the ``images`` dir of one topic package; each generator names its own.
"""

from __future__ import annotations

import sys
from pathlib import Path

_ROOT = Path(__file__).resolve().parent.parent  # AI-ML-learning-resources/

#: Sections that hold LLM topic packages. Their ``<sub-area>/<topic>/code`` dirs carry the
#: demo modules the generators import.
_TOPIC_ROOTS = (
    _ROOT / "llms-applications-and-agents",
    _ROOT / "deep-learning" / "attention-and-transformers",
)

# Put every topic code/ dir (for demo-module imports) and tools/ (for sibling-generator
# imports) on sys.path, so a generator run from tools/ resolves the same names it used to.
_code_dirs: list[Path] = []
for _root in _TOPIC_ROOTS:
    _code_dirs.extend(_root.glob("*/code"))
    _code_dirs.extend(_root.glob("*/*/code"))
for _code in sorted(set(_code_dirs)):
    _p = str(_code)
    if _p not in sys.path:
        sys.path.insert(0, _p)
_TOOLS = str(Path(__file__).resolve().parent)
if _TOOLS not in sys.path:
    sys.path.insert(0, _TOOLS)


def topic_images(topic: str) -> Path:
    """Return a topic package's ``images`` dir, given its repo-relative folder."""
    return _ROOT / topic / "images"
