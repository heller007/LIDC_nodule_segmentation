"""Generate the standalone profiler from the package sources.

Run from the repo root:  python build/make_standalone.py

Stitches lidcseg/data/index.py and lidcseg/profile_data.py into one file that
needs no package install, for environments without internet (e.g. a Kaggle
notebook that cannot clone). Regenerate after changing either source.
"""

from __future__ import annotations

import pathlib
import re
import sys

HEADER = '''
# ======================================================================
# LIDC-IDRI dataset profiler -- STANDALONE (generated; do not edit)
#
# Regenerate with:  python build/make_standalone.py
#
# Needs only numpy, OpenCV and matplotlib, all preinstalled on Kaggle.
# No internet, no repo clone, no extra dataset -- just the LIDC slices.
#
# Measures:
#   1. positive pixel fraction, per reader-vote threshold
#   2. empty-merged-mask fraction, per vote threshold
#   3. reader-mask histogram
#   4. native resolution (before any resize)
#   5. nodule size distribution, native AND after the training resize
#   6. slice/patient/nodule counts, all nodules vs nodule-0 only
#   7. inter-reader pairwise Dice -- the achievable-Dice ceiling
# ======================================================================

# Must stay first: the stitched bodies annotate with names defined later.
from __future__ import annotations

import json
import re
import time
from collections import Counter
from dataclasses import dataclass, asdict
from pathlib import Path
from types import SimpleNamespace

import cv2
import numpy as np
'''.lstrip("\n")

FOOTER = '''

# ---------------------------------------------------------------- config shim
def make_cfg(root, img_size=128, vote_threshold=2, n_readers=4,
             out_dir="/kaggle/working", name="standalone"):
    """Mirrors the lidcseg.config fields the profiler reads."""
    return SimpleNamespace(
        name=name,
        out_dir=out_dir,
        data=SimpleNamespace(
            root=str(root),
            img_size=img_size,
            vote_threshold=vote_threshold,
            n_readers=n_readers,
            nodules=None,
        ),
    )
'''


def strip_head(src: str) -> str:
    """Drop the module docstring and TOP-LEVEL imports only.

    Function-local imports (matplotlib inside write_figures, cv2 inside
    remove_small_components) must survive, so the pattern is anchored at
    column 0 rather than allowing leading whitespace.
    """
    m = re.match(r'\s*(""".*?"""|\'\'\'.*?\'\'\')', src, re.S)
    if m:
        src = src[m.end():]
    return "\n".join(
        ln for ln in src.splitlines() if not re.match(r"(import |from )", ln)
    )


def build(repo: pathlib.Path) -> str:
    idx = (repo / "src/lidcseg/data/index.py").read_text(encoding="utf-8")
    prof = (repo / "src/lidcseg/profile_data.py").read_text(encoding="utf-8")

    body = strip_head(prof)
    # The CLI belongs to the package version; the notebook calls run() directly.
    body = body.split("def main(argv")[0]
    # `Config` is a type-only reference here; the shim is a SimpleNamespace.
    body = body.replace("cfg: Config", "cfg")

    return HEADER + "\n" + strip_head(idx) + "\n" + body + FOOTER


def main() -> int:
    repo = pathlib.Path(__file__).resolve().parents[1]
    out = repo / "build" / "standalone_profile.py"
    out.parent.mkdir(parents=True, exist_ok=True)
    text = build(repo)
    out.write_text(text, encoding="utf-8")

    # Fail loudly if a needed import was stripped.
    for name in ("matplotlib", "cv2", "np."):
        if name not in text:
            print(f"WARNING: {name!r} missing from generated file")
    print(f"wrote {out} ({len(text.splitlines())} lines)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
