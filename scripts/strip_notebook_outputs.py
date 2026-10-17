"""Strip outputs and execution counts from notebooks before committing.

The source notebook this work came from was 4.5 MB, almost all of it embedded
PNG output. Run this before committing any .ipynb:

    python scripts/strip_notebook_outputs.py notebooks/*.ipynb
"""

from __future__ import annotations

import json
import sys
from pathlib import Path


def strip(path: Path) -> tuple[int, int]:
    before = path.stat().st_size
    nb = json.loads(path.read_text(encoding="utf-8"))
    for cell in nb.get("cells", []):
        if cell.get("cell_type") == "code":
            cell["outputs"] = []
            cell["execution_count"] = None
    path.write_text(json.dumps(nb, indent=1), encoding="utf-8")
    return before, path.stat().st_size


def main(argv: list[str]) -> int:
    if not argv:
        print(__doc__)
        return 1
    for arg in argv:
        p = Path(arg)
        if not p.exists():
            print(f"skip (missing): {p}")
            continue
        b, a = strip(p)
        print(f"{p}: {b/1024:.0f} KB -> {a/1024:.0f} KB")
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
