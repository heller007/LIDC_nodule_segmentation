"""Dataset discovery.

Walks the LIDC slice tree once and records, for every slice, the image path,
its reader-mask paths, and the patient it belongs to. The patient id is what
makes a leak-free split possible, so it is carried from the start rather than
reconstructed later.

Expected layout (Kaggle `zhangweiled/lidcidri`):

    <root>/LIDC-IDRI-0001/nodule-0/images/0.png
    <root>/LIDC-IDRI-0001/nodule-0/mask-0/0.png
    <root>/LIDC-IDRI-0001/nodule-0/mask-1/0.png   ... mask-3
    <root>/LIDC-IDRI-0001/nodule-1/...
"""

from __future__ import annotations

import json
import re
from dataclasses import dataclass, asdict
from pathlib import Path


@dataclass(frozen=True)
class SliceRecord:
    patient: str      # "LIDC-IDRI-0001" -- the grouping key for splits
    nodule: str       # "nodule-0"
    slice_name: str   # "0.png"
    image: str        # absolute path
    masks: tuple[str, ...]  # absolute paths, one per reader that has this slice

    @property
    def uid(self) -> str:
        return f"{self.patient}/{self.nodule}/{self.slice_name}"


def _natural_key(s: str):
    """Sort "10.png" after "9.png" so slice order is anatomical."""
    return [int(t) if t.isdigit() else t.lower() for t in re.split(r"(\d+)", s)]


# A patient directory is "LIDC-IDRI-" plus digits and nothing else. Matching
# the prefix alone also matches the dataset's own "LIDC-IDRI-slices" wrapper,
# which then gets treated as a patient with no nodules.
_PATIENT_RE = re.compile(r"^LIDC-IDRI-\d+$")


def is_patient_dir(p: Path) -> bool:
    return p.is_dir() and bool(_PATIENT_RE.match(p.name))


def find_dataset_root(root: str | Path) -> Path:
    """Resolve the directory that directly contains LIDC-IDRI-* folders.

    Kaggle mount paths vary. `kagglehub.dataset_download` mounts under
    `/kaggle/input/datasets/<owner>/<slug>/`, which is what the original
    notebook hard-coded and it was correct for that setup. Adding the same
    dataset through the Add Data panel instead gives `/kaggle/input/<slug>/`,
    and either may or may not nest a `LIDC-IDRI-slices/` directory.

    So rather than hard-code any of them, search a few levels down for
    directories matching `LIDC-IDRI-<number>`. This is about portability
    across mount styles, not about the old path being broken.
    """
    root = Path(root)
    if not root.exists():
        raise FileNotFoundError(
            f"dataset root does not exist: {root}\n"
            "On Kaggle, add the 'zhangweiled/lidcidri' dataset to the notebook and "
            "pass --set data.root=/kaggle/input/<mounted-name>"
        )

    def has_patients(d: Path) -> bool:
        try:
            return any(is_patient_dir(p) for p in d.iterdir())
        except (PermissionError, OSError):
            return False

    if has_patients(root):
        return root
    for depth in range(1, 4):
        for cand in sorted(root.glob("/".join(["*"] * depth))):
            if cand.is_dir() and has_patients(cand):
                return cand
    raise FileNotFoundError(
        f"no LIDC-IDRI-<number> patient directories found under {root} "
        "(searched 3 levels deep)"
    )


def build_index(
    root: str | Path,
    nodules: list[str] | None = None,
    n_readers: int = 4,
) -> list[SliceRecord]:
    """Scan the tree and return one record per slice.

    `nodules=None` takes every nodule directory per patient. The notebook only
    read `nodule-0`, which silently dropped the second and later nodules of
    every multi-nodule patient.
    """
    root = find_dataset_root(root)
    records: list[SliceRecord] = []

    patients = sorted(
        (p for p in root.iterdir() if is_patient_dir(p)),
        key=lambda p: _natural_key(p.name),
    )

    for pdir in patients:
        if nodules is None:
            ndirs = sorted(
                (d for d in pdir.iterdir() if d.is_dir() and d.name.startswith("nodule-")),
                key=lambda d: _natural_key(d.name),
            )
        else:
            ndirs = [pdir / n for n in nodules]

        for ndir in ndirs:
            img_dir = ndir / "images"
            if not img_dir.is_dir():
                continue

            mask_dirs = [ndir / f"mask-{i}" for i in range(n_readers)]
            mask_dirs = [d for d in mask_dirs if d.is_dir()]
            if not mask_dirs:
                continue

            for img in sorted(img_dir.iterdir(), key=lambda p: _natural_key(p.name)):
                if img.suffix.lower() not in {".png", ".jpg", ".jpeg", ".tif", ".tiff", ".bmp"}:
                    continue
                masks = tuple(str(d / img.name) for d in mask_dirs if (d / img.name).exists())
                if not masks:
                    continue
                records.append(
                    SliceRecord(
                        patient=pdir.name,
                        nodule=ndir.name,
                        slice_name=img.name,
                        image=str(img),
                        masks=masks,
                    )
                )

    if not records:
        raise RuntimeError(f"index is empty -- found no usable slices under {root}")
    return records


def index_stats(records: list[SliceRecord]) -> dict:
    patients = {r.patient for r in records}
    nodules = {(r.patient, r.nodule) for r in records}
    per_reader_counts: dict[int, int] = {}
    for r in records:
        per_reader_counts[len(r.masks)] = per_reader_counts.get(len(r.masks), 0) + 1
    return {
        "slices": len(records),
        "patients": len(patients),
        "nodules": len(nodules),
        "slices_per_patient": round(len(records) / max(len(patients), 1), 2),
        "mask_count_histogram": dict(sorted(per_reader_counts.items())),
    }


def save_index(records: list[SliceRecord], path: str | Path) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = {"stats": index_stats(records), "records": [asdict(r) for r in records]}
    path.write_text(json.dumps(payload, indent=1), encoding="utf-8")


def load_index(path: str | Path) -> list[SliceRecord]:
    payload = json.loads(Path(path).read_text(encoding="utf-8"))
    return [
        SliceRecord(
            patient=d["patient"],
            nodule=d["nodule"],
            slice_name=d["slice_name"],
            image=d["image"],
            masks=tuple(d["masks"]),
        )
        for d in payload["records"]
    ]
