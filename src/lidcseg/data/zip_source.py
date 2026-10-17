"""Read the dataset straight out of its Kaggle zip, without extracting.

The export is ~78,000 files averaging 1.9 KB. Extracting costs far more than
its 153 MB suggests -- on a 4 KB-cluster filesystem it lands around 437 MB --
and writing 78k tiny files is slow on any disk and slower on a network mount.
Reading from the zip avoids both: one sequential file, no extraction step, and
no per-file filesystem latency.

`ZipSource` mirrors `index.build_index`'s traversal rules exactly, so an index
built from a zip is identical to one built from the extracted tree.
"""

from __future__ import annotations

import threading
import zipfile
from pathlib import Path, PurePosixPath

import cv2
import numpy as np

from .index import IMAGE_SUFFIXES, SliceRecord, _natural_key, _PATIENT_RE


class ZipSource:
    """Indexed, thread-safe read access to a dataset zip.

    `zipfile.ZipFile` is not thread-safe, so each thread gets its own handle;
    the profiler reads with a thread pool.
    """

    def __init__(self, zip_path: str | Path):
        self.zip_path = str(zip_path)
        self._local = threading.local()
        with zipfile.ZipFile(self.zip_path) as z:
            self._names = [n for n in z.namelist() if not n.endswith("/")]
        if not self._names:
            raise RuntimeError(f"zip is empty: {self.zip_path}")

    # ---------------------------------------------------------------- io

    @property
    def _zf(self) -> zipfile.ZipFile:
        zf = getattr(self._local, "zf", None)
        if zf is None:
            zf = zipfile.ZipFile(self.zip_path)
            self._local.zf = zf
        return zf

    def read_bytes(self, name: str) -> bytes:
        return self._zf.read(name)

    def imread_gray(self, name: str):
        """Decode a zip member as a grayscale image, or None if unreadable.

        Mirrors `cv2.imread(path, IMREAD_GRAYSCALE)`, which also returns None
        on failure, so callers need no special-casing.
        """
        try:
            buf = np.frombuffer(self.read_bytes(name), dtype=np.uint8)
        except KeyError:
            return None
        if buf.size == 0:
            return None
        return cv2.imdecode(buf, cv2.IMREAD_GRAYSCALE)

    def close(self) -> None:
        zf = getattr(self._local, "zf", None)
        if zf is not None:
            zf.close()
            self._local.zf = None

    # ------------------------------------------------------------- index

    def build_index(
        self,
        nodules: list[str] | None = None,
        n_readers: int = 4,
    ) -> list[SliceRecord]:
        """Index the zip, applying the same rules as `index.build_index`.

        Members are grouped into patient -> nodule -> {images, mask-N} and then
        filtered identically: a slice is kept only when it has at least one
        reader mask present.
        """
        present = set(self._names)

        # patient -> nodule -> subdir -> {filename: full member name}
        tree: dict[str, dict[str, dict[str, dict[str, str]]]] = {}
        for name in self._names:
            parts = PurePosixPath(name).parts
            # Tolerate an optional wrapper directory above the patients.
            for off in (0, 1, 2):
                if len(parts) >= off + 4 and _PATIENT_RE.match(parts[off]):
                    pat, nod, sub, fname = parts[off], parts[off + 1], parts[off + 2], parts[off + 3]
                    if not nod.startswith("nodule-"):
                        break
                    tree.setdefault(pat, {}).setdefault(nod, {}).setdefault(sub, {})[fname] = name
                    break

        if not tree:
            raise RuntimeError(f"no LIDC-IDRI-<number> entries found in {self.zip_path}")

        records: list[SliceRecord] = []
        for pat in sorted(tree, key=_natural_key):
            nods = tree[pat]
            chosen = sorted(nods, key=_natural_key) if nodules is None else [
                n for n in nodules if n in nods
            ]
            for nod in chosen:
                subs = nods[nod]
                images = subs.get("images")
                if not images:
                    continue
                mask_subs = [f"mask-{i}" for i in range(n_readers) if f"mask-{i}" in subs]
                if not mask_subs:
                    continue

                for fname in sorted(images, key=_natural_key):
                    if PurePosixPath(fname).suffix.lower() not in IMAGE_SUFFIXES:
                        continue
                    masks = tuple(
                        subs[ms][fname] for ms in mask_subs if fname in subs[ms]
                    )
                    if not masks:
                        continue
                    records.append(
                        SliceRecord(
                            patient=pat,
                            nodule=nod,
                            slice_name=fname,
                            image=images[fname],
                            masks=masks,
                        )
                    )

        if not records:
            raise RuntimeError(f"index is empty -- no usable slices in {self.zip_path}")
        return records
