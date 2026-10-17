"""Train/val/test splitting.

The notebook split with `train_test_split` over the flat slice array. Adjacent
CT slices through one nodule are near-duplicates, so that puts copies of the
same nodule in train and test and the reported Dice measures partly memorised
data. Grouping by patient is the fix, and the `"slice"` mode is kept so the
size of the inflation can be quantified rather than argued about.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from sklearn.model_selection import GroupKFold, GroupShuffleSplit, KFold, ShuffleSplit

from .index import SliceRecord


@dataclass
class Split:
    train: np.ndarray
    val: np.ndarray
    test: np.ndarray

    def summary(self, records: list[SliceRecord]) -> dict:
        def pats(idx: np.ndarray) -> set[str]:
            return {records[i].patient for i in idx}

        tr, va, te = pats(self.train), pats(self.val), pats(self.test)
        return {
            "slices": {"train": len(self.train), "val": len(self.val), "test": len(self.test)},
            "patients": {"train": len(tr), "val": len(va), "test": len(te)},
            # Non-zero overlap in "patient" mode is a bug; in "slice" mode it
            # is the leak being demonstrated.
            "patient_overlap": {
                "train_val": len(tr & va),
                "train_test": len(tr & te),
                "val_test": len(va & te),
            },
        }


def make_split(
    records: list[SliceRecord],
    mode: str = "patient",
    val_frac: float = 0.15,
    test_frac: float = 0.15,
    seed: int = 42,
    folds: int = 1,
    fold: int = 0,
) -> Split:
    """Build index arrays for the three sets.

    With ``folds > 1`` the data is partitioned into K folds; fold `fold`
    becomes test, the next one becomes val, and the rest train. This keeps
    every patient in test exactly once across the K runs.
    """
    n = len(records)
    idx = np.arange(n)
    groups = np.array([r.patient for r in records])

    if mode not in {"patient", "slice"}:
        raise ValueError(f"split.mode must be 'patient' or 'slice', got {mode!r}")

    if folds and folds > 1:
        if not 0 <= fold < folds:
            raise ValueError(f"fold must be in [0,{folds}), got {fold}")
        splitter = GroupKFold(n_splits=folds) if mode == "patient" else KFold(
            n_splits=folds, shuffle=True, random_state=seed
        )
        parts = [te for _, te in splitter.split(idx, groups=groups)]
        test = parts[fold]
        val = parts[(fold + 1) % folds]
        train = np.concatenate([p for i, p in enumerate(parts) if i not in {fold, (fold + 1) % folds}])
        return Split(np.sort(train), np.sort(val), np.sort(test))

    # Single hold-out: carve out test, then val from what remains.
    if mode == "patient":
        gss = GroupShuffleSplit(n_splits=1, test_size=test_frac, random_state=seed)
        rest, test = next(gss.split(idx, groups=groups))
        # val_frac is a fraction of the whole, so rescale against `rest`.
        inner = val_frac / max(1.0 - test_frac, 1e-9)
        gss2 = GroupShuffleSplit(n_splits=1, test_size=inner, random_state=seed + 1)
        tr_rel, va_rel = next(gss2.split(rest, groups=groups[rest]))
        train, val = rest[tr_rel], rest[va_rel]
    else:
        ss = ShuffleSplit(n_splits=1, test_size=test_frac, random_state=seed)
        rest, test = next(ss.split(idx))
        inner = val_frac / max(1.0 - test_frac, 1e-9)
        ss2 = ShuffleSplit(n_splits=1, test_size=inner, random_state=seed + 1)
        tr_rel, va_rel = next(ss2.split(rest))
        train, val = rest[tr_rel], rest[va_rel]

    return Split(np.sort(train), np.sort(val), np.sort(test))


def assert_no_leakage(split: Split, records: list[SliceRecord]) -> None:
    """Hard check that no patient spans two sets. Call this in patient mode."""
    s = split.summary(records)
    overlap = s["patient_overlap"]
    if any(overlap.values()):
        raise AssertionError(f"patient-level leakage detected: {overlap}")
