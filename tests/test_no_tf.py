"""Tests for everything that does not need TensorFlow.

Run anywhere:  python tests/test_no_tf.py
These cover the parts most likely to be silently wrong -- the split protocol,
the vote fusion and the metric conventions.
"""

from __future__ import annotations

import sys
import tempfile
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from lidcseg.config import load_config  # noqa: E402
from lidcseg.data.index import SliceRecord, build_index, index_stats  # noqa: E402
from lidcseg.data.preprocess import anisotropic_diffusion, load_vote_mask  # noqa: E402
from lidcseg.data.splits import assert_no_leakage, make_split  # noqa: E402
from lidcseg.metrics import all_metrics, per_image_dice, per_image_iou  # noqa: E402

PASS, FAIL = [], []


def check(name: str, cond: bool, detail: str = "") -> None:
    (PASS if cond else FAIL).append(name)
    print(f"  {'PASS' if cond else 'FAIL'}  {name}" + (f"  -- {detail}" if detail and not cond else ""))


# ----------------------------------------------------------------- configs
print("\nconfigs")
for cfg_file in sorted((ROOT / "configs").glob("*.yaml")):
    try:
        c = load_config(cfg_file)
        check(f"load {cfg_file.name}", True)
        if cfg_file.name == "improved.yaml":
            check("improved uses patient split", c.split.mode == "patient", c.split.mode)
            check("improved inherits tversky", c.loss.name == "tversky_focal", c.loss.name)
            check("improved keeps all nodules", c.data.nodules is None, str(c.data.nodules))
        if cfg_file.name == "notebook_best.yaml":
            check("notebook_best keeps slice split", c.split.mode == "slice", c.split.mode)
            check("notebook_best lr 5e-5", abs(c.train.lr - 5e-5) < 1e-12, str(c.train.lr))
            check("notebook_best nodule-0 only", c.data.nodules == ["nodule-0"], str(c.data.nodules))
        if cfg_file.name == "notebook_best_patient.yaml":
            # Two-level _base_ chain must resolve.
            check("2-level base chain", c.split.mode == "patient" and c.train.batch_size == 4,
                  f"{c.split.mode} {c.train.batch_size}")
    except Exception as e:
        check(f"load {cfg_file.name}", False, f"{type(e).__name__}: {e}")

try:
    c = load_config(ROOT / "configs" / "improved.yaml", ["train.lr=1e-4", "model.attention=cab"])
    check("CLI overrides apply", c.train.lr == 1e-4 and c.model.attention == "cab")
except Exception as e:
    check("CLI overrides apply", False, str(e))

try:
    load_config(ROOT / "configs" / "improved.yaml", ["train.nonsense=1"])
    check("unknown key rejected", False, "no error raised")
except ValueError:
    check("unknown key rejected", True)
except Exception as e:
    check("unknown key rejected", False, type(e).__name__)


# ------------------------------------------------------------------ splits
print("\nsplits")
# 40 patients x 25 slices: enough that fractions land near their targets.
recs = [
    SliceRecord(f"LIDC-IDRI-{p:04d}", "nodule-0", f"{s}.png", f"/i/{p}_{s}.png", (f"/m/{p}_{s}.png",))
    for p in range(40)
    for s in range(25)
]
check("index_stats counts", index_stats(recs)["patients"] == 40 and index_stats(recs)["slices"] == 1000)

sp = make_split(recs, mode="patient", val_frac=0.15, test_frac=0.15, seed=42)
summary = sp.summary(recs)
check("patient split: no overlap", not any(summary["patient_overlap"].values()), str(summary))
check("patient split: covers all", len(sp.train) + len(sp.val) + len(sp.test) == 1000)
check("patient split: test ~15%", 0.10 <= len(sp.test) / 1000 <= 0.20, str(len(sp.test) / 1000))
check("patient split: val ~15%", 0.10 <= len(sp.val) / 1000 <= 0.20, str(len(sp.val) / 1000))
try:
    assert_no_leakage(sp, recs)
    check("assert_no_leakage passes clean split", True)
except AssertionError as e:
    check("assert_no_leakage passes clean split", False, str(e))

sp_slice = make_split(recs, mode="slice", seed=42)
ov = sp_slice.summary(recs)["patient_overlap"]
check("slice split DOES leak (as expected)", ov["train_test"] > 0, str(ov))
try:
    assert_no_leakage(sp_slice, recs)
    check("assert_no_leakage catches leak", False, "did not raise")
except AssertionError:
    check("assert_no_leakage catches leak", True)

# Cross-validation: every patient must be tested exactly once across folds.
seen: dict[str, int] = {}
for f in range(5):
    s = make_split(recs, mode="patient", folds=5, fold=f, seed=42)
    assert_no_leakage(s, recs)
    for i in s.test:
        seen[recs[i].patient] = seen.get(recs[i].patient, 0) + 1
check("CV5 tests each patient once", len(seen) == 40 and all(v == 25 for v in seen.values()),
      f"{len(seen)} patients")


# -------------------------------------------------------------- preprocess
print("\npreprocess")
img = np.zeros((32, 32), dtype=np.uint8)
img[8:24, 8:24] = 200
sm = anisotropic_diffusion(img, num_iter=5, kappa=50, gamma=0.1)
check("diffusion preserves shape/dtype", sm.shape == (32, 32) and sm.dtype == np.float32)
check("diffusion stays in range", -1 <= sm.min() and sm.max() <= 256, f"{sm.min()},{sm.max()}")
check("diffusion keeps edge contrast", sm[16, 16] - sm[2, 2] > 100, str(sm[16, 16] - sm[2, 2]))

# No wrap-around: a bright top row must not bleed into the bottom row.
edge = np.zeros((16, 16), dtype=np.uint8)
edge[0, :] = 255
sm_e = anisotropic_diffusion(edge, num_iter=5, kappa=50, gamma=0.1)
check("diffusion does not wrap edges", sm_e[-1, 8] < 1.0, str(sm_e[-1, 8]))

try:
    anisotropic_diffusion(img, gamma=0.5)
    check("diffusion rejects unstable gamma", False, "no error")
except ValueError:
    check("diffusion rejects unstable gamma", True)

# Vote fusion: 3 readers agree on a 4x4 block, 1 dissents elsewhere.
with tempfile.TemporaryDirectory() as td:
    import cv2

    paths = []
    for r in range(4):
        m = np.zeros((16, 16), dtype=np.uint8)
        if r < 3:
            m[4:8, 4:8] = 255           # 3 readers agree here
        else:
            m[12:14, 12:14] = 255       # lone dissenter
        p = str(Path(td) / f"m{r}.png")
        cv2.imwrite(p, m)
        paths.append(p)

    mask2, n = load_vote_mask(paths, 16, vote_threshold=2)
    check("vote found all readers", n == 4, str(n))
    check("vote>=2 keeps agreed region", mask2[4:8, 4:8, 0].all())
    check("vote>=2 drops lone reader", not mask2[12:14, 12:14, 0].any())

    mask4, _ = load_vote_mask(paths, 16, vote_threshold=4)
    check("vote>=4 drops 3-reader region", not mask4.any())

    mask1, _ = load_vote_mask(paths, 16, vote_threshold=1)
    check("vote>=1 keeps lone reader", mask1[12:14, 12:14, 0].all())

    missing, n_missing = load_vote_mask(paths[:2] + ["/does/not/exist.png"], 16, 2)
    check("vote tolerates missing files", n_missing == 2 and missing[4:8, 4:8, 0].all(), str(n_missing))


# ----------------------------------------------------------------- metrics
print("\nmetrics")
y = np.zeros((4, 8, 8, 1), dtype=np.float32)
y[0, 0:4, 0:4] = 1      # big object
y[1, 0:2, 0:2] = 1      # small object
y[2, 0:4, 0:4] = 1
# y[3] stays empty

p = np.zeros_like(y)
p[0, 0:4, 0:4] = 1      # perfect
p[1] = 0                # missed entirely
p[2, 0:2, 0:4] = 1      # half
# p[3] empty -> correct

d = per_image_dice(y, p)
check("per-image dice perfect=1", abs(d[0] - 1.0) < 1e-9, str(d[0]))
check("per-image dice missed=0", abs(d[1]) < 1e-9, str(d[1]))
check("per-image dice half=2/3", abs(d[2] - 2 / 3) < 1e-9, str(d[2]))
check("empty-and-correct scores 1", abs(d[3] - 1.0) < 1e-9, str(d[3]))

i_ = per_image_iou(y, p)
check("per-image iou half=0.5", abs(i_[2] - 0.5) < 1e-9, str(i_[2]))

m = all_metrics(y, p)
# Global pools pixels: TP=16+8=24, FP=0, FN=0+4+8=12 -> 2*24/(2*24+0+12)=0.8
check("global dice pools pixels", abs(m["dice_global"] - 0.8) < 1e-9, str(m["dice_global"]))
check("per-image mean < global here", m["dice"] < m["dice_global"],
      f"{m['dice']:.4f} vs {m['dice_global']:.4f}")
check("recall correct", abs(m["recall"] - 24 / 36) < 1e-9, str(m["recall"]))
check("precision 1.0 (no FP)", abs(m["precision"] - 1.0) < 1e-9, str(m["precision"]))

# All-empty truth and prediction must not divide by zero.
z = np.zeros((2, 4, 4, 1), dtype=np.float32)
mz = all_metrics(z, z)
check("all-empty is finite", np.isfinite(mz["dice"]) and mz["dice"] == 1.0, str(mz["dice"]))


# ---------------------------------------------------------- index on disk
print("\nindex (synthetic tree)")
with tempfile.TemporaryDirectory() as td:
    import cv2

    root = Path(td) / "LIDC-IDRI-slices"
    for pid in (1, 2):
        for nod in (0, 1):
            base = root / f"LIDC-IDRI-{pid:04d}" / f"nodule-{nod}"
            (base / "images").mkdir(parents=True)
            for r in range(4):
                (base / f"mask-{r}").mkdir(parents=True)
            for s in range(3):
                cv2.imwrite(str(base / "images" / f"{s}.png"), np.full((16, 16), 100, np.uint8))
                for r in range(4):
                    cv2.imwrite(str(base / f"mask-{r}" / f"{s}.png"), np.zeros((16, 16), np.uint8))

    idx_all = build_index(root)
    check("index finds all nodules", len(idx_all) == 12, str(len(idx_all)))
    check("index finds 2 patients", index_stats(idx_all)["patients"] == 2)
    check("index finds 4 nodules", index_stats(idx_all)["nodules"] == 4)

    idx_n0 = build_index(root, nodules=["nodule-0"])
    check("nodules filter works", len(idx_n0) == 6, str(len(idx_n0)))

    # Root discovery must dig through an extra wrapper directory.
    idx_deep = build_index(Path(td))
    check("finds nested dataset root", len(idx_deep) == 12, str(len(idx_deep)))

    check("mask paths attached", all(len(r.masks) == 4 for r in idx_all))
    check("patient ids parsed", {r.patient for r in idx_all} == {"LIDC-IDRI-0001", "LIDC-IDRI-0002"})


print(f"\n{'=' * 52}\n{len(PASS)} passed, {len(FAIL)} failed")
if FAIL:
    for f in FAIL:
        print(f"  FAILED: {f}")
    sys.exit(1)
print("all green")
