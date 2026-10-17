"""Tests for the dataset profiler, on a synthetic tree with known properties.

    python tests/test_profile.py

The tree is built with nodule sizes, reader agreement and mask availability
chosen so every aggregate the profiler reports has a hand-checkable expected
value. No TensorFlow needed.
"""

from __future__ import annotations

import json
import shutil
import sys
import tempfile
from pathlib import Path

import cv2
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from lidcseg.config import load_config  # noqa: E402
from lidcseg.profile_data import SIZE_BIN_LABELS, aggregate, profile_slice, run  # noqa: E402
from lidcseg.data.index import build_index  # noqa: E402

PASS, FAIL = [], []


def check(name: str, cond: bool, detail: str = "") -> None:
    (PASS if cond else FAIL).append(name)
    print(f"  {'PASS' if cond else 'FAIL'}  {name}" + (f"  -- {detail}" if detail and not cond else ""))


NATIVE = 256


def disc(size: int, cy: int, cx: int, r: int) -> np.ndarray:
    yy, xx = np.ogrid[:size, :size]
    return (((yy - cy) ** 2 + (xx - cx) ** 2) <= r * r).astype(np.uint8) * 255


def build_tree(root: Path) -> dict:
    """Synthetic dataset with deliberately known properties.

    3 patients. Patient 1 has 2 nodules, patients 2-3 have 1 each -> 4 nodules.
    4 slices per nodule -> 16 slices total; nodule-0 only -> 12 slices.

    Slice 0 of each nodule: radius 20 disc, all 4 readers identical.
    Slice 1: radius 20 disc, only 3 readers have a mask on disk.
    Slice 2: radius 2 disc (tiny) -- tests resize survival.
    Slice 3: empty for all readers -- tests the empty-mask count.
    """
    spec = [("LIDC-IDRI-0001", 2), ("LIDC-IDRI-0002", 1), ("LIDC-IDRI-0003", 1)]
    for pid, n_nod in spec:
        for k in range(n_nod):
            base = root / pid / f"nodule-{k}"
            (base / "images").mkdir(parents=True)
            for r in range(4):
                (base / f"mask-{r}").mkdir(parents=True)

            for s in range(4):
                img = np.full((NATIVE, NATIVE), 90, np.uint8)
                cv2.imwrite(str(base / "images" / f"{s}.png"), img)

                if s == 0:
                    m, readers = disc(NATIVE, 128, 128, 20), 4
                elif s == 1:
                    m, readers = disc(NATIVE, 128, 128, 20), 3
                elif s == 2:
                    m, readers = disc(NATIVE, 128, 128, 2), 4
                else:
                    m, readers = np.zeros((NATIVE, NATIVE), np.uint8), 4

                for r in range(readers):
                    cv2.imwrite(str(base / f"mask-{r}" / f"{s}.png"), m)

    return {"patients": 3, "nodules": 4, "slices_all": 16, "slices_n0": 12}


print("building synthetic tree")
tmp = Path(tempfile.mkdtemp())
try:
    data_root = tmp / "LIDC-IDRI-slices"
    data_root.mkdir()
    expect = build_tree(data_root)
    print(f"  {expect}")

    cfg = load_config(
        ROOT / "configs" / "improved.yaml",
        [f"data.root={data_root}", "data.img_size=128", "out_dir=" + str(tmp / "out")],
    )

    # ------------------------------------------------------- per-slice
    print("\nper-slice profiling")
    recs = build_index(data_root, None, 4)
    check("index finds 16 slices", len(recs) == 16, str(len(recs)))

    by_name = {}
    for r in recs:
        by_name.setdefault(r.slice_name, []).append(r)

    p0 = profile_slice(by_name["0.png"][0], 128, 2, 4)
    check("native resolution read", (p0.height, p0.width) == (NATIVE, NATIVE),
          f"{p0.height}x{p0.width}")
    check("slice 0 has 4 masks", p0.n_masks == 4, str(p0.n_masks))
    # pi*20^2 = 1257, discretised
    check("slice 0 area ~1257px", 1200 < p0.component_areas[0] < 1320, str(p0.component_areas))
    check("slice 0 identical readers -> Dice 1.0",
          abs(p0.mean_pairwise_dice - 1.0) < 1e-9, str(p0.mean_pairwise_dice))
    # Identical readers: every vote threshold gives the same mask.
    check("slice 0 vote px equal across thresholds",
          len(set(p0.vote_px.values())) == 1, str(p0.vote_px))
    # 256 -> 128 halves each axis, so area scales by ~1/4.
    check("slice 0 resized area ~1/4",
          0.2 < p0.resized_px / p0.component_areas[0] < 0.3,
          f"{p0.resized_px}/{p0.component_areas[0]}")

    p1 = profile_slice(by_name["1.png"][0], 128, 2, 4)
    check("slice 1 has 3 masks", p1.n_masks == 3, str(p1.n_masks))
    check("slice 1 vote>=4 is empty", p1.vote_px[4] == 0, str(p1.vote_px))
    check("slice 1 vote>=3 is not empty", p1.vote_px[3] > 0, str(p1.vote_px))

    p2 = profile_slice(by_name["2.png"][0], 128, 2, 4)
    # pi*2^2 ~ 13px natively; a quarter of that is ~3px after resize.
    check("slice 2 tiny natively", 0 < p2.component_areas[0] < 30, str(p2.component_areas))
    check("slice 2 shrinks on resize", p2.resized_px < p2.component_areas[0],
          f"{p2.resized_px} vs {p2.component_areas[0]}")

    p3 = profile_slice(by_name["3.png"][0], 128, 2, 4)
    check("slice 3 empty", p3.component_areas == () and p3.resized_px == 0, str(p3.component_areas))
    check("empty readers agree -> Dice 1.0", abs(p3.mean_pairwise_dice - 1.0) < 1e-9,
          str(p3.mean_pairwise_dice))

    # ------------------------------------------------------- aggregate
    print("\naggregate")
    profiles = [profile_slice(r, 128, 2, 4) for r in recs]
    agg = aggregate(profiles, cfg, 4)

    res = agg["resolution"]
    check("resolution uniform", res["uniform"] and res["distinct_shapes"] == 1, str(res))
    check("resolution reported correctly",
          res["shapes"][0]["h"] == NATIVE and res["shapes"][0]["slices"] == 16, str(res["shapes"][0]))

    rm = agg["reader_masks"]
    # 4 nodules x (3 slices with 4 masks + 1 slice with 3) = 12 and 4.
    check("mask histogram 4->12, 3->4", rm["histogram"] == {"3": 4, "4": 12}, str(rm["histogram"]))
    check("fraction with all readers = 0.75", abs(rm["fraction_with_all_readers"] - 0.75) < 1e-9,
          str(rm["fraction_with_all_readers"]))
    check("none below vote threshold 2", rm["slices_below_vote_threshold"] == 0,
          str(rm["slices_below_vote_threshold"]))

    vs = {r["vote_threshold"]: r for r in agg["vote_sensitivity"]}
    # Empty at vote>=t: the 4 all-empty slices, plus at t=4 the 4 three-reader ones.
    check("empty slices at vote>=2 is 4", vs[2]["empty_slices"] == 4, str(vs[2]["empty_slices"]))
    check("empty slices at vote>=4 is 8", vs[4]["empty_slices"] == 8, str(vs[4]["empty_slices"]))
    check("empty fraction at vote>=2 = 0.25", abs(vs[2]["empty_slice_fraction"] - 0.25) < 1e-9,
          str(vs[2]["empty_slice_fraction"]))
    check("positive px frac decreases with threshold",
          vs[1]["positive_pixel_fraction"] >= vs[4]["positive_pixel_fraction"],
          f"{vs[1]['positive_pixel_fraction']} vs {vs[4]['positive_pixel_fraction']}")
    check("imbalance ratio computed", vs[2]["imbalance_ratio"] and vs[2]["imbalance_ratio"] > 1,
          str(vs[2]["imbalance_ratio"]))

    s = agg["sizes"]
    # 8 big discs (slices 0 and 1 of 4 nodules) + 4 tiny = 12 components.
    check("12 components natively", s["components_total_native"] == 12,
          str(s["components_total_native"]))
    check("resized area mean < native mean",
          s["resized_component_area_px"]["mean"] < s["native_component_area_px"]["mean"],
          f"{s['resized_component_area_px']['mean']} vs {s['native_component_area_px']['mean']}")
    check("diameter histogram sums to component count",
          sum(s["diameter_histogram_native"].values()) == 12,
          str(s["diameter_histogram_native"]))
    check("histogram uses the declared bins",
          list(s["diameter_histogram_native"]) == SIZE_BIN_LABELS, str(list(s["diameter_histogram_native"])))
    check("resize survival table present",
          len(s["resize_survival"]) == 5 and all("retained_fraction" in r for r in s["resize_survival"]))
    # The radius-2 disc (~13px) drops to ~3px, so it is lost at min_area 9 and 16.
    surv9 = next(r for r in s["resize_survival"] if r["min_area_px"] == 9)
    check("tiny nodules lost at min_area 9",
          surv9["components_resized"] < surv9["components_native"],
          f"{surv9['components_resized']} vs {surv9['components_native']}")

    ag = agg["inter_reader_agreement"]
    check("agreement computed for all 16", ag["n"] == 16, str(ag["n"]))
    check("agreement is 1.0 (identical readers)", abs(ag["mean"] - 1.0) < 1e-9, str(ag["mean"]))

    # --------------------------------------------- disagreeing readers
    print("\ninter-reader disagreement")
    dis_root = tmp / "dis" / "LIDC-IDRI-slices" / "LIDC-IDRI-0009" / "nodule-0"
    (dis_root / "images").mkdir(parents=True)
    for r in range(4):
        (dis_root / f"mask-{r}").mkdir(parents=True)
    cv2.imwrite(str(dis_root / "images" / "0.png"), np.full((NATIVE, NATIVE), 90, np.uint8))
    # Two readers mark one disc, two mark a disjoint disc: pairwise Dice over
    # 6 pairs = 4 disjoint pairs (0.0) + 2 identical pairs (1.0) -> 1/3.
    for r in (0, 1):
        cv2.imwrite(str(dis_root / f"mask-{r}" / "0.png"), disc(NATIVE, 60, 60, 15))
    for r in (2, 3):
        cv2.imwrite(str(dis_root / f"mask-{r}" / "0.png"), disc(NATIVE, 200, 200, 15))

    drecs = build_index(tmp / "dis" / "LIDC-IDRI-slices", None, 4)
    dp = profile_slice(drecs[0], 128, 2, 4)
    check("disjoint readers -> pairwise Dice 1/3",
          abs(dp.mean_pairwise_dice - 1 / 3) < 1e-6, str(dp.mean_pairwise_dice))
    check("vote>=2 keeps both discs", len(dp.component_areas) == 2, str(dp.component_areas))
    check("vote>=3 drops both", dp.vote_px[3] == 0, str(dp.vote_px))

    # ----------------------------------------------------- end to end
    print("\nend-to-end run")
    result = run(cfg, max_slices=None, out_dir=tmp / "prof")
    check("run completes", result["slices_profiled"] == 16, str(result["slices_profiled"]))
    check("no read failures", result["n_failures"] == 0, str(result["n_failures"]))
    check("counts: all nodules = 16", result["counts"]["all_nodules"]["slices"] == 16)
    check("counts: nodule-0 only = 12", result["counts"]["nodule_0_only"]["slices"] == 12,
          str(result["counts"]["nodule_0_only"]["slices"]))
    check("profile.json written", (tmp / "prof" / "profile.json").exists())
    check("profile.md written", (tmp / "prof" / "profile.md").exists())

    md = (tmp / "prof" / "profile.md").read_text(encoding="utf-8")
    for heading in ("Native resolution", "Reader-mask histogram", "Vote threshold",
                    "Nodule size distribution", "Inter-reader agreement",
                    "all nodules vs nodule-0"):
        check(f"report covers '{heading}'", heading in md)
    check("report states the +33% nodule gain", "+33.3%" in md,
          [ln for ln in md.splitlines() if "adds" in ln][:1])

    # JSON must round-trip -- no numpy scalars left in it.
    reloaded = json.loads((tmp / "prof" / "profile.json").read_text(encoding="utf-8"))
    check("json is serialisable and complete",
          reloaded["measurements"]["sizes"]["components_total_native"] == 12)

    print("\nsampling")
    r2 = run(cfg, max_slices=8, out_dir=tmp / "prof2")
    check("sampling honours max-slices", r2["slices_profiled"] == 8, str(r2["slices_profiled"]))
    check("sampling flagged in output", r2["sampled"] is True)

finally:
    shutil.rmtree(tmp, ignore_errors=True)

print(f"\n{'=' * 52}\n{len(PASS)} passed, {len(FAIL)} failed")
if FAIL:
    for f in FAIL:
        print(f"  FAILED: {f}")
    sys.exit(1)
print("all green")
