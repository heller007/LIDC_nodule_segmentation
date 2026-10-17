"""Dataset profiler -- measures the quantities RESEARCH_NOTES lists as unverified.

    python -m lidcseg.profile_data --config configs/improved.yaml
    python -m lidcseg.profile_data --config configs/improved.yaml --max-slices 1500

Answers, with numbers:

1. positive pixel fraction, and how it moves with the reader-vote threshold
2. fraction of slices whose merged mask is empty, per vote threshold
3. reader-mask histogram -- how many slices have 4 masks, 3, fewer
4. native resolution of the exported slices, before any resize
5. nodule size distribution in pixels, native *and* after the training resize,
   which is what says whether 128x128 is costing small nodules
6. slice/patient/nodule counts with all nodules vs `nodule-0` only

Masks are read at **native** resolution; the training cache has already
resized, so it cannot answer (4) or the native half of (5). Pure numpy and
OpenCV -- no TensorFlow, so this runs anywhere.

Writes `profile.json`, `profile.md` (tables ready to paste into the notes) and
figures, into `<out_dir>/profile/` by default.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from concurrent.futures import ThreadPoolExecutor
from collections import Counter
from dataclasses import dataclass, field
from pathlib import Path

import cv2
import numpy as np

from .config import Config, load_config
from .data.index import SliceRecord, build_index, index_stats

# Equivalent-diameter bins in pixels, for the size-stratified metrics that
# RESEARCH_NOTES section 6 item 4 calls for. Kept here so the profiler and the
# later evaluation use identical bins.
SIZE_BINS_PX = [0, 4, 8, 16, 32, 64, np.inf]
SIZE_BIN_LABELS = ["<4", "4-8", "8-16", "16-32", "32-64", ">=64"]

# "Is this nodule still there after the resize?" thresholds, in pixels of area.
SURVIVAL_THRESHOLDS = [1, 4, 9, 16, 25]


@dataclass
class SliceProfile:
    """Per-slice measurements. Scalars only -- no masks are retained."""

    patient: str
    nodule: str
    height: int
    width: int
    n_masks: int
    # Positive pixels for each reader present, at native resolution.
    reader_px: tuple[int, ...]
    # Positive pixels of the merged mask at each vote threshold 1..n_readers.
    vote_px: dict[int, int]
    total_px: int
    # Connected components of the merged mask at the configured threshold.
    component_areas: tuple[int, ...]
    # Same mask binarised then resized to img_size, as the pipeline does.
    resized_px: int
    resized_component_areas: tuple[int, ...]
    # Mean pairwise Dice between readers; None when fewer than 2 masks exist.
    mean_pairwise_dice: float | None
    # Each reader vs the others' majority consensus; None with fewer than 3.
    reader_vs_consensus_dice: float | None


def _dice(a: np.ndarray, b: np.ndarray) -> float:
    """Dice between two boolean masks; both-empty counts as perfect."""
    a, b = a.astype(bool), b.astype(bool)
    denom = a.sum() + b.sum()
    return 1.0 if denom == 0 else 2.0 * np.count_nonzero(a & b) / denom


def _reader_vs_consensus_dice(masks: list[np.ndarray], vote: int) -> float | None:
    """Leave-one-reader-out: each reader scored against the others' consensus.

    This is the *correct* ceiling for a model trained on a fused target, and it
    is higher than mean pairwise agreement. Predicting a consensus is easier
    than predicting an individual, because fusing several readers averages out
    each one's idiosyncratic noise -- so comparing a model's score against the
    fused mask to reader-vs-reader Dice understates what is achievable.

    Here reader r is compared against the majority vote of the remaining
    readers, which is the same kind of quantity the model is asked to produce.
    """
    if len(masks) < 3:
        return None
    scores = []
    for r in range(len(masks)):
        others = [m for i, m in enumerate(masks) if i != r]
        # Scale the vote threshold to the smaller panel, keeping "majority".
        thr = max(1, min(vote, len(others)))
        consensus = (np.sum(others, axis=0) >= thr)
        scores.append(_dice(masks[r], consensus))
    return float(np.mean(scores))


def _pairwise_dice(masks: list[np.ndarray]) -> float | None:
    """Mean Dice over every pair of reader masks -- raw annotator agreement."""
    if len(masks) < 2:
        return None
    scores = []
    for i in range(len(masks)):
        for j in range(i + 1, len(masks)):
            # Two readers who both marked nothing agree perfectly.
            scores.append(_dice(masks[i], masks[j]))
    return float(np.mean(scores))


def _components(mask: np.ndarray) -> tuple[int, ...]:
    """Areas of the 8-connected foreground components, largest first."""
    if not mask.any():
        return ()
    n, _, stats, _ = cv2.connectedComponentsWithStats(mask.astype(np.uint8), connectivity=8)
    areas = sorted((int(stats[k, cv2.CC_STAT_AREA]) for k in range(1, n)), reverse=True)
    return tuple(areas)


def profile_slice(
    rec: SliceRecord,
    img_size: int,
    vote_threshold: int,
    n_readers: int,
    source=None,
) -> SliceProfile:
    """Measure one slice.

    `source` is an optional ZipSource; when given, members are decoded from
    the zip instead of read from the filesystem. Both paths produce identical
    arrays, so every number below is unaffected by the choice.
    """
    read = source.imread_gray if source is not None else (
        lambda p: cv2.imread(p, cv2.IMREAD_GRAYSCALE)
    )

    img = read(rec.image)
    if img is None:
        raise FileNotFoundError(f"could not read image: {rec.image}")
    h, w = img.shape[:2]

    masks: list[np.ndarray] = []
    for mp in rec.masks:
        m = read(mp)
        if m is None:
            continue
        if m.shape[:2] != (h, w):
            # A reader mask that disagrees with its image on shape would
            # silently misalign; resize it to the image and keep going.
            m = cv2.resize(m, (w, h), interpolation=cv2.INTER_NEAREST)
        masks.append((m > 127).astype(np.uint8))

    votes = np.zeros((h, w), dtype=np.uint8)
    for m in masks:
        votes += m

    vote_px = {t: int(np.count_nonzero(votes >= t)) for t in range(1, n_readers + 1)}

    merged = (votes >= vote_threshold).astype(np.uint8)

    # Binarise then nearest-resize -- exactly what data.preprocess does, so the
    # survival numbers describe the masks the model actually trains on.
    resized = (
        cv2.resize(merged, (img_size, img_size), interpolation=cv2.INTER_NEAREST)
        if (h, w) != (img_size, img_size)
        else merged
    )

    return SliceProfile(
        patient=rec.patient,
        nodule=rec.nodule,
        height=h,
        width=w,
        n_masks=len(masks),
        reader_px=tuple(int(m.sum()) for m in masks),
        vote_px=vote_px,
        total_px=h * w,
        component_areas=_components(merged),
        resized_px=int(np.count_nonzero(resized)),
        resized_component_areas=_components(resized),
        mean_pairwise_dice=_pairwise_dice(masks),
        reader_vs_consensus_dice=_reader_vs_consensus_dice(masks, vote_threshold),
    )


# ------------------------------------------------------------------ aggregate


def _pct(values, qs=(5, 25, 50, 75, 95)) -> dict:
    if len(values) == 0:
        return {}
    a = np.asarray(values, dtype=np.float64)
    return {f"p{q}": round(float(np.percentile(a, q)), 3) for q in qs}


def _describe(values, qs=(5, 25, 50, 75, 95)) -> dict:
    if len(values) == 0:
        return {"n": 0}
    a = np.asarray(values, dtype=np.float64)
    return {
        "n": int(a.size),
        "mean": round(float(a.mean()), 4),
        "std": round(float(a.std()), 4),
        "min": round(float(a.min()), 4),
        "max": round(float(a.max()), 4),
        **_pct(a, qs),
    }


def aggregate(profiles: list[SliceProfile], cfg: Config, n_readers: int) -> dict:
    size = cfg.data.img_size
    vt = cfg.data.vote_threshold
    n = len(profiles)

    # (4) native resolution
    shapes = Counter((p.height, p.width) for p in profiles)
    resolution = {
        "distinct_shapes": len(shapes),
        "shapes": [{"h": h, "w": w, "slices": c} for (h, w), c in shapes.most_common(10)],
        "uniform": len(shapes) == 1,
    }

    # (3) reader-mask histogram
    mask_hist = Counter(p.n_masks for p in profiles)
    reader_masks = {
        "histogram": {str(k): mask_hist[k] for k in sorted(mask_hist)},
        "fraction_with_all_readers": round(mask_hist[n_readers] / n, 4) if n else 0.0,
        # Below the vote threshold a slice can never be positive: its label is
        # all-background by construction, not by anatomy.
        "slices_below_vote_threshold": sum(c for k, c in mask_hist.items() if k < vt),
        "fraction_below_vote_threshold": round(
            sum(c for k, c in mask_hist.items() if k < vt) / n, 4
        )
        if n
        else 0.0,
    }

    # (1) + (2) vote-threshold sensitivity
    total_px = sum(p.total_px for p in profiles)
    vote_rows = []
    for t in range(1, n_readers + 1):
        pos = sum(p.vote_px[t] for p in profiles)
        empty = sum(1 for p in profiles if p.vote_px[t] == 0)
        vote_rows.append(
            {
                "vote_threshold": t,
                "positive_pixel_fraction": round(pos / total_px, 8) if total_px else 0.0,
                "positive_pixels": pos,
                "empty_slices": empty,
                "empty_slice_fraction": round(empty / n, 4) if n else 0.0,
                "imbalance_ratio": round((total_px - pos) / pos, 1) if pos else None,
            }
        )

    # (5) nodule size distribution, native and resized
    native_areas = [a for p in profiles for a in p.component_areas]
    resized_areas = [a for p in profiles for a in p.resized_component_areas]
    # Equivalent diameter of a circle with the same area.
    native_diam = [2.0 * np.sqrt(a / np.pi) for a in native_areas]
    resized_diam = [2.0 * np.sqrt(a / np.pi) for a in resized_areas]

    diam_hist, _ = np.histogram(native_diam, bins=SIZE_BINS_PX) if native_diam else (np.zeros(6, int), None)
    rdiam_hist, _ = np.histogram(resized_diam, bins=SIZE_BINS_PX) if resized_diam else (np.zeros(6, int), None)

    survival = []
    for thr in SURVIVAL_THRESHOLDS:
        before = sum(1 for a in native_areas if a >= thr)
        after = sum(1 for a in resized_areas if a >= thr)
        survival.append(
            {
                "min_area_px": thr,
                "components_native": before,
                "components_resized": after,
                "retained_fraction": round(after / before, 4) if before else None,
            }
        )

    lost = sum(1 for p in profiles if p.component_areas and not p.resized_component_areas)

    sizes = {
        "native_component_area_px": _describe(native_areas),
        "resized_component_area_px": _describe(resized_areas),
        "native_equiv_diameter_px": _describe(native_diam),
        "resized_equiv_diameter_px": _describe(resized_diam),
        "diameter_histogram_native": dict(zip(SIZE_BIN_LABELS, [int(x) for x in diam_hist])),
        "diameter_histogram_resized": dict(zip(SIZE_BIN_LABELS, [int(x) for x in rdiam_hist])),
        "components_total_native": len(native_areas),
        "components_total_resized": len(resized_areas),
        "resize_survival": survival,
        "slices_losing_all_foreground_on_resize": lost,
        "fraction_losing_all_foreground": round(lost / n, 5) if n else 0.0,
        "components_per_nonempty_slice": round(
            len(native_areas) / max(sum(1 for p in profiles if p.component_areas), 1), 3
        ),
    }

    # Inter-reader agreement -- the practical Dice ceiling.
    agreements = [p.mean_pairwise_dice for p in profiles if p.mean_pairwise_dice is not None]
    agreement = {
        **_describe(agreements),
        "note": "mean pairwise Dice between radiologists -- raw annotator "
        "agreement, NOT the ceiling for a model predicting a fused target",
    }

    loo = [p.reader_vs_consensus_dice for p in profiles
           if p.reader_vs_consensus_dice is not None]
    consensus_ceiling = {
        **_describe(loo),
        "note": "each reader vs the majority consensus of the others -- the "
        "correct ceiling for a model trained on the fused mask, since "
        "predicting a consensus is easier than predicting an individual",
    }

    return {
        "resolution": resolution,
        "reader_masks": reader_masks,
        "vote_sensitivity": vote_rows,
        "configured_vote_threshold": vt,
        "train_img_size": size,
        "sizes": sizes,
        "inter_reader_agreement": agreement,
        "consensus_ceiling": consensus_ceiling,
    }


# -------------------------------------------------------------------- report


def _md_table(headers: list[str], rows: list[list]) -> str:
    out = ["| " + " | ".join(headers) + " |", "|" + "|".join(["---"] * len(headers)) + "|"]
    for r in rows:
        out.append("| " + " | ".join("" if v is None else str(v) for v in r) + " |")
    return "\n".join(out)


def write_report(result: dict, out_dir: Path) -> Path:
    counts = result["counts"]
    agg = result["measurements"]
    size = agg["train_img_size"]
    vt = agg["configured_vote_threshold"]

    L: list[str] = [
        "# Dataset profile",
        "",
        f"Root: `{result['root']}`  ",
        f"Generated: {result['generated']}  ",
        f"Slices profiled: **{result['slices_profiled']}**"
        + (f" (sampled from {counts['all_nodules']['slices']})" if result["sampled"] else "")
        + "  ",
        f"Training resize target: {size}x{size}, vote threshold >={vt}",
        "",
        "## 6. Counts: all nodules vs nodule-0 only",
        "",
    ]

    a, z = counts["all_nodules"], counts["nodule_0_only"]
    L.append(
        _md_table(
            ["scope", "slices", "patients", "nodules", "slices/patient"],
            [
                ["all nodules", a["slices"], a["patients"], a["nodules"], a["slices_per_patient"]],
                ["`nodule-0` only", z["slices"], z["patients"], z["nodules"], z["slices_per_patient"]],
            ],
        )
    )
    extra = a["slices"] - z["slices"]
    pct = 100.0 * extra / z["slices"] if z["slices"] else 0.0
    L += [
        "",
        f"Using every nodule adds **{extra} slices ({pct:+.1f}%)** and "
        f"{a['nodules'] - z['nodules']} nodules over the notebook's `nodule-0` filter.",
        "",
        "## 4. Native resolution",
        "",
    ]

    res = agg["resolution"]
    L.append(
        _md_table(
            ["height", "width", "slices"],
            [[s["h"], s["w"], s["slices"]] for s in res["shapes"]],
        )
    )
    L += ["", f"Distinct shapes: {res['distinct_shapes']} "
          f"({'uniform' if res['uniform'] else 'MIXED -- resize behaviour varies per slice'})", ""]

    if not res["uniform"]:
        L += [
            "> Mixed native resolutions mean a fixed resize changes scale "
            "non-uniformly across slices, so pixel areas are not comparable "
            "between patients without normalising by native shape.",
            "",
        ]

    L += ["## 3. Reader-mask histogram", ""]
    rm = agg["reader_masks"]
    L.append(_md_table(["masks present", "slices"], [[k, v] for k, v in rm["histogram"].items()]))
    L += [
        "",
        f"Slices with the full reader set: **{100 * rm['fraction_with_all_readers']:.1f}%**  ",
        f"Slices with fewer than {vt} masks: **{rm['slices_below_vote_threshold']} "
        f"({100 * rm['fraction_below_vote_threshold']:.2f}%)**",
        "",
    ]
    if rm["slices_below_vote_threshold"]:
        L += [
            f"> These cannot be positive under a >={vt} vote regardless of anatomy -- "
            "their label is all-background by construction. They inflate the "
            "negative class and should probably be dropped rather than trained on.",
            "",
        ]

    L += ["## 1 + 2. Vote threshold: imbalance and empty masks", ""]
    L.append(
        _md_table(
            ["vote >=", "positive pixel frac", "bg:fg ratio", "empty slices", "empty frac"],
            [
                [
                    r["vote_threshold"],
                    f"{r['positive_pixel_fraction']:.6f}",
                    r["imbalance_ratio"],
                    r["empty_slices"],
                    f"{r['empty_slice_fraction']:.4f}",
                ]
                for r in agg["vote_sensitivity"]
            ],
        )
    )
    cur = next(r for r in agg["vote_sensitivity"] if r["vote_threshold"] == vt)
    L += [
        "",
        f"At the configured threshold (>={vt}): **{100 * cur['positive_pixel_fraction']:.4f}%** of "
        f"pixels are nodule, a background:foreground ratio of about "
        f"**{cur['imbalance_ratio']}:1**, and "
        f"**{100 * cur['empty_slice_fraction']:.1f}%** of slices have an empty mask.",
        "",
        "> This is why pixel accuracy is meaningless here: predicting all "
        "background scores "
        f"{100 * (1 - cur['positive_pixel_fraction']):.3f}%.",
        "",
        "## 5. Nodule size distribution",
        "",
    ]

    s = agg["sizes"]
    L.append(
        _md_table(
            ["metric", "n", "mean", "p5", "p25", "p50", "p75", "p95", "max"],
            [
                [
                    name,
                    d.get("n"),
                    d.get("mean"),
                    d.get("p5"),
                    d.get("p25"),
                    d.get("p50"),
                    d.get("p75"),
                    d.get("p95"),
                    d.get("max"),
                ]
                for name, d in (
                    ("area px (native)", s["native_component_area_px"]),
                    (f"area px ({size}x{size})", s["resized_component_area_px"]),
                    ("equiv diameter px (native)", s["native_equiv_diameter_px"]),
                    (f"equiv diameter px ({size}x{size})", s["resized_equiv_diameter_px"]),
                )
            ],
        )
    )

    L += ["", "### Equivalent-diameter histogram (px)", ""]
    L.append(
        _md_table(
            ["diameter bin", "native", f"after {size}x{size} resize"],
            [
                [lab, s["diameter_histogram_native"].get(lab, 0), s["diameter_histogram_resized"].get(lab, 0)]
                for lab in SIZE_BIN_LABELS
            ],
        )
    )

    L += ["", "### Does the resize destroy small nodules?", ""]
    L.append(
        _md_table(
            ["min area px", "components native", "after resize", "retained"],
            [
                [
                    r["min_area_px"],
                    r["components_native"],
                    r["components_resized"],
                    f"{100 * r['retained_fraction']:.1f}%" if r["retained_fraction"] is not None else None,
                ]
                for r in s["resize_survival"]
            ],
        )
    )
    L += [
        "",
        f"Slices that had foreground natively and none after resize: "
        f"**{s['slices_losing_all_foreground_on_resize']} "
        f"({100 * s['fraction_losing_all_foreground']:.3f}%)**  ",
        f"Components per non-empty slice: {s['components_per_nonempty_slice']}",
        "",
        "## Inter-reader agreement",
        "",
    ]

    ag = agg["inter_reader_agreement"]
    cc = agg.get("consensus_ceiling", {})
    if ag.get("n"):
        L += [
            _md_table(
                ["quantity", "mean", "sd", "p5", "median", "p95"],
                [
                    ["reader vs reader (pairwise)", f"{ag['mean']:.4f}", f"{ag['std']:.4f}",
                     ag.get("p5"), ag.get("p50"), ag.get("p95")],
                ] + ([
                    ["reader vs others' consensus", f"{cc['mean']:.4f}", f"{cc['std']:.4f}",
                     cc.get("p5"), cc.get("p50"), cc.get("p95")],
                ] if cc.get("n") else []),
            ),
            "",
            "**Read the second row, not the first.** Pairwise reader agreement "
            "measures how much the annotators disagree with each other. It is "
            "not the ceiling for a model, because the model is trained against "
            f"the vote->={vt} *fusion* of all readers, and predicting a consensus "
            "is easier than predicting any individual -- fusing averages out "
            "each reader's idiosyncratic noise.",
            "",
            "The second row is the right comparison: each reader scored against "
            "the majority consensus of the others, which is the same kind of "
            "target the model produces.",
        ]
        if cc.get("n"):
            L += [
                "",
                f"So a model scoring much above **{cc['mean']:.3f}** per-image Dice "
                "on this target is doing something a radiologist cannot do against "
                "their colleagues' consensus -- which points at leakage, at an "
                "easier metric (global rather than per-image Dice), or at fitting "
                "the fusion rule rather than the anatomy.",
            ]
    else:
        L.append("Not computable -- fewer than 2 reader masks available.")

    L += ["", "---", "", "Generated by `python -m lidcseg.profile_data`.", ""]

    path = out_dir / "profile.md"
    path.write_text("\n".join(L), encoding="utf-8")
    return path


def write_figures(profiles: list[SliceProfile], agg: dict, out_dir: Path) -> None:
    try:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        print("[profile] matplotlib unavailable -- skipping figures")
        return

    size = agg["train_img_size"]
    native = [a for p in profiles for a in p.component_areas]
    resized = [a for p in profiles for a in p.resized_component_areas]

    fig, axes = plt.subplots(2, 2, figsize=(13, 8))

    ax = axes[0, 0]
    if native:
        nd = [2 * np.sqrt(a / np.pi) for a in native]
        rd = [2 * np.sqrt(a / np.pi) for a in resized] or [0]
        bins = np.linspace(0, max(max(nd), max(rd)) or 1, 50)
        ax.hist(nd, bins=bins, alpha=0.65, label="native")
        ax.hist(rd, bins=bins, alpha=0.65, label=f"{size}x{size}")
        ax.set_xlabel("equivalent diameter (px)")
        ax.set_ylabel("components")
        ax.set_title("Nodule size distribution")
        ax.legend()
    ax.grid(alpha=0.3)

    ax = axes[0, 1]
    rows = agg["vote_sensitivity"]
    ts = [r["vote_threshold"] for r in rows]
    ax.plot(ts, [r["positive_pixel_fraction"] for r in rows], marker="o", label="positive px frac")
    ax.set_yscale("log")
    ax.set_xlabel("reader vote threshold")
    ax.set_ylabel("positive pixel fraction (log)")
    ax.set_xticks(ts)
    ax2 = ax.twinx()
    ax2.plot(ts, [r["empty_slice_fraction"] for r in rows], marker="s", color="#C44E52",
             label="empty slice frac")
    ax2.set_ylabel("empty slice fraction")
    ax.set_title("Vote threshold sensitivity")
    ax.grid(alpha=0.3)
    lines = ax.get_lines() + ax2.get_lines()
    ax.legend(lines, [l.get_label() for l in lines], fontsize=8)

    ax = axes[1, 0]
    hist = agg["reader_masks"]["histogram"]
    ax.bar(list(hist.keys()), list(hist.values()), color="#55A868")
    ax.set_xlabel("reader masks present")
    ax.set_ylabel("slices")
    ax.set_title("Reader-mask availability")
    ax.grid(axis="y", alpha=0.3)

    ax = axes[1, 1]
    ags = [p.mean_pairwise_dice for p in profiles if p.mean_pairwise_dice is not None]
    if ags:
        ax.hist(ags, bins=40, color="#8172B2")
        ax.axvline(float(np.mean(ags)), color="#C44E52", ls="--",
                   label=f"mean {np.mean(ags):.3f}")
        ax.legend()
    ax.set_xlabel("mean pairwise Dice between readers")
    ax.set_ylabel("slices")
    ax.set_title("Inter-reader agreement (Dice ceiling)")
    ax.grid(alpha=0.3)

    fig.tight_layout()
    fig.savefig(out_dir / "profile.png", dpi=140)
    plt.close(fig)
    print(f"[profile] figures -> {out_dir / 'profile.png'}")


# ---------------------------------------------------------------------- main


def default_workers() -> int:
    """Thread count for the mask reads.

    I/O-bound against a network filesystem, so useful parallelism far exceeds
    the core count; capped at 32 so a huge machine does not thrash the mount.
    """
    return max(4, min(32, (os.cpu_count() or 4) * 4))


def _profile_all(
    subset: list[SliceRecord],
    cfg,
    workers: int,
    source=None,
) -> tuple[list[SliceProfile], list[dict]]:
    """Profile every record, in parallel, preserving input order.

    Order is preserved via `executor.map` so the output is identical to the
    serial version regardless of thread scheduling -- aggregates would be the
    same either way, but a reproducible `profile.json` is worth more than the
    microseconds saved by `as_completed`.
    """
    size, vt, nr = cfg.data.img_size, cfg.data.vote_threshold, cfg.data.n_readers
    total = len(subset)

    def work(rec: SliceRecord):
        try:
            return profile_slice(rec, size, vt, nr, source), None
        except Exception as e:
            return None, {"uid": rec.uid, "error": f"{type(e).__name__}: {e}"}

    profiles: list[SliceProfile] = []
    failures: list[dict] = []
    t0 = time.time()
    step = max(total // 20, 1)

    if workers <= 1:
        results = (work(r) for r in subset)
    else:
        pool = ThreadPoolExecutor(max_workers=workers)
        results = pool.map(work, subset)

    try:
        for i, (prof, err) in enumerate(results, 1):
            if prof is not None:
                profiles.append(prof)
            else:
                failures.append(err)
            if i % step == 0 or i == total:
                dt = time.time() - t0
                rate = i / dt if dt > 0 else 0.0
                eta = (total - i) / rate if rate > 0 else 0.0
                print(
                    f"  {i}/{total}  ({100 * i / total:4.1f}%)  "
                    f"{rate:6.1f} slices/s  eta {eta / 60:4.1f} min",
                    flush=True,
                )
    finally:
        if workers > 1:
            pool.shutdown(wait=True)

    return profiles, failures


def run(
    cfg: Config,
    max_slices: int | None = None,
    out_dir: Path | None = None,
    workers: int | None = None,
    zip_path: str | Path | None = None,
) -> dict:
    t0 = time.time()
    workers = default_workers() if workers is None else max(1, int(workers))
    out_dir = Path(out_dir or (Path(cfg.out_dir) / "profile"))
    out_dir.mkdir(parents=True, exist_ok=True)

    source = None
    if zip_path:
        from .data.zip_source import ZipSource

        print(f"[profile] reading from zip (no extraction): {zip_path}")
        source = ZipSource(zip_path)
        index_all = lambda nods: source.build_index(nods, cfg.data.n_readers)  # noqa: E731
    else:
        index_all = lambda nods: build_index(cfg.data.root, nods, cfg.data.n_readers)  # noqa: E731

    print("[profile] indexing with all nodules")
    recs_all = index_all(None)
    print("[profile] indexing with nodule-0 only")
    try:
        recs_n0 = index_all(["nodule-0"])
    except RuntimeError:
        recs_n0 = []

    counts = {
        "all_nodules": index_stats(recs_all),
        "nodule_0_only": index_stats(recs_n0) if recs_n0 else {
            "slices": 0, "patients": 0, "nodules": 0, "slices_per_patient": 0.0
        },
    }
    print(f"[profile] all nodules  : {json.dumps(counts['all_nodules'])}")
    print(f"[profile] nodule-0 only: {json.dumps(counts['nodule_0_only'])}")

    # Profile the full set unless asked to sample. Sampling is stratified by
    # patient order via a fixed stride, so it is not biased toward low ids.
    sampled = bool(max_slices and max_slices < len(recs_all))
    if sampled:
        stride = len(recs_all) / max_slices
        idx = sorted({int(i * stride) for i in range(max_slices)})
        subset = [recs_all[i] for i in idx if i < len(recs_all)]
        print(f"[profile] sampling {len(subset)} of {len(recs_all)} slices")
    else:
        subset = recs_all

    print(f"[profile] reading {len(subset)} slices x {cfg.data.n_readers + 1} files "
          f"with {workers} threads")

    # Each slice costs 5 file reads (1 image + 4 reader masks), so the full set
    # is ~29k reads of ~6KB files. On Kaggle's network-backed /kaggle/input the
    # per-file latency dominates completely -- the dataset is only 170MB, but
    # serial reads take tens of minutes. The work is I/O-bound and cv2.imread
    # releases the GIL, so threads give a near-linear speedup.
    profiles, failures = _profile_all(subset, cfg, workers, source)

    if not profiles:
        raise RuntimeError(f"profiled nothing; {len(failures)} failures, first: {failures[:1]}")

    agg = aggregate(profiles, cfg, cfg.data.n_readers)

    result = {
        "root": str(zip_path) if zip_path else str(cfg.data.root),
        "source": "zip" if zip_path else "filesystem",
        "generated": time.strftime("%Y-%m-%d %H:%M:%S"),
        "config": cfg.name,
        "slices_profiled": len(profiles),
        "sampled": sampled,
        "workers": workers,
        "failures": failures[:50],
        "n_failures": len(failures),
        "counts": counts,
        "measurements": agg,
        "minutes": round((time.time() - t0) / 60, 2),
    }

    (out_dir / "profile.json").write_text(json.dumps(result, indent=2), encoding="utf-8")
    md = write_report(result, out_dir)
    write_figures(profiles, agg, out_dir)

    if source is not None:
        source.close()

    if failures:
        print(f"[profile] {len(failures)} slices failed to read (see profile.json)")
    print(f"\n[profile] {result['minutes']} min -> {out_dir}")
    print(f"[profile] report: {md}")
    return result


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description="Profile the LIDC slice dataset")
    ap.add_argument("--config", required=True,
                    help="YAML config; supplies img_size, vote_threshold, n_readers")
    ap.add_argument("--set", dest="overrides", nargs="*", default=[], metavar="KEY=VALUE")
    ap.add_argument("--max-slices", type=int, default=None,
                    help="profile a strided sample instead of everything")
    ap.add_argument("--workers", type=int, default=None,
                    help=f"reader threads (default {default_workers()}); the reads are "
                         "I/O-bound on Kaggle, so more than the core count helps")
    ap.add_argument("--zip", dest="zip_path", default=None,
                    help="read directly from a dataset .zip instead of an extracted "
                         "tree; avoids writing ~78k tiny files")
    ap.add_argument("--out-dir", default=None)
    ap.add_argument("--print-report", action="store_true",
                    help="echo the markdown report to stdout when done")
    args = ap.parse_args(argv)

    cfg = load_config(args.config, args.overrides)
    result = run(cfg, args.max_slices, Path(args.out_dir) if args.out_dir else None,
                 workers=args.workers, zip_path=args.zip_path)

    if args.print_report:
        out = Path(args.out_dir or (Path(cfg.out_dir) / "profile")) / "profile.md"
        print("\n" + out.read_text(encoding="utf-8"))
    return 0


if __name__ == "__main__":
    sys.exit(main())
