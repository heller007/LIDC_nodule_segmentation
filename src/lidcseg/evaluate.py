"""Inference and evaluation: TTA, threshold tuning, final metrics.

The threshold is tuned on validation and then applied unchanged to test. That
ordering is what makes the test number honest, and it is one thing the
notebook got right -- worth keeping explicit so it does not drift.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from .config import EvalConfig
from .metrics import all_metrics, bootstrap_ci, per_image_dice

# The 8 dihedral transforms, as (n_rot90, flip) pairs.
_TTA_OPS = [(k, f) for f in (False, True) for k in range(4)]


def _apply(x: np.ndarray, k: int, flip: bool) -> np.ndarray:
    if flip:
        x = x[:, :, ::-1]
    return np.rot90(x, k, axes=(1, 2))


def _unapply(x: np.ndarray, k: int, flip: bool) -> np.ndarray:
    x = np.rot90(x, -k, axes=(1, 2))
    if flip:
        x = x[:, :, ::-1]
    return x


def predict(model, x: np.ndarray, batch_size: int = 32, tta: bool = False) -> np.ndarray:
    """Predict probabilities, optionally averaged over 8 dihedral transforms.

    Averaging in probability space (not after thresholding) keeps the soft
    scores usable for the threshold sweep.
    """

    def _raw(arr: np.ndarray) -> np.ndarray:
        out = model.predict(arr, batch_size=batch_size, verbose=0)
        # Deep supervision returns a dict or list; the main head is what counts.
        if isinstance(out, dict):
            out = out["main_output"]
        elif isinstance(out, (list, tuple)):
            out = out[0]
        return np.asarray(out, dtype=np.float32)

    if not tta:
        return _raw(x)

    acc = np.zeros((x.shape[0], x.shape[1], x.shape[2], 1), dtype=np.float32)
    for k, flip in _TTA_OPS:
        pred = _raw(np.ascontiguousarray(_apply(x, k, flip)))
        acc += _unapply(pred, k, flip)
    return acc / len(_TTA_OPS)


def remove_small_components(mask: np.ndarray, min_px: int) -> np.ndarray:
    """Drop connected components below `min_px` pixels, per slice.

    Scattered single-pixel responses cost precision without ever being a real
    nodule; LIDC nodules are >=3mm, which is several pixels at this resolution.
    """
    if min_px <= 0:
        return mask
    try:
        import cv2
    except ImportError:
        return mask

    out = np.zeros_like(mask)
    for i in range(mask.shape[0]):
        m = (mask[i, ..., 0] > 0).astype(np.uint8)
        n, labels, stats, _ = cv2.connectedComponentsWithStats(m, connectivity=8)
        keep = np.zeros_like(m)
        for lab in range(1, n):
            if stats[lab, cv2.CC_STAT_AREA] >= min_px:
                keep[labels == lab] = 1
        out[i, ..., 0] = keep
    return out.astype(mask.dtype)


def sweep_threshold(
    y_true: np.ndarray,
    y_prob: np.ndarray,
    cfg: EvalConfig,
) -> tuple[float, list[dict]]:
    """Find the threshold maximising mean per-image Dice on this split."""
    if not cfg.tune_threshold:
        return cfg.default_threshold, []

    thresholds = np.linspace(cfg.threshold_min, cfg.threshold_max, cfg.threshold_steps)
    trace: list[dict] = []
    best_t, best_d = cfg.default_threshold, -1.0

    for t in thresholds:
        pred = (y_prob > t).astype(np.uint8)
        if cfg.min_component_px > 0:
            pred = remove_small_components(pred, cfg.min_component_px)
        d = float(per_image_dice(y_true, pred).mean())
        trace.append({"threshold": round(float(t), 4), "dice": round(d, 5)})
        if d > best_d:
            best_t, best_d = float(t), d

    return best_t, trace


def evaluate_split(
    model,
    x: np.ndarray,
    y: np.ndarray,
    threshold: float,
    cfg: EvalConfig,
    batch_size: int = 32,
    bootstrap: bool = False,
) -> dict:
    """Predict and score one split at a fixed threshold."""
    prob = predict(model, x, batch_size=batch_size, tta=cfg.tta)
    pred = (prob > threshold).astype(np.uint8)
    if cfg.min_component_px > 0:
        pred = remove_small_components(pred, cfg.min_component_px)

    res = all_metrics(y, pred)
    res["threshold"] = float(threshold)
    res["n_slices"] = int(len(y))
    if bootstrap and cfg.bootstrap > 0:
        res.update(bootstrap_ci(y, pred, n_boot=cfg.bootstrap))
    return res


def full_evaluation(
    model,
    x_val: np.ndarray,
    y_val: np.ndarray,
    x_test: np.ndarray,
    y_test: np.ndarray,
    cfg: EvalConfig,
    batch_size: int = 32,
    out_dir: str | Path | None = None,
) -> dict:
    """Tune on val, report both splits, write results.json."""
    val_prob = predict(model, x_val, batch_size=batch_size, tta=cfg.tta)
    threshold, trace = sweep_threshold(y_val, val_prob, cfg)

    val_pred = (val_prob > threshold).astype(np.uint8)
    if cfg.min_component_px > 0:
        val_pred = remove_small_components(val_pred, cfg.min_component_px)

    results = {
        "threshold": float(threshold),
        "tta": bool(cfg.tta),
        "val": {**all_metrics(y_val, val_pred), "threshold": float(threshold), "n_slices": int(len(y_val))},
        "test": evaluate_split(
            model, x_test, y_test, threshold, cfg, batch_size=batch_size, bootstrap=True
        ),
        "threshold_sweep": trace,
    }

    if out_dir:
        out = Path(out_dir)
        out.mkdir(parents=True, exist_ok=True)
        (out / "results.json").write_text(json.dumps(results, indent=2), encoding="utf-8")

    return results


def format_results(results: dict) -> str:
    """Compact text table for the training log."""
    keys = ["dice", "iou", "dice_global", "iou_global", "precision", "recall", "f1", "accuracy"]
    lines = [
        f"threshold = {results['threshold']:.3f}   TTA = {results['tta']}",
        f"{'metric':<14}{'val':>10}{'test':>10}",
        "-" * 34,
    ]
    for k in keys:
        v, t = results["val"].get(k), results["test"].get(k)
        if v is None or t is None:
            continue
        lines.append(f"{k:<14}{v:>10.4f}{t:>10.4f}")
    ci = results["test"].get("dice_ci")
    if ci:
        lines.append(f"\ntest per-image Dice 95% CI: [{ci[0]:.4f}, {ci[1]:.4f}]")
    return "\n".join(lines)
