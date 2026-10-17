"""Metrics (pure numpy -- importable without TensorFlow).

Two Dice numbers exist and they are not interchangeable:

* **global Dice** pools every pixel of the split into one ratio. Large nodules
  dominate it. This is what the notebook reported (0.8611-0.8826).
* **per-image Dice** scores each slice, then averages. Every slice counts the
  same, so a model that handles big nodules but misses small ones scores
  lower. This is the number clinical papers usually mean.

Both are computed and reported side by side. `val_dice`, used for model
selection, is the per-image one.

The streaming Keras metrics live in `metrics_tf.py` so this module stays
usable for offline analysis of saved predictions.
"""

from __future__ import annotations

import numpy as np

def confusion_counts(y_true: np.ndarray, y_pred: np.ndarray) -> tuple[int, int, int, int]:
    t = y_true.astype(bool).ravel()
    p = y_pred.astype(bool).ravel()
    tp = int(np.count_nonzero(t & p))
    fp = int(np.count_nonzero(~t & p))
    fn = int(np.count_nonzero(t & ~p))
    tn = int(t.size - tp - fp - fn)
    return tp, fp, fn, tn


def per_image_dice(y_true: np.ndarray, y_pred: np.ndarray) -> np.ndarray:
    """Dice for each slice. Empty-and-correct scores 1.0."""
    n = y_true.shape[0]
    t = y_true.reshape(n, -1).astype(bool)
    p = y_pred.reshape(n, -1).astype(bool)
    inter = np.count_nonzero(t & p, axis=1).astype(np.float64)
    denom = t.sum(axis=1) + p.sum(axis=1)
    out = np.ones(n, dtype=np.float64)
    nz = denom > 0
    out[nz] = 2.0 * inter[nz] / denom[nz]
    return out


def per_image_iou(y_true: np.ndarray, y_pred: np.ndarray) -> np.ndarray:
    n = y_true.shape[0]
    t = y_true.reshape(n, -1).astype(bool)
    p = y_pred.reshape(n, -1).astype(bool)
    inter = np.count_nonzero(t & p, axis=1).astype(np.float64)
    union = np.count_nonzero(t | p, axis=1).astype(np.float64)
    out = np.ones(n, dtype=np.float64)
    nz = union > 0
    out[nz] = inter[nz] / union[nz]
    return out


def all_metrics(y_true: np.ndarray, y_pred_bin: np.ndarray) -> dict[str, float]:
    """Full metric set for a binarised prediction stack."""
    tp, fp, fn, tn = confusion_counts(y_true, y_pred_bin)

    precision = tp / (tp + fp) if tp + fp else 0.0
    recall = tp / (tp + fn) if tp + fn else 0.0
    f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
    dice_g = 2 * tp / (2 * tp + fp + fn) if 2 * tp + fp + fn else 0.0
    iou_g = tp / (tp + fp + fn) if tp + fp + fn else 0.0
    specificity = tn / (tn + fp) if tn + fp else 0.0

    pid = per_image_dice(y_true, y_pred_bin)
    pii = per_image_iou(y_true, y_pred_bin)

    return {
        "dice": float(pid.mean()),
        "dice_std": float(pid.std()),
        "dice_median": float(np.median(pid)),
        "iou": float(pii.mean()),
        "dice_global": float(dice_g),
        "iou_global": float(iou_g),
        "precision": float(precision),
        "recall": float(recall),
        "f1": float(f1),
        "specificity": float(specificity),
        "accuracy": float((tp + tn) / max(tp + tn + fp + fn, 1)),
    }


def bootstrap_ci(
    y_true: np.ndarray,
    y_pred_bin: np.ndarray,
    n_boot: int = 1000,
    alpha: float = 0.05,
    seed: int = 42,
) -> dict[str, list[float]]:
    """Percentile CI for per-image Dice and IoU by resampling slices.

    A single test Dice with no interval cannot tell a real improvement from
    split noise, which is the main reason the notebook's 0.8611 vs 0.8826
    comparison is hard to act on.
    """
    if n_boot <= 0:
        return {}
    pid = per_image_dice(y_true, y_pred_bin)
    pii = per_image_iou(y_true, y_pred_bin)
    rng = np.random.default_rng(seed)
    n = len(pid)
    d, i = np.empty(n_boot), np.empty(n_boot)
    for b in range(n_boot):
        idx = rng.integers(0, n, n)
        d[b] = pid[idx].mean()
        i[b] = pii[idx].mean()
    lo, hi = 100 * alpha / 2, 100 * (1 - alpha / 2)
    return {
        "dice_ci": [float(np.percentile(d, lo)), float(np.percentile(d, hi))],
        "iou_ci": [float(np.percentile(i, lo)), float(np.percentile(i, hi))],
    }
