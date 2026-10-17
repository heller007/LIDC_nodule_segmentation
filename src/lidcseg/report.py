"""Run artifacts: training curves, metric bars, qualitative panels.

Kept out of `train.py` so a plotting failure can never lose a finished run.
Figures land in the run directory next to results.json.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np


def _mpl():
    import matplotlib

    matplotlib.use("Agg")  # headless; Kaggle has no display
    import matplotlib.pyplot as plt

    return plt


def plot_history(history: dict, out: Path) -> None:
    plt = _mpl()

    # Deep supervision prefixes metric names; strip it for readable legends.
    def find(*cands: str) -> str | None:
        """Locate a history key, tolerating the `main_output_` prefix.

        A plain suffix match would let `val_loss` answer a lookup for the
        training `loss`, so validation keys are only eligible when the
        candidate itself asks for them.
        """
        for c in cands:
            want_val = c.startswith("val_")
            for key in history:
                if key.startswith("val_") != want_val:
                    continue
                if key == c or key.endswith("_" + c.removeprefix("val_")):
                    return key
        return None

    panels = [
        ("loss", find("loss"), find("val_loss")),
        ("per-image Dice", find("dice"), find("val_dice")),
        ("IoU", find("iou"), find("val_iou")),
        ("precision / recall", find("precision"), find("val_precision")),
    ]

    fig, axes = plt.subplots(2, 2, figsize=(13, 8))
    for ax, (title, tr_key, va_key) in zip(axes.ravel(), panels):
        plotted = False
        if tr_key and tr_key in history:
            ax.plot(history[tr_key], label="train", lw=1.6)
            plotted = True
        if va_key and va_key in history:
            ax.plot(history[va_key], label="val", lw=1.6)
            plotted = True
        if title.startswith("precision"):
            rc, vrc = None, None
            for key in history:
                if key.endswith("recall") and not key.startswith("val_"):
                    rc = key
                if key.startswith("val_") and key.endswith("recall"):
                    vrc = key
            if rc:
                ax.plot(history[rc], label="train recall", ls="--", lw=1.2)
            if vrc:
                ax.plot(history[vrc], label="val recall", ls="--", lw=1.2)
        ax.set_title(title)
        ax.set_xlabel("epoch")
        ax.grid(alpha=0.3)
        if plotted:
            ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(out / "training_curves.png", dpi=140)
    plt.close(fig)


def plot_metrics(results: dict, out: Path) -> None:
    plt = _mpl()
    keys = ["dice", "iou", "dice_global", "precision", "recall", "f1"]
    test = results["test"]
    vals = [test.get(k, 0.0) for k in keys]

    fig, ax = plt.subplots(figsize=(8.5, 4.5))
    bars = ax.bar(keys, vals, color="#4C72B0")

    ci = test.get("dice_ci")
    if ci:
        ax.errorbar(
            [0], [test["dice"]],
            yerr=[[test["dice"] - ci[0]], [ci[1] - test["dice"]]],
            fmt="none", ecolor="#333", capsize=5, lw=1.4,
        )

    ax.set_ylim(0, 1)
    ax.set_title(f"Test metrics (threshold {results['threshold']:.2f}, TTA={results['tta']})")
    ax.grid(axis="y", alpha=0.3)
    for b, v in zip(bars, vals):
        ax.text(b.get_x() + b.get_width() / 2, v + 0.015, f"{v:.3f}", ha="center", fontsize=9)
    fig.tight_layout()
    fig.savefig(out / "test_metrics.png", dpi=140)
    plt.close(fig)


def plot_threshold_sweep(results: dict, out: Path) -> None:
    trace = results.get("threshold_sweep") or []
    if len(trace) < 2:
        return
    plt = _mpl()
    ts = [p["threshold"] for p in trace]
    ds = [p["dice"] for p in trace]

    fig, ax = plt.subplots(figsize=(7, 4))
    ax.plot(ts, ds, marker="o", ms=3)
    ax.axvline(results["threshold"], color="#C44E52", ls="--",
               label=f"chosen {results['threshold']:.3f}")
    ax.set_xlabel("threshold")
    ax.set_ylabel("val per-image Dice")
    ax.set_title("Threshold sweep (validation)")
    ax.grid(alpha=0.3)
    ax.legend()
    fig.tight_layout()
    fig.savefig(out / "threshold_sweep.png", dpi=140)
    plt.close(fig)


def plot_predictions(
    model,
    x: np.ndarray,
    y: np.ndarray,
    threshold: float,
    out: Path,
    n: int = 6,
    seed: int = 42,
) -> None:
    """Qualitative panel, biased toward slices that actually contain nodules.

    Random sampling mostly returns empty slices, which look like a perfect
    score and show nothing.
    """
    from .evaluate import predict

    plt = _mpl()
    has_pos = np.array([bool(m.any()) for m in y])
    pool = np.flatnonzero(has_pos)
    if len(pool) == 0:
        pool = np.arange(len(y))
    rng = np.random.default_rng(seed)
    idx = rng.choice(pool, size=min(n, len(pool)), replace=False)

    xb = x[idx]
    prob = predict(model, xb, batch_size=8, tta=False)
    pred = (prob > threshold).astype(np.uint8)

    # Undo the caffe-style shift so the display image is viewable.
    disp = xb[..., 0]
    if disp.min() < -1.0:
        disp = (disp + 103.939) / 255.0
    disp = np.clip(disp, 0, 1)

    fig, axes = plt.subplots(len(idx), 3, figsize=(9, 3 * len(idx)))
    axes = np.atleast_2d(axes)
    from .metrics import per_image_dice

    dices = per_image_dice(y[idx], pred)

    for r in range(len(idx)):
        axes[r, 0].imshow(disp[r], cmap="gray")
        axes[r, 0].set_title("input", fontsize=9)
        axes[r, 1].imshow(y[idx][r, ..., 0], cmap="gray", vmin=0, vmax=1)
        axes[r, 1].set_title("ground truth", fontsize=9)
        axes[r, 2].imshow(pred[r, ..., 0], cmap="gray", vmin=0, vmax=1)
        axes[r, 2].set_title(f"prediction (Dice {dices[r]:.3f})", fontsize=9)
        for c in range(3):
            axes[r, c].axis("off")

    fig.tight_layout()
    fig.savefig(out / "predictions.png", dpi=130)
    plt.close(fig)


def write_report(
    run_dir: str | Path,
    history: dict,
    results: dict,
    model=None,
    x_test: np.ndarray | None = None,
    y_test: np.ndarray | None = None,
) -> None:
    out = Path(run_dir)
    out.mkdir(parents=True, exist_ok=True)

    if history:
        plot_history(history, out)
    plot_metrics(results, out)
    plot_threshold_sweep(results, out)
    if model is not None and x_test is not None and y_test is not None:
        plot_predictions(model, x_test, y_test, results["threshold"], out)

    print(f"[report] figures written to {out}")
