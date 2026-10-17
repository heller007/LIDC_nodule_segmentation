"""Training entry point.

    python -m lidcseg.train --config configs/improved.yaml
    python -m lidcseg.train --config configs/improved.yaml --set train.lr=1e-4

Everything needed to reproduce a run lands in the run directory: the resolved
config, the split summary, history.csv, results.json and the best weights.
"""

from __future__ import annotations

import argparse
import json
import os
import platform
import random
import sys
import time
from pathlib import Path

import numpy as np

from .config import Config, load_config


def set_seeds(seed: int) -> None:
    os.environ["PYTHONHASHSEED"] = str(seed)
    random.seed(seed)
    np.random.seed(seed)
    import tensorflow as tf

    tf.random.set_seed(seed)
    tf.keras.utils.set_random_seed(seed)


def setup_hardware(mixed_precision: bool) -> dict:
    """Configure GPU memory growth and mixed precision; report what we got."""
    import tensorflow as tf

    gpus = tf.config.list_physical_devices("GPU")
    for g in gpus:
        try:
            tf.config.experimental.set_memory_growth(g, True)
        except RuntimeError:
            pass  # Already initialised; harmless.

    info = {
        "tensorflow": tf.__version__,
        "keras": tf.keras.__version__,
        "python": platform.python_version(),
        "gpus": [g.name for g in gpus],
    }

    if mixed_precision and gpus:
        from tensorflow.keras import mixed_precision as mp

        mp.set_global_policy("mixed_float16")
        info["policy"] = "mixed_float16"
    else:
        info["policy"] = "float32"
        if mixed_precision and not gpus:
            print("[hw] no GPU found -- mixed precision disabled")

    # Kaggle's dual T4 setup: use both when present.
    if len(gpus) > 1:
        info["strategy"] = "MirroredStrategy"
    else:
        info["strategy"] = "default"

    return info


def build_optimizer(cfg: Config, steps_per_epoch: int):
    import tensorflow as tf

    lr: float | tf.keras.optimizers.schedules.LearningRateSchedule = cfg.train.lr

    if cfg.train.schedule == "cosine":
        warmup_steps = max(cfg.train.warmup_epochs * steps_per_epoch, 1)
        total_steps = max(cfg.train.epochs * steps_per_epoch, warmup_steps + 1)
        lr = tf.keras.optimizers.schedules.CosineDecay(
            initial_learning_rate=0.0,
            decay_steps=total_steps - warmup_steps,
            alpha=cfg.train.min_lr / max(cfg.train.lr, 1e-12),
            warmup_target=cfg.train.lr,
            warmup_steps=warmup_steps,
        )

    kw = {"clipnorm": cfg.train.clipnorm} if cfg.train.clipnorm else {}
    if cfg.train.optimizer == "adamw":
        return tf.keras.optimizers.AdamW(
            learning_rate=lr, weight_decay=cfg.train.weight_decay, **kw
        )
    return tf.keras.optimizers.Adam(learning_rate=lr, **kw)


def compile_model(model, cfg: Config, steps_per_epoch: int):
    from .losses import build_loss
    from .metrics_tf import build_metrics

    loss = build_loss(cfg.loss)
    opt = build_optimizer(cfg, steps_per_epoch)

    if cfg.model.deep_supervision:
        model.compile(
            optimizer=opt,
            loss={"main_output": loss, "aux_output": loss},
            loss_weights={"main_output": 1.0, "aux_output": cfg.model.aux_weight},
            metrics={"main_output": build_metrics(cfg.eval.default_threshold)},
        )
    else:
        model.compile(optimizer=opt, loss=loss, metrics=build_metrics(cfg.eval.default_threshold))
    return model


def build_callbacks(cfg: Config, run_dir: Path):
    import tensorflow as tf

    monitor = cfg.train.monitor
    if cfg.model.deep_supervision and not monitor.startswith("val_main_output"):
        # Keras prefixes per-output metric names when there are several heads.
        monitor = monitor.replace("val_", "val_main_output_", 1)

    weights_path = run_dir / "best.weights.h5"
    cbs = [
        tf.keras.callbacks.ModelCheckpoint(
            str(weights_path),
            monitor=monitor,
            mode="max",
            save_best_only=True,
            save_weights_only=True,
            verbose=1,
        ),
        tf.keras.callbacks.EarlyStopping(
            monitor=monitor,
            mode="max",
            patience=cfg.train.early_stopping_patience,
            restore_best_weights=True,
            verbose=1,
        ),
        tf.keras.callbacks.CSVLogger(str(run_dir / "history.csv")),
        tf.keras.callbacks.TerminateOnNaN(),
    ]

    # ReduceLROnPlateau fights a cosine schedule -- only attach it when the LR
    # is otherwise constant.
    if cfg.train.schedule == "none":
        cbs.append(
            tf.keras.callbacks.ReduceLROnPlateau(
                monitor="val_loss",
                factor=cfg.train.reduce_lr_factor,
                patience=cfg.train.reduce_lr_patience,
                min_lr=cfg.train.min_lr,
                verbose=1,
            )
        )
    return cbs, monitor, weights_path


def run(cfg: Config) -> dict:
    from .data.cache import build_cache, filter_empty
    from .data.index import build_index, index_stats, save_index
    from .data.pipeline import make_dataset, raw_eval_arrays
    from .data.splits import assert_no_leakage, make_split
    from .evaluate import format_results, full_evaluation
    from .models.dplinknet import build_model, set_encoder_trainable

    t0 = time.time()
    run_dir = cfg.run_dir
    run_dir.mkdir(parents=True, exist_ok=True)
    cfg.save(run_dir / "config.yaml")

    set_seeds(cfg.seed)
    hw = setup_hardware(cfg.train.mixed_precision)
    print(f"[hw] {json.dumps(hw)}")

    # ---- data ----
    print("\n[data] indexing")
    records = build_index(cfg.data.root, cfg.data.nodules, cfg.data.n_readers)
    print(f"[data] {json.dumps(index_stats(records))}")
    save_index(records, run_dir / "index.json")

    cache = build_cache(records, cfg.data)

    split = make_split(
        records,
        mode=cfg.split.mode,
        val_frac=cfg.split.val_frac,
        test_frac=cfg.split.test_frac,
        seed=cfg.split.seed,
        folds=cfg.split.folds,
        fold=cfg.split.fold,
    )
    summary = split.summary(records)
    print(f"[split] {json.dumps(summary)}")
    if cfg.split.mode == "patient":
        assert_no_leakage(split, records)
        print("[split] verified: no patient appears in two sets")
    else:
        print("[split] WARNING slice-level split -- patients span sets, metrics are optimistic")
    (run_dir / "split.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")

    train_idx = filter_empty(
        cache,
        split.train,
        cfg.data.drop_empty_masks,
        cfg.data.empty_mask_keep_frac,
        seed=cfg.seed,
    )
    if len(train_idx) != len(split.train):
        print(f"[data] train slices {len(split.train)} -> {len(train_idx)} after empty-mask filter")

    train_ds = make_dataset(
        cache, train_idx, cfg.train.batch_size, cfg.aug, cfg.model, training=True, seed=cfg.seed
    )
    val_ds = make_dataset(
        cache, split.val, cfg.train.batch_size, None, cfg.model, training=False, seed=cfg.seed
    )

    steps_per_epoch = max(len(train_idx) // cfg.train.batch_size, 1)

    # ---- model ----
    print("\n[model] building")
    model = build_model(cfg.model, cfg.data.img_size)
    n_params = model.count_params()
    print(f"[model] {model.name}  params={n_params:,}  outputs={cfg.model.deep_supervision and 2 or 1}")

    callbacks, monitor, weights_path = build_callbacks(cfg, run_dir)

    history_all: dict[str, list] = {}

    # ---- optional frozen-encoder warmup ----
    if cfg.train.freeze_encoder_epochs > 0:
        import tensorflow as tf

        n_frozen = set_encoder_trainable(model, False)
        compile_model(model, cfg, steps_per_epoch)
        print(f"\n[train] phase 1: {n_frozen} encoder layers frozen, "
              f"{cfg.train.freeze_encoder_epochs} epochs")
        # Warmup must not early-stop or checkpoint: the decoder is still random
        # and its val score says nothing about the final model. CSVLogger is
        # left out too, since phase 2 opens the same file in write mode.
        warmup_cbs = [
            c
            for c in callbacks
            if not isinstance(
                c,
                (
                    tf.keras.callbacks.EarlyStopping,
                    tf.keras.callbacks.ModelCheckpoint,
                    tf.keras.callbacks.CSVLogger,
                ),
            )
        ]
        h = model.fit(
            train_ds,
            validation_data=val_ds,
            epochs=cfg.train.freeze_encoder_epochs,
            callbacks=warmup_cbs,
            verbose=1,
        )
        history_all = {k: list(v) for k, v in h.history.items()}

        set_encoder_trainable(model, True)
        compile_model(model, cfg, steps_per_epoch)
        print("[train] phase 2: encoder unfrozen")

    else:
        compile_model(model, cfg, steps_per_epoch)

    # ---- main training ----
    remaining = max(cfg.train.epochs - cfg.train.freeze_encoder_epochs, 1)
    print(f"\n[train] {remaining} epochs, batch={cfg.train.batch_size}, "
          f"steps/epoch={steps_per_epoch}, monitor={monitor}")

    hist = model.fit(
        train_ds,
        validation_data=val_ds,
        epochs=remaining,
        callbacks=callbacks,
        verbose=1,
    )
    for k, v in hist.history.items():
        history_all.setdefault(k, []).extend(list(v))

    if weights_path.exists():
        model.load_weights(str(weights_path))
        print(f"[train] restored best weights from {weights_path.name}")

    # ---- evaluation ----
    print("\n[eval] tuning threshold on val, scoring test")
    x_val, y_val = raw_eval_arrays(cache, split.val, cfg.model)
    x_test, y_test = raw_eval_arrays(cache, split.test, cfg.model)

    results = full_evaluation(
        model,
        x_val,
        y_val,
        x_test,
        y_test,
        cfg.eval,
        batch_size=max(cfg.train.batch_size, 16),
        out_dir=run_dir,
    )
    print("\n" + format_results(results))

    meta = {
        "name": cfg.name,
        "hardware": hw,
        "params": int(n_params),
        "split": summary,
        "epochs_run": len(history_all.get("loss", [])),
        "train_slices": int(len(train_idx)),
        "minutes": round((time.time() - t0) / 60, 2),
        "best_monitor": monitor,
        "results": {"val": results["val"], "test": results["test"], "threshold": results["threshold"]},
    }
    (run_dir / "summary.json").write_text(json.dumps(meta, indent=2), encoding="utf-8")
    print(f"\n[done] {meta['minutes']} min -- artifacts in {run_dir}")

    try:
        from .report import write_report

        write_report(run_dir, history_all, results, model, x_test, y_test)
    except Exception as e:  # plotting must never fail a finished run
        print(f"[report] skipped: {type(e).__name__}: {e}")

    return meta


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description="Train DPLinkNet50 on LIDC-IDRI slices")
    ap.add_argument("--config", required=True, help="path to a YAML config")
    ap.add_argument(
        "--set",
        dest="overrides",
        nargs="*",
        default=[],
        metavar="KEY=VALUE",
        help="dotted overrides, e.g. train.lr=1e-4 model.attention=cab",
    )
    ap.add_argument("--name", default=None, help="override the run name")
    ap.add_argument("--fold", type=int, default=None, help="override split.fold")
    args = ap.parse_args(argv)

    cfg = load_config(args.config, args.overrides)
    if args.name:
        cfg.name = args.name
    if args.fold is not None:
        cfg.split.fold = args.fold

    run(cfg)
    return 0


if __name__ == "__main__":
    sys.exit(main())
