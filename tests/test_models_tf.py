"""Model-construction tests. Needs TensorFlow; run on Kaggle or any TF box.

    python tests/test_models_tf.py

Builds every architecture, checks output shapes and head wiring, runs one
forward and one backward pass, and confirms the loss and metric conventions
agree with their numpy counterparts. Cheap compared with a training run, and
it catches the shape and naming mistakes that otherwise surface an hour in.
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

PASS, FAIL = [], []


def check(name: str, cond: bool, detail: str = "") -> None:
    (PASS if cond else FAIL).append(name)
    print(f"  {'PASS' if cond else 'FAIL'}  {name}" + (f"  -- {detail}" if detail and not cond else ""))


import tensorflow as tf  # noqa: E402

from lidcseg.config import ModelConfig, load_config  # noqa: E402
from lidcseg.losses import build_loss  # noqa: E402
from lidcseg.metrics import all_metrics, per_image_dice  # noqa: E402
from lidcseg.metrics_tf import DiceMetric, IoUMetric  # noqa: E402
from lidcseg.models.factory import BUILDERS, build_model  # noqa: E402
from lidcseg.models.dplinknet import set_encoder_trainable  # noqa: E402

print(f"tensorflow {tf.__version__}, keras {tf.keras.__version__}")
SIZE = 128

# Keep it quick: no ImageNet download for the shape checks.
BASE = dict(pretrained=False, imagenet_preprocess=False)


# -------------------------------------------------------------- build all
print("\nbuild every architecture")
built = {}
for arch in sorted(BUILDERS):
    try:
        cfg = ModelConfig(arch=arch, deep_supervision=True, **BASE)
        m = build_model(cfg, SIZE)
        built[arch] = m
        check(f"{arch} builds", True)

        out = m.output
        check(f"{arch} has 2 heads", isinstance(out, dict) and set(out) == {"main_output", "aux_output"},
              str(type(out)))
        for head, t in out.items():
            check(f"{arch}.{head} shape {SIZE}x{SIZE}x1",
                  tuple(t.shape[1:]) == (SIZE, SIZE, 1), str(t.shape))
            check(f"{arch}.{head} is float32", t.dtype == tf.float32, str(t.dtype))
        print(f"       params: {m.count_params():,}")
    except Exception as e:
        check(f"{arch} builds", False, f"{type(e).__name__}: {e}")

# Single-head variant must return a tensor, not a dict.
for arch in sorted(BUILDERS):
    try:
        m = build_model(ModelConfig(arch=arch, deep_supervision=False, **BASE), SIZE)
        check(f"{arch} single-head returns tensor", not isinstance(m.output, dict))
        check(f"{arch} single-head named main_output",
              m.output_names == ["main_output"], str(m.output_names))
    except Exception as e:
        check(f"{arch} single-head", False, f"{type(e).__name__}: {e}")

try:
    build_model(ModelConfig(arch="does_not_exist", **BASE), SIZE)
    check("unknown arch rejected", False, "no error raised")
except ValueError:
    check("unknown arch rejected", True)


# ------------------------------------------------------- encoder freezing
print("\nencoder freeze/unfreeze")
for arch, expect_encoder in (("dplinknet", True), ("resnet50_unet", True), ("unet", False)):
    m = built.get(arch)
    if m is None:
        continue
    names = getattr(m, "encoder_layer_names", None)
    check(f"{arch} reports encoder_layer_names", names is not None)
    check(f"{arch} has encoder = {expect_encoder}", bool(names) is expect_encoder,
          f"{len(names or [])} layers")
    n = set_encoder_trainable(m, False)
    check(f"{arch} freeze returns {'>0' if expect_encoder else '0'}",
          (n > 0) is expect_encoder, str(n))
    set_encoder_trainable(m, True)


# --------------------------------------------------------- attention stages
print("\nattention wiring")
for att in ("scanext", "cab", "none"):
    try:
        m = build_model(ModelConfig(arch="dplinknet", attention=att, **BASE), SIZE)
        att_layers = [l.name for l in m.layers if l.name.endswith("_att")]
        want = 0 if att == "none" else 4
        check(f"dplinknet+{att} has {want} attention stages",
              len(att_layers) == want, str(att_layers))
    except Exception as e:
        check(f"dplinknet+{att}", False, f"{type(e).__name__}: {e}")

# Baselines must ignore `attention` rather than silently gaining stages.
for arch in ("resnet50_unet", "unet"):
    m = build_model(ModelConfig(arch=arch, attention="scanext", **BASE), SIZE)
    check(f"{arch} ignores attention", not [l for l in m.layers if l.name.endswith("_att")])


# ------------------------------------------------------ forward / backward
print("\nforward and backward pass")
cfg = load_config(ROOT / "configs" / "improved.yaml")
cfg.model.pretrained = False
cfg.model.imagenet_preprocess = False

x = np.random.rand(2, SIZE, SIZE, 3).astype(np.float32)
y = np.zeros((2, SIZE, SIZE, 1), dtype=np.float32)
y[0, 40:70, 40:70] = 1.0
y[1, 10:18, 90:98] = 1.0

for arch in sorted(BUILDERS):
    try:
        mc = ModelConfig(arch=arch, deep_supervision=True, **BASE)
        m = build_model(mc, SIZE)
        loss = build_loss(cfg.loss)
        m.compile(
            optimizer=tf.keras.optimizers.Adam(1e-4),
            loss={"main_output": loss, "aux_output": loss},
            loss_weights={"main_output": 1.0, "aux_output": mc.aux_weight},
            metrics={"main_output": [DiceMetric(name="dice"), IoUMetric(name="iou")]},
        )
        p = m.predict(x, verbose=0)
        main = p["main_output"] if isinstance(p, dict) else p[0]
        check(f"{arch} predict shape", main.shape == (2, SIZE, SIZE, 1), str(main.shape))
        check(f"{arch} predict in [0,1]", main.min() >= 0.0 and main.max() <= 1.0,
              f"{main.min():.3f}..{main.max():.3f}")

        h = m.fit(x, {"main_output": y, "aux_output": y}, epochs=1, batch_size=2, verbose=0)
        lv = h.history["loss"][0]
        check(f"{arch} one train step, finite loss", np.isfinite(lv), str(lv))
        check(f"{arch} reports dice metric",
              any("dice" in k for k in h.history), str(list(h.history)))
    except Exception as e:
        check(f"{arch} forward/backward", False, f"{type(e).__name__}: {e}")


# ------------------------------------------------- loss / metric agreement
print("\nloss and metric conventions")
for name in ("tversky_focal", "dice_focal", "dice_bce"):
    c = load_config(ROOT / "configs" / "improved.yaml", [f"loss.name={name}"])
    fn = build_loss(c.loss)
    # A perfect prediction must cost less than an inverted one.
    good = fn(tf.constant(y), tf.constant(y * 0.99 + 0.005)).numpy()
    bad = fn(tf.constant(y), tf.constant(1.0 - y * 0.99 - 0.005)).numpy()
    check(f"{name}: perfect < inverted", float(np.mean(good)) < float(np.mean(bad)),
          f"{np.mean(good):.4f} vs {np.mean(bad):.4f}")
    check(f"{name}: finite", np.all(np.isfinite(good)) and np.all(np.isfinite(bad)))

# The streaming metric must agree with the numpy one it is named after.
pred = np.zeros_like(y)
pred[0, 40:70, 40:55] = 1.0   # half of object 0
pred[1, 10:18, 90:98] = 1.0   # all of object 1

dm = DiceMetric(threshold=0.5, per_image=True)
dm.update_state(tf.constant(y), tf.constant(pred))
tf_dice = float(dm.result().numpy())
np_dice = float(per_image_dice(y, pred).mean())
check("streaming dice == numpy dice", abs(tf_dice - np_dice) < 1e-5, f"{tf_dice} vs {np_dice}")

# Updating in two batches must give the same answer as one.
dm2 = DiceMetric(threshold=0.5, per_image=True)
dm2.update_state(tf.constant(y[:1]), tf.constant(pred[:1]))
dm2.update_state(tf.constant(y[1:]), tf.constant(pred[1:]))
check("streaming dice batch-invariant", abs(float(dm2.result().numpy()) - np_dice) < 1e-5,
      str(dm2.result().numpy()))

im = IoUMetric(threshold=0.5, per_image=True)
im.update_state(tf.constant(y), tf.constant(pred))
check("streaming iou finite and <= dice",
      0 <= float(im.result().numpy()) <= tf_dice + 1e-6, str(im.result().numpy()))

gm = DiceMetric(threshold=0.5, per_image=False)
gm.update_state(tf.constant(y), tf.constant(pred))
check("global dice == numpy global",
      abs(float(gm.result().numpy()) - all_metrics(y, pred)["dice_global"]) < 1e-5)


# --------------------------------------------------------------- pipeline
print("\ntf.data pipeline")
try:
    from lidcseg.data.cache import ArrayCache
    from lidcseg.data.index import SliceRecord
    from lidcseg.data.pipeline import make_dataset, to_encoder_input

    n = 8
    imgs = np.random.rand(n, SIZE, SIZE, 1).astype(np.float32)
    msks = (np.random.rand(n, SIZE, SIZE, 1) > 0.9).astype(np.float32)
    recs = [SliceRecord(f"LIDC-IDRI-{i:04d}", "nodule-0", "0.png", "i", ("m",)) for i in range(n)]
    cache = ArrayCache(imgs, msks, recs)

    ds = make_dataset(cache, np.arange(n), 4, cfg.aug, cfg.model, training=True, seed=1)
    xb, yb = next(iter(ds))
    check("pipeline batch x shape", tuple(xb.shape) == (4, SIZE, SIZE, 3), str(xb.shape))
    check("pipeline yields both heads", isinstance(yb, dict) and set(yb) == {"main_output", "aux_output"})
    check("pipeline mask stays binary",
          bool(np.all(np.isin(yb["main_output"].numpy(), [0.0, 1.0]))))

    # Augmentation must actually vary between epochs.
    e1 = np.concatenate([b[0].numpy().ravel() for b in ds.take(2)])
    e2 = np.concatenate([b[0].numpy().ravel() for b in ds.take(2)])
    check("augmentation differs across epochs", not np.allclose(e1, e2))

    # ImageNet scaling path.
    mc = ModelConfig(imagenet_preprocess=True)
    xi = to_encoder_input(tf.constant(imgs[:1]), True).numpy()
    check("imagenet scaling shifts range", xi.min() < -50, f"{xi.min():.1f}..{xi.max():.1f}")
except Exception as e:
    check("tf.data pipeline", False, f"{type(e).__name__}: {e}")


print(f"\n{'=' * 52}\n{len(PASS)} passed, {len(FAIL)} failed")
if FAIL:
    for f in FAIL:
        print(f"  FAILED: {f}")
    sys.exit(1)
print("all green")
