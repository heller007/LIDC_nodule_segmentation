"""Segmentation losses.

All losses support a `per_image` mode. The notebook flattened the whole batch
with `tf.reshape(y, [-1])` and computed one ratio over every pixel in it, so a
batch containing one large nodule and three small ones was dominated by the
large one, and empty-mask slices contributed no gradient signal at all. Scoring
each image then averaging gives every slice equal weight.
"""

from __future__ import annotations

import tensorflow as tf
from tensorflow import keras

from .config import LossConfig

EPS = 1e-7


def _flatten(y_true, y_pred, per_image: bool):
    """Shape both tensors to (N, P): N scoring units, P pixels each."""
    y_true = tf.cast(y_true, y_pred.dtype)
    if per_image:
        n = tf.shape(y_pred)[0]
        return tf.reshape(y_true, (n, -1)), tf.reshape(y_pred, (n, -1))
    return tf.reshape(y_true, (1, -1)), tf.reshape(y_pred, (1, -1))


@keras.saving.register_keras_serializable(package="lidcseg")
class TverskyFocalLoss(keras.losses.Loss):
    """Focal Tversky loss.

    Tversky index generalises Dice with asymmetric error costs:

        TI = TP / (TP + alpha*FP + beta*FN)

    beta > alpha penalises missed nodule pixels more than false alarms, which
    is the right trade-off for screening. The focal exponent gamma < 1 is
    *expansive*, not focusing: it raises the loss for already-good predictions
    and keeps gradients alive late in training. gamma=0.75 is what won the
    notebook's sweep.
    """

    def __init__(
        self,
        alpha: float = 0.3,
        beta: float = 0.7,
        gamma: float = 0.75,
        smooth: float = 1e-6,
        per_image: bool = True,
        name: str = "tversky_focal_loss",
        **kw,
    ):
        super().__init__(name=name, **kw)
        self.alpha = alpha
        self.beta = beta
        self.gamma = gamma
        self.smooth = smooth
        self.per_image = per_image

    def call(self, y_true, y_pred):
        y_pred = tf.clip_by_value(y_pred, EPS, 1.0 - EPS)
        t, p = _flatten(y_true, y_pred, self.per_image)

        tp = tf.reduce_sum(t * p, axis=-1)
        fp = tf.reduce_sum((1.0 - t) * p, axis=-1)
        fn = tf.reduce_sum(t * (1.0 - p), axis=-1)

        ti = (tp + self.smooth) / (tp + self.alpha * fp + self.beta * fn + self.smooth)
        return tf.pow(tf.maximum(1.0 - ti, 0.0), self.gamma)

    def get_config(self):
        return {
            **super().get_config(),
            "alpha": self.alpha,
            "beta": self.beta,
            "gamma": self.gamma,
            "smooth": self.smooth,
            "per_image": self.per_image,
        }


@keras.saving.register_keras_serializable(package="lidcseg")
class DiceFocalLoss(keras.losses.Loss):
    """`dice_weight * Dice + (1 - dice_weight) * Focal`.

    Dice supplies a region-level signal that tolerates class imbalance; focal
    BCE supplies a pixel-level one that concentrates on hard pixels. The
    notebook's baseline used this and scored 0.8611.
    """

    def __init__(
        self,
        dice_weight: float = 0.5,
        focal_alpha: float = 0.25,
        focal_gamma: float = 2.0,
        smooth: float = 1e-6,
        per_image: bool = True,
        name: str = "dice_focal_loss",
        **kw,
    ):
        super().__init__(name=name, **kw)
        self.dice_weight = dice_weight
        self.focal_alpha = focal_alpha
        self.focal_gamma = focal_gamma
        self.smooth = smooth
        self.per_image = per_image

    def call(self, y_true, y_pred):
        y_pred = tf.clip_by_value(y_pred, EPS, 1.0 - EPS)
        t, p = _flatten(y_true, y_pred, self.per_image)

        inter = tf.reduce_sum(t * p, axis=-1)
        dice = 1.0 - (2.0 * inter + self.smooth) / (
            tf.reduce_sum(t, axis=-1) + tf.reduce_sum(p, axis=-1) + self.smooth
        )

        fg = -self.focal_alpha * tf.pow(1.0 - p, self.focal_gamma) * t * tf.math.log(p)
        bg = -(1.0 - self.focal_alpha) * tf.pow(p, self.focal_gamma) * (1.0 - t) * tf.math.log(1.0 - p)
        focal = tf.reduce_mean(fg + bg, axis=-1)

        return self.dice_weight * dice + (1.0 - self.dice_weight) * focal

    def get_config(self):
        return {
            **super().get_config(),
            "dice_weight": self.dice_weight,
            "focal_alpha": self.focal_alpha,
            "focal_gamma": self.focal_gamma,
            "smooth": self.smooth,
            "per_image": self.per_image,
        }


@keras.saving.register_keras_serializable(package="lidcseg")
class DiceBCELoss(keras.losses.Loss):
    """Plain Dice + unweighted BCE -- a sanity baseline."""

    def __init__(
        self,
        dice_weight: float = 0.5,
        smooth: float = 1e-6,
        per_image: bool = True,
        name: str = "dice_bce_loss",
        **kw,
    ):
        super().__init__(name=name, **kw)
        self.dice_weight = dice_weight
        self.smooth = smooth
        self.per_image = per_image

    def call(self, y_true, y_pred):
        y_pred = tf.clip_by_value(y_pred, EPS, 1.0 - EPS)
        t, p = _flatten(y_true, y_pred, self.per_image)

        inter = tf.reduce_sum(t * p, axis=-1)
        dice = 1.0 - (2.0 * inter + self.smooth) / (
            tf.reduce_sum(t, axis=-1) + tf.reduce_sum(p, axis=-1) + self.smooth
        )
        bce = -tf.reduce_mean(t * tf.math.log(p) + (1.0 - t) * tf.math.log(1.0 - p), axis=-1)
        return self.dice_weight * dice + (1.0 - self.dice_weight) * bce

    def get_config(self):
        return {
            **super().get_config(),
            "dice_weight": self.dice_weight,
            "smooth": self.smooth,
            "per_image": self.per_image,
        }


def build_loss(cfg: LossConfig) -> keras.losses.Loss:
    if cfg.name == "tversky_focal":
        return TverskyFocalLoss(
            alpha=cfg.tversky_alpha,
            beta=cfg.tversky_beta,
            gamma=cfg.tversky_gamma,
            smooth=cfg.smooth,
            per_image=cfg.per_image,
        )
    if cfg.name == "dice_focal":
        return DiceFocalLoss(
            dice_weight=cfg.dice_weight,
            focal_alpha=cfg.focal_alpha,
            focal_gamma=cfg.focal_gamma,
            smooth=cfg.smooth,
            per_image=cfg.per_image,
        )
    if cfg.name == "dice_bce":
        return DiceBCELoss(
            dice_weight=cfg.dice_weight, smooth=cfg.smooth, per_image=cfg.per_image
        )
    raise ValueError(f"unknown loss {cfg.name!r}")
