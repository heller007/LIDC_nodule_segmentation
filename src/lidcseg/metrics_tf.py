"""Streaming Keras metrics used during training.

Stateful rather than the notebook's stateless `dice_coef` function. Keras
averages a stateless metric over batches, and the mean of per-batch ratios is
not the ratio over the epoch -- for Dice on small objects the two differ
noticeably. The numpy equivalents live in `metrics.py`.
"""

from __future__ import annotations

import numpy as np
import tensorflow as tf
from tensorflow import keras


@keras.saving.register_keras_serializable(package="lidcseg")
class DiceMetric(keras.metrics.Metric):
    """Streaming per-image (or global) Dice at a fixed threshold.

    Implemented as a stateful metric rather than the notebook's stateless
    `dice_coef` function. A stateless metric is averaged over batches by Keras,
    which for a ratio is not the ratio over the epoch.
    """

    def __init__(self, threshold: float = 0.5, per_image: bool = True, name: str = "dice", **kw):
        super().__init__(name=name, **kw)
        self.threshold = threshold
        self.per_image = per_image
        if per_image:
            self.total = self.add_weight(name="total", initializer="zeros")
            self.count = self.add_weight(name="count", initializer="zeros")
        else:
            self.inter = self.add_weight(name="inter", initializer="zeros")
            self.denom = self.add_weight(name="denom", initializer="zeros")

    def update_state(self, y_true, y_pred, sample_weight=None):
        y_true = tf.cast(tf.cast(y_true, tf.float32) > 0.5, tf.float32)
        y_pred = tf.cast(tf.cast(y_pred, tf.float32) > self.threshold, tf.float32)

        if self.per_image:
            n = tf.shape(y_pred)[0]
            t = tf.reshape(y_true, (n, -1))
            p = tf.reshape(y_pred, (n, -1))
            inter = tf.reduce_sum(t * p, axis=-1)
            denom = tf.reduce_sum(t, axis=-1) + tf.reduce_sum(p, axis=-1)
            # An empty mask predicted empty is a perfect score, not 0/0.
            dice = tf.where(denom > 0, (2.0 * inter) / tf.maximum(denom, 1e-7), tf.ones_like(denom))
            self.total.assign_add(tf.reduce_sum(dice))
            self.count.assign_add(tf.cast(n, tf.float32))
        else:
            self.inter.assign_add(tf.reduce_sum(y_true * y_pred))
            self.denom.assign_add(tf.reduce_sum(y_true) + tf.reduce_sum(y_pred))

    def result(self):
        if self.per_image:
            return self.total / tf.maximum(self.count, 1e-7)
        return (2.0 * self.inter) / tf.maximum(self.denom, 1e-7)

    def reset_state(self):
        for v in self.variables:
            v.assign(tf.zeros_like(v))

    def get_config(self):
        return {**super().get_config(), "threshold": self.threshold, "per_image": self.per_image}


@keras.saving.register_keras_serializable(package="lidcseg")
class IoUMetric(DiceMetric):
    """Streaming IoU (Jaccard), same conventions as DiceMetric."""

    def __init__(self, threshold: float = 0.5, per_image: bool = True, name: str = "iou", **kw):
        super().__init__(threshold=threshold, per_image=per_image, name=name, **kw)

    def update_state(self, y_true, y_pred, sample_weight=None):
        y_true = tf.cast(tf.cast(y_true, tf.float32) > 0.5, tf.float32)
        y_pred = tf.cast(tf.cast(y_pred, tf.float32) > self.threshold, tf.float32)

        if self.per_image:
            n = tf.shape(y_pred)[0]
            t = tf.reshape(y_true, (n, -1))
            p = tf.reshape(y_pred, (n, -1))
            inter = tf.reduce_sum(t * p, axis=-1)
            union = tf.reduce_sum(tf.cast((t + p) > 0, tf.float32), axis=-1)
            iou = tf.where(union > 0, inter / tf.maximum(union, 1e-7), tf.ones_like(union))
            self.total.assign_add(tf.reduce_sum(iou))
            self.count.assign_add(tf.cast(n, tf.float32))
        else:
            self.inter.assign_add(tf.reduce_sum(y_true * y_pred))
            self.denom.assign_add(tf.reduce_sum(tf.cast((y_true + y_pred) > 0, tf.float32)))

    def result(self):
        if self.per_image:
            return self.total / tf.maximum(self.count, 1e-7)
        return self.inter / tf.maximum(self.denom, 1e-7)


def build_metrics(threshold: float = 0.5) -> list[keras.metrics.Metric]:
    """Metrics attached to the main output during training."""
    return [
        DiceMetric(threshold=threshold, per_image=True, name="dice"),
        DiceMetric(threshold=threshold, per_image=False, name="dice_global"),
        IoUMetric(threshold=threshold, per_image=True, name="iou"),
        keras.metrics.Precision(thresholds=threshold, name="precision"),
        keras.metrics.Recall(thresholds=threshold, name="recall"),
    ]


