"""Attention blocks.

Both blocks come from the notebook; they are reimplemented as registered
`keras.layers.Layer` subclasses instead of `Lambda` wrappers around raw `tf`
ops. Lambda layers do not survive a `model.save()` round-trip without
`safe_mode=False`, which is why the notebook could only checkpoint weights.
These serialise cleanly.
"""

from __future__ import annotations

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers


@keras.saving.register_keras_serializable(package="lidcseg")
class ReduceOp(layers.Layer):
    """Mean or max reduction over one axis, keeping rank."""

    def __init__(self, axis: int, op: str = "mean", **kw):
        super().__init__(**kw)
        if op not in {"mean", "max"}:
            raise ValueError(f"op must be 'mean' or 'max', got {op!r}")
        self.axis = axis
        self.op = op

    def call(self, x):
        fn = tf.reduce_mean if self.op == "mean" else tf.reduce_max
        return fn(x, axis=self.axis, keepdims=True)

    def compute_output_shape(self, input_shape):
        shape = list(input_shape)
        ax = self.axis if self.axis >= 0 else len(shape) + self.axis
        shape[ax] = 1
        return tuple(shape)

    def get_config(self):
        return {**super().get_config(), "axis": self.axis, "op": self.op}


@keras.saving.register_keras_serializable(package="lidcseg")
class SCANeXtAttention(layers.Layer):
    """Dual-path channel + strip-spatial attention.

    Branch 1 (channel): global average pool, then a 1x1 bottleneck and a
    sigmoid gate per channel -- squeeze-and-excitation.

    Branch 2 (spatial): average over height and over width separately, convolve
    each strip (1x3 and 3x1), add the two broadcasts, and reduce to a single
    sigmoid gate per pixel. Strip pooling gives a wide receptive field cheaply,
    which suits elongated vessel/nodule context.

    The two refined tensors are summed and projected back to C channels.
    """

    def __init__(self, reduction: int = 8, **kw):
        super().__init__(**kw)
        self.reduction = reduction

    def build(self, input_shape):
        c = int(input_shape[-1])
        # At least one channel in the bottleneck, whatever the reduction.
        hidden = max(c // self.reduction, 1)

        self.ca_pool = layers.GlobalAveragePooling2D(keepdims=True)
        self.ca_down = layers.Conv2D(hidden, 1, activation="relu", use_bias=False)
        self.ca_up = layers.Conv2D(c, 1, activation="sigmoid", use_bias=False)

        self.h_pool = ReduceOp(axis=1, op="mean")
        self.h_conv = layers.Conv2D(c, (1, 3), padding="same", activation="relu", use_bias=False)
        self.v_pool = ReduceOp(axis=2, op="mean")
        self.v_conv = layers.Conv2D(c, (3, 1), padding="same", activation="relu", use_bias=False)
        self.sa_gate = layers.Conv2D(1, 1, activation="sigmoid", use_bias=False)

        self.proj = layers.Conv2D(c, 1, padding="same", use_bias=False)
        self.bn = layers.BatchNormalization()
        self.act = layers.Activation("relu")
        super().build(input_shape)

    def call(self, x):
        ca = self.ca_up(self.ca_down(self.ca_pool(x)))
        x_ca = x * ca

        sa = self.h_conv(self.h_pool(x)) + self.v_conv(self.v_pool(x))
        x_sa = x * self.sa_gate(sa)

        return self.act(self.bn(self.proj(x_ca + x_sa)))

    def compute_output_shape(self, input_shape):
        return input_shape

    def get_config(self):
        return {**super().get_config(), "reduction": self.reduction}


@keras.saving.register_keras_serializable(package="lidcseg")
class CascadedAttention(layers.Layer):
    """Three attention gates in series, with a residual path.

    Gate 1 recalibrates channels (SE). Gate 2 gates space using the avg- and
    max-pooled channel descriptors through a 7x7 conv (CBAM-style), applied to
    gate 1's output. Gate 3 builds multi-scale context with dilated 3x3 convs
    at rates 1/3/5 and gates with it.

    The residual `x + gated` keeps the un-gated features available, so three
    multiplicative gates in a row cannot collapse the signal.
    """

    def __init__(self, reduction: int = 4, dilations: tuple[int, ...] = (1, 3, 5), **kw):
        super().__init__(**kw)
        self.reduction = reduction
        self.dilations = tuple(dilations)

    def build(self, input_shape):
        c = int(input_shape[-1])
        hidden = max(c // self.reduction, 1)
        ctx_ch = max(c // 2, 1)

        self.ca_pool = layers.GlobalAveragePooling2D(keepdims=True)
        self.ca_down = layers.Conv2D(hidden, 1, activation="relu", use_bias=False)
        self.ca_up = layers.Conv2D(c, 1, activation="sigmoid", use_bias=False)

        self.sa_avg = ReduceOp(axis=-1, op="mean")
        self.sa_max = ReduceOp(axis=-1, op="max")
        self.sa_conv = layers.Conv2D(1, 7, padding="same", activation="sigmoid", use_bias=False)

        self.ctx_convs = [
            layers.Conv2D(ctx_ch, 3, dilation_rate=d, padding="same", activation="relu", use_bias=False)
            for d in self.dilations
        ]
        self.ctx_gate = layers.Conv2D(c, 1, padding="same", activation="sigmoid", use_bias=False)

        self.proj = layers.Conv2D(c, 1, padding="same", use_bias=False)
        self.bn = layers.BatchNormalization()
        self.act = layers.Activation("relu")
        super().build(input_shape)

    def call(self, x):
        ca = self.ca_up(self.ca_down(self.ca_pool(x)))
        x_ca = x * ca

        sa = self.sa_conv(tf.concat([self.sa_avg(x_ca), self.sa_max(x_ca)], axis=-1))
        x_sa = x_ca * sa

        ctx = tf.concat([conv(x_sa) for conv in self.ctx_convs], axis=-1)
        x_ctx = x_sa * self.ctx_gate(ctx)

        return self.act(self.bn(self.proj(x + x_ctx)))

    def compute_output_shape(self, input_shape):
        return input_shape

    def get_config(self):
        return {**super().get_config(), "reduction": self.reduction, "dilations": list(self.dilations)}


def get_attention(kind: str, reduction: int = 8, name: str | None = None):
    """Factory used by the model builder; `None` means identity.

    `name` is passed to the constructor -- a Keras 3 layer's name is read-only
    once built, so it cannot be assigned afterwards.
    """
    kind = (kind or "none").lower()
    if kind == "scanext":
        return SCANeXtAttention(reduction=reduction, name=name)
    if kind == "cab":
        return CascadedAttention(reduction=max(reduction // 2, 1), name=name)
    if kind == "none":
        return None
    raise ValueError(f"unknown attention {kind!r}; expected scanext, cab or none")
