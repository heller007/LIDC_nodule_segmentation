"""DPLinkNet50: ResNet50 encoder, dual-path centre, LinkNet-style decoder.

Encoder taps at 128x128 input:

    conv1_relu          64x64     64ch   skip -> dec1
    conv2_block3_out    32x32    256ch   skip -> dec2
    conv3_block4_out    16x16    512ch   skip -> dec3   (unused in notebook)
    conv4_block6_out     8x8    1024ch   skip -> dec4   (unused in notebook)
    conv5_block3_out     4x4   2048ch    centre

The notebook wired skips only into dec1 and dec2, leaving the two deepest
encoder stages unconnected -- their features reached the decoder only through
the 4x4 bottleneck. `use_skips` restores all four, which is what LinkNet
actually specifies.
"""

from __future__ import annotations

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers

from ..config import ModelConfig
from .attention import get_attention

SKIP_LAYERS = {
    "dec1": "conv1_relu",
    "dec2": "conv2_block3_out",
    "dec3": "conv3_block4_out",
    "dec4": "conv4_block6_out",
    "center": "conv5_block3_out",
}


def dual_path_block(x, filters: int, name: str):
    """Two parallel 3x3 convs, one dilated, summed.

    The dilated path widens the receptive field at the bottleneck without
    another downsample -- at 4x4 there is no resolution left to lose.
    """
    p1 = layers.Conv2D(filters, 3, padding="same", use_bias=False, name=f"{name}_p1")(x)
    p1 = layers.BatchNormalization(name=f"{name}_p1_bn")(p1)
    p1 = layers.Activation("relu", name=f"{name}_p1_relu")(p1)

    p2 = layers.Conv2D(filters, 3, dilation_rate=2, padding="same", use_bias=False, name=f"{name}_p2")(x)
    p2 = layers.BatchNormalization(name=f"{name}_p2_bn")(p2)
    p2 = layers.Activation("relu", name=f"{name}_p2_relu")(p2)

    out = layers.Add(name=f"{name}_add")([p1, p2])
    return layers.Activation("relu", name=f"{name}_out")(out)


def decoder_block(
    x,
    filters: int,
    name: str,
    skip=None,
    attention: str = "scanext",
    attention_reduction: int = 8,
    dropout: float = 0.0,
):
    """LinkNet decoder stage: 1x1 squeeze, transposed-conv upsample, 1x1 expand.

    The notebook put BN+ReLU before adding the skip in the no-skip variant but
    only BN in the skip variant; here the add always precedes the activation,
    which is the standard residual ordering.
    """
    mid = max(filters // 4, 1)

    x = layers.Conv2D(mid, 1, padding="same", use_bias=False, name=f"{name}_sq")(x)
    x = layers.BatchNormalization(name=f"{name}_sq_bn")(x)
    x = layers.Activation("relu", name=f"{name}_sq_relu")(x)

    x = layers.Conv2DTranspose(mid, 3, strides=2, padding="same", use_bias=False, name=f"{name}_up")(x)
    x = layers.BatchNormalization(name=f"{name}_up_bn")(x)
    x = layers.Activation("relu", name=f"{name}_up_relu")(x)

    x = layers.Conv2D(filters, 1, padding="same", use_bias=False, name=f"{name}_ex")(x)
    x = layers.BatchNormalization(name=f"{name}_ex_bn")(x)

    if skip is not None:
        s = layers.Conv2D(filters, 1, padding="same", use_bias=False, name=f"{name}_skip")(skip)
        s = layers.BatchNormalization(name=f"{name}_skip_bn")(s)
        x = layers.Add(name=f"{name}_add")([x, s])

    x = layers.Activation("relu", name=f"{name}_relu")(x)

    if dropout > 0:
        x = layers.SpatialDropout2D(dropout, name=f"{name}_drop")(x)

    att = get_attention(attention, attention_reduction, name=f"{name}_att")
    if att is not None:
        x = att(x)
    return x


def build_dplinknet(cfg: ModelConfig, img_size: int = 128) -> keras.Model:
    """Assemble DPLinkNet50 as described by `cfg`."""
    if cfg.encoder != "resnet50":
        raise ValueError(f"only resnet50 is wired up, got {cfg.encoder!r}")
    if img_size % 32 != 0:
        raise ValueError(f"img_size must be divisible by 32 for ResNet50 taps, got {img_size}")

    base = keras.applications.ResNet50(
        weights="imagenet" if cfg.pretrained else None,
        include_top=False,
        input_shape=(img_size, img_size, 3),
    )

    taps = {k: base.get_layer(v).output for k, v in SKIP_LAYERS.items()}

    center = dual_path_block(taps["center"], 2048, name="center")

    skips = taps if cfg.use_skips else {}
    x = decoder_block(
        center, 1024, "dec4", skips.get("dec4"), cfg.attention, cfg.attention_reduction, cfg.dropout
    )
    x = decoder_block(
        x, 512, "dec3", skips.get("dec3"), cfg.attention, cfg.attention_reduction, cfg.dropout
    )
    dec2 = decoder_block(
        x, 256, "dec2", taps["dec2"], cfg.attention, cfg.attention_reduction, cfg.dropout
    )
    dec1 = decoder_block(
        dec2, 64, "dec1", taps["dec1"], cfg.attention, cfg.attention_reduction, cfg.dropout
    )

    head = layers.Conv2DTranspose(32, 3, strides=2, padding="same", use_bias=False, name="head_up")(dec1)
    head = layers.BatchNormalization(name="head_bn")(head)
    head = layers.Activation("relu", name="head_relu")(head)

    # float32 on the final activation keeps the sigmoid and the loss in full
    # precision when mixed_float16 is on; a float16 sigmoid saturates and the
    # Dice gradient vanishes.
    main = layers.Conv2D(1, 1, activation="sigmoid", dtype="float32", name="main_output")(head)

    outputs = main
    if cfg.deep_supervision:
        aux = layers.Conv2D(1, 1, padding="same", use_bias=True, name="aux_proj")(dec2)
        aux = layers.UpSampling2D(4, interpolation="bilinear", name="aux_up")(aux)
        aux = layers.Activation("sigmoid", dtype="float32", name="aux_output")(aux)
        outputs = {"main_output": main, "aux_output": aux}

    model = keras.Model(base.input, outputs, name=f"dplinknet50_{cfg.attention}")
    model.encoder_layer_names = [l.name for l in base.layers]
    return model


def set_encoder_trainable(model: keras.Model, trainable: bool) -> int:
    """Freeze or unfreeze the ResNet50 stack. Returns how many layers changed.

    Used for the warmup phase: train the fresh decoder first against frozen
    ImageNet features, then fine-tune everything.
    """
    if not hasattr(model, "encoder_layer_names"):
        raise RuntimeError("model was not built by this package; encoder names unknown")
    names = set(model.encoder_layer_names)
    if not names:
        # A from-scratch model (plain U-Net) has no pretrained stack to freeze.
        return 0
    n = 0
    for layer in model.layers:
        if layer.name in names:
            layer.trainable = trainable
            n += 1
    return n
