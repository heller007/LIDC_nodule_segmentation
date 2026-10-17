"""U-Net baselines.

Two reference points for the attention decoder, both trained through the same
data pipeline, split, loss, schedule and evaluation so a difference in score
is a difference in architecture and nothing else:

* `unet` -- the textbook U-Net, random init. Says what the ImageNet encoder is
  worth.
* `resnet50_unet` -- the same ResNet50 encoder as DPLinkNet50 with a plain
  U-Net decoder and no attention. Says what the LinkNet decoder and the
  attention blocks are worth on top of the encoder.

Both support the deep-supervision head, so that can be held constant across
the comparison instead of being a confound.
"""

from __future__ import annotations

from tensorflow import keras
from tensorflow.keras import layers

from ..config import ModelConfig


def conv_block(x, filters: int, name: str, dropout: float = 0.0):
    """Two 3x3 convs with BN+ReLU -- the U-Net unit.

    BN is used where the original paper had none; without it a from-scratch
    net at this depth and batch size trains poorly.
    """
    for i in (1, 2):
        x = layers.Conv2D(filters, 3, padding="same", use_bias=False, name=f"{name}_c{i}")(x)
        x = layers.BatchNormalization(name=f"{name}_bn{i}")(x)
        x = layers.Activation("relu", name=f"{name}_relu{i}")(x)
    if dropout > 0:
        x = layers.SpatialDropout2D(dropout, name=f"{name}_drop")(x)
    return x


def upconv_block(x, skip, filters: int, name: str, dropout: float = 0.0):
    """Upsample, concatenate the skip, then two convs.

    Concatenation is the U-Net way, as against LinkNet's addition: it keeps
    encoder and decoder features separable at the cost of more parameters.
    """
    x = layers.Conv2DTranspose(filters, 2, strides=2, padding="same", use_bias=False, name=f"{name}_up")(x)
    x = layers.BatchNormalization(name=f"{name}_up_bn")(x)
    x = layers.Activation("relu", name=f"{name}_up_relu")(x)
    if skip is not None:
        x = layers.Concatenate(name=f"{name}_cat")([x, skip])
    return conv_block(x, filters, name, dropout)


def _attach_heads(x, aux_source, cfg: ModelConfig, aux_upsample: int):
    """Final 1x1 sigmoid head, plus the optional auxiliary head.

    float32 on both activations: under mixed_float16 a float16 sigmoid
    saturates and the Dice gradient disappears.
    """
    main = layers.Conv2D(1, 1, activation="sigmoid", dtype="float32", name="main_output")(x)
    if not cfg.deep_supervision:
        return main
    aux = layers.Conv2D(1, 1, padding="same", use_bias=True, name="aux_proj")(aux_source)
    if aux_upsample > 1:
        aux = layers.UpSampling2D(aux_upsample, interpolation="bilinear", name="aux_up")(aux)
    aux = layers.Activation("sigmoid", dtype="float32", name="aux_output")(aux)
    return {"main_output": main, "aux_output": aux}


def build_unet(cfg: ModelConfig, img_size: int = 128) -> keras.Model:
    """Textbook U-Net, random initialisation.

    Five encoder stages at base, 2x, 4x, 8x, 16x width. At base_filters=64
    this is ~31M parameters, against ~80M for DPLinkNet50 -- noted in the
    leaderboard so the comparison is read with that in mind.
    """
    if img_size % 16 != 0:
        raise ValueError(f"img_size must be divisible by 16 for a 4-level U-Net, got {img_size}")

    f = cfg.base_filters
    inp = layers.Input((img_size, img_size, 3), name="input")

    e1 = conv_block(inp, f, "enc1")
    p1 = layers.MaxPooling2D(2, name="pool1")(e1)
    e2 = conv_block(p1, f * 2, "enc2")
    p2 = layers.MaxPooling2D(2, name="pool2")(e2)
    e3 = conv_block(p2, f * 4, "enc3")
    p3 = layers.MaxPooling2D(2, name="pool3")(e3)
    e4 = conv_block(p3, f * 8, "enc4", cfg.dropout)
    p4 = layers.MaxPooling2D(2, name="pool4")(e4)

    bottleneck = conv_block(p4, f * 16, "bottleneck", cfg.dropout)

    d4 = upconv_block(bottleneck, e4 if cfg.use_skips else None, f * 8, "dec4", cfg.dropout)
    d3 = upconv_block(d4, e3 if cfg.use_skips else None, f * 4, "dec3", cfg.dropout)
    d2 = upconv_block(d3, e2 if cfg.use_skips else None, f * 2, "dec2")
    d1 = upconv_block(d2, e1 if cfg.use_skips else None, f, "dec1")

    # d2 sits at 64x64 for a 128x128 input, so the aux head upsamples by 2.
    outputs = _attach_heads(d1, d2, cfg, aux_upsample=2)

    model = keras.Model(inp, outputs, name=f"unet_f{f}")
    # No pretrained encoder, so the freeze/unfreeze phase has nothing to act on.
    model.encoder_layer_names = []
    return model


def build_resnet50_unet(cfg: ModelConfig, img_size: int = 128) -> keras.Model:
    """ResNet50 encoder with a plain U-Net decoder, no attention.

    Shares the encoder and its taps with DPLinkNet50, so the only differences
    are the decoder style (concat + 3x3 convs vs additive + 1x1 squeeze) and
    the absence of attention blocks.
    """
    if img_size % 32 != 0:
        raise ValueError(f"img_size must be divisible by 32 for ResNet50 taps, got {img_size}")

    base = keras.applications.ResNet50(
        weights="imagenet" if cfg.pretrained else None,
        include_top=False,
        input_shape=(img_size, img_size, 3),
    )

    # Same taps as dplinknet.SKIP_LAYERS.
    s1 = base.get_layer("conv1_relu").output        # 64x64,   64ch
    s2 = base.get_layer("conv2_block3_out").output  # 32x32,  256ch
    s3 = base.get_layer("conv3_block4_out").output  # 16x16,  512ch
    s4 = base.get_layer("conv4_block6_out").output  # 8x8,   1024ch
    bottleneck = base.get_layer("conv5_block3_out").output  # 4x4, 2048ch

    use = cfg.use_skips
    d4 = upconv_block(bottleneck, s4 if use else None, 512, "dec4", cfg.dropout)  # 8x8
    d3 = upconv_block(d4, s3 if use else None, 256, "dec3", cfg.dropout)          # 16x16
    d2 = upconv_block(d3, s2 if use else None, 128, "dec2")                        # 32x32
    d1 = upconv_block(d2, s1 if use else None, 64, "dec1")                         # 64x64

    # One more upsample: the deepest ResNet50 tap is at stride 2, so the
    # decoder ends at half resolution.
    head = layers.Conv2DTranspose(32, 2, strides=2, padding="same", use_bias=False, name="head_up")(d1)
    head = layers.BatchNormalization(name="head_bn")(head)
    head = layers.Activation("relu", name="head_relu")(head)

    # d2 is at 32x32 -> x4 to reach 128x128, matching DPLinkNet's aux head.
    outputs = _attach_heads(head, d2, cfg, aux_upsample=4)

    model = keras.Model(base.input, outputs, name="resnet50_unet")
    model.encoder_layer_names = [l.name for l in base.layers]
    return model
