"""tf.data input pipelines.

Augmentation happens inside the graph, freshly each epoch. The notebook
instead built one static augmented copy and concatenated it, so the model saw
the same 2x set every epoch -- half the regularisation benefit at twice the
memory. Here every epoch draws new transforms and nothing is duplicated in RAM.

Spatial transforms apply identically to image and mask; photometric ones touch
the image only.
"""

from __future__ import annotations

import numpy as np
import tensorflow as tf

from ..config import AugConfig, ModelConfig
from .cache import ArrayCache

AUTOTUNE = tf.data.AUTOTUNE


def _rand(seed_pair: tf.Tensor) -> tf.Tensor:
    return tf.random.stateless_uniform([], seed=seed_pair, minval=0.0, maxval=1.0)


def augment(img: tf.Tensor, mask: tf.Tensor, cfg: AugConfig, seed: tf.Tensor):
    """Apply the configured transforms to one (image, mask) pair.

    `seed` is a stateless-RNG seed pair, so the same element gets different
    draws each epoch while the whole pipeline stays reproducible from the run
    seed and parallelises without RNG contention.
    """
    s = tf.random.experimental.stateless_split(seed, num=8)

    if cfg.hflip > 0:
        do = _rand(s[0]) < cfg.hflip
        img = tf.cond(do, lambda: tf.image.flip_left_right(img), lambda: img)
        mask = tf.cond(do, lambda: tf.image.flip_left_right(mask), lambda: mask)

    if cfg.vflip > 0:
        do = _rand(s[1]) < cfg.vflip
        img = tf.cond(do, lambda: tf.image.flip_up_down(img), lambda: img)
        mask = tf.cond(do, lambda: tf.image.flip_up_down(mask), lambda: mask)

    if cfg.rot90:
        k = tf.random.stateless_uniform([], seed=s[2], minval=0, maxval=4, dtype=tf.int32)
        img = tf.image.rot90(img, k)
        mask = tf.image.rot90(mask, k)

    if cfg.zoom > 0:
        def _zoom():
            scale = tf.random.stateless_uniform([], seed=s[3], minval=cfg.zoom_min, maxval=1.0)
            h = tf.cast(tf.shape(img)[0], tf.float32)
            w = tf.cast(tf.shape(img)[1], tf.float32)
            nh = tf.cast(h * scale, tf.int32)
            nw = tf.cast(w * scale, tf.int32)
            y0 = tf.random.stateless_uniform(
                [], seed=s[4], minval=0, maxval=tf.maximum(tf.cast(h, tf.int32) - nh, 1), dtype=tf.int32
            )
            x0 = tf.random.stateless_uniform(
                [], seed=s[5], minval=0, maxval=tf.maximum(tf.cast(w, tf.int32) - nw, 1), dtype=tf.int32
            )
            ic = tf.image.crop_to_bounding_box(img, y0, x0, nh, nw)
            mc = tf.image.crop_to_bounding_box(mask, y0, x0, nh, nw)
            size = tf.shape(img)[:2]
            ic = tf.image.resize(ic, size, method="bilinear")
            # Nearest for the mask: bilinear would create fractional labels.
            mc = tf.image.resize(mc, size, method="nearest")
            return ic, mc

        do = _rand(s[6]) < cfg.zoom
        img, mask = tf.cond(do, _zoom, lambda: (img, mask))

    # ---- photometric: image only ----
    # Four independent seeds: one probability and one value per transform.
    # Sharing a seed between a probability draw and a value draw correlates
    # "was it applied" with "how strongly", which is not the intended noise.
    p = tf.random.experimental.stateless_split(s[7], num=4)

    if cfg.brightness > 0:
        def _bright():
            r = cfg.brightness_range
            f = tf.random.stateless_uniform([], seed=p[0], minval=1.0 - r, maxval=1.0 + r)
            return tf.clip_by_value(img * f, 0.0, 1.0)

        img = tf.cond(_rand(p[1]) < cfg.brightness, _bright, lambda: img)

    if cfg.noise > 0:
        def _noise():
            n = tf.random.stateless_normal(tf.shape(img), seed=p[2], stddev=cfg.noise_sigma)
            return tf.clip_by_value(img + n, 0.0, 1.0)

        img = tf.cond(_rand(p[3]) < cfg.noise, _noise, lambda: img)

    img = tf.ensure_shape(img, [None, None, 1])
    mask = tf.ensure_shape(mask, [None, None, 1])
    return img, mask


def to_encoder_input(img: tf.Tensor, imagenet_preprocess: bool) -> tf.Tensor:
    """Expand 1 channel to 3 and match the encoder's expected scaling.

    ResNet50's ImageNet weights were trained on caffe-style inputs: BGR order,
    channel means subtracted, range roughly [-124, 152]. The notebook fed
    [0,1] RGB, so the pretrained filters saw inputs an order of magnitude off
    their training distribution. Replicating a grayscale channel makes RGB vs
    BGR irrelevant, but the mean subtraction still matters.
    """
    x = tf.repeat(img, 3, axis=-1)
    if imagenet_preprocess:
        x = x * 255.0
        mean = tf.constant([103.939, 116.779, 123.68], dtype=x.dtype)
        x = x - mean
    return x


def make_dataset(
    cache: ArrayCache,
    indices: np.ndarray,
    batch_size: int,
    aug_cfg: AugConfig | None,
    model_cfg: ModelConfig,
    training: bool,
    seed: int = 42,
    drop_remainder: bool | None = None,
) -> tf.data.Dataset:
    """Build a batched dataset over `indices` of `cache`.

    Reads go through a py_function so the memmapped arrays are touched lazily
    -- only the slices in the current batch are paged in.
    """
    indices = np.asarray(indices, dtype=np.int64)
    if len(indices) == 0:
        raise ValueError("cannot build a dataset from an empty index array")

    images, masks = cache.images, cache.masks

    def _read(i: np.int64):
        j = int(i)
        return images[j].astype(np.float32), masks[j].astype(np.float32)

    size = cache.images.shape[1]

    ds = tf.data.Dataset.from_tensor_slices(indices)
    if training:
        ds = ds.shuffle(len(indices), seed=seed, reshuffle_each_iteration=True)

    def _load(i):
        img, mask = tf.numpy_function(_read, [i], [tf.float32, tf.float32], stateful=False)
        img = tf.ensure_shape(img, [size, size, 1])
        mask = tf.ensure_shape(mask, [size, size, 1])
        return img, mask

    ds = ds.map(_load, num_parallel_calls=AUTOTUNE)

    if training and aug_cfg and aug_cfg.enabled:
        # Dataset.counter is TF>=2.13; fall back for older runtimes.
        counter = (
            tf.data.Dataset.counter()
            if hasattr(tf.data.Dataset, "counter")
            else tf.data.experimental.Counter()
        )
        seeds = counter.map(lambda c: tf.stack([tf.cast(c, tf.int32), seed]))
        ds = tf.data.Dataset.zip((ds, seeds))
        ds = ds.map(
            lambda pair, sd: augment(pair[0], pair[1], aug_cfg, sd),
            num_parallel_calls=AUTOTUNE,
        )

    def _finalize(img, mask):
        x = to_encoder_input(img, model_cfg.imagenet_preprocess)
        if model_cfg.deep_supervision:
            # Keras matches dict targets to output names.
            return x, {"main_output": mask, "aux_output": mask}
        return x, mask

    ds = ds.map(_finalize, num_parallel_calls=AUTOTUNE)

    if drop_remainder is None:
        drop_remainder = training
    ds = ds.batch(batch_size, drop_remainder=drop_remainder)
    return ds.prefetch(AUTOTUNE)


def raw_eval_arrays(cache: ArrayCache, indices: np.ndarray, model_cfg: ModelConfig):
    """Return (encoder-ready X, y) as in-memory arrays for evaluation.

    Evaluation needs the masks aligned with predictions and small enough to
    hold, so val/test are materialised rather than streamed.
    """
    indices = np.asarray(indices, dtype=np.int64)
    imgs = np.stack([cache.images[int(i)] for i in indices]).astype(np.float32)
    msks = np.stack([cache.masks[int(i)] for i in indices]).astype(np.float32)
    x = np.repeat(imgs, 3, axis=-1)
    if model_cfg.imagenet_preprocess:
        x = x * 255.0 - np.array([103.939, 116.779, 123.68], dtype=np.float32)
    return x, msks
