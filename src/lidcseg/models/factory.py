"""Model dispatch.

One entry point so `train.py` and the ablation runner never branch on
architecture. Adding a model means adding a builder and one registry line.
"""

from __future__ import annotations

from tensorflow import keras

from ..config import ModelConfig
from .dplinknet import build_dplinknet
from .unet import build_resnet50_unet, build_unet

BUILDERS = {
    "dplinknet": build_dplinknet,
    "resnet50_unet": build_resnet50_unet,
    "unet": build_unet,
}


def build_model(cfg: ModelConfig, img_size: int = 128) -> keras.Model:
    """Build the architecture named by `cfg.arch`."""
    try:
        builder = BUILDERS[cfg.arch]
    except KeyError:
        raise ValueError(
            f"unknown arch {cfg.arch!r}; expected one of {sorted(BUILDERS)}"
        ) from None

    model = builder(cfg, img_size)

    # The baselines ignore `attention`; say so rather than letting someone read
    # a leaderboard row as "cab did nothing here".
    if cfg.arch != "dplinknet" and cfg.attention not in (None, "none"):
        print(
            f"[model] note: arch={cfg.arch} has no attention stage -- "
            f"model.attention={cfg.attention!r} is ignored"
        )

    return model
