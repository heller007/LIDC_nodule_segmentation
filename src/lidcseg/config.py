"""Configuration objects.

Every experiment is fully described by a `Config`. Configs load from YAML and
accept dotted-key overrides from the CLI, so an ablation is a one-line change
and the exact settings of a run are always recoverable from `config.yaml` in
the run directory.
"""

from __future__ import annotations

import copy
import dataclasses
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Literal

import yaml


@dataclass
class DataConfig:
    # Root containing LIDC-IDRI-XXXX/ patient directories.
    root: str = "/kaggle/input/lidcidri/LIDC-IDRI-slices"
    # Where preprocessed .npy caches are written (fast re-runs).
    cache_dir: str = "/kaggle/working/cache"

    img_size: int = 128

    # Which nodule sub-directories to use per patient. The original notebook
    # read only "nodule-0" and discarded the rest; `None` takes all of them.
    nodules: list[str] | None = None

    # Radiologist agreement needed for a pixel to count as nodule.
    # LIDC ships 4 independent readers; >=2 is the common "majority" choice.
    n_readers: int = 4
    vote_threshold: int = 2

    # Anisotropic (Perona-Malik) diffusion applied before resize.
    diffusion_iters: int = 5
    diffusion_kappa: float = 50.0
    diffusion_gamma: float = 0.1

    # Drop slices whose merged mask is empty. The notebook kept them; they are
    # ~all background and dominate the pixel count.
    drop_empty_masks: bool = False
    # Keep at most this fraction of empty-mask slices (ignored when
    # drop_empty_masks is True). Acts as a negative-sample budget.
    empty_mask_keep_frac: float = 1.0


@dataclass
class SplitConfig:
    # "patient" groups every slice of a patient into one fold -- the correct
    # protocol. "slice" reproduces the notebook's leaky random split and
    # exists only so the gap can be measured.
    mode: Literal["patient", "slice"] = "patient"
    val_frac: float = 0.15
    test_frac: float = 0.15
    seed: int = 42

    # Cross-validation. folds<=1 means a single hold-out split.
    folds: int = 1
    fold: int = 0


@dataclass
class AugConfig:
    enabled: bool = True
    hflip: float = 0.5
    vflip: float = 0.5
    rot90: bool = True
    brightness: float = 0.5          # probability
    brightness_range: float = 0.15   # factor in [1-r, 1+r]
    noise: float = 0.5               # probability
    noise_sigma: float = 0.005
    zoom: float = 0.5                # probability
    zoom_min: float = 0.92
    elastic: float = 0.0             # probability; 0 disables
    elastic_alpha: float = 20.0
    elastic_sigma: float = 4.0


@dataclass
class ModelConfig:
    # Architecture. The two baselines exist so the attention decoder can be
    # compared against something, not just against its own ablations:
    #   dplinknet      ResNet50 encoder + LinkNet decoder (additive skips,
    #                  1x1 squeeze/expand) + an attention block per stage
    #   resnet50_unet  same encoder, plain U-Net decoder (concat skips, two
    #                  3x3 convs), no attention -- isolates the decoder and
    #                  the attention block while holding the encoder fixed
    #   unet           textbook U-Net trained from scratch -- isolates what
    #                  the ImageNet encoder is worth
    arch: Literal["dplinknet", "resnet50_unet", "unet"] = "dplinknet"

    encoder: Literal["resnet50"] = "resnet50"
    # Attention block inserted at every decoder stage. `resnet50_unet` and
    # `unet` ignore this; they are deliberately attention-free baselines.
    attention: Literal["scanext", "cab", "none"] = "scanext"
    attention_reduction: int = 8
    # ImageNet weights for the encoder. Ignored by `unet`, which has no
    # pretrained encoder to load.
    pretrained: bool = True
    # Caffe-style ImageNet preprocessing inside the graph. The notebook fed
    # raw [0,1] tensors to ImageNet weights, a train/pretrain mismatch.
    imagenet_preprocess: bool = True
    # Auxiliary head off dec2 (deep supervision).
    deep_supervision: bool = True
    aux_weight: float = 0.2
    dropout: float = 0.0
    # Which encoder taps feed decoder skips.
    use_skips: bool = True
    # `unet` only: channel width of the first encoder stage; each stage
    # doubles it. 64 is the width in the original U-Net paper.
    base_filters: int = 64


@dataclass
class LossConfig:
    name: Literal["tversky_focal", "dice_focal", "dice_bce"] = "tversky_focal"
    # Tversky: alpha penalises false positives, beta false negatives.
    tversky_alpha: float = 0.3
    tversky_beta: float = 0.7
    tversky_gamma: float = 0.75
    # Dice+Focal
    dice_weight: float = 0.5
    focal_alpha: float = 0.25
    focal_gamma: float = 2.0
    # Per-image losses average the score over the batch instead of pooling all
    # pixels into one global ratio. Small nodules get a fair share this way.
    per_image: bool = True
    smooth: float = 1e-6


@dataclass
class TrainConfig:
    epochs: int = 200
    batch_size: int = 16
    lr: float = 5e-5
    optimizer: Literal["adam", "adamw"] = "adamw"
    weight_decay: float = 1e-4
    clipnorm: float = 1.0

    # Cosine decay with linear warmup. "none" keeps a constant LR and lets
    # ReduceLROnPlateau handle decay, as the notebook did.
    schedule: Literal["cosine", "none"] = "cosine"
    warmup_epochs: int = 3
    min_lr: float = 1e-7

    # Train the decoder against a frozen encoder for this many epochs first,
    # then unfreeze. Avoids wrecking ImageNet features early on.
    freeze_encoder_epochs: int = 0

    mixed_precision: bool = True
    early_stopping_patience: int = 25
    reduce_lr_patience: int = 10
    reduce_lr_factor: float = 0.5
    monitor: str = "val_dice"

    # Exponential moving average of weights, evaluated at the end.
    ema: bool = False
    ema_decay: float = 0.999


@dataclass
class EvalConfig:
    # Sweep thresholds on validation, apply the winner to test.
    tune_threshold: bool = True
    threshold_min: float = 0.10
    threshold_max: float = 0.90
    threshold_steps: int = 33
    default_threshold: float = 0.5

    # 8-way flip/rotation test-time augmentation.
    tta: bool = True

    # Discard predicted components smaller than this many pixels.
    min_component_px: int = 0

    # Bootstrap resamples for confidence intervals on test metrics.
    bootstrap: int = 1000


@dataclass
class Config:
    name: str = "default"
    out_dir: str = "/kaggle/working/runs"
    seed: int = 42

    data: DataConfig = field(default_factory=DataConfig)
    split: SplitConfig = field(default_factory=SplitConfig)
    aug: AugConfig = field(default_factory=AugConfig)
    model: ModelConfig = field(default_factory=ModelConfig)
    loss: LossConfig = field(default_factory=LossConfig)
    train: TrainConfig = field(default_factory=TrainConfig)
    eval: EvalConfig = field(default_factory=EvalConfig)

    # ------------------------------------------------------------------ #

    @property
    def run_dir(self) -> Path:
        d = Path(self.out_dir) / self.name
        if self.split.folds > 1:
            d = d / f"fold{self.split.fold}"
        return d

    def to_dict(self) -> dict[str, Any]:
        return dataclasses.asdict(self)

    def save(self, path: str | Path) -> None:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(yaml.safe_dump(self.to_dict(), sort_keys=False), encoding="utf-8")


_SECTIONS = {
    "data": DataConfig,
    "split": SplitConfig,
    "aug": AugConfig,
    "model": ModelConfig,
    "loss": LossConfig,
    "train": TrainConfig,
    "eval": EvalConfig,
}


def _merge(base: dict, extra: dict) -> dict:
    """Recursive dict merge; `extra` wins."""
    out = copy.deepcopy(base)
    for k, v in extra.items():
        if isinstance(v, dict) and isinstance(out.get(k), dict):
            out[k] = _merge(out[k], v)
        else:
            out[k] = v
    return out


_TRUE = {"true", "yes", "on", "1"}
_FALSE = {"false", "no", "off", "0"}


def _coerce(value: Any, field: dataclasses.Field) -> Any:
    """Cast a loaded value to the field's declared type.

    Needed because YAML 1.1 only recognises a float when it has a decimal
    point: `yaml.safe_load("1e-4")` is the *string* "1e-4", so a CLI override
    like `train.lr=1e-4` would otherwise set the learning rate to a string and
    fail deep inside the optimizer.
    """
    if value is None:
        return None
    ann = field.type
    # Dataclass annotations may arrive as strings under `from __future__`.
    ann = ann if isinstance(ann, str) else getattr(ann, "__name__", str(ann))

    if ann.startswith("bool"):
        if isinstance(value, bool):
            return value
        s = str(value).strip().lower()
        if s in _TRUE:
            return True
        if s in _FALSE:
            return False
        raise ValueError(f"cannot read {value!r} as a bool for '{field.name}'")

    if ann.startswith("int") and not isinstance(value, bool):
        return int(float(value))

    if ann.startswith("float"):
        return float(value)

    # Literal[...] and str fields: leave as-is; the dataclass validates later.
    return value


def from_dict(d: dict[str, Any]) -> Config:
    """Build a Config from a plain dict, validating section/field names."""
    d = dict(d)
    kwargs: dict[str, Any] = {}
    for key, cls in _SECTIONS.items():
        section = d.pop(key, None) or {}
        if not isinstance(section, dict):
            raise TypeError(f"config section '{key}' must be a mapping, got {type(section).__name__}")
        fields = {f.name: f for f in dataclasses.fields(cls)}
        unknown = set(section) - set(fields)
        if unknown:
            raise ValueError(f"unknown key(s) in '{key}': {sorted(unknown)}")
        section = {k: _coerce(v, fields[k]) for k, v in section.items()}
        kwargs[key] = cls(**section)

    top_valid = {f.name for f in dataclasses.fields(Config)} - set(_SECTIONS)
    unknown_top = set(d) - top_valid
    if unknown_top:
        raise ValueError(f"unknown top-level config key(s): {sorted(unknown_top)}")
    kwargs.update(d)
    return Config(**kwargs)


def load_config(path: str | Path, overrides: list[str] | None = None) -> Config:
    """Load YAML, follow an optional `_base_` chain, then apply CLI overrides.

    Overrides are dotted assignments parsed as YAML scalars, e.g.
    ``train.lr=1e-4``, ``model.attention=cab``, ``aug.enabled=false``.
    """
    path = Path(path)
    raw = yaml.safe_load(path.read_text(encoding="utf-8")) or {}

    # A config may inherit from another via `_base_` (relative to itself).
    seen: set[Path] = set()
    while "_base_" in raw:
        base_rel = raw.pop("_base_")
        base_path = (path.parent / base_rel).resolve()
        if base_path in seen:
            raise ValueError(f"circular _base_ chain at {base_path}")
        seen.add(base_path)
        base_raw = yaml.safe_load(base_path.read_text(encoding="utf-8")) or {}
        raw = _merge(base_raw, raw)
        path = base_path

    for ov in overrides or []:
        if "=" not in ov:
            raise ValueError(f"override must look like key.sub=value, got {ov!r}")
        key, _, val = ov.partition("=")
        node = raw
        parts = key.strip().split(".")
        for p in parts[:-1]:
            node = node.setdefault(p, {})
            if not isinstance(node, dict):
                raise ValueError(f"cannot descend into non-mapping at {p!r} in override {ov!r}")
        node[parts[-1]] = yaml.safe_load(val)

    return from_dict(raw)
