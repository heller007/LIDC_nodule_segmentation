"""Preprocessed array cache.

Decoding PNGs and running diffusion over ~6k slices costs minutes and the
notebook paid it on every run, inside the training script, holding the whole
float32 stack in RAM twice (once raw, once augmented). Here it is paid once
into `.npy` memmaps, so later runs start in seconds and the arrays stay on
disk until a batch actually needs them.

The cache key covers every preprocessing knob, so changing `img_size` or a
diffusion parameter produces a new cache rather than silently reusing stale
arrays.
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np

from ..config import DataConfig
from .index import SliceRecord, index_stats
from .preprocess import load_image, load_vote_mask


def cache_key(cfg: DataConfig, n_records: int) -> str:
    payload = {
        "img_size": cfg.img_size,
        "n_readers": cfg.n_readers,
        "vote_threshold": cfg.vote_threshold,
        "diffusion_iters": cfg.diffusion_iters,
        "diffusion_kappa": cfg.diffusion_kappa,
        "diffusion_gamma": cfg.diffusion_gamma,
        "nodules": cfg.nodules,
        "n_records": n_records,
    }
    blob = json.dumps(payload, sort_keys=True).encode()
    return hashlib.sha1(blob).hexdigest()[:12]


class ArrayCache:
    """Holds images (N,H,W,1) and masks (N,H,W,1) as float32 memmaps."""

    def __init__(self, images: np.ndarray, masks: np.ndarray, records: list[SliceRecord]):
        if len(images) != len(masks) or len(images) != len(records):
            raise ValueError("images, masks and records must have equal length")
        self.images = images
        self.masks = masks
        self.records = records

    def __len__(self) -> int:
        return len(self.records)

    @property
    def positive_fraction(self) -> float:
        """Share of slices whose merged mask has any positive pixel."""
        return float(np.mean([m.any() for m in self.masks]))


def build_cache(
    records: list[SliceRecord],
    cfg: DataConfig,
    cache_dir: str | Path | None = None,
    verbose: bool = True,
) -> ArrayCache:
    """Materialise (or reuse) the preprocessed arrays for `records`."""
    cache_dir = Path(cache_dir or cfg.cache_dir)
    cache_dir.mkdir(parents=True, exist_ok=True)
    key = cache_key(cfg, len(records))

    img_path = cache_dir / f"img_{key}.npy"
    msk_path = cache_dir / f"msk_{key}.npy"
    meta_path = cache_dir / f"meta_{key}.json"

    size = cfg.img_size
    n = len(records)

    if img_path.exists() and msk_path.exists() and meta_path.exists():
        if verbose:
            print(f"[cache] reusing {img_path.name} ({n} slices)")
        images = np.load(img_path, mmap_mode="r")
        masks = np.load(msk_path, mmap_mode="r")
        if images.shape[0] == n and masks.shape[0] == n:
            return ArrayCache(images, masks, records)
        if verbose:
            print("[cache] length mismatch -- rebuilding")

    if verbose:
        print(f"[cache] building {n} slices at {size}x{size} -> {cache_dir}")

    images = np.lib.format.open_memmap(img_path, mode="w+", dtype=np.float32, shape=(n, size, size, 1))
    masks = np.lib.format.open_memmap(msk_path, mode="w+", dtype=np.float32, shape=(n, size, size, 1))

    diffusion = (
        {
            "num_iter": cfg.diffusion_iters,
            "kappa": cfg.diffusion_kappa,
            "gamma": cfg.diffusion_gamma,
        }
        if cfg.diffusion_iters > 0
        else None
    )

    incomplete = 0
    for i, rec in enumerate(records):
        images[i] = load_image(rec.image, size, diffusion)
        mask, n_found = load_vote_mask(list(rec.masks), size, cfg.vote_threshold)
        masks[i] = mask
        if n_found < cfg.vote_threshold:
            # Fewer readers than the vote needs -- this slice can never be
            # positive, which is a labelling artefact rather than a finding.
            incomplete += 1
        if verbose and (i + 1) % 500 == 0:
            print(f"  {i + 1}/{n}")

    images.flush()
    masks.flush()

    stats = index_stats(records)
    stats["incomplete_reader_slices"] = incomplete
    stats["positive_slice_fraction"] = round(float(np.mean([m.any() for m in masks])), 4)
    stats["positive_pixel_fraction"] = round(float(np.mean(masks)), 6)
    meta_path.write_text(json.dumps(stats, indent=1), encoding="utf-8")

    if verbose:
        print(f"[cache] done. {json.dumps(stats)}")
        if incomplete:
            print(f"[cache] warning: {incomplete} slices had < {cfg.vote_threshold} reader masks")

    return ArrayCache(
        np.load(img_path, mmap_mode="r"),
        np.load(msk_path, mmap_mode="r"),
        records,
    )


def filter_empty(
    cache: ArrayCache,
    indices: np.ndarray,
    drop_empty: bool,
    keep_frac: float,
    seed: int = 42,
) -> np.ndarray:
    """Thin out all-background slices within `indices`.

    Applied to the training indices only -- val and test must stay untouched or
    the metrics stop being comparable across experiments.
    """
    if drop_empty is False and keep_frac >= 1.0:
        return indices

    has_pos = np.array([bool(cache.masks[i].any()) for i in indices])
    pos = indices[has_pos]
    neg = indices[~has_pos]

    if drop_empty:
        return np.sort(pos)

    rng = np.random.default_rng(seed)
    n_keep = int(round(len(neg) * float(np.clip(keep_frac, 0.0, 1.0))))
    kept_neg = rng.choice(neg, size=n_keep, replace=False) if n_keep else np.array([], dtype=indices.dtype)
    return np.sort(np.concatenate([pos, kept_neg]).astype(indices.dtype))
