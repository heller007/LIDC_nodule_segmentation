"""Slice-level preprocessing: Perona-Malik diffusion and reader-vote fusion."""

from __future__ import annotations

import cv2
import numpy as np


def anisotropic_diffusion(
    img: np.ndarray,
    num_iter: int = 5,
    kappa: float = 50.0,
    gamma: float = 0.1,
) -> np.ndarray:
    """Perona-Malik anisotropic diffusion -- edge-preserving denoise.

    Vectorised form of the notebook's loop. Two differences that matter:

    * Gradients at the borders use edge replication rather than `np.roll`'s
      wrap-around, so the top row no longer diffuses into the bottom row.
    * `gamma` is checked against the stability bound (0.25 for 4-neighbour
      updates); above it the scheme diverges.

    Operates on the image's native intensity scale, so `kappa` is in the same
    units as the input (50 suits 0-255 CT slices).
    """
    if not 0 < gamma <= 0.25:
        raise ValueError(f"gamma must be in (0, 0.25] for stability, got {gamma}")

    out = img.astype(np.float32, copy=True)
    for _ in range(num_iter):
        # Edge-replicated neighbour differences.
        d_n = np.vstack([out[:1], out[:-1]]) - out
        d_s = np.vstack([out[1:], out[-1:]]) - out
        d_w = np.hstack([out[:, :1], out[:, :-1]]) - out
        d_e = np.hstack([out[:, 1:], out[:, -1:]]) - out

        inv_k2 = 1.0 / (kappa * kappa)
        c_n = np.exp(-(d_n * d_n) * inv_k2)
        c_s = np.exp(-(d_s * d_s) * inv_k2)
        c_w = np.exp(-(d_w * d_w) * inv_k2)
        c_e = np.exp(-(d_e * d_e) * inv_k2)

        out += gamma * (c_n * d_n + c_s * d_s + c_w * d_w + c_e * d_e)
    return out


def load_image(path: str, img_size: int, diffusion: dict | None = None) -> np.ndarray:
    """Read a grayscale slice, denoise, resize, scale to [0,1].

    Returns a single-channel float32 array of shape (img_size, img_size, 1).
    The 3-channel expansion the encoder needs happens in the graph, so the
    cache stays a third of the size.

    Diffusion runs at native resolution *before* resize. Denoising after
    downsampling (as the notebook did) works on already-mixed pixels.
    """
    img = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
    if img is None:
        raise FileNotFoundError(f"could not read image: {path}")

    if diffusion:
        img = anisotropic_diffusion(img, **diffusion)

    img = cv2.resize(img.astype(np.float32), (img_size, img_size), interpolation=cv2.INTER_LINEAR)
    img = np.clip(img / 255.0, 0.0, 1.0)
    return img[..., None].astype(np.float32)


def load_vote_mask(
    mask_paths: list[str],
    img_size: int,
    vote_threshold: int = 2,
) -> tuple[np.ndarray, int]:
    """Fuse per-radiologist masks by majority vote.

    Each reader's mask is binarised, resized with nearest-neighbour (so no
    interpolated half-labels appear), then summed. A pixel is positive when at
    least `vote_threshold` readers marked it.

    Returns (mask, n_found) where `n_found` is how many reader masks actually
    existed on disk -- callers use it to spot slices with missing annotations.
    """
    votes = np.zeros((img_size, img_size), dtype=np.uint8)
    n_found = 0

    for path in mask_paths:
        m = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
        if m is None:
            continue
        n_found += 1
        # Binarise first, then resize nearest -- resizing a soft mask and
        # thresholding afterwards invents labels along nodule boundaries.
        m = (m > 127).astype(np.uint8)
        if m.shape != (img_size, img_size):
            m = cv2.resize(m, (img_size, img_size), interpolation=cv2.INTER_NEAREST)
        votes += m

    mask = (votes >= vote_threshold).astype(np.float32)
    return mask[..., None], n_found
