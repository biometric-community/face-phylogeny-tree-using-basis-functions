"""Photometric edits + Chebyshev/Legendre/Gabor-style α fitting."""

from __future__ import annotations

from typing import Callable, Dict, List, Tuple

import numpy as np
from PIL import Image, ImageEnhance, ImageFilter


def to_gray_array(img: Image.Image, size: int = 64) -> np.ndarray:
    g = img.convert("L").resize((size, size), Image.BILINEAR)
    return np.asarray(g, dtype=np.float64) / 255.0


def apply_photometric(img: Image.Image, kind: str, strength: float) -> Image.Image:
    """Apply one photometric transform (paper Table 2 family)."""
    out = img.convert("RGB")
    if kind == "brightness":
        return ImageEnhance.Brightness(out).enhance(strength)
    if kind == "contrast":
        return ImageEnhance.Contrast(out).enhance(strength)
    if kind == "gamma":
        arr = np.asarray(out).astype(np.float64) / 255.0
        arr = np.clip(arr ** strength, 0, 1)
        return Image.fromarray((arr * 255).astype(np.uint8))
    if kind == "gaussian":
        # strength as radius proxy
        return out.filter(ImageFilter.GaussianBlur(radius=max(0.5, strength)))
    if kind == "median":
        k = int(max(3, round(strength)))
        if k % 2 == 0:
            k += 1
        return out.filter(ImageFilter.MedianFilter(size=k))
    raise ValueError(kind)


def chebyshev_values(x: np.ndarray, order: int) -> np.ndarray:
    """Chebyshev T_0..T_{order-1} evaluated at ``x`` in [-1,1]."""
    B = np.zeros((order, x.size))
    B[0] = 1.0
    if order > 1:
        B[1] = x
    for k in range(2, order):
        B[k] = 2 * x * B[k - 1] - B[k - 2]
    return B


def legendre_values(x: np.ndarray, order: int) -> np.ndarray:
    """Legendre P_0..P_{order-1} evaluated at ``x`` in [-1,1]."""
    B = np.zeros((order, x.size))
    B[0] = 1.0
    if order > 1:
        B[1] = x
    for k in range(2, order):
        B[k] = ((2 * k - 1) * x * B[k - 1] - (k - 1) * B[k - 2]) / k
    return B


def chebyshev_basis_1d(n: int, order: int) -> np.ndarray:
    """Chebyshev T_0..T_{order-1} on n samples in [-1,1]."""
    return chebyshev_values(np.linspace(-1, 1, n), order)


def legendre_basis_1d(n: int, order: int) -> np.ndarray:
    return legendre_values(np.linspace(-1, 1, n), order)


def fit_separable_poly(
    src: np.ndarray, dst: np.ndarray, family: str = "chebyshev", order: int = 6, ridge: float = 1e-6
) -> np.ndarray:
    """
    Fit dst(p) ≈ sum_h α_h B_h[src(p)] per pixel (Eq. 1). Returns α of length ``order``.

    Pixel values in [0,1] are mapped to the polynomial domain [-1,1]. Regularized least
    squares solves the same objective as the ICE updates of Eq. (4) in closed form.
    """
    x = 2.0 * np.asarray(src, dtype=np.float64).ravel() - 1.0
    y = np.asarray(dst, dtype=np.float64).ravel()
    values = chebyshev_values if family == "chebyshev" else legendre_values
    A = values(x, order).T
    return np.linalg.solve(A.T @ A + ridge * np.eye(order), A.T @ y)


PHOTOMETRIC_KINDS = ("brightness", "gamma", "gaussian", "median")  # paper Table 2


def make_chain(root: Image.Image, length: int, rng: np.random.RandomState) -> List[Tuple[Image.Image, str]]:
    """Generate ancestral chain: root → edit1 → edit2 → ..."""
    chain = [(root.copy(), "root")]
    cur = root
    for _ in range(length - 1):
        kind = PHOTOMETRIC_KINDS[int(rng.randint(0, len(PHOTOMETRIC_KINDS)))]
        if kind == "brightness":
            s = float(rng.uniform(0.9, 1.5))
        elif kind == "gamma":
            s = float(rng.uniform(0.5, 1.5))
        elif kind == "gaussian":
            s = float(rng.uniform(1.0, 3.0))
        else:
            s = float(rng.choice([3, 5]))
        cur = apply_photometric(cur, kind, s)
        chain.append((cur.copy(), f"{kind}:{s:.2f}"))
    return chain
