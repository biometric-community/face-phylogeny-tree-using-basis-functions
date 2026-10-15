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


def chebyshev_basis_1d(n: int, order: int) -> np.ndarray:
    """Chebyshev T_0..T_{order-1} on n samples in [-1,1]."""
    x = np.linspace(-1, 1, n)
    B = np.zeros((order, n))
    B[0] = 1.0
    if order > 1:
        B[1] = x
    for k in range(2, order):
        B[k] = 2 * x * B[k - 1] - B[k - 2]
    return B


def legendre_basis_1d(n: int, order: int) -> np.ndarray:
    x = np.linspace(-1, 1, n)
    B = np.zeros((order, n))
    B[0] = 1.0
    if order > 1:
        B[1] = x
    for k in range(2, order):
        B[k] = ((2 * k - 1) * x * B[k - 1] - (k - 1) * B[k - 2]) / k
    return B


def fit_separable_poly(src: np.ndarray, dst: np.ndarray, family: str = "chebyshev", order: int = 6) -> np.ndarray:
    """
    Fit Ii ≈ sum α_h B_h[Ij] via separable 1D bases on rows/cols average (ICE-style proxy).
    Returns α vector (order*2,).
    """
    h, w = src.shape
    if family == "chebyshev":
        By = chebyshev_basis_1d(h, order)
        Bx = chebyshev_basis_1d(w, order)
    else:
        By = legendre_basis_1d(h, order)
        Bx = legendre_basis_1d(w, order)
    # project row-means and col-means of dst onto bases conditioned on src structure
    # Simplified: regress dst flattened onto Kronecker-lite features from src
    feats = []
    for k in range(order):
        # modulate src with basis maps
        row_mod = (By[k][:, None] * src).mean()
        col_mod = (Bx[k][None, :] * src).mean()
        feats.append(row_mod)
        feats.append(col_mod)
    F = np.asarray(feats, dtype=np.float64)
    # target scalar summary + residual map energy → use least squares on patch means
    # Build design matrix over 8x8 grid cells
    gh, gw = 8, 8
    ys = np.linspace(0, h, gh + 1, dtype=int)
    xs = np.linspace(0, w, gw + 1, dtype=int)
    A, b = [], []
    for i in range(gh):
        for j in range(gw):
            s = src[ys[i] : ys[i + 1], xs[j] : xs[j + 1]].ravel()
            d = dst[ys[i] : ys[i + 1], xs[j] : xs[j + 1]].ravel()
            if s.size == 0:
                continue
            # features: moments of s under polynomial weights
            feat = []
            yy = np.linspace(-1, 1, max(ys[i + 1] - ys[i], 1))
            xx = np.linspace(-1, 1, max(xs[j + 1] - xs[j], 1))
            if family == "chebyshev":
                By_c = chebyshev_basis_1d(len(yy), order)
                Bx_c = chebyshev_basis_1d(len(xx), order)
            else:
                By_c = legendre_basis_1d(len(yy), order)
                Bx_c = legendre_basis_1d(len(xx), order)
            patch = src[ys[i] : ys[i + 1], xs[j] : xs[j + 1]]
            for k in range(order):
                feat.append(float((By_c[k][:, None] * patch).mean()))
                feat.append(float((Bx_c[k][None, :] * patch).mean()))
            A.append(feat)
            b.append(float(d.mean()))
    A = np.asarray(A, dtype=np.float64)
    b = np.asarray(b, dtype=np.float64)
    alpha, *_ = np.linalg.lstsq(A, b, rcond=None)
    return alpha.astype(np.float64)


PHOTOMETRIC_KINDS = ("brightness", "contrast", "gamma", "gaussian", "median")


def make_chain(root: Image.Image, length: int, rng: np.random.RandomState) -> List[Tuple[Image.Image, str]]:
    """Generate ancestral chain: root → edit1 → edit2 → ..."""
    chain = [(root.copy(), "root")]
    cur = root
    for _ in range(length - 1):
        kind = PHOTOMETRIC_KINDS[int(rng.randint(0, len(PHOTOMETRIC_KINDS)))]
        if kind == "brightness":
            s = float(rng.uniform(0.6, 1.5))
        elif kind == "contrast":
            s = float(rng.uniform(0.6, 1.5))
        elif kind == "gamma":
            s = float(rng.uniform(0.5, 1.8))
        elif kind == "gaussian":
            s = float(rng.uniform(0.8, 2.5))
        else:
            s = float(rng.choice([3, 5]))
        cur = apply_photometric(cur, kind, s)
        chain.append((cur.copy(), f"{kind}:{s:.2f}"))
    return chain
