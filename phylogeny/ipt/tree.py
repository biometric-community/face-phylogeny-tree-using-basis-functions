"""Likelihood-ratio asymmetric scores + IPT root / edge reconstruction."""

from __future__ import annotations

from typing import Dict, List, Sequence, Tuple

import numpy as np


def parzen_logpdf(x: np.ndarray, samples: np.ndarray, bandwidth: float = 0.3) -> float:
    """Isotropic Gaussian Parzen density log-pdf for one vector."""
    if len(samples) == 0:
        return -1e6
    d = x.shape[0]
    diff = samples - x[None, :]
    dist2 = np.sum(diff * diff, axis=1) / (bandwidth ** 2)
    # log mean exp(-0.5 dist2) - 0.5 d log(2π h^2)
    m = -0.5 * dist2
    mmax = m.max()
    log_mean = mmax + np.log(np.mean(np.exp(m - mmax)) + 1e-300)
    return float(log_mean - 0.5 * d * np.log(2 * np.pi * bandwidth ** 2 + 1e-12))


def build_score_matrix(
    alphas_fwd: np.ndarray,
    alphas_rev: np.ndarray,
    pf_samples: np.ndarray,
    pb_samples: np.ndarray,
    bandwidth: float = 0.3,
) -> np.ndarray:
    """
    alphas_*: (N,N,D) with [i,j]=α fitting Ii from Ij (i≈B(j)).
    Λij = pf(αij)/pb(αij); Sij = log Λ.
    """
    n = alphas_fwd.shape[0]
    S = np.zeros((n, n), dtype=np.float64)
    for i in range(n):
        for j in range(n):
            if i == j:
                continue
            a = alphas_fwd[i, j]
            # forward: j is ancestor of i → α matches pf
            lf = parzen_logpdf(a, pf_samples, bandwidth)
            lb = parzen_logpdf(a, pb_samples, bandwidth)
            S[i, j] = lf - lb
    return S


def root_candidates(S: np.ndarray, top_k: int = 3) -> List[int]:
    """
    Root has many outgoing ancestral edges: for edge j→i (j ancestor of i),
    we use S[i,j] > 0 meaning αij prefers forward (j→i).
    Out-degree of node r = count_i 1[S[i,r] > 0].
    """
    n = S.shape[0]
    out_deg = np.zeros(n, dtype=np.int64)
    for r in range(n):
        out_deg[r] = int(np.sum(S[:, r] > 0))
    order = np.argsort(-out_deg)
    return [int(i) for i in order[:top_k]]


def reconstruct_edges(S: np.ndarray, threshold: float = 0.0) -> List[Tuple[int, int]]:
    """Directed edges parent→child where S[child, parent] > threshold."""
    n = S.shape[0]
    edges = []
    for i in range(n):
        for j in range(n):
            if i == j:
                continue
            if S[i, j] > threshold:
                edges.append((j, i))  # j ancestor of i
    return edges


def edge_accuracy(pred: Sequence[Tuple[int, int]], gt: Sequence[Tuple[int, int]]) -> float:
    if not gt:
        return float("nan")
    gt_set = set(gt)
    pred_set = set(pred)
    return float(len(gt_set & pred_set) / len(gt_set))


def rank_hit(candidates: Sequence[int], true_root: int, k: int) -> bool:
    return true_root in list(candidates)[:k]
