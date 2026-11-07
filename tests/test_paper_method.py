"""Basis functions, alpha fitting and IPT steps, checked against Sec. 3 of the paper."""

from __future__ import annotations

import math
import re

import numpy as np
import pytest
from numpy.polynomial import chebyshev, legendre
from PIL import Image

from phylogeny.basis.transforms import (
    apply_photometric,
    chebyshev_basis_1d,
    fit_separable_poly,
    legendre_basis_1d,
    make_chain,
)
from phylogeny.ipt.tree import (
    build_score_matrix,
    edge_accuracy,
    parzen_logpdf,
    rank_hit,
    reconstruct_edges,
    root_candidates,
)


def test_chebyshev_basis_matches_numpy():
    x = np.linspace(-1, 1, 21)
    expected = np.stack([chebyshev.chebval(x, np.eye(6)[k]) for k in range(6)])
    assert np.allclose(chebyshev_basis_1d(21, 6), expected)


def test_legendre_basis_matches_numpy():
    x = np.linspace(-1, 1, 21)
    expected = np.stack([legendre.legval(x, np.eye(6)[k]) for k in range(6)])
    assert np.allclose(legendre_basis_1d(21, 6), expected)


@pytest.fixture
def pair():
    rng = np.random.RandomState(0)
    src = rng.rand(64, 64)
    return src, np.clip(0.8 * src + 0.1, 0, 1) ** 1.2


def test_alpha_has_the_paper_dimension(pair, targets):
    src, dst = pair
    assert fit_separable_poly(src, dst, order=6).shape == (targets["alpha_dimension"],)


def test_alpha_depends_only_on_pixel_values(pair):
    """Eq. (1) applies T to each pixel, so moving pixels together cannot change alpha."""
    src, dst = pair
    order = np.random.RandomState(1).permutation(src.size)
    shuffled_src, shuffled_dst = src.ravel()[order].reshape(src.shape), dst.ravel()[order].reshape(src.shape)
    assert np.allclose(fit_separable_poly(src, dst), fit_separable_poly(shuffled_src, shuffled_dst), atol=1e-6)


def test_parzen_density_of_one_sample_is_a_gaussian():
    x = np.array([0.3, -0.2])
    h = 0.5
    expected = -0.5 * (x @ x) / h**2 - math.log(2 * math.pi * h**2)
    assert parzen_logpdf(x, np.zeros((1, 2)), bandwidth=h) == pytest.approx(expected)


def test_parzen_density_averages_its_samples():
    x = np.zeros(1)
    samples = np.array([[0.0], [100.0]])
    single = parzen_logpdf(x, samples[:1], bandwidth=1.0)
    assert parzen_logpdf(x, samples, bandwidth=1.0) == pytest.approx(single + math.log(0.5), abs=1e-6)


def test_likelihood_ratio_favours_the_forward_distribution():
    forward, backward = np.zeros((5, 2)), np.full((5, 2), 3.0)
    alphas = np.zeros((2, 2, 2))
    alphas[1, 0] = 0.0  # looks forward
    alphas[0, 1] = 3.0  # looks backward
    scores = build_score_matrix(alphas, alphas, forward, backward)
    assert scores[1, 0] > 0 > scores[0, 1]
    assert scores[0, 0] == scores[1, 1] == 0.0


def chain_scores() -> np.ndarray:
    """0 -> 1 -> 2 -> 3, ancestral links included: S[child, parent] > 0."""
    scores = -np.ones((4, 4))
    for child in range(4):
        scores[child, :child] = 1.0
    return scores


def test_root_has_the_most_outgoing_links():
    assert root_candidates(chain_scores(), top_k=3) == [0, 1, 2]


def test_edges_point_from_parent_to_child():
    assert (0, 1) in reconstruct_edges(chain_scores())
    assert (1, 0) not in reconstruct_edges(chain_scores())


def test_edge_accuracy_and_rank_hits():
    assert edge_accuracy([(0, 1), (2, 3)], [(0, 1), (1, 2)]) == 0.5
    assert rank_hit([2, 0, 1], 0, 2) and not rank_hit([2, 0, 1], 0, 1)


def test_chain_starts_at_the_root_and_has_the_requested_length():
    root = Image.fromarray(np.random.RandomState(2).randint(0, 255, (32, 32, 3), dtype=np.uint8))
    chain = make_chain(root, 4, np.random.RandomState(3))
    assert len(chain) == 4 and chain[0][1] == "root"


def test_gamma_stays_in_the_table_2_range(targets):
    root = Image.new("RGB", (8, 8), (120, 120, 120))
    low, high = targets["gamma_range"]
    rng = np.random.RandomState(4)
    values = []
    for _ in range(100):
        values += [float(m) for _, tag in make_chain(root, 5, rng) for m in re.findall(r"gamma:([\d.]+)", tag)]
    assert values
    assert min(values) >= low and max(values) <= high


def test_brightness_edit_scales_the_image():
    image = Image.new("RGB", (4, 4), (100, 100, 100))
    assert np.asarray(apply_photometric(image, "brightness", 1.5))[0, 0, 0] == 150
