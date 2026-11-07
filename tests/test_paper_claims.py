"""The paper's central claim, reproduced with this repo's pipeline on synthetic faces.

Sec. 3.2: forward and reverse parameters differ, so a likelihood ratio finds the root.
"""

from __future__ import annotations

import numpy as np
from PIL import Image

from phylogeny.basis.transforms import fit_separable_poly, make_chain, to_gray_array
from phylogeny.ipt.tree import build_score_matrix, rank_hit, root_candidates

SIZE, CHAIN = 64, 4


def smooth_image(rng: np.random.RandomState) -> Image.Image:
    small = rng.randint(0, 255, (8, 8, 3), dtype=np.uint8)
    return Image.fromarray(small).resize((SIZE, SIZE), Image.BICUBIC)


def test_root_identification_beats_chance():
    rng = np.random.RandomState(0)
    forward, backward = [], []
    for _ in range(40):
        chain = make_chain(smooth_image(rng), CHAIN, rng)
        grays = [to_gray_array(image, SIZE) for image, _ in chain]
        for parent, child in zip(grays[:-1], grays[1:]):
            forward.append(fit_separable_poly(parent, child))
            backward.append(fit_separable_poly(child, parent))
    forward, backward = np.stack(forward), np.stack(backward)

    hits, sets = 0, 40
    for _ in range(sets):
        chain = make_chain(smooth_image(rng), CHAIN, rng)
        order = rng.permutation(CHAIN)
        grays = [to_gray_array(chain[i][0], SIZE) for i in order]
        alphas = np.zeros((CHAIN, CHAIN, forward.shape[1]))
        for i in range(CHAIN):
            for j in range(CHAIN):
                if i != j:
                    alphas[i, j] = fit_separable_poly(grays[j], grays[i])
        scores = build_score_matrix(alphas, alphas, forward, backward, bandwidth=0.05)
        hits += rank_hit(root_candidates(scores), int(np.where(order == 0)[0][0]), 1)
    assert hits / sets >= 0.45  # Table 5 Chebyshev Rank-1 is 45.18%; chance here is 25%
