"""IPT root identification + edge reconstruction on held-out LFW chains."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import yaml
from tqdm import tqdm

_ROOT = Path(__file__).resolve().parents[1]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from phylogeny.basis.transforms import fit_separable_poly, make_chain, to_gray_array
from phylogeny.data.lfw_sets import list_subjects, load_image, resolve_lfw
from phylogeny.ipt.tree import (
    build_score_matrix,
    edge_accuracy,
    rank_hit,
    reconstruct_edges,
    root_candidates,
)


def load_config(path: str) -> dict:
    with open(path) as f:
        return yaml.safe_load(f)


def pairwise_alphas(grays: List[np.ndarray], family: str, order: int) -> np.ndarray:
    n = len(grays)
    d = order
    out = np.zeros((n, n, d), dtype=np.float64)
    for i in range(n):
        for j in range(n):
            if i == j:
                continue
            # α fitting Ii from Ij
            out[i, j] = fit_separable_poly(grays[j], grays[i], family=family, order=order)
    return out


def evaluate(cfg: dict) -> Dict:
    root = resolve_lfw(cfg)
    seed = int(cfg["train"].get("seed", 42)) + 7
    rng = np.random.RandomState(seed)
    subjects = list_subjects(root, min_images=int(cfg["data"].get("min_images", 5)))
    all_names = sorted(subjects.keys())
    start = int(cfg["eval"].get("subject_offset", 0))
    n_eval = int(cfg["eval"].get("n_sets", 40))
    names = all_names[start : start + n_eval]
    if len(names) < n_eval:
        # wrap / take from end if offset past end
        names = all_names[-n_eval:]

    blob = np.load(Path(cfg["paths"]["checkpoint_dir"]) / "densities.npz", allow_pickle=True)
    pf = blob["pf"]
    pb = blob["pb"]
    family = str(blob["family"]) if "family" in blob else cfg["model"].get("basis", "chebyshev")
    order = int(blob["order"]) if "order" in blob else int(cfg["model"].get("order", 6))
    gray_size = int(cfg["model"].get("gray_size", 64))
    chain_len = int(cfg["eval"].get("chain_length", 4))
    bandwidth = float(cfg["eval"].get("bandwidth", 0.05))

    rank1 = rank2 = rank3 = 0
    edge_accs = []
    n_sets = 0
    details = []

    for name in tqdm(names, desc="ipt-eval"):
        paths = subjects[name]
        root_img = load_image(paths[int(rng.randint(0, len(paths)))])
        chain = make_chain(root_img, chain_len, rng)
        # shuffle presentation order (root at unknown index)
        order_idx = list(range(chain_len))
        rng.shuffle(order_idx)
        true_root_pos = order_idx.index(0)  # where original root landed
        imgs = [chain[i][0] for i in order_idx]
        grays = [to_gray_array(im, gray_size) for im in imgs]
        alphas = pairwise_alphas(grays, family, order)
        S = build_score_matrix(alphas, alphas, pf, pb, bandwidth=bandwidth)
        cands = root_candidates(S, top_k=3)
        if rank_hit(cands, true_root_pos, 1):
            rank1 += 1
        if rank_hit(cands, true_root_pos, 2):
            rank2 += 1
        if rank_hit(cands, true_root_pos, 3):
            rank3 += 1
        # GT edges in shuffled index space: consecutive in original chain
        # original: 0→1→2→... ; mapped via order_idx
        pos_of = {orig: pos for pos, orig in enumerate(order_idx)}
        gt_edges = [(pos_of[i], pos_of[i + 1]) for i in range(chain_len - 1)]
        pred_edges = reconstruct_edges(S, threshold=0.0)
        # keep only tree-like: highest score parent per child
        n = chain_len
        best_parent = {}
        for p, c in pred_edges:
            sc = S[c, p]
            if c not in best_parent or sc > best_parent[c][1]:
                best_parent[c] = (p, sc)
        pred_tree = [(p, c) for c, (p, _) in best_parent.items()]
        ea = edge_accuracy(pred_tree, gt_edges)
        edge_accs.append(ea)
        n_sets += 1
        details.append(
            {
                "subject": name,
                "true_root": true_root_pos,
                "candidates": cands,
                "edge_acc": ea,
            }
        )

    results = {
        "n_sets": n_sets,
        "chain_length": chain_len,
        "basis": family,
        "rank1": rank1 / max(n_sets, 1),
        "rank2": rank2 / max(n_sets, 1),
        "rank3": rank3 / max(n_sets, 1),
        "ipt_edge_accuracy": float(np.nanmean(edge_accs)) if edge_accs else 0.0,
        "details": details[:20],
    }
    print(
        f"[ipt] n={n_sets} R1={results['rank1']:.3f} R2={results['rank2']:.3f} "
        f"R3={results['rank3']:.3f} edge={results['ipt_edge_accuracy']:.3f}"
    )
    log_dir = Path(cfg["paths"]["log_dir"])
    log_dir.mkdir(parents=True, exist_ok=True)
    out = log_dir / "eval.json"
    with open(out, "w") as f:
        json.dump(results, f, indent=2)
        f.write("\n")
    print(f"Wrote {out}")
    return results


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/default.yaml")
    args = ap.parse_args()
    evaluate(load_config(args.config))


if __name__ == "__main__":
    main()
