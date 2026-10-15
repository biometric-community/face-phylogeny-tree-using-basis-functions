"""Fit forward/backward Parzen densities on known photometric transform pairs."""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path
from typing import Dict, Optional

import numpy as np
import yaml
from tqdm import tqdm

_ROOT = Path(__file__).resolve().parents[1]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from phylogeny.basis.transforms import fit_separable_poly, make_chain, to_gray_array
from phylogeny.data.lfw_sets import list_subjects, load_image, resolve_lfw


def load_config(path: str) -> dict:
    with open(path) as f:
        return yaml.safe_load(f)


def train(cfg: dict) -> Dict:
    root = resolve_lfw(cfg)
    seed = int(cfg["train"].get("seed", 42))
    rng = np.random.RandomState(seed)
    min_imgs = int(cfg["data"].get("min_images", 5))
    subjects = list_subjects(root, min_images=min_imgs)
    names = sorted(subjects.keys())
    max_subj = cfg["data"].get("max_subjects")
    if max_subj is not None:
        names = names[: int(max_subj)]

    family = cfg["model"].get("basis", "chebyshev")
    order = int(cfg["model"].get("order", 6))
    gray_size = int(cfg["model"].get("gray_size", 64))
    chain_len = int(cfg["train"].get("chain_length", 4))
    pairs_per_subj = int(cfg["train"].get("pairs_per_subject", 3))

    pf, pb = [], []
    t0 = time.time()
    for name in tqdm(names, desc="fit-densities"):
        paths = subjects[name]
        for _ in range(pairs_per_subj):
            root_img = load_image(paths[int(rng.randint(0, len(paths)))])
            chain = make_chain(root_img, chain_len, rng)
            # collect consecutive forward (parent→child) and reverse (child→parent) α
            for i in range(len(chain) - 1):
                parent = to_gray_array(chain[i][0], gray_size)
                child = to_gray_array(chain[i + 1][0], gray_size)
                # forward: child ≈ B(parent) → α_fwd fits child from parent
                a_f = fit_separable_poly(parent, child, family=family, order=order)
                a_b = fit_separable_poly(child, parent, family=family, order=order)
                pf.append(a_f)
                pb.append(a_b)

    pf_arr = np.stack(pf, axis=0)
    pb_arr = np.stack(pb, axis=0)
    elapsed = time.time() - t0

    ckpt_dir = Path(cfg["paths"]["checkpoint_dir"])
    log_dir = Path(cfg["paths"]["log_dir"])
    ckpt_dir.mkdir(parents=True, exist_ok=True)
    log_dir.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(ckpt_dir / "densities.npz", pf=pf_arr, pb=pb_arr, family=family, order=order)
    summary = {
        "protocol": cfg["train"].get("protocol", "full"),
        "n_subjects": len(names),
        "n_forward": int(len(pf_arr)),
        "n_backward": int(len(pb_arr)),
        "basis": family,
        "order": order,
        "elapsed_sec": elapsed,
        "checkpoint": str(ckpt_dir / "densities.npz"),
    }
    with open(log_dir / "train_summary.json", "w") as f:
        json.dump(summary, f, indent=2)
        f.write("\n")
    print(f"Wrote {log_dir / 'train_summary.json'} n_fwd={len(pf_arr)} elapsed={elapsed:.1f}s")
    return summary


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/default.yaml")
    args = ap.parse_args()
    train(load_config(args.config))


if __name__ == "__main__":
    main()
