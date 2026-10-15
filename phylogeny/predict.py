"""Score root candidates for a folder of near-duplicate face images."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import yaml
from PIL import Image

_ROOT = Path(__file__).resolve().parents[1]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from phylogeny.basis.transforms import fit_separable_poly, to_gray_array
from phylogeny.ipt.tree import build_score_matrix, root_candidates


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/default.yaml")
    ap.add_argument("--images", required=True, help="Directory of near-duplicate images")
    args = ap.parse_args()
    cfg = yaml.safe_load(Path(args.config).read_text())
    paths = sorted(Path(args.images).glob("*.jpg")) + sorted(Path(args.images).glob("*.png"))
    if len(paths) < 2:
        raise SystemExit("need >=2 images")
    blob = np.load(Path(cfg["paths"]["checkpoint_dir"]) / "densities.npz", allow_pickle=True)
    family = str(blob["family"])
    order = int(blob["order"])
    gray_size = int(cfg["model"].get("gray_size", 64))
    grays = [to_gray_array(Image.open(p).convert("RGB"), gray_size) for p in paths]
    n, d = len(grays), order * 2
    alphas = np.zeros((n, n, d))
    for i in range(n):
        for j in range(n):
            if i != j:
                alphas[i, j] = fit_separable_poly(grays[j], grays[i], family=family, order=order)
    S = build_score_matrix(alphas, alphas, blob["pf"], blob["pb"], bandwidth=float(cfg["eval"].get("bandwidth", 0.3)))
    cands = root_candidates(S, top_k=3)
    out_dir = Path(cfg["paths"]["pred_dir"])
    out_dir.mkdir(parents=True, exist_ok=True)
    payload = {"candidates": [{"index": c, "path": str(paths[c])} for c in cands]}
    (out_dir / "roots.json").write_text(json.dumps(payload, indent=2) + "\n")
    print(f"Wrote {out_dir / 'roots.json'} cands={cands}")


if __name__ == "__main__":
    main()
