"""LFW multi-image subjects for phylogeny experiments."""

from __future__ import annotations

from pathlib import Path
from typing import Dict, List, Optional

from PIL import Image


def resolve_lfw(cfg: dict) -> Path:
    root = Path(cfg["data"].get("lfw_root", "../../datasets/lfw"))
    if not root.is_absolute():
        root = (Path(__file__).resolve().parents[2] / root).resolve()
    if not (root / "lfw").is_dir():
        raise FileNotFoundError(root)
    return root


def list_subjects(lfw_root: Path, min_images: int = 5) -> Dict[str, List[Path]]:
    img_root = lfw_root / "lfw"
    out: Dict[str, List[Path]] = {}
    for person in sorted(img_root.iterdir()):
        if not person.is_dir():
            continue
        imgs = sorted(person.glob("*.jpg"))
        if len(imgs) >= min_images:
            out[person.name] = imgs
    return out


def load_image(path: Path) -> Image.Image:
    return Image.open(path).convert("RGB")
