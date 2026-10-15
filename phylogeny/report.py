"""Figures + REPORT.md."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import yaml

_ROOT = Path(__file__).resolve().parents[1]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from phylogeny.plot_style import FIGSIZE, MULTI_SERIES_COLORS, apply_rcparams, apply_style


def plot_ranks(eval_res: dict, out: Path) -> None:
    apply_rcparams()
    fig, ax = plt.subplots(figsize=FIGSIZE)
    names = ["Rank-1", "Rank-2", "Rank-3", "IPT edge"]
    vals = [
        eval_res["rank1"] * 100,
        eval_res["rank2"] * 100,
        eval_res["rank3"] * 100,
        eval_res["ipt_edge_accuracy"] * 100,
    ]
    x = np.arange(len(names))
    ax.bar(x, vals, color=MULTI_SERIES_COLORS[0], edgecolor="#002080")
    ax.set_xticks(x)
    ax.set_xticklabels(names)
    ax.set_ylabel("%")
    ax.set_title(f"Face IPT ({eval_res.get('basis')}, n={eval_res.get('n_sets')})")
    apply_style(ax, grid_axis="y")
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(out, format="svg")
    plt.close(fig)


def write_report(cfg, train_s, eval_res, fig_dir: Path, path: Path) -> None:
    protocol = cfg["train"].get("protocol", "full")
    lines = [
        "# REPORT — Face Phylogeny Tree Using Basis Functions",
        "",
        f"**Protocol:** `{protocol}`",
        "",
        "Lab twin: photometric near-duplicate chains on LFW; Chebyshev/Legendre",
        "α fitting; Parzen likelihood-ratio scores; IPT root + edge recovery.",
        "",
        "## Density fitting",
        "",
        f"- Subjects: {train_s.get('n_subjects')}",
        f"- Forward / backward pairs: {train_s.get('n_forward')} / {train_s.get('n_backward')}",
        f"- Basis: {train_s.get('basis')} order={train_s.get('order')}",
        f"- Elapsed (s): {train_s.get('elapsed_sec', 0):.1f}",
        "",
        "## IPT evaluation",
        "",
        f"- Sets: {eval_res.get('n_sets')} (chain length {eval_res.get('chain_length')})",
        f"- Rank-1 root: **{eval_res.get('rank1', 0)*100:.2f}%**",
        f"- Rank-2 root: **{eval_res.get('rank2', 0)*100:.2f}%**",
        f"- Rank-3 root: **{eval_res.get('rank3', 0)*100:.2f}%**",
        f"- IPT edge accuracy: **{eval_res.get('ipt_edge_accuracy', 0)*100:.2f}%**",
        "",
        f"![Ranks]({fig_dir.as_posix()}/ipt_ranks.svg)",
        "",
        "## Notes",
        "",
        "- Paper also evaluates iris/fingerprint and geometric edits; this twin focuses on LFW photometric chains.",
        "- See `DEVIATIONS.md`.",
        "",
    ]
    path.write_text("\n".join(lines))
    print(f"Wrote {path}")


def generate(cfg: dict) -> None:
    log_dir = Path(cfg["paths"]["log_dir"])
    fig_dir = Path(cfg["paths"]["figure_dir"])
    fig_dir.mkdir(parents=True, exist_ok=True)
    train_s = json.loads((log_dir / "train_summary.json").read_text())
    eval_res = json.loads((log_dir / "eval.json").read_text())
    plot_ranks(eval_res, fig_dir / "ipt_ranks.svg")
    write_report(cfg, train_s, eval_res, fig_dir, Path("REPORT.md"))


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/default.yaml")
    args = ap.parse_args()
    generate(yaml.safe_load(Path(args.config).read_text()))


if __name__ == "__main__":
    main()
