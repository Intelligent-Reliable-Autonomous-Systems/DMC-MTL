#!/usr/bin/env python3
"""Enumerate non-nyus grape_coldhardiness pickles into Ferguson STL train commands.

Walks ``_data/processed_data/grape_coldhardiness``, skips any ``.pkl`` whose
name contains ``nyus``, and for each unique (Region, Station, Site, Cultivar)
writes five ``trainers.train_model`` lines (seeds 0–4).
"""

from __future__ import annotations

import argparse
from pathlib import Path

SEEDS = (0, 1, 2, 3, 4)
CONFIG = "aaai/ferguson.yaml"
MARKER = "_grape_coldhardiness_"


def collect_combos(data_root: Path) -> list[tuple[str, str, str, str]]:
    combos: set[tuple[str, str, str, str]] = set()
    for path in data_root.rglob("*.pkl"):
        if "nyus" in path.name.lower():
            continue
        rel = path.relative_to(data_root)
        if len(rel.parts) != 4:
            raise ValueError(f"Expected Region/Station/Site/file.pkl, got: {rel}")
        region, station, site, _ = rel.parts
        stem = path.stem
        if MARKER not in stem:
            raise ValueError(f"Missing '{MARKER}' in pickle stem: {path}")
        cultivar = stem.split(MARKER, 1)[1]
        combos.add((region, station, site, cultivar))
    return sorted(combos)


def format_commands(combos: list[tuple[str, str, str, str]]) -> str:
    blocks = []
    for region, station, site, cultivar in combos:
        lines = [
            (
                f"python3 -m trainers.train_model --config {CONFIG} --seed {seed} "
                f"--region {region} --station {station} --site {site} --cultivar {cultivar}"
            )
            for seed in SEEDS
        ]
        blocks.append("\n".join(lines))
    return "\n\n".join(blocks) + "\n"


def main() -> None:
    here = Path(__file__).resolve().parent
    repo = here.parents[2]
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument(
        "--data-root",
        type=Path,
        default=repo / "_data" / "processed_data" / "grape_coldhardiness",
        help="Directory of Region/Station/Site/*.pkl files",
    )
    p.add_argument(
        "-o",
        "--output",
        type=Path,
        default=here / "ferguson_stl.txt",
        help="Output experiments text file",
    )
    args = p.parse_args()

    if not args.data_root.is_dir():
        raise SystemExit(f"Data root not found: {args.data_root}")

    combos = collect_combos(args.data_root)
    text = format_commands(combos)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(text)
    print(f"Wrote {len(combos)} combinations × {len(SEEDS)} seeds = {len(combos) * len(SEEDS)} commands")
    print(f"  -> {args.output}")


if __name__ == "__main__":
    main()
