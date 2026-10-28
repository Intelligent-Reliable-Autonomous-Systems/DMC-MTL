#!/usr/bin/env python3
"""Enumerate Ferguson STL run dirs into results.compile_runs commands.

Walks ``_runs/AAAI/Ferguson``, skips the top-level ``All`` folder, and for each
Region/Station/Site/Cultivar leaf writes one ``results.compile_runs`` line.
"""

from __future__ import annotations

import argparse
from pathlib import Path


def collect_leaves(runs_root: Path) -> list[Path]:
    leaves: list[Path] = []
    for region_dir in sorted(p for p in runs_root.iterdir() if p.is_dir()):
        if region_dir.name == "All":
            continue
        for path in sorted(region_dir.rglob("*")):
            if not path.is_dir():
                continue
            rel = path.relative_to(runs_root)
            # Region / Station / Site / Cultivar
            if len(rel.parts) == 4:
                leaves.append(path)
    return leaves


def format_commands(leaves: list[Path], repo: Path) -> str:
    lines = []
    for leaf in leaves:
        rel = leaf.relative_to(repo)
        lines.append(f"python3 -m results.compile_runs --start_dir {rel.as_posix()}")
    return "\n".join(lines) + ("\n" if lines else "")


def main() -> None:
    here = Path(__file__).resolve().parent
    repo = here.parents[2]
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument(
        "--runs-root",
        type=Path,
        default=repo / "_runs" / "AAAI" / "Ferguson",
        help="Ferguson runs directory (Region/Station/Site/Cultivar)",
    )
    p.add_argument(
        "-o",
        "--output",
        type=Path,
        default=here / "ferguson_compile.txt",
        help="Output compile_runs command text file",
    )
    args = p.parse_args()

    if not args.runs_root.is_dir():
        raise SystemExit(f"Runs root not found: {args.runs_root}")

    leaves = collect_leaves(args.runs_root)
    text = format_commands(leaves, repo)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(text)
    print(f"Wrote {len(leaves)} compile_runs commands")
    print(f"  -> {args.output}")


if __name__ == "__main__":
    main()
