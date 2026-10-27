#!/usr/bin/env python3
"""Split experiments.txt into SLURM sbatch .sh scripts.

Groups consecutive python commands into chunks of N (default 5) and writes one
sbatch file per chunk with a header of the form:

    #!/bin/bash
    #SBATCH -J <NAME>
    #SBATCH -o output/<NAME>.out
    ...
"""

from __future__ import annotations

import argparse
import re
from pathlib import Path

HEADER = """\
#!/bin/bash
#SBATCH -J {name}
#SBATCH -o output/{name}.out
#SBATCH -e output/{name}.err
#SBATCH -p eecs,gpu
#SBATCH -t 2-00:00:00
#SBATCH --gres=gpu:1
"""

CONFIG_RE = re.compile(r"--config\s+(\S+)")
WITHOLD_RE = re.compile(r"--withold\s+\"\{Region:\s*\[([^\]]*)\]\}\"")
REGION_RE = re.compile(r"[A-Za-z0-9_]+")


def parse_commands(path: Path) -> list[str]:
    cmds = []
    for line in path.read_text().splitlines():
        line = line.strip()
        if line and not line.startswith("#"):
            cmds.append(line)
    return cmds


def all_regions(cmds: list[str]) -> set[str]:
    regions: set[str] = set()
    for cmd in cmds:
        m = WITHOLD_RE.search(cmd)
        if m:
            regions.update(REGION_RE.findall(m.group(1)))
    return regions


def chunk_name(cmd: str, universe: set[str], used: set[str]) -> str:
    config_m = CONFIG_RE.search(cmd)
    stem = Path(config_m.group(1)).name if config_m else "job"

    withold_m = WITHOLD_RE.search(cmd)
    if withold_m and universe:
        withheld = set(REGION_RE.findall(withold_m.group(1)))
        held_in = sorted(universe - withheld)
        if held_in:
            stem = f"{stem}_{'_'.join(held_in)}"

    name = stem
    i = 2
    while name in used:
        name = f"{stem}_{i}"
        i += 1
    used.add(name)
    return name


def write_sbatch(out_dir: Path, name: str, cmds: list[str]) -> Path:
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "output").mkdir(exist_ok=True)
    path = out_dir / f"{name}.sh"
    body = HEADER.format(name=name) + "\n" + "\n".join(cmds) + "\n"
    path.write_text(body)
    path.chmod(path.stat().st_mode | 0o111)
    return path


def main() -> None:
    here = Path(__file__).resolve().parent
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument(
        "-i",
        "--input",
        type=Path,
        default=here / "experiments.txt",
        help="experiments file (one python command per line)",
    )
    p.add_argument(
        "-o",
        "--out-dir",
        type=Path,
        default=here / "sbatch",
        help="directory for generated .sh files",
    )
    p.add_argument(
        "-n",
        "--chunk-size",
        type=int,
        default=5,
        help="python commands per sbatch script (default: 5)",
    )
    args = p.parse_args()

    cmds = parse_commands(args.input)
    if not cmds:
        raise SystemExit(f"No commands found in {args.input}")
    if args.chunk_size < 1:
        raise SystemExit("--chunk-size must be >= 1")

    universe = all_regions(cmds)
    used: set[str] = set()
    written = []
    for i in range(0, len(cmds), args.chunk_size):
        chunk = cmds[i : i + args.chunk_size]
        name = chunk_name(chunk[0], universe, used)
        written.append(write_sbatch(args.out_dir, name, chunk))

    print(f"Wrote {len(written)} sbatch scripts to {args.out_dir}:")
    for path in written:
        print(f"  {path.name}")


if __name__ == "__main__":
    main()
