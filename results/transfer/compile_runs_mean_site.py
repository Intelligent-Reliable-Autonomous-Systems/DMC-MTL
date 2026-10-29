"""Thin wrapper — delegates to compile_runs_transfer --strategy mean_site."""

from results.transfer.compile_runs_transfer import main

if __name__ == "__main__":
    import sys

    if "--strategy" not in sys.argv:
        sys.argv.extend(["--strategy", "mean_site"])
    main(strategy="mean_site")
