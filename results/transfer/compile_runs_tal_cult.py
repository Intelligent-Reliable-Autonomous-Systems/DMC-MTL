"""Thin wrapper — delegates to compile_runs_transfer --strategy tal_cult."""

from results.transfer.compile_runs_transfer import main

if __name__ == "__main__":
    import sys
    if "--strategy" not in sys.argv:
        sys.argv.extend(["--strategy", "tal_cult"])
    main(strategy="tal_cult")
