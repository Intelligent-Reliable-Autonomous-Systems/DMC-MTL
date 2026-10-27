"""Thin wrapper — delegates to compile_runs_transfer --strategy best_source."""

from results.transfer.compile_runs_transfer import main

if __name__ == "__main__":
    import sys
    if "--strategy" not in sys.argv:
        sys.argv.extend(["--strategy", "best_source"])
    main(strategy="best_source")
