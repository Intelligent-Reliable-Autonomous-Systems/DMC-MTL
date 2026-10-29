"""Thin wrapper — delegates to compile_runs_transfer --strategy cossim_true_site."""

from results.transfer.compile_runs_transfer import main

if __name__ == "__main__":
    import sys

    if "--strategy" not in sys.argv:
        sys.argv.extend(["--strategy", "cossim_true_site"])
    main(strategy="cossim_true_site")
