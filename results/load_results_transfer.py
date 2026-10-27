"""Thin wrapper around load_results for transfer-style pickle names."""

import sys

from results.load_results import main

if __name__ == "__main__":
    if "--transfer_pickles" not in sys.argv:
        sys.argv.append("--transfer_pickles")
    main()
