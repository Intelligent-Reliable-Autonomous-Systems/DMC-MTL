"""Thin wrapper — prefer `python -m plotters.embeddings --mode hybrid`."""

from plotters.embeddings import main

if __name__ == "__main__":
    main(defaults={"mode": "hybrid", "e_mode": "direct", "n_components": 2})
