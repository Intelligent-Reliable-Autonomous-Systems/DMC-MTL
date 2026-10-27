"""Thin wrapper — prefer `python -m plotters.embeddings --method pca`."""
from plotters.embeddings import main

if __name__ == "__main__":
    main(defaults={"method": "pca", "mode": "default", "e_mode": "forward", "n_components": 2})
