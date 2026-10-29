"""Thin wrapper — prefer `python -m plotters.embeddings --method tsne`."""

from plotters.embeddings import main

if __name__ == "__main__":
    main(defaults={"method": "tsne", "mode": "default", "e_mode": "direct", "n_components": 2})
