"""Thin wrapper — prefer `python -m plotters.embeddings --mode counterfactual`."""
from plotters.embeddings import main

if __name__ == "__main__":
    main(defaults={"mode": "counterfactual", "break_early": True})
