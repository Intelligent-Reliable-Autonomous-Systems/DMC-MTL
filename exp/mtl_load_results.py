"""Thin wrapper — prefer `python -m exp.job_runner --preset mtl_load_bcov`."""
from exp.job_runner import main

if __name__ == "__main__":
    main(preset="mtl_load_bcov")
