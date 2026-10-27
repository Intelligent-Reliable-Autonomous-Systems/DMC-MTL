"""Thin wrapper — prefer `python -m exp.job_runner --preset mtl_bcov_deep`."""
from exp.job_runner import main

if __name__ == "__main__":
    main(preset="mtl_bcov_deep")
