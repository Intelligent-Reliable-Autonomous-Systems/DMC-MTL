"""Thin wrapper — prefer `python -m exp.job_runner --preset stl_load_mthl_gd`."""
from exp.job_runner import main

if __name__ == "__main__":
    main(preset="stl_load_mthl_gd")
