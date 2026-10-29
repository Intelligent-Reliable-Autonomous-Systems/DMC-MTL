"""
train_model_checkpoint.py

Train a model from an existing run checkpoint in place.
"""

import argparse
from pathlib import Path

import utils


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True, type=str, help="Existing run directory containing config.yaml")
    parser.add_argument(
        "--checkpoint-name",
        default="rnn_model_best.pt",
        type=str,
        help="Checkpoint filename within the run directory",
    )
    parser.add_argument("--epochs", default=None, type=int, help="Optional runtime epoch override")
    parser.add_argument("--seed", default=0, type=int, help="Seed of Experiment")
    parser.add_argument("--region", default=None, type=str, help="Region")
    parser.add_argument("--station", default=None, type=str, help="Station")
    parser.add_argument("--site", default=None, type=str, help="Site")
    parser.add_argument("--cultivar", default=None, type=str, help="Cultivar Type")
    parser.add_argument("--withold", default=None, type=str, help="Cultivar Type")
    args = parser.parse_args()

    log_dir = Path(args.config).expanduser().resolve()
    if not (log_dir / "config.yaml").is_file():
        raise FileNotFoundError(f"Run config not found: {log_dir / 'config.yaml'}")
    checkpoint = log_dir / args.checkpoint_name
    if not checkpoint.is_file():
        raise FileNotFoundError(f"Checkpoint not found: {checkpoint}")

    args.config = str(log_dir.relative_to(Path.cwd())) if log_dir.is_relative_to(Path.cwd()) else str(log_dir)
    config, data, _ = utils.load_config_data_fpath(args)
    config.DConfig.transfer_start = 0
    config.DConfig.transfer_type = "adapt_text"
    if args.epochs is not None:
        config.DConfig.epochs = args.epochs

    calibrator = utils.load_model_from_config(config, data)
    calibrator.load_model(str(log_dir), name=args.checkpoint_name)
    calibrator._reuse_log_path = str(log_dir)
    calibrator.optimize()


if __name__ == "__main__":
    main()
