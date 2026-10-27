#!/bin/bash
#SBATCH -J deep_foundation_NY
#SBATCH -o output/deep_foundation_NY.out
#SBATCH -e output/deep_foundation_NY.err
#SBATCH -p eecs,gpu
#SBATCH -t 2-00:00:00
#SBATCH --gres=gpu:1

python3 -m trainers.train_model --config aaai/rebuttal/deep_foundation --seed 0 --withold "{Region: [BCOV, ONNP, WA, MI, NS, WI, QC, PA, TX]}"
python3 -m trainers.train_model --config aaai/rebuttal/deep_foundation --seed 1 --withold "{Region: [BCOV, ONNP, WA, MI, NS, WI, QC, PA, TX]}"
python3 -m trainers.train_model --config aaai/rebuttal/deep_foundation --seed 2 --withold "{Region: [BCOV, ONNP, WA, MI, NS, WI, QC, PA, TX]}"
python3 -m trainers.train_model --config aaai/rebuttal/deep_foundation --seed 3 --withold "{Region: [BCOV, ONNP, WA, MI, NS, WI, QC, PA, TX]}"
python3 -m trainers.train_model --config aaai/rebuttal/deep_foundation --seed 4 --withold "{Region: [BCOV, ONNP, WA, MI, NS, WI, QC, PA, TX]}"
