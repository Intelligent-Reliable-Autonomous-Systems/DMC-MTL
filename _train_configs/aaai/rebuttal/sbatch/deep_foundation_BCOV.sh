#!/bin/bash
#SBATCH -J deep_foundation_BCOV
#SBATCH -o output/deep_foundation_BCOV.out
#SBATCH -e output/deep_foundation_BCOV.err
#SBATCH -p eecs,gpu
#SBATCH -t 2-00:00:00
#SBATCH --gres=gpu:1

python3 -m trainers.train_model --config aaai/rebuttal/deep_foundation --seed 0 --withold "{Region: [ONNP, WA, NY, MI, NS, WI, QC, PA, TX]}"
python3 -m trainers.train_model --config aaai/rebuttal/deep_foundation --seed 1 --withold "{Region: [ONNP, WA, NY, MI, NS, WI, QC, PA, TX]}"
python3 -m trainers.train_model --config aaai/rebuttal/deep_foundation --seed 2 --withold "{Region: [ONNP, WA, NY, MI, NS, WI, QC, PA, TX]}"
python3 -m trainers.train_model --config aaai/rebuttal/deep_foundation --seed 3 --withold "{Region: [ONNP, WA, NY, MI, NS, WI, QC, PA, TX]}"
python3 -m trainers.train_model --config aaai/rebuttal/deep_foundation --seed 4 --withold "{Region: [ONNP, WA, NY, MI, NS, WI, QC, PA, TX]}"
