#!/bin/bash
#SBATCH -J ferguson_BCOV
#SBATCH -o output/ferguson_BCOV.out
#SBATCH -e output/ferguson_BCOV.err
#SBATCH -p eecs,gpu
#SBATCH -t 2-00:00:00
#SBATCH --gres=gpu:1

python3 -m trainers.train_model --config aaai/rebuttal/ferguson --seed 0 --withold "{Region: [ONNP, WA, NY, MI, NS, WI, QC, PA, TX]}"
python3 -m trainers.train_model --config aaai/rebuttal/ferguson --seed 1 --withold "{Region: [ONNP, WA, NY, MI, NS, WI, QC, PA, TX]}"
python3 -m trainers.train_model --config aaai/rebuttal/ferguson --seed 2 --withold "{Region: [ONNP, WA, NY, MI, NS, WI, QC, PA, TX]}"
python3 -m trainers.train_model --config aaai/rebuttal/ferguson --seed 3 --withold "{Region: [ONNP, WA, NY, MI, NS, WI, QC, PA, TX]}"
python3 -m trainers.train_model --config aaai/rebuttal/ferguson --seed 4 --withold "{Region: [ONNP, WA, NY, MI, NS, WI, QC, PA, TX]}"
