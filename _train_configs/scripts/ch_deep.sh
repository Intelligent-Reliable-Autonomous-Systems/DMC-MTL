#!/bin/bash
#SBATCH -J ch_deep
#SBATCH -o output/ch_deep.out
#SBATCH -e output/ch_deep.err
#SBATCH -p eecs,share,gpu
#SBATCH -t 2-12:00:00
#SBATCH --gres=gpu:1

python3 -m trainers.train_model --config dmc_mtl/grape_coldhardiness/ch_deep --seed 0
python3 -m trainers.train_model --config dmc_mtl/grape_coldhardiness/ch_deep --seed 1
python3 -m trainers.train_model --config dmc_mtl/grape_coldhardiness/ch_deep --seed 2
python3 -m trainers.train_model --config dmc_mtl/grape_coldhardiness/ch_deep --seed 3
python3 -m trainers.train_model --config dmc_mtl/grape_coldhardiness/ch_deep --seed 4