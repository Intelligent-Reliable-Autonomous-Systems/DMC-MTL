#!/bin/bash
#SBATCH -J ch_pinn
#SBATCH -o output/ch_pinn.out
#SBATCH -e output/ch_pinn.err
#SBATCH -p eecs,share,gpu
#SBATCH -t 2-12:00:00
#SBATCH --gres=gpu:1

python3 -m trainers.train_model --config dmc_mtl/grape_coldhardiness/ch_pinn --seed 0
python3 -m trainers.train_model --config dmc_mtl/grape_coldhardiness/ch_pinn --seed 1
python3 -m trainers.train_model --config dmc_mtl/grape_coldhardiness/ch_pinn --seed 2
python3 -m trainers.train_model --config dmc_mtl/grape_coldhardiness/ch_pinn --seed 3
python3 -m trainers.train_model --config dmc_mtl/grape_coldhardiness/ch_pinn --seed 4