#!/bin/bash
#SBATCH -J ch_param_mult
#SBATCH -o output/ch_param_mult.out
#SBATCH -e output/ch_param_mult.err
#SBATCH -p eecs,share,gpu
#SBATCH -t 2-12:00:00
#SBATCH --gres=gpu:1

python3 -m trainers.train_model --config dmc_mtl/grape_coldhardiness/ch_param_mult --seed 0
python3 -m trainers.train_model --config dmc_mtl/grape_coldhardiness/ch_param_mult --seed 1
python3 -m trainers.train_model --config dmc_mtl/grape_coldhardiness/ch_param_mult --seed 2
python3 -m trainers.train_model --config dmc_mtl/grape_coldhardiness/ch_param_mult --seed 3
python3 -m trainers.train_model --config dmc_mtl/grape_coldhardiness/ch_param_mult --seed 4