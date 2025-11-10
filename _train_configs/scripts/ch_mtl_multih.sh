#!/bin/bash
#SBATCH -J ch_param_multihead
#SBATCH -o output/ch_param_multihead.out
#SBATCH -e output/ch_param_multihead.err
#SBATCH -p eecs,share,gpu
#SBATCH -t 2-12:00:00
#SBATCH --gres=gpu:1

python3 -m trainers.train_model --config dmc_mtl/grape_coldhardiness/ch_param_multihead --seed 0
python3 -m trainers.train_model --config dmc_mtl/grape_coldhardiness/ch_param_multihead --seed 1
python3 -m trainers.train_model --config dmc_mtl/grape_coldhardiness/ch_param_multihead --seed 2
python3 -m trainers.train_model --config dmc_mtl/grape_coldhardiness/ch_param_multihead --seed 3
python3 -m trainers.train_model --config dmc_mtl/grape_coldhardiness/ch_param_multihead --seed 4