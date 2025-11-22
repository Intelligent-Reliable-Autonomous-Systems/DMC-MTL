#!/bin/bash
#SBATCH -J ch_param_mtl_ff
#SBATCH -o output/ch_param_mtl_ff.out
#SBATCH -e output/ch_param_mtl_ff.err
#SBATCH -p eecs,share,gpu
#SBATCH -t 1-12:00:00
#SBATCH --gres=gpu:1

python3 -m trainers.train_model --config dmc_mtl/grape_coldhardiness/ch_param_mtl_ff --seed 0
python3 -m trainers.train_model --config dmc_mtl/grape_coldhardiness/ch_param_mtl_ff --seed 1
python3 -m trainers.train_model --config dmc_mtl/grape_coldhardiness/ch_param_mtl_ff --seed 2
python3 -m trainers.train_model --config dmc_mtl/grape_coldhardiness/ch_param_mtl_ff --seed 3
python3 -m trainers.train_model --config dmc_mtl/grape_coldhardiness/ch_param_mtl_ff --seed 4