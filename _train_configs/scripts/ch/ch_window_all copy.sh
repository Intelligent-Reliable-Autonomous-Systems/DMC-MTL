#!/bin/bash
#SBATCH -J ch_param_window_all
#SBATCH -o output/ch_param_window_all.out
#SBATCH -e output/ch_param_window_all.err
#SBATCH -p eecs,share,gpu
#SBATCH -t 2-12:00:00
#SBATCH --gres=gpu:1

python3 -m trainers.train_model --config dmc_mtl/window/ch/ch_param_window_all --seed 0
python3 -m trainers.train_model --config dmc_mtl/window/ch/ch_param_window_all --seed 1
python3 -m trainers.train_model --config dmc_mtl/window/ch/ch_param_window_all --seed 2
python3 -m trainers.train_model --config dmc_mtl/window/ch/ch_param_window_all --seed 3
python3 -m trainers.train_model --config dmc_mtl/window/ch/ch_param_window_all --seed 4