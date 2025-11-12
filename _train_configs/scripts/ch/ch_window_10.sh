#!/bin/bash
#SBATCH -J ch_param_mtl_window_10
#SBATCH -o output/ch_param_mtl_window_10.out
#SBATCH -e output/ch_param_mtl_window_10.err
#SBATCH -p eecs,share,gpu
#SBATCH -t 2-12:00:00
#SBATCH --gres=gpu:1

python3 -m trainers.train_model --config dmc_mtl/window/ch/ch_param_mtl_window_10 --seed 0
python3 -m trainers.train_model --config dmc_mtl/window/ch/ch_param_mtl_window_10 --seed 1
python3 -m trainers.train_model --config dmc_mtl/window/ch/ch_param_mtl_window_10 --seed 2
python3 -m trainers.train_model --config dmc_mtl/window/ch/ch_param_mtl_window_10 --seed 3
python3 -m trainers.train_model --config dmc_mtl/window/ch/ch_param_mtl_window_10 --seed 4