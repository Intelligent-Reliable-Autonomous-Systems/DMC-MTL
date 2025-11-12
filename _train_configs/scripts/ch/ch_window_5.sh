#!/bin/bash
#SBATCH -J ch_param_mtl_window_5
#SBATCH -o output/ch_param_mtl_window_5.out
#SBATCH -e output/ch_param_mtl_window_5.err
#SBATCH -p eecs,share,gpu
#SBATCH -t 2-12:00:00
#SBATCH --gres=gpu:1

python3 -m trainers.train_model --config dmc_mtl/window/ch/ch_param_mtl_window_5 --seed 0
python3 -m trainers.train_model --config dmc_mtl/window/ch/ch_param_mtl_window_5 --seed 1
python3 -m trainers.train_model --config dmc_mtl/window/ch/ch_param_mtl_window_5 --seed 2
python3 -m trainers.train_model --config dmc_mtl/window/ch/ch_param_mtl_window_5 --seed 3
python3 -m trainers.train_model --config dmc_mtl/window/ch/ch_param_mtl_window_5 --seed 4