#!/bin/bash
#SBATCH -J ch/ch_param_data_3
#SBATCH -o output/ch/ch_param_data_3.out
#SBATCH -e output/ch/ch_param_data_3.err
#SBATCH -p eecs,share,gpu
#SBATCH -t 2-12:00:00
#SBATCH --gres=gpu:1

python3 -m trainers.train_model --config dmc_mtl/lim_data/ch/ch_param_data_3 --seed 0
python3 -m trainers.train_model --config dmc_mtl/lim_data/ch/ch_param_data_3 --seed 1
python3 -m trainers.train_model --config dmc_mtl/lim_data/ch/ch_param_data_3 --seed 2
python3 -m trainers.train_model --config dmc_mtl/lim_data/ch/ch_param_data_3 --seed 3
python3 -m trainers.train_model --config dmc_mtl/lim_data/ch/ch_param_data_3 --seed 4
