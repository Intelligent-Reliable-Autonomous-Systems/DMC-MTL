#!/bin/bash
#SBATCH -J ch/ch_deep_data_5
#SBATCH -o output/ch/ch_deep_data_5.out
#SBATCH -e output/ch/ch_deep_data_5.err
#SBATCH -p eecs,share,gpu
#SBATCH -t 2-12:00:00
#SBATCH --gres=gpu:1

python3 -m trainers.train_model --config dmc_mtl/lim_data/ch/ch_deep_data_5 --seed 0
python3 -m trainers.train_model --config dmc_mtl/lim_data/ch/ch_deep_data_5 --seed 1
python3 -m trainers.train_model --config dmc_mtl/lim_data/ch/ch_deep_data_5 --seed 2
python3 -m trainers.train_model --config dmc_mtl/lim_data/ch/ch_deep_data_5 --seed 3
python3 -m trainers.train_model --config dmc_mtl/lim_data/ch/ch_deep_data_5 --seed 4
