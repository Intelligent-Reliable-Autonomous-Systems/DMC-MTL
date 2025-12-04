#!/bin/bash
#SBATCH -J ch_smoothing_0.0001
#SBATCH -o output/ch_smoothing_0.0001.out
#SBATCH -e output/ch_smoothing_0.0001.err
#SBATCH -p eecs,share,gpu
#SBATCH -t 2-12:00:00
#SBATCH --gres=gpu:1

python3 -m trainers.train_model --config dmc_mtl/smoothing/ch_smoothing_0.0001 --seed 0
python3 -m trainers.train_model --config dmc_mtl/smoothing/ch_smoothing_0.0001 --seed 1
python3 -m trainers.train_model --config dmc_mtl/smoothing/ch_smoothing_0.0001 --seed 2
python3 -m trainers.train_model --config dmc_mtl/smoothing/ch_smoothing_0.0001 --seed 3
python3 -m trainers.train_model --config dmc_mtl/smoothing/ch_smoothing_0.0001 --seed 4