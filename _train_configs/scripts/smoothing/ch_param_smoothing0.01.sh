#!/bin/bash
#SBATCH -J ch_smoothing0.01
#SBATCH -o output/ch_smoothing0.01.out
#SBATCH -e output/ch_smoothing0.01.err
#SBATCH -p eecs,share,gpu
#SBATCH -t 2-12:00:00
#SBATCH --gres=gpu:1

python3 -m trainers.train_model --config dmc_mtl/smoothing/ch_smoothing0.01 --seed 0
python3 -m trainers.train_model --config dmc_mtl/smoothing/ch_smoothing0.01 --seed 1
python3 -m trainers.train_model --config dmc_mtl/smoothing/ch_smoothing0.01 --seed 2
python3 -m trainers.train_model --config dmc_mtl/smoothing/ch_smoothing0.01 --seed 3
python3 -m trainers.train_model --config dmc_mtl/smoothing/ch_smoothing0.01 --seed 4