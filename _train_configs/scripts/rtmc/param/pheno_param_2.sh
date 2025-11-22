#!/bin/bash
#SBATCH -J param_param_2
#SBATCH -o output/param_param_2.out
#SBATCH -e output/param_param_2.err
#SBATCH -p eecs,share,gpu
#SBATCH -t 2-12:00:00
#SBATCH --gres=gpu:1

python3 -m trainers.train_model --config rtmc/phenology/param_param_2 --seed 0
python3 -m trainers.train_model --config rtmc/phenology/param_param_2 --seed 1
python3 -m trainers.train_model --config rtmc/phenology/param_param_2 --seed 2
python3 -m trainers.train_model --config rtmc/phenology/param_param_2 --seed 3
python3 -m trainers.train_model --config rtmc/phenology/param_param_2 --seed 4