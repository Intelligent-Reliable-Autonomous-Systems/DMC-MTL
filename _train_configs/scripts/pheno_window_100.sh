#!/bin/bash
#SBATCH -J pheno_param_mtl_window_100
#SBATCH -o output/pheno_param_mtl_window_100.out
#SBATCH -e output/pheno_param_mtl_window_100.err
#SBATCH -p eecs,share,gpu
#SBATCH -t 2-12:00:00
#SBATCH --gres=gpu:1

python3 -m trainers.train_model --config dmc_mtl/window/pheno_param_mtl_window_100 --seed 0
python3 -m trainers.train_model --config dmc_mtl/window/pheno_param_mtl_window_100 --seed 1
python3 -m trainers.train_model --config dmc_mtl/window/pheno_param_mtl_window_100 --seed 2
python3 -m trainers.train_model --config dmc_mtl/window/pheno_param_mtl_window_100 --seed 3
python3 -m trainers.train_model --config dmc_mtl/window/pheno_param_mtl_window_100 --seed 4