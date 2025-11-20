#!/bin/bash
#SBATCH -J pheno_deep_data_1
#SBATCH -o output/pheno_deep_data_1.out
#SBATCH -e output/pheno_deep_data_1.err
#SBATCH -p eecs,share,gpu
#SBATCH -t 2-12:00:00
#SBATCH --gres=gpu:1

python3 -m trainers.train_model --config dmc_mtl/lim_data/deep/pheno_deep_data_1 --seed 0
python3 -m trainers.train_model --config dmc_mtl/lim_data/deep/pheno_deep_data_1 --seed 1
python3 -m trainers.train_model --config dmc_mtl/lim_data/deep/pheno_deep_data_1 --seed 2
python3 -m trainers.train_model --config dmc_mtl/lim_data/deep/pheno_deep_data_1 --seed 3
python3 -m trainers.train_model --config dmc_mtl/lim_data/deep/pheno_deep_data_1 --seed 4
