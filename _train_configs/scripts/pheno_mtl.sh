#!/bin/bash
#SBATCH -J pheno_mtl
#SBATCH -o output/pheno_mtl.out
#SBATCH -e output/pheno_mtl.err
#SBATCH -p eecs,share,gpu
#SBATCH -t 2-12:00:00
#SBATCH --gres=gpu:1

python3 -m trainers.train_model --config dmc_mtl/phenology/pheno_param_mtl --seed 0
python3 -m trainers.train_model --config dmc_mtl/phenology/pheno_param_mtl --seed 1
python3 -m trainers.train_model --config dmc_mtl/phenology/pheno_param_mtl --seed 2
python3 -m trainers.train_model --config dmc_mtl/phenology/pheno_param_mtl --seed 3
python3 -m trainers.train_model --config dmc_mtl/phenology/pheno_param_mtl --seed 4