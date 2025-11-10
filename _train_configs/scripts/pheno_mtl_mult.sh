#!/bin/bash
#SBATCH -J pheno_param_mtl_mult
#SBATCH -o output/pheno_param_mtl_mult.out
#SBATCH -e output/pheno_param_mtl_mult.err
#SBATCH -p eecs,share,gpu
#SBATCH -t 2-12:00:00
#SBATCH --gres=gpu:1

python3 -m trainers.train_model --config dmc_mtl/phenology/pheno_param_mtl_mult --seed 0
python3 -m trainers.train_model --config dmc_mtl/phenology/pheno_param_mtl_mult --seed 1
python3 -m trainers.train_model --config dmc_mtl/phenology/pheno_param_mtl_mult --seed 2
python3 -m trainers.train_model --config dmc_mtl/phenology/pheno_param_mtl_mult --seed 3
python3 -m trainers.train_model --config dmc_mtl/phenology/pheno_param_mtl_mult --seed 4