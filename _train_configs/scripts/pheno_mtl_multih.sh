#!/bin/bash
#SBATCH -J pheno_param_mtl_multihead
#SBATCH -o output/pheno_param_mtl_multihead.out
#SBATCH -e output/pheno_param_mtl_multihead.err
#SBATCH -p eecs,share,gpu
#SBATCH -t 2-12:00:00
#SBATCH --gres=gpu:1

python3 -m trainers.train_model --config dmc_mtl/phenology/pheno_param_mtl_multihead --seed 0
python3 -m trainers.train_model --config dmc_mtl/phenology/pheno_param_mtl_multihead --seed 1
python3 -m trainers.train_model --config dmc_mtl/phenology/pheno_param_mtl_multihead --seed 2
python3 -m trainers.train_model --config dmc_mtl/phenology/pheno_param_mtl_multihead --seed 3
python3 -m trainers.train_model --config dmc_mtl/phenology/pheno_param_mtl_multihead --seed 4