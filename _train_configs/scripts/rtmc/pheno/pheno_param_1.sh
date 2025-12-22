#!/bin/bash
#SBATCH -J pheno_param_1
#SBATCH -o output/pheno_param_1.out
#SBATCH -e output/pheno_param_1.err
#SBATCH -p eecs,share,gpu
#SBATCH -t 2-12:00:00
#SBATCH --gres=gpu:1

python3 -m trainers.train_model --config rtmc/phenology/pheno_param_1 --seed 0
python3 -m trainers.train_model --config rtmc/phenology/pheno_param_1 --seed 1
python3 -m trainers.train_model --config rtmc/phenology/pheno_param_1 --seed 2
python3 -m trainers.train_model --config rtmc/phenology/pheno_param_1 --seed 3
python3 -m trainers.train_model --config rtmc/phenology/pheno_param_1 --seed 4