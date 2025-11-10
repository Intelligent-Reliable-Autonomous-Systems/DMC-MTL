#!/bin/bash
#SBATCH -J pheno_deep_class
#SBATCH -o output/pheno_deep_class.out
#SBATCH -e output/pheno_deep_class.err
#SBATCH -p eecs,share,gpu
#SBATCH -t 2-12:00:00
#SBATCH --gres=gpu:1

python3 -m trainers.train_model --config dmc_mtl/phenology/pheno_deep_class --seed 0
python3 -m trainers.train_model --config dmc_mtl/phenology/pheno_deep_class --seed 1
python3 -m trainers.train_model --config dmc_mtl/phenology/pheno_deep_class --seed 2
python3 -m trainers.train_model --config dmc_mtl/phenology/pheno_deep_class --seed 3
python3 -m trainers.train_model --config dmc_mtl/phenology/pheno_deep_class --seed 4