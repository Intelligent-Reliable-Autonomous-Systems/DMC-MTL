#!/bin/bash
#SBATCH -J pheno_smoothing_0.001
#SBATCH -o output/pheno_smoothing_0.001.out
#SBATCH -e output/pheno_smoothing_0.001.err
#SBATCH -p eecs,share,gpu
#SBATCH -t 2-12:00:00
#SBATCH --gres=gpu:1

python3 -m trainers.train_model --config dmc_mtl/smoothing/pheno_smoothing_0.001 --seed 0
python3 -m trainers.train_model --config dmc_mtl/smoothing/pheno_smoothing_0.001 --seed 1
python3 -m trainers.train_model --config dmc_mtl/smoothing/pheno_smoothing_0.001 --seed 2
python3 -m trainers.train_model --config dmc_mtl/smoothing/pheno_smoothing_0.001 --seed 3
python3 -m trainers.train_model --config dmc_mtl/smoothing/pheno_smoothing_0.001 --seed 4