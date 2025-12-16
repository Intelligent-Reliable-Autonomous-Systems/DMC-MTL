#!/bin/bash
#SBATCH -J pinn/pheno_pinn_2
#SBATCH -o output/pinn/pheno_pinn_2.out
#SBATCH -e output/pinn/pheno_pinn_2.err
#SBATCH -p eecs,share,gpu
#SBATCH -t 2-12:00:00
#SBATCH --gres=gpu:1

python3 -m trainers.train_model --config dmc_mtl/lim_data/pinn/pheno_pinn_2 --seed 0
python3 -m trainers.train_model --config dmc_mtl/lim_data/pinn/pheno_pinn_2 --seed 1
python3 -m trainers.train_model --config dmc_mtl/lim_data/pinn/pheno_pinn_2 --seed 2
python3 -m trainers.train_model --config dmc_mtl/lim_data/pinn/pheno_pinn_2 --seed 3
python3 -m trainers.train_model --config dmc_mtl/lim_data/pinn/pheno_pinn_2 --seed 4
