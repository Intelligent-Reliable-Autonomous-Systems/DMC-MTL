#!/bin/bash
#SBATCH -J pheno_pinn_mtl_5
#SBATCH -o output/pheno_pinn_mtl_5.out
#SBATCH -e output/pheno_pinn_mtl_5.err
#SBATCH -p eecs,share,gpu
#SBATCH -t 2-00:00:00
#SBATCH --gres=gpu:1  

python3 -m trainers.train_model --config paper_experiments/pheno_weather_abl_pinn/pheno_pinn_mtl_5 --seed 0
python3 -m trainers.train_model --config paper_experiments/pheno_weather_abl_pinn/pheno_pinn_mtl_5 --seed 1
python3 -m trainers.train_model --config paper_experiments/pheno_weather_abl_pinn/pheno_pinn_mtl_5 --seed 2
python3 -m trainers.train_model --config paper_experiments/pheno_weather_abl_pinn/pheno_pinn_mtl_5 --seed 3
python3 -m trainers.train_model --config paper_experiments/pheno_weather_abl_pinn/pheno_pinn_mtl_5 --seed 4

