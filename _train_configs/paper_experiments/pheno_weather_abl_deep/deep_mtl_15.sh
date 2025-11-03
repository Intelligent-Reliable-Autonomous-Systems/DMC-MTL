#!/bin/bash
#SBATCH -J pheno_deep_mtl_15
#SBATCH -o output/pheno_deep_mtl_15.out
#SBATCH -e output/pheno_deep_mtl_15.err
#SBATCH -p eecs,share,gpu
#SBATCH -t 2-00:00:00
#SBATCH --gres=gpu:1  

python3 -m trainers.train_model --config paper_experiments/pheno_weather_abl_deep/pheno_deep_mtl_15 --seed 0
python3 -m trainers.train_model --config paper_experiments/pheno_weather_abl_deep/pheno_deep_mtl_15 --seed 1
python3 -m trainers.train_model --config paper_experiments/pheno_weather_abl_deep/pheno_deep_mtl_15 --seed 2
python3 -m trainers.train_model --config paper_experiments/pheno_weather_abl_deep/pheno_deep_mtl_15 --seed 3
python3 -m trainers.train_model --config paper_experiments/pheno_weather_abl_deep/pheno_deep_mtl_15 --seed 4

