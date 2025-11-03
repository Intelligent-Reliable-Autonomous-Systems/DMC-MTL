#!/bin/bash
#SBATCH -J pheno_param_mtl_10
#SBATCH -o output/pheno_param_mtl_10.out
#SBATCH -e output/pheno_param_mtl_10.err
#SBATCH -p eecs,share,gpu
#SBATCH -t 2-12:00:00
#SBATCH --gres=gpu:1  

python3 -m trainers.train_model --config paper_experiments/pheno_weather_abl_param/pheno_param_mtl_10 --seed 0
python3 -m trainers.train_model --config paper_experiments/pheno_weather_abl_param/pheno_param_mtl_10 --seed 1
python3 -m trainers.train_model --config paper_experiments/pheno_weather_abl_param/pheno_param_mtl_10 --seed 2
python3 -m trainers.train_model --config paper_experiments/pheno_weather_abl_param/pheno_param_mtl_10 --seed 3
python3 -m trainers.train_model --config paper_experiments/pheno_weather_abl_param/pheno_param_mtl_10 --seed 4

