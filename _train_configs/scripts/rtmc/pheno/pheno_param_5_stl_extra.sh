#!/bin/bash
#SBATCH -J pheno_param_5_stl_extra
#SBATCH -o output/pheno_param_5_stl_extra.out
#SBATCH -e output/pheno_param_5_stl_extra.err
#SBATCH -p eecs,share,gpu
#SBATCH -t 2-12:00:00
#SBATCH --gres=gpu:1
#SBATCH --exclude=cn-gpu3

python3 -m trainers.train_model --config rtmc/phenology/pheno_param_5_stl --seed 2 --cultivar Mourvedre
python3 -m trainers.train_model --config rtmc/phenology/pheno_param_5_stl --seed 3 --cultivar Mourvedre 
python3 -m trainers.train_model --config rtmc/phenology/pheno_param_5_stl --seed 4 --cultivar Mourvedre

python3 -m trainers.train_model --config rtmc/phenology/pheno_param_5_stl --seed 0 --cultivar Muscat_Blanc
python3 -m trainers.train_model --config rtmc/phenology/pheno_param_5_stl --seed 1 --cultivar Muscat_Blanc
python3 -m trainers.train_model --config rtmc/phenology/pheno_param_5_stl --seed 2 --cultivar Muscat_Blanc
python3 -m trainers.train_model --config rtmc/phenology/pheno_param_5_stl --seed 3 --cultivar Muscat_Blanc 
python3 -m trainers.train_model --config rtmc/phenology/pheno_param_5_stl --seed 4 --cultivar Muscat_Blanc

python3 -m trainers.train_model --config rtmc/phenology/pheno_param_5_stl --seed 0 --cultivar Nebbiolo
python3 -m trainers.train_model --config rtmc/phenology/pheno_param_5_stl --seed 1 --cultivar Nebbiolo
python3 -m trainers.train_model --config rtmc/phenology/pheno_param_5_stl --seed 2 --cultivar Nebbiolo
python3 -m trainers.train_model --config rtmc/phenology/pheno_param_5_stl --seed 3 --cultivar Nebbiolo 
python3 -m trainers.train_model --config rtmc/phenology/pheno_param_5_stl --seed 4 --cultivar Nebbiolo

python3 -m trainers.train_model --config rtmc/phenology/pheno_param_5_stl --seed 0 --cultivar Petit_Verdot
python3 -m trainers.train_model --config rtmc/phenology/pheno_param_5_stl --seed 1 --cultivar Petit_Verdot
python3 -m trainers.train_model --config rtmc/phenology/pheno_param_5_stl --seed 2 --cultivar Petit_Verdot
python3 -m trainers.train_model --config rtmc/phenology/pheno_param_5_stl --seed 3 --cultivar Petit_Verdot 
python3 -m trainers.train_model --config rtmc/phenology/pheno_param_5_stl --seed 4 --cultivar Petit_Verdot

python3 -m trainers.train_model --config rtmc/phenology/pheno_param_5_stl --seed 0 --cultivar Pinot_Blanc
python3 -m trainers.train_model --config rtmc/phenology/pheno_param_5_stl --seed 1 --cultivar Pinot_Blanc
python3 -m trainers.train_model --config rtmc/phenology/pheno_param_5_stl --seed 2 --cultivar Pinot_Blanc
python3 -m trainers.train_model --config rtmc/phenology/pheno_param_5_stl --seed 3 --cultivar Pinot_Blanc 
python3 -m trainers.train_model --config rtmc/phenology/pheno_param_5_stl --seed 4 --cultivar Pinot_Blanc

python3 -m trainers.train_model --config rtmc/phenology/pheno_param_5_stl --seed 0 --cultivar Pinot_Gris
python3 -m trainers.train_model --config rtmc/phenology/pheno_param_5_stl --seed 1 --cultivar Pinot_Gris
python3 -m trainers.train_model --config rtmc/phenology/pheno_param_5_stl --seed 2 --cultivar Pinot_Gris
python3 -m trainers.train_model --config rtmc/phenology/pheno_param_5_stl --seed 3 --cultivar Pinot_Gris 
python3 -m trainers.train_model --config rtmc/phenology/pheno_param_5_stl --seed 4 --cultivar Pinot_Gris

python3 -m trainers.train_model --config rtmc/phenology/pheno_param_5_stl --seed 0 --cultivar Pinot_Noir
python3 -m trainers.train_model --config rtmc/phenology/pheno_param_5_stl --seed 1 --cultivar Pinot_Noir
python3 -m trainers.train_model --config rtmc/phenology/pheno_param_5_stl --seed 2 --cultivar Pinot_Noir
python3 -m trainers.train_model --config rtmc/phenology/pheno_param_5_stl --seed 3 --cultivar Pinot_Noir 
python3 -m trainers.train_model --config rtmc/phenology/pheno_param_5_stl --seed 4 --cultivar Pinot_Noir

python3 -m trainers.train_model --config rtmc/phenology/pheno_param_5_stl --seed 0 --cultivar Riesling
python3 -m trainers.train_model --config rtmc/phenology/pheno_param_5_stl --seed 1 --cultivar Riesling
python3 -m trainers.train_model --config rtmc/phenology/pheno_param_5_stl --seed 2 --cultivar Riesling
python3 -m trainers.train_model --config rtmc/phenology/pheno_param_5_stl --seed 3 --cultivar Riesling 
python3 -m trainers.train_model --config rtmc/phenology/pheno_param_5_stl --seed 4 --cultivar Riesling

python3 -m trainers.train_model --config rtmc/phenology/pheno_param_5_stl --seed 0 --cultivar Sangiovese
python3 -m trainers.train_model --config rtmc/phenology/pheno_param_5_stl --seed 1 --cultivar Sangiovese
python3 -m trainers.train_model --config rtmc/phenology/pheno_param_5_stl --seed 2 --cultivar Sangiovese
python3 -m trainers.train_model --config rtmc/phenology/pheno_param_5_stl --seed 3 --cultivar Sangiovese 
python3 -m trainers.train_model --config rtmc/phenology/pheno_param_5_stl --seed 4 --cultivar Sangiovese

python3 -m trainers.train_model --config rtmc/phenology/pheno_param_5_stl --seed 0 --cultivar Sauvignon_Blanc
python3 -m trainers.train_model --config rtmc/phenology/pheno_param_5_stl --seed 1 --cultivar Sauvignon_Blanc
python3 -m trainers.train_model --config rtmc/phenology/pheno_param_5_stl --seed 2 --cultivar Sauvignon_Blanc
python3 -m trainers.train_model --config rtmc/phenology/pheno_param_5_stl --seed 3 --cultivar Sauvignon_Blanc 
python3 -m trainers.train_model --config rtmc/phenology/pheno_param_5_stl --seed 4 --cultivar Sauvignon_Blanc

python3 -m trainers.train_model --config rtmc/phenology/pheno_param_5_stl --seed 0 --cultivar Semillon
python3 -m trainers.train_model --config rtmc/phenology/pheno_param_5_stl --seed 1 --cultivar Semillon
python3 -m trainers.train_model --config rtmc/phenology/pheno_param_5_stl --seed 2 --cultivar Semillon
python3 -m trainers.train_model --config rtmc/phenology/pheno_param_5_stl --seed 3 --cultivar Semillon 
python3 -m trainers.train_model --config rtmc/phenology/pheno_param_5_stl --seed 4 --cultivar Semillon

python3 -m trainers.train_model --config rtmc/phenology/pheno_param_5_stl --seed 0 --cultivar Tempranillo
python3 -m trainers.train_model --config rtmc/phenology/pheno_param_5_stl --seed 1 --cultivar Tempranillo
python3 -m trainers.train_model --config rtmc/phenology/pheno_param_5_stl --seed 2 --cultivar Tempranillo
python3 -m trainers.train_model --config rtmc/phenology/pheno_param_5_stl --seed 3 --cultivar Tempranillo 
python3 -m trainers.train_model --config rtmc/phenology/pheno_param_5_stl --seed 4 --cultivar Tempranillo

python3 -m trainers.train_model --config rtmc/phenology/pheno_param_5_stl --seed 0 --cultivar Viognier
python3 -m trainers.train_model --config rtmc/phenology/pheno_param_5_stl --seed 1 --cultivar Viognier
python3 -m trainers.train_model --config rtmc/phenology/pheno_param_5_stl --seed 2 --cultivar Viognier
python3 -m trainers.train_model --config rtmc/phenology/pheno_param_5_stl --seed 3 --cultivar Viognier 
python3 -m trainers.train_model --config rtmc/phenology/pheno_param_5_stl --seed 4 --cultivar Viognier

python3 -m trainers.train_model --config rtmc/phenology/pheno_param_5_stl --seed 0 --cultivar Zinfandel
python3 -m trainers.train_model --config rtmc/phenology/pheno_param_5_stl --seed 1 --cultivar Zinfandel
python3 -m trainers.train_model --config rtmc/phenology/pheno_param_5_stl --seed 2 --cultivar Zinfandel
python3 -m trainers.train_model --config rtmc/phenology/pheno_param_5_stl --seed 3 --cultivar Zinfandel 
python3 -m trainers.train_model --config rtmc/phenology/pheno_param_5_stl --seed 4 --cultivar Zinfandel

