#!/bin/bash
#SBATCH -J ch_param_10_stl_extra
#SBATCH -o output/ch_param_10_stl_extra.out
#SBATCH -e output/ch_param_10_stl_extra.err
#SBATCH -p eecs,share,gpu
#SBATCH -t 2-12:00:00
#SBATCH --gres=gpu:1
#SBATCH --exclude=cn-gpu3

python3 -m trainers.train_model --config rtmc/cold_hardiness/ch_param_10_stl --seed 4 --cultivar Nebbiolo

python3 -m trainers.train_model --config rtmc/cold_hardiness/ch_param_10_stl --seed 0 --cultivar Pinot_Gris
python3 -m trainers.train_model --config rtmc/cold_hardiness/ch_param_10_stl --seed 1 --cultivar Pinot_Gris
python3 -m trainers.train_model --config rtmc/cold_hardiness/ch_param_10_stl --seed 2 --cultivar Pinot_Gris
python3 -m trainers.train_model --config rtmc/cold_hardiness/ch_param_10_stl --seed 3 --cultivar Pinot_Gris 
python3 -m trainers.train_model --config rtmc/cold_hardiness/ch_param_10_stl --seed 4 --cultivar Pinot_Gris

python3 -m trainers.train_model --config rtmc/cold_hardiness/ch_param_10_stl --seed 0 --cultivar Riesling
python3 -m trainers.train_model --config rtmc/cold_hardiness/ch_param_10_stl --seed 1 --cultivar Riesling
python3 -m trainers.train_model --config rtmc/cold_hardiness/ch_param_10_stl --seed 2 --cultivar Riesling
python3 -m trainers.train_model --config rtmc/cold_hardiness/ch_param_10_stl --seed 3 --cultivar Riesling 
python3 -m trainers.train_model --config rtmc/cold_hardiness/ch_param_10_stl --seed 4 --cultivar Riesling

python3 -m trainers.train_model --config rtmc/cold_hardiness/ch_param_10_stl --seed 0 --cultivar Sangiovese
python3 -m trainers.train_model --config rtmc/cold_hardiness/ch_param_10_stl --seed 1 --cultivar Sangiovese
python3 -m trainers.train_model --config rtmc/cold_hardiness/ch_param_10_stl --seed 2 --cultivar Sangiovese
python3 -m trainers.train_model --config rtmc/cold_hardiness/ch_param_10_stl --seed 3 --cultivar Sangiovese 
python3 -m trainers.train_model --config rtmc/cold_hardiness/ch_param_10_stl --seed 4 --cultivar Sangiovese

python3 -m trainers.train_model --config rtmc/cold_hardiness/ch_param_10_stl --seed 0 --cultivar Sauvignon_Blanc
python3 -m trainers.train_model --config rtmc/cold_hardiness/ch_param_10_stl --seed 1 --cultivar Sauvignon_Blanc
python3 -m trainers.train_model --config rtmc/cold_hardiness/ch_param_10_stl --seed 2 --cultivar Sauvignon_Blanc
python3 -m trainers.train_model --config rtmc/cold_hardiness/ch_param_10_stl --seed 3 --cultivar Sauvignon_Blanc 
python3 -m trainers.train_model --config rtmc/cold_hardiness/ch_param_10_stl --seed 4 --cultivar Sauvignon_Blanc

python3 -m trainers.train_model --config rtmc/cold_hardiness/ch_param_10_stl --seed 0 --cultivar Semillon
python3 -m trainers.train_model --config rtmc/cold_hardiness/ch_param_10_stl --seed 1 --cultivar Semillon
python3 -m trainers.train_model --config rtmc/cold_hardiness/ch_param_10_stl --seed 2 --cultivar Semillon
python3 -m trainers.train_model --config rtmc/cold_hardiness/ch_param_10_stl --seed 3 --cultivar Semillon 
python3 -m trainers.train_model --config rtmc/cold_hardiness/ch_param_10_stl --seed 4 --cultivar Semillon

python3 -m trainers.train_model --config rtmc/cold_hardiness/ch_param_10_stl --seed 0 --cultivar Syrah
python3 -m trainers.train_model --config rtmc/cold_hardiness/ch_param_10_stl --seed 1 --cultivar Syrah
python3 -m trainers.train_model --config rtmc/cold_hardiness/ch_param_10_stl --seed 2 --cultivar Syrah
python3 -m trainers.train_model --config rtmc/cold_hardiness/ch_param_10_stl --seed 3 --cultivar Syrah 
python3 -m trainers.train_model --config rtmc/cold_hardiness/ch_param_10_stl --seed 4 --cultivar Syrah

python3 -m trainers.train_model --config rtmc/cold_hardiness/ch_param_10_stl --seed 0 --cultivar Tempranillo
python3 -m trainers.train_model --config rtmc/cold_hardiness/ch_param_10_stl --seed 1 --cultivar Tempranillo
python3 -m trainers.train_model --config rtmc/cold_hardiness/ch_param_10_stl --seed 2 --cultivar Tempranillo
python3 -m trainers.train_model --config rtmc/cold_hardiness/ch_param_10_stl --seed 3 --cultivar Tempranillo 
python3 -m trainers.train_model --config rtmc/cold_hardiness/ch_param_10_stl --seed 4 --cultivar Tempranillo

python3 -m trainers.train_model --config rtmc/cold_hardiness/ch_param_10_stl --seed 0 --cultivar Viognier
python3 -m trainers.train_model --config rtmc/cold_hardiness/ch_param_10_stl --seed 1 --cultivar Viognier
python3 -m trainers.train_model --config rtmc/cold_hardiness/ch_param_10_stl --seed 2 --cultivar Viognier
python3 -m trainers.train_model --config rtmc/cold_hardiness/ch_param_10_stl --seed 3 --cultivar Viognier 
python3 -m trainers.train_model --config rtmc/cold_hardiness/ch_param_10_stl --seed 4 --cultivar Viognier

python3 -m trainers.train_model --config rtmc/cold_hardiness/ch_param_10_stl --seed 0 --cultivar Zinfandel
python3 -m trainers.train_model --config rtmc/cold_hardiness/ch_param_10_stl --seed 1 --cultivar Zinfandel
python3 -m trainers.train_model --config rtmc/cold_hardiness/ch_param_10_stl --seed 2 --cultivar Zinfandel
python3 -m trainers.train_model --config rtmc/cold_hardiness/ch_param_10_stl --seed 3 --cultivar Zinfandel 
python3 -m trainers.train_model --config rtmc/cold_hardiness/ch_param_10_stl --seed 4 --cultivar Zinfandel