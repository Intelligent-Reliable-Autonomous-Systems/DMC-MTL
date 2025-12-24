#!/bin/bash
#SBATCH -J ch_param_2_stl_stl
#SBATCH -o output/ch_param_2_stl.out
#SBATCH -e output/ch_param_2_stl.err
#SBATCH -p eecs,share,gpu
#SBATCH -t 2-12:00:00
#SBATCH --gres=gpu:1

python3 -m trainers.train_model --config rtmc/cold_hardiness/ch_param_2_stl --seed 0 --cultivar Barbera
python3 -m trainers.train_model --config rtmc/cold_hardiness/ch_param_2_stl --seed 1 --cultivar Barbera
python3 -m trainers.train_model --config rtmc/cold_hardiness/ch_param_2_stl --seed 2 --cultivar Barbera
python3 -m trainers.train_model --config rtmc/cold_hardiness/ch_param_2_stl --seed 3 --cultivar Barbera 
python3 -m trainers.train_model --config rtmc/cold_hardiness/ch_param_2_stl --seed 4 --cultivar Barbera

python3 -m trainers.train_model --config rtmc/cold_hardiness/ch_param_2_stl --seed 0 --cultivar Cabernet_Franc
python3 -m trainers.train_model --config rtmc/cold_hardiness/ch_param_2_stl --seed 1 --cultivar Cabernet_Franc
python3 -m trainers.train_model --config rtmc/cold_hardiness/ch_param_2_stl --seed 2 --cultivar Cabernet_Franc
python3 -m trainers.train_model --config rtmc/cold_hardiness/ch_param_2_stl --seed 3 --cultivar Cabernet_Franc 
python3 -m trainers.train_model --config rtmc/cold_hardiness/ch_param_2_stl --seed 4 --cultivar Cabernet_Franc

python3 -m trainers.train_model --config rtmc/cold_hardiness/ch_param_2_stl --seed 0 --cultivar Cabernet_Sauvignon
python3 -m trainers.train_model --config rtmc/cold_hardiness/ch_param_2_stl --seed 1 --cultivar Cabernet_Sauvignon
python3 -m trainers.train_model --config rtmc/cold_hardiness/ch_param_2_stl --seed 2 --cultivar Cabernet_Sauvignon
python3 -m trainers.train_model --config rtmc/cold_hardiness/ch_param_2_stl --seed 3 --cultivar Cabernet_Sauvignon 
python3 -m trainers.train_model --config rtmc/cold_hardiness/ch_param_2_stl --seed 4 --cultivar Cabernet_Sauvignon

python3 -m trainers.train_model --config rtmc/cold_hardiness/ch_param_2_stl --seed 0 --cultivar Chardonnay
python3 -m trainers.train_model --config rtmc/cold_hardiness/ch_param_2_stl --seed 1 --cultivar Chardonnay
python3 -m trainers.train_model --config rtmc/cold_hardiness/ch_param_2_stl --seed 2 --cultivar Chardonnay
python3 -m trainers.train_model --config rtmc/cold_hardiness/ch_param_2_stl --seed 3 --cultivar Chardonnay 
python3 -m trainers.train_model --config rtmc/cold_hardiness/ch_param_2_stl --seed 4 --cultivar Chardonnay

python3 -m trainers.train_model --config rtmc/cold_hardiness/ch_param_2_stl --seed 0 --cultivar Chenin_Blanc
python3 -m trainers.train_model --config rtmc/cold_hardiness/ch_param_2_stl --seed 1 --cultivar Chenin_Blanc
python3 -m trainers.train_model --config rtmc/cold_hardiness/ch_param_2_stl --seed 2 --cultivar Chenin_Blanc
python3 -m trainers.train_model --config rtmc/cold_hardiness/ch_param_2_stl --seed 3 --cultivar Chenin_Blanc 
python3 -m trainers.train_model --config rtmc/cold_hardiness/ch_param_2_stl --seed 4 --cultivar Chenin_Blanc

python3 -m trainers.train_model --config rtmc/cold_hardiness/ch_param_2_stl --seed 0 --cultivar Concord
python3 -m trainers.train_model --config rtmc/cold_hardiness/ch_param_2_stl --seed 1 --cultivar Concord
python3 -m trainers.train_model --config rtmc/cold_hardiness/ch_param_2_stl --seed 2 --cultivar Concord
python3 -m trainers.train_model --config rtmc/cold_hardiness/ch_param_2_stl --seed 3 --cultivar Concord 
python3 -m trainers.train_model --config rtmc/cold_hardiness/ch_param_2_stl --seed 4 --cultivar Concord

python3 -m trainers.train_model --config rtmc/cold_hardiness/ch_param_2_stl --seed 0 --cultivar Gewurztraminer
python3 -m trainers.train_model --config rtmc/cold_hardiness/ch_param_2_stl --seed 1 --cultivar Gewurztraminer
python3 -m trainers.train_model --config rtmc/cold_hardiness/ch_param_2_stl --seed 2 --cultivar Gewurztraminer
python3 -m trainers.train_model --config rtmc/cold_hardiness/ch_param_2_stl --seed 3 --cultivar Gewurztraminer 
python3 -m trainers.train_model --config rtmc/cold_hardiness/ch_param_2_stl --seed 4 --cultivar Gewurztraminer

python3 -m trainers.train_model --config rtmc/cold_hardiness/ch_param_2_stl --seed 0 --cultivar Lemberger
python3 -m trainers.train_model --config rtmc/cold_hardiness/ch_param_2_stl --seed 1 --cultivar Lemberger
python3 -m trainers.train_model --config rtmc/cold_hardiness/ch_param_2_stl --seed 2 --cultivar Lemberger
python3 -m trainers.train_model --config rtmc/cold_hardiness/ch_param_2_stl --seed 3 --cultivar Lemberger 
python3 -m trainers.train_model --config rtmc/cold_hardiness/ch_param_2_stl --seed 4 --cultivar Lemberger

python3 -m trainers.train_model --config rtmc/cold_hardiness/ch_param_2_stl --seed 0 --cultivar Malbec
python3 -m trainers.train_model --config rtmc/cold_hardiness/ch_param_2_stl --seed 1 --cultivar Malbec
python3 -m trainers.train_model --config rtmc/cold_hardiness/ch_param_2_stl --seed 2 --cultivar Malbec
python3 -m trainers.train_model --config rtmc/cold_hardiness/ch_param_2_stl --seed 3 --cultivar Malbec 
python3 -m trainers.train_model --config rtmc/cold_hardiness/ch_param_2_stl --seed 4 --cultivar Malbec

python3 -m trainers.train_model --config rtmc/cold_hardiness/ch_param_2_stl --seed 0 --cultivar Merlot
python3 -m trainers.train_model --config rtmc/cold_hardiness/ch_param_2_stl --seed 1 --cultivar Merlot
python3 -m trainers.train_model --config rtmc/cold_hardiness/ch_param_2_stl --seed 2 --cultivar Merlot
python3 -m trainers.train_model --config rtmc/cold_hardiness/ch_param_2_stl --seed 3 --cultivar Merlot 
python3 -m trainers.train_model --config rtmc/cold_hardiness/ch_param_2_stl --seed 4 --cultivar Merlot

python3 -m trainers.train_model --config rtmc/cold_hardiness/ch_param_2_stl --seed 0 --cultivar Mourvedre
python3 -m trainers.train_model --config rtmc/cold_hardiness/ch_param_2_stl --seed 1 --cultivar Mourvedre
python3 -m trainers.train_model --config rtmc/cold_hardiness/ch_param_2_stl --seed 2 --cultivar Mourvedre
python3 -m trainers.train_model --config rtmc/cold_hardiness/ch_param_2_stl --seed 3 --cultivar Mourvedre 
python3 -m trainers.train_model --config rtmc/cold_hardiness/ch_param_2_stl --seed 4 --cultivar Mourvedre

python3 -m trainers.train_model --config rtmc/cold_hardiness/ch_param_2_stl --seed 0 --cultivar Nebbiolo
python3 -m trainers.train_model --config rtmc/cold_hardiness/ch_param_2_stl --seed 1 --cultivar Nebbiolo
python3 -m trainers.train_model --config rtmc/cold_hardiness/ch_param_2_stl --seed 2 --cultivar Nebbiolo
python3 -m trainers.train_model --config rtmc/cold_hardiness/ch_param_2_stl --seed 3 --cultivar Nebbiolo 
python3 -m trainers.train_model --config rtmc/cold_hardiness/ch_param_2_stl --seed 4 --cultivar Nebbiolo

python3 -m trainers.train_model --config rtmc/cold_hardiness/ch_param_2_stl --seed 0 --cultivar Pinot_Gris
python3 -m trainers.train_model --config rtmc/cold_hardiness/ch_param_2_stl --seed 1 --cultivar Pinot_Gris
python3 -m trainers.train_model --config rtmc/cold_hardiness/ch_param_2_stl --seed 2 --cultivar Pinot_Gris
python3 -m trainers.train_model --config rtmc/cold_hardiness/ch_param_2_stl --seed 3 --cultivar Pinot_Gris 
python3 -m trainers.train_model --config rtmc/cold_hardiness/ch_param_2_stl --seed 4 --cultivar Pinot_Gris

python3 -m trainers.train_model --config rtmc/cold_hardiness/ch_param_2_stl --seed 0 --cultivar Riesling
python3 -m trainers.train_model --config rtmc/cold_hardiness/ch_param_2_stl --seed 1 --cultivar Riesling
python3 -m trainers.train_model --config rtmc/cold_hardiness/ch_param_2_stl --seed 2 --cultivar Riesling
python3 -m trainers.train_model --config rtmc/cold_hardiness/ch_param_2_stl --seed 3 --cultivar Riesling 
python3 -m trainers.train_model --config rtmc/cold_hardiness/ch_param_2_stl --seed 4 --cultivar Riesling

python3 -m trainers.train_model --config rtmc/cold_hardiness/ch_param_2_stl --seed 0 --cultivar Sangiovese
python3 -m trainers.train_model --config rtmc/cold_hardiness/ch_param_2_stl --seed 1 --cultivar Sangiovese
python3 -m trainers.train_model --config rtmc/cold_hardiness/ch_param_2_stl --seed 2 --cultivar Sangiovese
python3 -m trainers.train_model --config rtmc/cold_hardiness/ch_param_2_stl --seed 3 --cultivar Sangiovese 
python3 -m trainers.train_model --config rtmc/cold_hardiness/ch_param_2_stl --seed 4 --cultivar Sangiovese

python3 -m trainers.train_model --config rtmc/cold_hardiness/ch_param_2_stl --seed 0 --cultivar Sauvignon_Blanc
python3 -m trainers.train_model --config rtmc/cold_hardiness/ch_param_2_stl --seed 1 --cultivar Sauvignon_Blanc
python3 -m trainers.train_model --config rtmc/cold_hardiness/ch_param_2_stl --seed 2 --cultivar Sauvignon_Blanc
python3 -m trainers.train_model --config rtmc/cold_hardiness/ch_param_2_stl --seed 3 --cultivar Sauvignon_Blanc 
python3 -m trainers.train_model --config rtmc/cold_hardiness/ch_param_2_stl --seed 4 --cultivar Sauvignon_Blanc

python3 -m trainers.train_model --config rtmc/cold_hardiness/ch_param_2_stl --seed 0 --cultivar Semillon
python3 -m trainers.train_model --config rtmc/cold_hardiness/ch_param_2_stl --seed 1 --cultivar Semillon
python3 -m trainers.train_model --config rtmc/cold_hardiness/ch_param_2_stl --seed 2 --cultivar Semillon
python3 -m trainers.train_model --config rtmc/cold_hardiness/ch_param_2_stl --seed 3 --cultivar Semillon 
python3 -m trainers.train_model --config rtmc/cold_hardiness/ch_param_2_stl --seed 4 --cultivar Semillon

python3 -m trainers.train_model --config rtmc/cold_hardiness/ch_param_2_stl --seed 0 --cultivar Syrah
python3 -m trainers.train_model --config rtmc/cold_hardiness/ch_param_2_stl --seed 1 --cultivar Syrah
python3 -m trainers.train_model --config rtmc/cold_hardiness/ch_param_2_stl --seed 2 --cultivar Syrah
python3 -m trainers.train_model --config rtmc/cold_hardiness/ch_param_2_stl --seed 3 --cultivar Syrah 
python3 -m trainers.train_model --config rtmc/cold_hardiness/ch_param_2_stl --seed 4 --cultivar Syrah

python3 -m trainers.train_model --config rtmc/cold_hardiness/ch_param_2_stl --seed 0 --cultivar Tempranillo
python3 -m trainers.train_model --config rtmc/cold_hardiness/ch_param_2_stl --seed 1 --cultivar Tempranillo
python3 -m trainers.train_model --config rtmc/cold_hardiness/ch_param_2_stl --seed 2 --cultivar Tempranillo
python3 -m trainers.train_model --config rtmc/cold_hardiness/ch_param_2_stl --seed 3 --cultivar Tempranillo 
python3 -m trainers.train_model --config rtmc/cold_hardiness/ch_param_2_stl --seed 4 --cultivar Tempranillo

python3 -m trainers.train_model --config rtmc/cold_hardiness/ch_param_2_stl --seed 0 --cultivar Viognier
python3 -m trainers.train_model --config rtmc/cold_hardiness/ch_param_2_stl --seed 1 --cultivar Viognier
python3 -m trainers.train_model --config rtmc/cold_hardiness/ch_param_2_stl --seed 2 --cultivar Viognier
python3 -m trainers.train_model --config rtmc/cold_hardiness/ch_param_2_stl --seed 3 --cultivar Viognier 
python3 -m trainers.train_model --config rtmc/cold_hardiness/ch_param_2_stl --seed 4 --cultivar Viognier

python3 -m trainers.train_model --config rtmc/cold_hardiness/ch_param_2_stl --seed 0 --cultivar Zinfandel
python3 -m trainers.train_model --config rtmc/cold_hardiness/ch_param_2_stl --seed 1 --cultivar Zinfandel
python3 -m trainers.train_model --config rtmc/cold_hardiness/ch_param_2_stl --seed 2 --cultivar Zinfandel
python3 -m trainers.train_model --config rtmc/cold_hardiness/ch_param_2_stl --seed 3 --cultivar Zinfandel 
python3 -m trainers.train_model --config rtmc/cold_hardiness/ch_param_2_stl --seed 4 --cultivar Zinfandel