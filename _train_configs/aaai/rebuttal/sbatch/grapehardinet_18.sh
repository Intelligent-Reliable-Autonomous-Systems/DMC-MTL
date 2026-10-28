#!/bin/bash
#SBATCH -J grapehardinet_18
#SBATCH -o output/grapehardinet_18.out
#SBATCH -e output/grapehardinet_18.err
#SBATCH -p eecs,gpu
#SBATCH -t 2-00:00:00
#SBATCH --gres=gpu:1

python3 -m trainers.train_model --config aaai/grapehardinet --seed 0 --region BCOV --station Summerland --site NaramataBench
python3 -m trainers.train_model --config aaai/grapehardinet --seed 1 --region BCOV --station Summerland --site NaramataBench
python3 -m trainers.train_model --config aaai/grapehardinet --seed 2 --region BCOV --station Summerland --site NaramataBench
python3 -m trainers.train_model --config aaai/grapehardinet --seed 3 --region BCOV --station Summerland --site NaramataBench
python3 -m trainers.train_model --config aaai/grapehardinet --seed 4 --region BCOV --station Summerland --site NaramataBench
