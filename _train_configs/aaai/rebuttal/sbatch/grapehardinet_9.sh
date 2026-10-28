#!/bin/bash
#SBATCH -J grapehardinet_9
#SBATCH -o output/grapehardinet_9.out
#SBATCH -e output/grapehardinet_9.err
#SBATCH -p eecs,gpu
#SBATCH -t 2-00:00:00
#SBATCH --gres=gpu:1

python3 -m trainers.train_model --config aaai/grapehardinet --seed 0 --region BCOV --station Kelowna_UBC --site OKFallsEast
python3 -m trainers.train_model --config aaai/grapehardinet --seed 1 --region BCOV --station Kelowna_UBC --site OKFallsEast
python3 -m trainers.train_model --config aaai/grapehardinet --seed 2 --region BCOV --station Kelowna_UBC --site OKFallsEast
python3 -m trainers.train_model --config aaai/grapehardinet --seed 3 --region BCOV --station Kelowna_UBC --site OKFallsEast
python3 -m trainers.train_model --config aaai/grapehardinet --seed 4 --region BCOV --station Kelowna_UBC --site OKFallsEast
