#!/bin/bash
#SBATCH -J grapehardinet_7
#SBATCH -o output/grapehardinet_7.out
#SBATCH -e output/grapehardinet_7.err
#SBATCH -p eecs,gpu
#SBATCH -t 2-00:00:00
#SBATCH --gres=gpu:1

python3 -m trainers.train_model --config aaai/grapehardinet --seed 0 --region NS --station Beaver --site Beaver
python3 -m trainers.train_model --config aaai/grapehardinet --seed 1 --region NS --station Beaver --site Beaver
python3 -m trainers.train_model --config aaai/grapehardinet --seed 2 --region NS --station Beaver --site Beaver
python3 -m trainers.train_model --config aaai/grapehardinet --seed 3 --region NS --station Beaver --site Beaver
python3 -m trainers.train_model --config aaai/grapehardinet --seed 4 --region NS --station Beaver --site Beaver
