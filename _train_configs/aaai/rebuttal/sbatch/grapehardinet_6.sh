#!/bin/bash
#SBATCH -J grapehardinet_6
#SBATCH -o output/grapehardinet_6.out
#SBATCH -e output/grapehardinet_6.err
#SBATCH -p eecs,gpu
#SBATCH -t 2-00:00:00
#SBATCH --gres=gpu:1

python3 -m trainers.train_model --config aaai/grapehardinet --seed 0 --region MI --station SWMREC --site Mich1
python3 -m trainers.train_model --config aaai/grapehardinet --seed 1 --region MI --station SWMREC --site Mich1
python3 -m trainers.train_model --config aaai/grapehardinet --seed 2 --region MI --station SWMREC --site Mich1
python3 -m trainers.train_model --config aaai/grapehardinet --seed 3 --region MI --station SWMREC --site Mich1
python3 -m trainers.train_model --config aaai/grapehardinet --seed 4 --region MI --station SWMREC --site Mich1
