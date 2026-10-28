#!/bin/bash
#SBATCH -J grapehardinet_15
#SBATCH -o output/grapehardinet_15.out
#SBATCH -e output/grapehardinet_15.err
#SBATCH -p eecs,gpu
#SBATCH -t 2-00:00:00
#SBATCH --gres=gpu:1

python3 -m trainers.train_model --config aaai/grapehardinet --seed 0 --region BCOV --station Osoyoos_CS --site OsoyoosNortheast
python3 -m trainers.train_model --config aaai/grapehardinet --seed 1 --region BCOV --station Osoyoos_CS --site OsoyoosNortheast
python3 -m trainers.train_model --config aaai/grapehardinet --seed 2 --region BCOV --station Osoyoos_CS --site OsoyoosNortheast
python3 -m trainers.train_model --config aaai/grapehardinet --seed 3 --region BCOV --station Osoyoos_CS --site OsoyoosNortheast
python3 -m trainers.train_model --config aaai/grapehardinet --seed 4 --region BCOV --station Osoyoos_CS --site OsoyoosNortheast
