#!/bin/bash
#SBATCH -J grapehardinet_1
#SBATCH -o output/grapehardinet_1.out
#SBATCH -e output/grapehardinet_1.err
#SBATCH -p eecs,gpu
#SBATCH -t 2-00:00:00
#SBATCH --gres=gpu:1

python3 -m trainers.train_model --config aaai/grapehardinet --seed 0 --region WA --station Roza2 --site Prosser
python3 -m trainers.train_model --config aaai/grapehardinet --seed 1 --region WA --station Roza2 --site Prosser
python3 -m trainers.train_model --config aaai/grapehardinet --seed 2 --region WA --station Roza2 --site Prosser
python3 -m trainers.train_model --config aaai/grapehardinet --seed 3 --region WA --station Roza2 --site Prosser
python3 -m trainers.train_model --config aaai/grapehardinet --seed 4 --region WA --station Roza2 --site Prosser
