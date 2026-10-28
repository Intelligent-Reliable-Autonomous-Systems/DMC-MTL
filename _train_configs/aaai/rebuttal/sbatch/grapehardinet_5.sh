#!/bin/bash
#SBATCH -J grapehardinet_5
#SBATCH -o output/grapehardinet_5.out
#SBATCH -e output/grapehardinet_5.err
#SBATCH -p eecs,gpu
#SBATCH -t 2-00:00:00
#SBATCH --gres=gpu:1

python3 -m trainers.train_model --config aaai/grapehardinet --seed 0 --region NY --station Hudson --site Hudson
python3 -m trainers.train_model --config aaai/grapehardinet --seed 1 --region NY --station Hudson --site Hudson
python3 -m trainers.train_model --config aaai/grapehardinet --seed 2 --region NY --station Hudson --site Hudson
python3 -m trainers.train_model --config aaai/grapehardinet --seed 3 --region NY --station Hudson --site Hudson
python3 -m trainers.train_model --config aaai/grapehardinet --seed 4 --region NY --station Hudson --site Hudson
