#!/bin/bash
#SBATCH -J grapehardinet_19
#SBATCH -o output/grapehardinet_19.out
#SBATCH -e output/grapehardinet_19.err
#SBATCH -p eecs,gpu
#SBATCH -t 2-00:00:00
#SBATCH --gres=gpu:1

python3 -m trainers.train_model --config aaai/grapehardinet --seed 0 --region ONNP --station Grimbsy_Mtn --site BeamsvilleBench
python3 -m trainers.train_model --config aaai/grapehardinet --seed 1 --region ONNP --station Grimbsy_Mtn --site BeamsvilleBench
python3 -m trainers.train_model --config aaai/grapehardinet --seed 2 --region ONNP --station Grimbsy_Mtn --site BeamsvilleBench
python3 -m trainers.train_model --config aaai/grapehardinet --seed 3 --region ONNP --station Grimbsy_Mtn --site BeamsvilleBench
python3 -m trainers.train_model --config aaai/grapehardinet --seed 4 --region ONNP --station Grimbsy_Mtn --site BeamsvilleBench
