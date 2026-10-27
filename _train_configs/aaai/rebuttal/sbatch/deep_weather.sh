#!/bin/bash
#SBATCH -J deep_weather
#SBATCH -o output/deep_weather.out
#SBATCH -e output/deep_weather.err
#SBATCH -p eecs,gpu
#SBATCH -t 2-00:00:00
#SBATCH --gres=gpu:1

python3 -m trainers.train_model --config aaai/rebuttal/deep_weather --seed 0
python3 -m trainers.train_model --config aaai/rebuttal/deep_weather --seed 1
python3 -m trainers.train_model --config aaai/rebuttal/deep_weather --seed 2
python3 -m trainers.train_model --config aaai/rebuttal/deep_weather --seed 3
python3 -m trainers.train_model --config aaai/rebuttal/deep_weather --seed 4
