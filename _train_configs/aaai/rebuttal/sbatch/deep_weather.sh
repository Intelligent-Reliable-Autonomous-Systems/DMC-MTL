#!/bin/bash
#SBATCH -J deep_weather
#SBATCH -o output/deep_weather.out
#SBATCH -e output/deep_weather.err
#SBATCH -p eecs,gpu
#SBATCH -t 2-00:00:00
#SBATCH --gres=gpu:1

python3 -m trainers.train_model --config aaai/rebuttal/deep_weather --seed 0 --withold "{Region: [WA, NY, MI, NS, WI, QC, PA, TX]}"
python3 -m trainers.train_model --config aaai/rebuttal/deep_weather --seed 1 --withold "{Region: [WA, NY, MI, NS, WI, QC, PA, TX]}"
python3 -m trainers.train_model --config aaai/rebuttal/deep_weather --seed 2 --withold "{Region: [WA, NY, MI, NS, WI, QC, PA, TX]}"
python3 -m trainers.train_model --config aaai/rebuttal/deep_weather --seed 3 --withold "{Region: [WA, NY, MI, NS, WI, QC, PA, TX]}"
python3 -m trainers.train_model --config aaai/rebuttal/deep_weather --seed 4 --withold "{Region: [WA, NY, MI, NS, WI, QC, PA, TX]}"