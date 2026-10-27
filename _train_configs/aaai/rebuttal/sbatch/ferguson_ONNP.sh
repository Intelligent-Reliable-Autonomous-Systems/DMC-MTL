#!/bin/bash
#SBATCH -J ferguson_ONNP
#SBATCH -o output/ferguson_ONNP.out
#SBATCH -e output/ferguson_ONNP.err
#SBATCH -p eecs,gpu
#SBATCH -t 2-00:00:00
#SBATCH --gres=gpu:1

python3 -m trainers.train_model --config aaai/rebuttal/ferguson --seed 0 --withold "{Region: [BCOV, WA, NY, MI, NS, WI, QC, PA, TX]}"
python3 -m trainers.train_model --config aaai/rebuttal/ferguson --seed 1 --withold "{Region: [BCOV, WA, NY, MI, NS, WI, QC, PA, TX]}"
python3 -m trainers.train_model --config aaai/rebuttal/ferguson --seed 2 --withold "{Region: [BCOV, WA, NY, MI, NS, WI, QC, PA, TX]}"
python3 -m trainers.train_model --config aaai/rebuttal/ferguson --seed 3 --withold "{Region: [BCOV, WA, NY, MI, NS, WI, QC, PA, TX]}"
python3 -m trainers.train_model --config aaai/rebuttal/ferguson --seed 4 --withold "{Region: [BCOV, WA, NY, MI, NS, WI, QC, PA, TX]}"
