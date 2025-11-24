#!/bin/bash
#SBATCH -J ch_deep_5_2
#SBATCH -o output/ch_deep_5_2.out
#SBATCH -e output/ch_deep_5_2.err
#SBATCH -p eecs,share,gpu
#SBATCH -t 2-12:00:00
#SBATCH --gres=gpu:1

python3 -m trainers.train_model_rtmc --config rtmc/finetune_ch_deep/ch_deep_5_2 --seed 0 --rnn_fpath _runs/RTMC/ColdHardiness/DeepMTL2/None/None/None/All/deep_mtl__1763805256
python3 -m trainers.train_model_rtmc --config rtmc/finetune_ch_deep/ch_deep_5_2 --seed 0 --rnn_fpath _runs/RTMC/ColdHardiness/DeepMTL2/None/None/None/All/deep_mtl__1763806601
python3 -m trainers.train_model_rtmc --config rtmc/finetune_ch_deep/ch_deep_5_2 --seed 0 --rnn_fpath _runs/RTMC/ColdHardiness/DeepMTL2/None/None/None/All/deep_mtl__1763807931
python3 -m trainers.train_model_rtmc --config rtmc/finetune_ch_deep/ch_deep_5_2 --seed 0 --rnn_fpath _runs/RTMC/ColdHardiness/DeepMTL2/None/None/None/All/deep_mtl__1763807931
python3 -m trainers.train_model_rtmc --config rtmc/finetune_ch_deep/ch_deep_5_2 --seed 0 --rnn_fpath _runs/RTMC/ColdHardiness/DeepMTL2/None/None/None/All/deep_mtl__1763810634
