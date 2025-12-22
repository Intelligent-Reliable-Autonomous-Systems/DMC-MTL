#!/bin/bash
#SBATCH -J ch_deep_1_5
#SBATCH -o output/ch_deep_1_5.out
#SBATCH -e output/ch_deep_1_5.err
#SBATCH -p eecs,share,gpu
#SBATCH -t 2-12:00:00
#SBATCH --gres=gpu:1

python3 -m trainers.train_model_rtmc --config rtmc/finetune_ch_deep/ch_deep_1_5 --seed 0 --rnn_fpath _runs/RTMC/ColdHardiness/DeepMTL5/None/None/None/All/deep_mtl__1763811986
python3 -m trainers.train_model_rtmc --config rtmc/finetune_ch_deep/ch_deep_1_5 --seed 0 --rnn_fpath _runs/RTMC/ColdHardiness/DeepMTL5/None/None/None/All/deep_mtl__1763814670
python3 -m trainers.train_model_rtmc --config rtmc/finetune_ch_deep/ch_deep_1_5 --seed 0 --rnn_fpath _runs/RTMC/ColdHardiness/DeepMTL5/None/None/None/All/deep_mtl__1763817351
python3 -m trainers.train_model_rtmc --config rtmc/finetune_ch_deep/ch_deep_1_5 --seed 0 --rnn_fpath _runs/RTMC/ColdHardiness/DeepMTL5/None/None/None/All/deep_mtl__1763820008
python3 -m trainers.train_model_rtmc --config rtmc/finetune_ch_deep/ch_deep_1_5 --seed 0 --rnn_fpath _runs/RTMC/ColdHardiness/DeepMTL5/None/None/None/All/deep_mtl__1763822694
