#!/bin/bash
#SBATCH -J ch_deep_2_10
#SBATCH -o output/ch_deep_2_10.out
#SBATCH -e output/ch_deep_2_10.err
#SBATCH -p eecs,share,gpu
#SBATCH -t 2-12:00:00
#SBATCH --gres=gpu:1

python3 -m trainers.train_model_rtmc --config rtmc/finetune_ch_deep/ch_deep_2_10 --seed 0 --rnn_fpath _runs/RTMC/ColdHardiness/DeepMTL10/None/None/None/All/deep_mtl__1763818000
python3 -m trainers.train_model_rtmc --config rtmc/finetune_ch_deep/ch_deep_2_10 --seed 1 --rnn_fpath _runs/RTMC/ColdHardiness/DeepMTL10/None/None/None/All/deep_mtl__1763823117
python3 -m trainers.train_model_rtmc --config rtmc/finetune_ch_deep/ch_deep_2_10 --seed 2 --rnn_fpath _runs/RTMC/ColdHardiness/DeepMTL10/None/None/None/All/deep_mtl__1763828229
python3 -m trainers.train_model_rtmc --config rtmc/finetune_ch_deep/ch_deep_2_10 --seed 3 --rnn_fpath _runs/RTMC/ColdHardiness/DeepMTL10/None/None/None/All/deep_mtl__1763833332
python3 -m trainers.train_model_rtmc --config rtmc/finetune_ch_deep/ch_deep_2_10 --seed 4 --rnn_fpath _runs/RTMC/ColdHardiness/DeepMTL10/None/None/None/All/deep_mtl__1763838479

