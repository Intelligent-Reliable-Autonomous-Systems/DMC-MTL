#!/bin/bash
#SBATCH -J ch_param_2_1
#SBATCH -o output/ch_param_2_1.out
#SBATCH -e output/ch_param_2_1.err
#SBATCH -p eecs,share,gpu
#SBATCH -t 2-12:00:00
#SBATCH --gres=gpu:1

python3 -m trainers.train_model_rtmc --config rtmc/finetune_ch/ch_param_2_1 --seed 0 --rnn_fpath _runs/RTMC/ColdHardiness/ParamMTL1/None/None/None/All/deep_mtl__1763775611
python3 -m trainers.train_model_rtmc --config rtmc/finetune_ch/ch_param_2_1 --seed 1 --rnn_fpath _runs/RTMC/ColdHardiness/ParamMTL1/None/None/None/All/deep_mtl__1763779710
python3 -m trainers.train_model_rtmc --config rtmc/finetune_ch/ch_param_2_1 --seed 2 --rnn_fpath _runs/RTMC/ColdHardiness/ParamMTL1/None/None/None/All/deep_mtl__1763783801
python3 -m trainers.train_model_rtmc --config rtmc/finetune_ch/ch_param_2_1 --seed 3 --rnn_fpath _runs/RTMC/ColdHardiness/ParamMTL1/None/None/None/All/deep_mtl__1763787847
python3 -m trainers.train_model_rtmc --config rtmc/finetune_ch/ch_param_2_1 --seed 4 --rnn_fpath _runs/RTMC/ColdHardiness/ParamMTL1/None/None/None/All/deep_mtl__1763791946

