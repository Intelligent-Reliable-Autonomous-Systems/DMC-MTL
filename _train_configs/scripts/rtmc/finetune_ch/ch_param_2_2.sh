#!/bin/bash
#SBATCH -J ch_param_2_2
#SBATCH -o output/ch_param_2_2.out
#SBATCH -e output/ch_param_2_2.err
#SBATCH -p eecs,share,gpu
#SBATCH -t 2-12:00:00
#SBATCH --gres=gpu:1


python3 -m trainers.train_model_rtmc --config rtmc/finetune_ch/ch_param_2_2 --seed 0 --rnn_fpath _runs/RTMC/ColdHardiness/ParamMTL2/None/None/None/All/deep_mtl__1763783042
python3 -m trainers.train_model_rtmc --config rtmc/finetune_ch/ch_param_2_2 --seed 0 --rnn_fpath _runs/RTMC/ColdHardiness/ParamMTL2/None/None/None/All/deep_mtl__1763789889
python3 -m trainers.train_model_rtmc --config rtmc/finetune_ch/ch_param_2_2 --seed 0 --rnn_fpath _runs/RTMC/ColdHardiness/ParamMTL2/None/None/None/All/deep_mtl__1763796839
python3 -m trainers.train_model_rtmc --config rtmc/finetune_ch/ch_param_2_2 --seed 0 --rnn_fpath _runs/RTMC/ColdHardiness/ParamMTL2/None/None/None/All/deep_mtl__1763803861
python3 -m trainers.train_model_rtmc --config rtmc/finetune_ch/ch_param_2_2 --seed 0 --rnn_fpath _runs/RTMC/ColdHardiness/ParamMTL2/None/None/None/All/deep_mtl__1763810966

