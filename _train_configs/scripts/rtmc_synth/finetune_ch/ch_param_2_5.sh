#!/bin/bash
#SBATCH -J ch_param_2_5
#SBATCH -o output/ch_param_2_5.out
#SBATCH -e output/ch_param_2_5.err
#SBATCH -p eecs,share,gpu
#SBATCH -t 2-12:00:00
#SBATCH --gres=gpu:1


python3 -m trainers.train_model_rtmc --config rtmc/finetune_ch/ch_param_2_5 --seed 0 --rnn_fpath _runs/RTMC/ColdHardiness/ParamMTL5/None/None/None/All/deep_mtl__1763792945
python3 -m trainers.train_model_rtmc --config rtmc/finetune_ch/ch_param_2_5 --seed 0 --rnn_fpath _runs/RTMC/ColdHardiness/ParamMTL5/None/None/None/All/deep_mtl__1763805410
python3 -m trainers.train_model_rtmc --config rtmc/finetune_ch/ch_param_2_5 --seed 0 --rnn_fpath _runs/RTMC/ColdHardiness/ParamMTL5/None/None/None/All/deep_mtl__1763817873
python3 -m trainers.train_model_rtmc --config rtmc/finetune_ch/ch_param_2_5 --seed 0 --rnn_fpath _runs/RTMC/ColdHardiness/ParamMTL5/None/None/None/All/deep_mtl__1763830052
python3 -m trainers.train_model_rtmc --config rtmc/finetune_ch/ch_param_2_5 --seed 0 --rnn_fpath _runs/RTMC/ColdHardiness/ParamMTL5/None/None/None/All/deep_mtl__1763842280

