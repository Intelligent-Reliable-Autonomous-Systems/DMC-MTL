#!/bin/bash
#SBATCH -J ch_param_10_10
#SBATCH -o output/ch_param_10_10.out
#SBATCH -e output/ch_param_10_10.err
#SBATCH -p eecs,share,gpu
#SBATCH -t 2-12:00:00
#SBATCH --gres=gpu:1

python3 -m trainers.train_model_rtmc --config rtmc/finetune_ch/ch_param_10_10 --seed 0 --rnn_fpath _runs/RTMC/ColdHardiness/ParamMTL10/None/None/None/All/deep_mtl__1763796062
python3 -m trainers.train_model_rtmc --config rtmc/finetune_ch/ch_param_10_10 --seed 1 --rnn_fpath _runs/RTMC/ColdHardiness/ParamMTL10/None/None/None/All/deep_mtl__1763814484
python3 -m trainers.train_model_rtmc --config rtmc/finetune_ch/ch_param_10_10 --seed 2 --rnn_fpath _runs/RTMC/ColdHardiness/ParamMTL10/None/None/None/All/deep_mtl__1763832793
python3 -m trainers.train_model_rtmc --config rtmc/finetune_ch/ch_param_10_10 --seed 3 --rnn_fpath _runs/RTMC/ColdHardiness/ParamMTL10/None/None/None/All/deep_mtl__1763851298
python3 -m trainers.train_model_rtmc --config rtmc/finetune_ch/ch_param_10_10 --seed 4 --rnn_fpath _runs/RTMC/ColdHardiness/ParamMTL10/None/None/None/All/deep_mtl__1763869920

