#!/bin/bash
#SBATCH -J ch_param_1_1
#SBATCH -o output/ch_param_1_1.out
#SBATCH -e output/ch_param_1_1.err
#SBATCH -p eecs,share,gpu
#SBATCH -t 2-12:00:00
#SBATCH --gres=gpu:1

python3 -m trainers.train_model_rtmc --config rtmc/finetune_ch/ch_param_1_1 --seed 0 --rnn_fpath _runs/RTMC/ColdHardiness/ParamMTL1/WA/Roza2/Prosser/All/param_mtl__1766424437
python3 -m trainers.train_model_rtmc --config rtmc/finetune_ch/ch_param_1_1 --seed 1 --rnn_fpath _runs/RTMC/ColdHardiness/ParamMTL1/WA/Roza2/Prosser/All/param_mtl__1766429446
python3 -m trainers.train_model_rtmc --config rtmc/finetune_ch/ch_param_1_1 --seed 2 --rnn_fpath _runs/RTMC/ColdHardiness/ParamMTL1/WA/Roza2/Prosser/All/param_mtl__1766434472
python3 -m trainers.train_model_rtmc --config rtmc/finetune_ch/ch_param_1_1 --seed 3 --rnn_fpath _runs/RTMC/ColdHardiness/ParamMTL1/WA/Roza2/Prosser/All/param_mtl__1766439433
python3 -m trainers.train_model_rtmc --config rtmc/finetune_ch/ch_param_1_1 --seed 4 --rnn_fpath _runs/RTMC/ColdHardiness/ParamMTL1/WA/Roza2/Prosser/All/param_mtl__1766444439
