#!/bin/bash
#SBATCH -J ch_param_10_10
#SBATCH -o output/ch_param_10_10.out
#SBATCH -e output/ch_param_10_10.err
#SBATCH -p eecs,share,gpu
#SBATCH -t 2-12:00:00
#SBATCH --gres=gpu:1

python3 -m trainers.train_model_rtmc --config rtmc/finetune_ch/ch_param_10_10 --seed 0 --rnn_fpath _runs/RTMC/ColdHardiness/ParamMTL10/WA/Roza2/Prosser/All/param_mtl__1766424430
python3 -m trainers.train_model_rtmc --config rtmc/finetune_ch/ch_param_10_10 --seed 1 --rnn_fpath _runs/RTMC/ColdHardiness/ParamMTL10/WA/Roza2/Prosser/All/param_mtl__1766439427
python3 -m trainers.train_model_rtmc --config rtmc/finetune_ch/ch_param_10_10 --seed 2 --rnn_fpath _runs/RTMC/ColdHardiness/ParamMTL10/WA/Roza2/Prosser/All/param_mtl__1766454570
python3 -m trainers.train_model_rtmc --config rtmc/finetune_ch/ch_param_10_10 --seed 3 --rnn_fpath _runs/RTMC/ColdHardiness/ParamMTL10/WA/Roza2/Prosser/All/param_mtl__1766469536
python3 -m trainers.train_model_rtmc --config rtmc/finetune_ch/ch_param_10_10 --seed 4 --rnn_fpath _runs/RTMC/ColdHardiness/ParamMTL10/WA/Roza2/Prosser/All/param_mtl__1766484614
