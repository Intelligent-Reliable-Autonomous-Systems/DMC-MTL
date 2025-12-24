#!/bin/bash
#SBATCH -J ch_param_5_5
#SBATCH -o output/ch_param_5_5.out
#SBATCH -e output/ch_param_5_5.err
#SBATCH -p eecs,share,gpu
#SBATCH -t 2-12:00:00
#SBATCH --gres=gpu:1


python3 -m trainers.train_model_rtmc --config rtmc/finetune_ch/ch_param_5_5 --seed 0 --rnn_fpath _runs/RTMC/ColdHardiness/ParamMTL5/WA/Roza2/Prosser/All/param_mtl__1766424437
python3 -m trainers.train_model_rtmc --config rtmc/finetune_ch/ch_param_5_5 --seed 0 --rnn_fpath _runs/RTMC/ColdHardiness/ParamMTL5/WA/Roza2/Prosser/All/param_mtl__1766436940
python3 -m trainers.train_model_rtmc --config rtmc/finetune_ch/ch_param_5_5 --seed 0 --rnn_fpath _runs/RTMC/ColdHardiness/ParamMTL5/WA/Roza2/Prosser/All/param_mtl__1766449435
python3 -m trainers.train_model_rtmc --config rtmc/finetune_ch/ch_param_5_5 --seed 0 --rnn_fpath _runs/RTMC/ColdHardiness/ParamMTL5/WA/Roza2/Prosser/All/param_mtl__1766461915
python3 -m trainers.train_model_rtmc --config rtmc/finetune_ch/ch_param_5_5 --seed 0 --rnn_fpath _runs/RTMC/ColdHardiness/ParamMTL5/WA/Roza2/Prosser/All/param_mtl__1766474367
