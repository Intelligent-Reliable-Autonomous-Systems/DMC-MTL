#!/bin/bash
#SBATCH -J ch_param_2_2
#SBATCH -o output/ch_param_2_2.out
#SBATCH -e output/ch_param_2_2.err
#SBATCH -p eecs,share,gpu
#SBATCH -t 2-12:00:00
#SBATCH --gres=gpu:1


python3 -m trainers.train_model_rtmc --config rtmc/finetune_ch/ch_param_2_2 --seed 0 --rnn_fpath _runs/RTMC/ColdHardiness/ParamMTL2/WA/Roza2/Prosser/All/param_mtl__1766424437
python3 -m trainers.train_model_rtmc --config rtmc/finetune_ch/ch_param_2_2 --seed 0 --rnn_fpath _runs/RTMC/ColdHardiness/ParamMTL2/WA/Roza2/Prosser/All/param_mtl__1766431580
python3 -m trainers.train_model_rtmc --config rtmc/finetune_ch/ch_param_2_2 --seed 0 --rnn_fpath _runs/RTMC/ColdHardiness/ParamMTL2/WA/Roza2/Prosser/All/param_mtl__1766438739
python3 -m trainers.train_model_rtmc --config rtmc/finetune_ch/ch_param_2_2 --seed 0 --rnn_fpath _runs/RTMC/ColdHardiness/ParamMTL2/WA/Roza2/Prosser/All/param_mtl__1766445884
python3 -m trainers.train_model_rtmc --config rtmc/finetune_ch/ch_param_2_2 --seed 0 --rnn_fpath _runs/RTMC/ColdHardiness/ParamMTL2/WA/Roza2/Prosser/All/param_mtl__1766452994
