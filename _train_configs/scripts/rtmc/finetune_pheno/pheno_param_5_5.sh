#!/bin/bash
#SBATCH -J pheno_param_5_5
#SBATCH -o output/pheno_param_5_5.out
#SBATCH -e output/pheno_param_5_5.err
#SBATCH -p eecs,share,gpu
#SBATCH -t 2-12:00:00
#SBATCH --gres=gpu:1

python3 -m trainers.train_model_rtmc --config rtmc/finetune_pheno/pheno_param_5_5 --seed 0 --rnn_fpath _runs/RTMC/Phenology/ParamMTL5/WA/Roza2/Prosser/All/param_mtl__1766447582
python3 -m trainers.train_model_rtmc --config rtmc/finetune_pheno/pheno_param_5_5 --seed 0 --rnn_fpath _runs/RTMC/Phenology/ParamMTL5/WA/Roza2/Prosser/All/param_mtl__1766458575
python3 -m trainers.train_model_rtmc --config rtmc/finetune_pheno/pheno_param_5_5 --seed 0 --rnn_fpath _runs/RTMC/Phenology/ParamMTL5/WA/Roza2/Prosser/All/param_mtl__1766469148
python3 -m trainers.train_model_rtmc --config rtmc/finetune_pheno/pheno_param_5_5 --seed 0 --rnn_fpath _runs/RTMC/Phenology/ParamMTL5/WA/Roza2/Prosser/All/param_mtl__1766479472
python3 -m trainers.train_model_rtmc --config rtmc/finetune_pheno/pheno_param_5_5 --seed 0 --rnn_fpath _runs/RTMC/Phenology/ParamMTL5/WA/Roza2/Prosser/All/param_mtl__1766489925
