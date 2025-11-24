#!/bin/bash
#SBATCH -J pheno_param_5_5
#SBATCH -o output/pheno_param_5_5.out
#SBATCH -e output/pheno_param_5_5.err
#SBATCH -p eecs,share,gpu
#SBATCH -t 2-12:00:00
#SBATCH --gres=gpu:1

python3 -m trainers.train_model_rtmc --config rtmc/finetune_pheno_deep/pheno_deep_5_5 --seed 0 --rnn_fpath _runs/RTMC/Phenology/DeepReg5/None/None/None/All/deep_mtl__1763862257
python3 -m trainers.train_model_rtmc --config rtmc/finetune_pheno_deep/pheno_deep_5_5 --seed 0 --rnn_fpath _runs/RTMC/Phenology/DeepReg5/None/None/None/All/deep_mtl__1763866231
python3 -m trainers.train_model_rtmc --config rtmc/finetune_pheno_deep/pheno_deep_5_5 --seed 0 --rnn_fpath _runs/RTMC/Phenology/DeepReg5/None/None/None/All/deep_mtl__1763870196
python3 -m trainers.train_model_rtmc --config rtmc/finetune_pheno_deep/pheno_deep_5_5 --seed 0 --rnn_fpath _runs/RTMC/Phenology/DeepReg5/None/None/None/All/deep_mtl__1763874169
python3 -m trainers.train_model_rtmc --config rtmc/finetune_pheno_deep/pheno_deep_5_5 --seed 0 --rnn_fpath _runs/RTMC/Phenology/DeepReg5/None/None/None/All/deep_mtl__1763878125
