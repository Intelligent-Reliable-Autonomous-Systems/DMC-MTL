#!/bin/bash
#SBATCH -J pheno_param_2_10
#SBATCH -o output/pheno_param_2_10.out
#SBATCH -e output/pheno_param_2_10.err
#SBATCH -p eecs,share,gpu
#SBATCH -t 2-12:00:00
#SBATCH --gres=gpu:1

python3 -m trainers.train_model_rtmc --config rtmc/finetune_pheno_deep/pheno_deep_2_10 --seed 0 --rnn_fpath _runs/RTMC/Phenology/DeepReg10/None/None/None/All/deep_mtl__1763871750
python3 -m trainers.train_model_rtmc --config rtmc/finetune_pheno_deep/pheno_deep_2_10 --seed 1 --rnn_fpath _runs/RTMC/Phenology/DeepReg10/None/None/None/All/deep_mtl__1763879081
python3 -m trainers.train_model_rtmc --config rtmc/finetune_pheno_deep/pheno_deep_2_10 --seed 2 --rnn_fpath _runs/RTMC/Phenology/DeepReg10/None/None/None/All/deep_mtl__1763886331
python3 -m trainers.train_model_rtmc --config rtmc/finetune_pheno_deep/pheno_deep_2_10 --seed 3 --rnn_fpath _runs/RTMC/Phenology/DeepReg10/None/None/None/All/deep_mtl__1763893581
python3 -m trainers.train_model_rtmc --config rtmc/finetune_pheno_deep/pheno_deep_2_10 --seed 4 --rnn_fpath _runs/RTMC/Phenology/DeepReg10/None/None/None/All/deep_mtl__1763900844
