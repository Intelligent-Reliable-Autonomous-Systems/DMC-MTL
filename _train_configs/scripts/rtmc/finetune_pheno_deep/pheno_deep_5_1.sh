#!/bin/bash
#SBATCH -J pheno_param_5_1
#SBATCH -o output/pheno_param_5_1.out
#SBATCH -e output/pheno_param_5_1.err
#SBATCH -p eecs,share,gpu
#SBATCH -t 2-12:00:00
#SBATCH --gres=gpu:1

python3 -m trainers.train_model_rtmc --config rtmc/finetune_pheno_deep/pheno_deep_5_1 --seed 0 --rnn_fpath _runs/RTMC/Phenology/DeepReg1/None/None/None/All/deep_mtl__1763854407
python3 -m trainers.train_model_rtmc --config rtmc/finetune_pheno_deep/pheno_deep_5_1 --seed 1 --rnn_fpath _runs/RTMC/Phenology/DeepReg1/None/None/None/All/deep_mtl__1763855757
python3 -m trainers.train_model_rtmc --config rtmc/finetune_pheno_deep/pheno_deep_5_1 --seed 2 --rnn_fpath _runs/RTMC/Phenology/DeepReg1/None/None/None/All/deep_mtl__1763857088
python3 -m trainers.train_model_rtmc --config rtmc/finetune_pheno_deep/pheno_deep_5_1 --seed 3 --rnn_fpath _runs/RTMC/Phenology/DeepReg1/None/None/None/All/deep_mtl__1763858443
python3 -m trainers.train_model_rtmc --config rtmc/finetune_pheno_deep/pheno_deep_5_1 --seed 4 --rnn_fpath _runs/RTMC/Phenology/DeepReg1/None/None/None/All/deep_mtl__1763859792
