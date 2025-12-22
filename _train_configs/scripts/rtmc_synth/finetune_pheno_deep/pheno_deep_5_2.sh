#!/bin/bash
#SBATCH -J pheno_param_5_2
#SBATCH -o output/pheno_param_5_2.out
#SBATCH -e output/pheno_param_5_2.err
#SBATCH -p eecs,share,gpu
#SBATCH -t 2-12:00:00
#SBATCH --gres=gpu:1

python3 -m trainers.train_model_rtmc --config rtmc/finetune_pheno_deep/pheno_deep_5_2 --seed 0 --rnn_fpath _runs/RTMC/Phenology/DeepReg2/None/None/None/All/deep_mtl__1763861137
python3 -m trainers.train_model_rtmc --config rtmc/finetune_pheno_deep/pheno_deep_5_2 --seed 0 --rnn_fpath _runs/RTMC/Phenology/DeepReg2/None/None/None/All/deep_mtl__1763863240
python3 -m trainers.train_model_rtmc --config rtmc/finetune_pheno_deep/pheno_deep_5_2 --seed 0 --rnn_fpath _runs/RTMC/Phenology/DeepReg2/None/None/None/All/deep_mtl__1763865372
python3 -m trainers.train_model_rtmc --config rtmc/finetune_pheno_deep/pheno_deep_5_2 --seed 0 --rnn_fpath _runs/RTMC/Phenology/DeepReg2/None/None/None/All/deep_mtl__1763867513
python3 -m trainers.train_model_rtmc --config rtmc/finetune_pheno_deep/pheno_deep_5_2 --seed 0 --rnn_fpath _runs/RTMC/Phenology/DeepReg2/None/None/None/All/deep_mtl__1763869643
