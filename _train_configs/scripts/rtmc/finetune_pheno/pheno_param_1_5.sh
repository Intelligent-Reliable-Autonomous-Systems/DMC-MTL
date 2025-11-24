#!/bin/bash
#SBATCH -J pheno_param_1_5
#SBATCH -o output/pheno_param_1_5.out
#SBATCH -e output/pheno_param_1_5.err
#SBATCH -p eecs,share,gpu
#SBATCH -t 2-12:00:00
#SBATCH --gres=gpu:1

python3 -m trainers.train_model_rtmc --config rtmc/finetune_pheno/pheno_param_1_5 --seed 0 --rnn_fpath _runs/RTMC/Phenology/ParamMTL5/None/None/None/All/param_mtl__1763846147
python3 -m trainers.train_model_rtmc --config rtmc/finetune_pheno/pheno_param_1_5 --seed 0 --rnn_fpath _runs/RTMC/Phenology/ParamMTL5/None/None/None/All/param_mtl__1763855961
python3 -m trainers.train_model_rtmc --config rtmc/finetune_pheno/pheno_param_1_5 --seed 0 --rnn_fpath _runs/RTMC/Phenology/ParamMTL5/None/None/None/All/param_mtl__1763865769
python3 -m trainers.train_model_rtmc --config rtmc/finetune_pheno/pheno_param_1_5 --seed 0 --rnn_fpath _runs/RTMC/Phenology/ParamMTL5/None/None/None/All/param_mtl__1763875653
python3 -m trainers.train_model_rtmc --config rtmc/finetune_pheno/pheno_param_1_5 --seed 0 --rnn_fpath _runs/RTMC/Phenology/ParamMTL5/None/None/None/All/param_mtl__1763885399
