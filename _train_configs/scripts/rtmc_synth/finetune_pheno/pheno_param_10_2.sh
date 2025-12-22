#!/bin/bash
#SBATCH -J pheno_param_10_2
#SBATCH -o output/pheno_param_10_2.out
#SBATCH -e output/pheno_param_10_2.err
#SBATCH -p eecs,share,gpu
#SBATCH -t 2-12:00:00
#SBATCH --gres=gpu:1

python3 -m trainers.train_model_rtmc --config rtmc/finetune_pheno/pheno_param_10_2 --seed 0 --rnn_fpath _runs/RTMC/Phenology/ParamMTL2/None/None/None/All/param_mtl__1763843594
python3 -m trainers.train_model_rtmc --config rtmc/finetune_pheno/pheno_param_10_2 --seed 0 --rnn_fpath _runs/RTMC/Phenology/ParamMTL2/None/None/None/All/param_mtl__1763851469
python3 -m trainers.train_model_rtmc --config rtmc/finetune_pheno/pheno_param_10_2 --seed 0 --rnn_fpath _runs/RTMC/Phenology/ParamMTL2/None/None/None/All/param_mtl__1763859286
python3 -m trainers.train_model_rtmc --config rtmc/finetune_pheno/pheno_param_10_2 --seed 0 --rnn_fpath _runs/RTMC/Phenology/ParamMTL2/None/None/None/All/param_mtl__1763867129
python3 -m trainers.train_model_rtmc --config rtmc/finetune_pheno/pheno_param_10_2 --seed 0 --rnn_fpath _runs/RTMC/Phenology/ParamMTL2/None/None/None/All/param_mtl__1763875005

