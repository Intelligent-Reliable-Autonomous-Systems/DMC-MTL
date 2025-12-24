#!/bin/bash
#SBATCH -J pheno_param_10_10
#SBATCH -o output/pheno_param_10_10.out
#SBATCH -e output/pheno_param_10_10.err
#SBATCH -p eecs,share,gpu
#SBATCH -t 2-12:00:00
#SBATCH --gres=gpu:1

python3 -m trainers.train_model_rtmc --config rtmc/finetune_pheno/pheno_param_10_10 --seed 0 --rnn_fpath _runs/RTMC/Phenology/ParamMTL10/WA/Roza2/Prosser/All/param_mtl__1766449473
python3 -m trainers.train_model_rtmc --config rtmc/finetune_pheno/pheno_param_10_10 --seed 1 --rnn_fpath _runs/RTMC/Phenology/ParamMTL10/WA/Roza2/Prosser/All/param_mtl__1766473913
python3 -m trainers.train_model_rtmc --config rtmc/finetune_pheno/pheno_param_10_10 --seed 2 --rnn_fpath _runs/RTMC/Phenology/ParamMTL10/WA/Roza2/Prosser/All/param_mtl__1766497775
python3 -m trainers.train_model_rtmc --config rtmc/finetune_pheno/pheno_param_10_10 --seed 3 --rnn_fpath _runs/RTMC/Phenology/ParamMTL10/WA/Roza2/Prosser/All/param_mtl__1766521102
python3 -m trainers.train_model_rtmc --config rtmc/finetune_pheno/pheno_param_10_10 --seed 4 --rnn_fpath _runs/RTMC/Phenology/ParamMTL10/WA/Roza2/Prosser/All/param_mtl__1766544434
