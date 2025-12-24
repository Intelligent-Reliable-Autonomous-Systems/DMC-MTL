#!/bin/bash
#SBATCH -J pheno_param_1_1
#SBATCH -o output/pheno_param_1_1.out
#SBATCH -e output/pheno_param_1_1.err
#SBATCH -p eecs,share,gpu
#SBATCH -t 2-12:00:00
#SBATCH --gres=gpu:1

python3 -m trainers.train_model_rtmc --config rtmc/finetune_pheno/pheno_param_1_1 --seed 0 --rnn_fpath _runs/RTMC/Phenology/ParamMTL1/WA/Roza2/Prosser/All/param_mtl__1766424582
python3 -m trainers.train_model_rtmc --config rtmc/finetune_pheno/pheno_param_1_1 --seed 1 --rnn_fpath _runs/RTMC/Phenology/ParamMTL1/WA/Roza2/Prosser/All/param_mtl__1766429186
python3 -m trainers.train_model_rtmc --config rtmc/finetune_pheno/pheno_param_1_1 --seed 2 --rnn_fpath _runs/RTMC/Phenology/ParamMTL1/WA/Roza2/Prosser/All/param_mtl__1766433780
python3 -m trainers.train_model_rtmc --config rtmc/finetune_pheno/pheno_param_1_1 --seed 3 --rnn_fpath _runs/RTMC/Phenology/ParamMTL1/WA/Roza2/Prosser/All/param_mtl__1766438376
python3 -m trainers.train_model_rtmc --config rtmc/finetune_pheno/pheno_param_1_1 --seed 4 --rnn_fpath _runs/RTMC/Phenology/ParamMTL1/WA/Roza2/Prosser/All/param_mtl__1766442966
