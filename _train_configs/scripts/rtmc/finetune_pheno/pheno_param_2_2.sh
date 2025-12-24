#!/bin/bash
#SBATCH -J pheno_param_2_2
#SBATCH -o output/pheno_param_2_2.out
#SBATCH -e output/pheno_param_2_2.err
#SBATCH -p eecs,share,gpu
#SBATCH -t 2-12:00:00
#SBATCH --gres=gpu:1

python3 -m trainers.train_model_rtmc --config rtmc/finetune_pheno/pheno_param_2_2 --seed 0 --rnn_fpath _runs/RTMC/Phenology/ParamMTL2/WA/Roza2/Prosser/All/param_mtl__1766424589
python3 -m trainers.train_model_rtmc --config rtmc/finetune_pheno/pheno_param_2_2 --seed 0 --rnn_fpath _runs/RTMC/Phenology/ParamMTL2/WA/Roza2/Prosser/All/param_mtl__1766431064
python3 -m trainers.train_model_rtmc --config rtmc/finetune_pheno/pheno_param_2_2 --seed 0 --rnn_fpath _runs/RTMC/Phenology/ParamMTL2/WA/Roza2/Prosser/All/param_mtl__1766437524
python3 -m trainers.train_model_rtmc --config rtmc/finetune_pheno/pheno_param_2_2 --seed 0 --rnn_fpath _runs/RTMC/Phenology/ParamMTL2/WA/Roza2/Prosser/All/param_mtl__1766443984
python3 -m trainers.train_model_rtmc --config rtmc/finetune_pheno/pheno_param_2_2 --seed 0 --rnn_fpath _runs/RTMC/Phenology/ParamMTL2/WA/Roza2/Prosser/All/param_mtl__1766450461

