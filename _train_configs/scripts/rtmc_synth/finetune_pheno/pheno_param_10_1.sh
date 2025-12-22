#!/bin/bash
#SBATCH -J pheno_param_10_1
#SBATCH -o output/pheno_param_10_1.out
#SBATCH -e output/pheno_param_10_1.err
#SBATCH -p eecs,share,gpu
#SBATCH -t 2-12:00:00
#SBATCH --gres=gpu:1

python3 -m trainers.train_model_rtmc --config rtmc/finetune_pheno/pheno_param_10_1 --seed 0 --rnn_fpath _runs/RTMC/Phenology/ParamMTL1/None/None/None/All/param_mtl__1763825377
python3 -m trainers.train_model_rtmc --config rtmc/finetune_pheno/pheno_param_10_1 --seed 1 --rnn_fpath _runs/RTMC/Phenology/ParamMTL1/None/None/None/All/param_mtl__1763829553
python3 -m trainers.train_model_rtmc --config rtmc/finetune_pheno/pheno_param_10_1 --seed 2 --rnn_fpath _runs/RTMC/Phenology/ParamMTL1/None/None/None/All/param_mtl__1763833676
python3 -m trainers.train_model_rtmc --config rtmc/finetune_pheno/pheno_param_10_1 --seed 3 --rnn_fpath _runs/RTMC/Phenology/ParamMTL1/None/None/None/All/param_mtl__1763837854
python3 -m trainers.train_model_rtmc --config rtmc/finetune_pheno/pheno_param_10_1 --seed 4 --rnn_fpath _runs/RTMC/Phenology/ParamMTL1/None/None/None/All/param_mtl__1763841968

