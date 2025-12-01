#!/bin/bash
#SBATCH -J ch_pinn
#SBATCH -o output/ch_pinn.out
#SBATCH -e output/ch_pinn.err
#SBATCH -p eecs,share,gpu
#SBATCH -t 2-12:00:00
#SBATCH --gres=gpu:1

python3 -m trainers.train_model_rtmc --config rtmc/finetune_pheno/pheno_param_1_1 --seed 3 --rnn_fpath _runs/RTMC/Phenology/ParamMTL1/None/None/None/All/param_mtl__1763837854
python3 -m trainers.train_model_rtmc --config rtmc/finetune_pheno/pheno_param_1_1 --seed 4 --rnn_fpath _runs/RTMC/Phenology/ParamMTL1/None/None/None/All/param_mtl__1763841968
