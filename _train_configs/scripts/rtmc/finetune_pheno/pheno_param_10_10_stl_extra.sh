#!/bin/bash
#SBATCH -J pheno_param_10_10_stl
#SBATCH -o output/pheno_param_10_10_stl.out
#SBATCH -e output/pheno_param_10_10_stl.err
#SBATCH -p eecs,share,gpu
#SBATCH -t 2-12:00:00
#SBATCH --gres=gpu:1
#SBATCH --exclude=cn-gpu3

python3 -m trainers.train_model_rtmc --config rtmc/finetune_pheno/pheno_param_10_10_stl --seed 3 --cultivar Pinot_Noir --rnn_fpath _runs/RTMC/Phenology/ParamMTL10/STL/WA/Roza2/Prosser/Pinot_Noir/param_mtl__1766641495
python3 -m trainers.train_model_rtmc --config rtmc/finetune_pheno/pheno_param_10_10_stl --seed 4 --cultivar Pinot_Noir --rnn_fpath _runs/RTMC/Phenology/ParamMTL10/STL/WA/Roza2/Prosser/Pinot_Noir/param_mtl__1766642879
 
python3 -m trainers.train_model_rtmc --config rtmc/finetune_pheno/pheno_param_10_10_stl --seed 0 --cultivar Riesling --rnn_fpath _runs/RTMC/Phenology/ParamMTL10/STL/WA/Roza2/Prosser/Riesling/param_mtl__1766644279
python3 -m trainers.train_model_rtmc --config rtmc/finetune_pheno/pheno_param_10_10_stl --seed 1 --cultivar Riesling --rnn_fpath _runs/RTMC/Phenology/ParamMTL10/STL/WA/Roza2/Prosser/Riesling/param_mtl__1766645684
python3 -m trainers.train_model_rtmc --config rtmc/finetune_pheno/pheno_param_10_10_stl --seed 2 --cultivar Riesling --rnn_fpath _runs/RTMC/Phenology/ParamMTL10/STL/WA/Roza2/Prosser/Riesling/param_mtl__1766647056
python3 -m trainers.train_model_rtmc --config rtmc/finetune_pheno/pheno_param_10_10_stl --seed 3 --cultivar Riesling --rnn_fpath _runs/RTMC/Phenology/ParamMTL10/STL/WA/Roza2/Prosser/Riesling/param_mtl__1766648433
python3 -m trainers.train_model_rtmc --config rtmc/finetune_pheno/pheno_param_10_10_stl --seed 4 --cultivar Riesling --rnn_fpath _runs/RTMC/Phenology/ParamMTL10/STL/WA/Roza2/Prosser/Riesling/param_mtl__1766649829
 
python3 -m trainers.train_model_rtmc --config rtmc/finetune_pheno/pheno_param_10_10_stl --seed 0 --cultivar Sangiovese --rnn_fpath _runs/RTMC/Phenology/ParamMTL10/STL/WA/Roza2/Prosser/Sangiovese/param_mtl__1766651221
python3 -m trainers.train_model_rtmc --config rtmc/finetune_pheno/pheno_param_10_10_stl --seed 1 --cultivar Sangiovese --rnn_fpath _runs/RTMC/Phenology/ParamMTL10/STL/WA/Roza2/Prosser/Sangiovese/param_mtl__1766652240
python3 -m trainers.train_model_rtmc --config rtmc/finetune_pheno/pheno_param_10_10_stl --seed 2 --cultivar Sangiovese --rnn_fpath _runs/RTMC/Phenology/ParamMTL10/STL/WA/Roza2/Prosser/Sangiovese/param_mtl__1766653250
python3 -m trainers.train_model_rtmc --config rtmc/finetune_pheno/pheno_param_10_10_stl --seed 3 --cultivar Sangiovese --rnn_fpath _runs/RTMC/Phenology/ParamMTL10/STL/WA/Roza2/Prosser/Sangiovese/param_mtl__1766654265
python3 -m trainers.train_model_rtmc --config rtmc/finetune_pheno/pheno_param_10_10_stl --seed 4 --cultivar Sangiovese --rnn_fpath _runs/RTMC/Phenology/ParamMTL10/STL/WA/Roza2/Prosser/Sangiovese/param_mtl__1766655285
 
python3 -m trainers.train_model_rtmc --config rtmc/finetune_pheno/pheno_param_10_10_stl --seed 0 --cultivar Sauvignon_Blanc --rnn_fpath _runs/RTMC/Phenology/ParamMTL10/STL/WA/Roza2/Prosser/Sauvignon_Blanc/param_mtl__1766656294
python3 -m trainers.train_model_rtmc --config rtmc/finetune_pheno/pheno_param_10_10_stl --seed 1 --cultivar Sauvignon_Blanc --rnn_fpath _runs/RTMC/Phenology/ParamMTL10/STL/WA/Roza2/Prosser/Sauvignon_Blanc/param_mtl__1766657311
python3 -m trainers.train_model_rtmc --config rtmc/finetune_pheno/pheno_param_10_10_stl --seed 2 --cultivar Sauvignon_Blanc --rnn_fpath _runs/RTMC/Phenology/ParamMTL10/STL/WA/Roza2/Prosser/Sauvignon_Blanc/param_mtl__1766658324
python3 -m trainers.train_model_rtmc --config rtmc/finetune_pheno/pheno_param_10_10_stl --seed 3 --cultivar Sauvignon_Blanc --rnn_fpath _runs/RTMC/Phenology/ParamMTL10/STL/WA/Roza2/Prosser/Sauvignon_Blanc/param_mtl__1766659340
python3 -m trainers.train_model_rtmc --config rtmc/finetune_pheno/pheno_param_10_10_stl --seed 4 --cultivar Sauvignon_Blanc --rnn_fpath _runs/RTMC/Phenology/ParamMTL10/STL/WA/Roza2/Prosser/Sauvignon_Blanc/param_mtl__1766660362
 
python3 -m trainers.train_model_rtmc --config rtmc/finetune_pheno/pheno_param_10_10_stl --seed 0 --cultivar Semillon --rnn_fpath _runs/RTMC/Phenology/ParamMTL10/STL/WA/Roza2/Prosser/Semillon/param_mtl__1766661379
python3 -m trainers.train_model_rtmc --config rtmc/finetune_pheno/pheno_param_10_10_stl --seed 1 --cultivar Semillon --rnn_fpath _runs/RTMC/Phenology/ParamMTL10/STL/WA/Roza2/Prosser/Semillon/param_mtl__1766662758
python3 -m trainers.train_model_rtmc --config rtmc/finetune_pheno/pheno_param_10_10_stl --seed 2 --cultivar Semillon --rnn_fpath _runs/RTMC/Phenology/ParamMTL10/STL/WA/Roza2/Prosser/Semillon/param_mtl__1766664152
python3 -m trainers.train_model_rtmc --config rtmc/finetune_pheno/pheno_param_10_10_stl --seed 3 --cultivar Semillon --rnn_fpath _runs/RTMC/Phenology/ParamMTL10/STL/WA/Roza2/Prosser/Semillon/param_mtl__1766665532
python3 -m trainers.train_model_rtmc --config rtmc/finetune_pheno/pheno_param_10_10_stl --seed 4 --cultivar Semillon --rnn_fpath _runs/RTMC/Phenology/ParamMTL10/STL/WA/Roza2/Prosser/Semillon/param_mtl__1766666905
 
python3 -m trainers.train_model_rtmc --config rtmc/finetune_pheno/pheno_param_10_10_stl --seed 0 --cultivar Tempranillo --rnn_fpath _runs/RTMC/Phenology/ParamMTL10/STL/WA/Roza2/Prosser/Tempranillo/param_mtl__1766668279
python3 -m trainers.train_model_rtmc --config rtmc/finetune_pheno/pheno_param_10_10_stl --seed 1 --cultivar Tempranillo --rnn_fpath _runs/RTMC/Phenology/ParamMTL10/STL/WA/Roza2/Prosser/Tempranillo/param_mtl__1766669299
python3 -m trainers.train_model_rtmc --config rtmc/finetune_pheno/pheno_param_10_10_stl --seed 2 --cultivar Tempranillo --rnn_fpath _runs/RTMC/Phenology/ParamMTL10/STL/WA/Roza2/Prosser/Tempranillo/param_mtl__1766670319
python3 -m trainers.train_model_rtmc --config rtmc/finetune_pheno/pheno_param_10_10_stl --seed 3 --cultivar Tempranillo --rnn_fpath _runs/RTMC/Phenology/ParamMTL10/STL/WA/Roza2/Prosser/Tempranillo/param_mtl__1766671336
python3 -m trainers.train_model_rtmc --config rtmc/finetune_pheno/pheno_param_10_10_stl --seed 4 --cultivar Tempranillo --rnn_fpath _runs/RTMC/Phenology/ParamMTL10/STL/WA/Roza2/Prosser/Tempranillo/param_mtl__1766672360
 
python3 -m trainers.train_model_rtmc --config rtmc/finetune_pheno/pheno_param_10_10_stl --seed 0 --cultivar Viognier --rnn_fpath _runs/RTMC/Phenology/ParamMTL10/STL/WA/Roza2/Prosser/Viognier/param_mtl__1766673376
python3 -m trainers.train_model_rtmc --config rtmc/finetune_pheno/pheno_param_10_10_stl --seed 1 --cultivar Viognier --rnn_fpath _runs/RTMC/Phenology/ParamMTL10/STL/WA/Roza2/Prosser/Viognier/param_mtl__1766674384
python3 -m trainers.train_model_rtmc --config rtmc/finetune_pheno/pheno_param_10_10_stl --seed 2 --cultivar Viognier --rnn_fpath _runs/RTMC/Phenology/ParamMTL10/STL/WA/Roza2/Prosser/Viognier/param_mtl__1766675394
python3 -m trainers.train_model_rtmc --config rtmc/finetune_pheno/pheno_param_10_10_stl --seed 3 --cultivar Viognier --rnn_fpath _runs/RTMC/Phenology/ParamMTL10/STL/WA/Roza2/Prosser/Viognier/param_mtl__1766676400
python3 -m trainers.train_model_rtmc --config rtmc/finetune_pheno/pheno_param_10_10_stl --seed 4 --cultivar Viognier --rnn_fpath _runs/RTMC/Phenology/ParamMTL10/STL/WA/Roza2/Prosser/Viognier/param_mtl__1766677412
 
python3 -m trainers.train_model_rtmc --config rtmc/finetune_pheno/pheno_param_10_10_stl --seed 0 --cultivar Zinfandel --rnn_fpath _runs/RTMC/Phenology/ParamMTL10/STL/WA/Roza2/Prosser/Zinfandel/param_mtl__1766678420
python3 -m trainers.train_model_rtmc --config rtmc/finetune_pheno/pheno_param_10_10_stl --seed 1 --cultivar Zinfandel --rnn_fpath _runs/RTMC/Phenology/ParamMTL10/STL/WA/Roza2/Prosser/Zinfandel/param_mtl__1766679780
python3 -m trainers.train_model_rtmc --config rtmc/finetune_pheno/pheno_param_10_10_stl --seed 2 --cultivar Zinfandel --rnn_fpath _runs/RTMC/Phenology/ParamMTL10/STL/WA/Roza2/Prosser/Zinfandel/param_mtl__1766681154
python3 -m trainers.train_model_rtmc --config rtmc/finetune_pheno/pheno_param_10_10_stl --seed 3 --cultivar Zinfandel --rnn_fpath _runs/RTMC/Phenology/ParamMTL10/STL/WA/Roza2/Prosser/Zinfandel/param_mtl__1766682530
python3 -m trainers.train_model_rtmc --config rtmc/finetune_pheno/pheno_param_10_10_stl --seed 4 --cultivar Zinfandel --rnn_fpath _runs/RTMC/Phenology/ParamMTL10/STL/WA/Roza2/Prosser/Zinfandel/param_mtl__1766683906

