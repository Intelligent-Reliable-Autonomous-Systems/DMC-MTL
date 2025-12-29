#!/bin/bash
#SBATCH -J pheno_param_10_10_stl
#SBATCH -o output/pheno_param_10_10_stl.out
#SBATCH -e output/pheno_param_10_10_stl.err
#SBATCH -p eecs,share,gpu
#SBATCH -t 2-12:00:00
#SBATCH --gres=gpu:1
#SBATCH --exclude=cn-gpu3

python3 -m trainers.train_model_rtmc --config rtmc/finetune_pheno/pheno_param_10_10_stl --seed 0 --cultivar Aligote --rnn_fpath _runs/RTMC/Phenology/ParamMTL10/STL/WA/Roza2/Prosser/Aligote/param_mtl__1766499491
python3 -m trainers.train_model_rtmc --config rtmc/finetune_pheno/pheno_param_10_10_stl --seed 1 --cultivar Aligote --rnn_fpath _runs/RTMC/Phenology/ParamMTL10/STL/WA/Roza2/Prosser/Aligote/param_mtl__1766500482
python3 -m trainers.train_model_rtmc --config rtmc/finetune_pheno/pheno_param_10_10_stl --seed 2 --cultivar Aligote --rnn_fpath _runs/RTMC/Phenology/ParamMTL10/STL/WA/Roza2/Prosser/Aligote/param_mtl__1766501481
python3 -m trainers.train_model_rtmc --config rtmc/finetune_pheno/pheno_param_10_10_stl --seed 3 --cultivar Aligote --rnn_fpath _runs/RTMC/Phenology/ParamMTL10/STL/WA/Roza2/Prosser/Aligote/param_mtl__1766502474
python3 -m trainers.train_model_rtmc --config rtmc/finetune_pheno/pheno_param_10_10_stl --seed 4 --cultivar Aligote --rnn_fpath _runs/RTMC/Phenology/ParamMTL10/STL/WA/Roza2/Prosser/Aligote/param_mtl__1766503467
 
python3 -m trainers.train_model_rtmc --config rtmc/finetune_pheno/pheno_param_10_10_stl --seed 0 --cultivar Alvarinho --rnn_fpath _runs/RTMC/Phenology/ParamMTL10/STL/WA/Roza2/Prosser/Alvarinho/param_mtl__1766504458
python3 -m trainers.train_model_rtmc --config rtmc/finetune_pheno/pheno_param_10_10_stl --seed 1 --cultivar Alvarinho --rnn_fpath _runs/RTMC/Phenology/ParamMTL10/STL/WA/Roza2/Prosser/Alvarinho/param_mtl__1766505458
python3 -m trainers.train_model_rtmc --config rtmc/finetune_pheno/pheno_param_10_10_stl --seed 2 --cultivar Alvarinho --rnn_fpath _runs/RTMC/Phenology/ParamMTL10/STL/WA/Roza2/Prosser/Alvarinho/param_mtl__1766506453
python3 -m trainers.train_model_rtmc --config rtmc/finetune_pheno/pheno_param_10_10_stl --seed 3 --cultivar Alvarinho --rnn_fpath _runs/RTMC/Phenology/ParamMTL10/STL/WA/Roza2/Prosser/Alvarinho/param_mtl__1766507450
python3 -m trainers.train_model_rtmc --config rtmc/finetune_pheno/pheno_param_10_10_stl --seed 4 --cultivar Alvarinho --rnn_fpath _runs/RTMC/Phenology/ParamMTL10/STL/WA/Roza2/Prosser/Alvarinho/param_mtl__1766508451
 
python3 -m trainers.train_model_rtmc --config rtmc/finetune_pheno/pheno_param_10_10_stl --seed 0 --cultivar Auxerrois --rnn_fpath _runs/RTMC/Phenology/ParamMTL10/STL/WA/Roza2/Prosser/Auxerrois/param_mtl__1766509449
python3 -m trainers.train_model_rtmc --config rtmc/finetune_pheno/pheno_param_10_10_stl --seed 1 --cultivar Auxerrois --rnn_fpath _runs/RTMC/Phenology/ParamMTL10/STL/WA/Roza2/Prosser/Auxerrois/param_mtl__1766510445
python3 -m trainers.train_model_rtmc --config rtmc/finetune_pheno/pheno_param_10_10_stl --seed 2 --cultivar Auxerrois --rnn_fpath _runs/RTMC/Phenology/ParamMTL10/STL/WA/Roza2/Prosser/Auxerrois/param_mtl__1766511438
python3 -m trainers.train_model_rtmc --config rtmc/finetune_pheno/pheno_param_10_10_stl --seed 3 --cultivar Auxerrois --rnn_fpath _runs/RTMC/Phenology/ParamMTL10/STL/WA/Roza2/Prosser/Auxerrois/param_mtl__1766512438
python3 -m trainers.train_model_rtmc --config rtmc/finetune_pheno/pheno_param_10_10_stl --seed 4 --cultivar Auxerrois --rnn_fpath _runs/RTMC/Phenology/ParamMTL10/STL/WA/Roza2/Prosser/Auxerrois/param_mtl__1766513445
 
python3 -m trainers.train_model_rtmc --config rtmc/finetune_pheno/pheno_param_10_10_stl --seed 0 --cultivar Barbera --rnn_fpath _runs/RTMC/Phenology/ParamMTL10/STL/WA/Roza2/Prosser/Barbera/param_mtl__1766514447
python3 -m trainers.train_model_rtmc --config rtmc/finetune_pheno/pheno_param_10_10_stl --seed 1 --cultivar Barbera --rnn_fpath _runs/RTMC/Phenology/ParamMTL10/STL/WA/Roza2/Prosser/Barbera/param_mtl__1766515452
python3 -m trainers.train_model_rtmc --config rtmc/finetune_pheno/pheno_param_10_10_stl --seed 2 --cultivar Barbera --rnn_fpath _runs/RTMC/Phenology/ParamMTL10/STL/WA/Roza2/Prosser/Barbera/param_mtl__1766516460
python3 -m trainers.train_model_rtmc --config rtmc/finetune_pheno/pheno_param_10_10_stl --seed 3 --cultivar Barbera --rnn_fpath _runs/RTMC/Phenology/ParamMTL10/STL/WA/Roza2/Prosser/Barbera/param_mtl__1766517471
python3 -m trainers.train_model_rtmc --config rtmc/finetune_pheno/pheno_param_10_10_stl --seed 4 --cultivar Barbera --rnn_fpath _runs/RTMC/Phenology/ParamMTL10/STL/WA/Roza2/Prosser/Barbera/param_mtl__1766518479
 
python3 -m trainers.train_model_rtmc --config rtmc/finetune_pheno/pheno_param_10_10_stl --seed 0 --cultivar Cabernet_Franc --rnn_fpath _runs/RTMC/Phenology/ParamMTL10/STL/WA/Roza2/Prosser/Cabernet_Franc/param_mtl__1766519487
python3 -m trainers.train_model_rtmc --config rtmc/finetune_pheno/pheno_param_10_10_stl --seed 1 --cultivar Cabernet_Franc --rnn_fpath _runs/RTMC/Phenology/ParamMTL10/STL/WA/Roza2/Prosser/Cabernet_Franc/param_mtl__1766520860
python3 -m trainers.train_model_rtmc --config rtmc/finetune_pheno/pheno_param_10_10_stl --seed 2 --cultivar Cabernet_Franc --rnn_fpath _runs/RTMC/Phenology/ParamMTL10/STL/WA/Roza2/Prosser/Cabernet_Franc/param_mtl__1766522242
python3 -m trainers.train_model_rtmc --config rtmc/finetune_pheno/pheno_param_10_10_stl --seed 3 --cultivar Cabernet_Franc --rnn_fpath _runs/RTMC/Phenology/ParamMTL10/STL/WA/Roza2/Prosser/Cabernet_Franc/param_mtl__1766523613
python3 -m trainers.train_model_rtmc --config rtmc/finetune_pheno/pheno_param_10_10_stl --seed 4 --cultivar Cabernet_Franc --rnn_fpath _runs/RTMC/Phenology/ParamMTL10/STL/WA/Roza2/Prosser/Cabernet_Franc/param_mtl__1766524978
 
python3 -m trainers.train_model_rtmc --config rtmc/finetune_pheno/pheno_param_10_10_stl --seed 0 --cultivar Cabernet_Sauvignon --rnn_fpath _runs/RTMC/Phenology/ParamMTL10/STL/WA/Roza2/Prosser/Cabernet_Sauvignon/param_mtl__1766526350
python3 -m trainers.train_model_rtmc --config rtmc/finetune_pheno/pheno_param_10_10_stl --seed 1 --cultivar Cabernet_Sauvignon --rnn_fpath _runs/RTMC/Phenology/ParamMTL10/STL/WA/Roza2/Prosser/Cabernet_Sauvignon/param_mtl__1766527711
python3 -m trainers.train_model_rtmc --config rtmc/finetune_pheno/pheno_param_10_10_stl --seed 2 --cultivar Cabernet_Sauvignon --rnn_fpath _runs/RTMC/Phenology/ParamMTL10/STL/WA/Roza2/Prosser/Cabernet_Sauvignon/param_mtl__1766529098
python3 -m trainers.train_model_rtmc --config rtmc/finetune_pheno/pheno_param_10_10_stl --seed 3 --cultivar Cabernet_Sauvignon --rnn_fpath _runs/RTMC/Phenology/ParamMTL10/STL/WA/Roza2/Prosser/Cabernet_Sauvignon/param_mtl__1766530481
python3 -m trainers.train_model_rtmc --config rtmc/finetune_pheno/pheno_param_10_10_stl --seed 4 --cultivar Cabernet_Sauvignon --rnn_fpath _runs/RTMC/Phenology/ParamMTL10/STL/WA/Roza2/Prosser/Cabernet_Sauvignon/param_mtl__1766531837
 
python3 -m trainers.train_model_rtmc --config rtmc/finetune_pheno/pheno_param_10_10_stl --seed 0 --cultivar Chardonnay --rnn_fpath _runs/RTMC/Phenology/ParamMTL10/STL/WA/Roza2/Prosser/Chardonnay/param_mtl__1766533192
python3 -m trainers.train_model_rtmc --config rtmc/finetune_pheno/pheno_param_10_10_stl --seed 1 --cultivar Chardonnay --rnn_fpath _runs/RTMC/Phenology/ParamMTL10/STL/WA/Roza2/Prosser/Chardonnay/param_mtl__1766534543
python3 -m trainers.train_model_rtmc --config rtmc/finetune_pheno/pheno_param_10_10_stl --seed 2 --cultivar Chardonnay --rnn_fpath _runs/RTMC/Phenology/ParamMTL10/STL/WA/Roza2/Prosser/Chardonnay/param_mtl__1766535924
python3 -m trainers.train_model_rtmc --config rtmc/finetune_pheno/pheno_param_10_10_stl --seed 3 --cultivar Chardonnay --rnn_fpath _runs/RTMC/Phenology/ParamMTL10/STL/WA/Roza2/Prosser/Chardonnay/param_mtl__1766537292
python3 -m trainers.train_model_rtmc --config rtmc/finetune_pheno/pheno_param_10_10_stl --seed 4 --cultivar Chardonnay --rnn_fpath _runs/RTMC/Phenology/ParamMTL10/STL/WA/Roza2/Prosser/Chardonnay/param_mtl__1766538655
 
python3 -m trainers.train_model_rtmc --config rtmc/finetune_pheno/pheno_param_10_10_stl --seed 0 --cultivar Chenin_Blanc --rnn_fpath _runs/RTMC/Phenology/ParamMTL10/STL/WA/Roza2/Prosser/Chenin_Blanc/param_mtl__1766540023
python3 -m trainers.train_model_rtmc --config rtmc/finetune_pheno/pheno_param_10_10_stl --seed 1 --cultivar Chenin_Blanc --rnn_fpath _runs/RTMC/Phenology/ParamMTL10/STL/WA/Roza2/Prosser/Chenin_Blanc/param_mtl__1766541412
python3 -m trainers.train_model_rtmc --config rtmc/finetune_pheno/pheno_param_10_10_stl --seed 2 --cultivar Chenin_Blanc --rnn_fpath _runs/RTMC/Phenology/ParamMTL10/STL/WA/Roza2/Prosser/Chenin_Blanc/param_mtl__1766542767
python3 -m trainers.train_model_rtmc --config rtmc/finetune_pheno/pheno_param_10_10_stl --seed 3 --cultivar Chenin_Blanc --rnn_fpath _runs/RTMC/Phenology/ParamMTL10/STL/WA/Roza2/Prosser/Chenin_Blanc/param_mtl__1766544135
python3 -m trainers.train_model_rtmc --config rtmc/finetune_pheno/pheno_param_10_10_stl --seed 4 --cultivar Chenin_Blanc --rnn_fpath _runs/RTMC/Phenology/ParamMTL10/STL/WA/Roza2/Prosser/Chenin_Blanc/param_mtl__1766545487
 
python3 -m trainers.train_model_rtmc --config rtmc/finetune_pheno/pheno_param_10_10_stl --seed 0 --cultivar Concord --rnn_fpath _runs/RTMC/Phenology/ParamMTL10/STL/WA/Roza2/Prosser/Concord/param_mtl__1766546857
python3 -m trainers.train_model_rtmc --config rtmc/finetune_pheno/pheno_param_10_10_stl --seed 1 --cultivar Concord --rnn_fpath _runs/RTMC/Phenology/ParamMTL10/STL/WA/Roza2/Prosser/Concord/param_mtl__1766548230
python3 -m trainers.train_model_rtmc --config rtmc/finetune_pheno/pheno_param_10_10_stl --seed 2 --cultivar Concord --rnn_fpath _runs/RTMC/Phenology/ParamMTL10/STL/WA/Roza2/Prosser/Concord/param_mtl__1766549612
python3 -m trainers.train_model_rtmc --config rtmc/finetune_pheno/pheno_param_10_10_stl --seed 3 --cultivar Concord --rnn_fpath _runs/RTMC/Phenology/ParamMTL10/STL/WA/Roza2/Prosser/Concord/param_mtl__1766550983
python3 -m trainers.train_model_rtmc --config rtmc/finetune_pheno/pheno_param_10_10_stl --seed 4 --cultivar Concord --rnn_fpath _runs/RTMC/Phenology/ParamMTL10/STL/WA/Roza2/Prosser/Concord/param_mtl__1766552355
 
python3 -m trainers.train_model_rtmc --config rtmc/finetune_pheno/pheno_param_10_10_stl --seed 0 --cultivar Durif --rnn_fpath _runs/RTMC/Phenology/ParamMTL10/STL/WA/Roza2/Prosser/Durif/param_mtl__1766553708
python3 -m trainers.train_model_rtmc --config rtmc/finetune_pheno/pheno_param_10_10_stl --seed 1 --cultivar Durif --rnn_fpath _runs/RTMC/Phenology/ParamMTL10/STL/WA/Roza2/Prosser/Durif/param_mtl__1766554720
python3 -m trainers.train_model_rtmc --config rtmc/finetune_pheno/pheno_param_10_10_stl --seed 2 --cultivar Durif --rnn_fpath _runs/RTMC/Phenology/ParamMTL10/STL/WA/Roza2/Prosser/Durif/param_mtl__1766555728
python3 -m trainers.train_model_rtmc --config rtmc/finetune_pheno/pheno_param_10_10_stl --seed 3 --cultivar Durif --rnn_fpath _runs/RTMC/Phenology/ParamMTL10/STL/WA/Roza2/Prosser/Durif/param_mtl__1766556735
python3 -m trainers.train_model_rtmc --config rtmc/finetune_pheno/pheno_param_10_10_stl --seed 4 --cultivar Durif --rnn_fpath _runs/RTMC/Phenology/ParamMTL10/STL/WA/Roza2/Prosser/Durif/param_mtl__1766557750
 
python3 -m trainers.train_model_rtmc --config rtmc/finetune_pheno/pheno_param_10_10_stl --seed 0 --cultivar Gewurztraminer --rnn_fpath _runs/RTMC/Phenology/ParamMTL10/STL/WA/Roza2/Prosser/Gewurztraminer/param_mtl__1766558760
python3 -m trainers.train_model_rtmc --config rtmc/finetune_pheno/pheno_param_10_10_stl --seed 1 --cultivar Gewurztraminer --rnn_fpath _runs/RTMC/Phenology/ParamMTL10/STL/WA/Roza2/Prosser/Gewurztraminer/param_mtl__1766560135
python3 -m trainers.train_model_rtmc --config rtmc/finetune_pheno/pheno_param_10_10_stl --seed 2 --cultivar Gewurztraminer --rnn_fpath _runs/RTMC/Phenology/ParamMTL10/STL/WA/Roza2/Prosser/Gewurztraminer/param_mtl__1766561505
python3 -m trainers.train_model_rtmc --config rtmc/finetune_pheno/pheno_param_10_10_stl --seed 3 --cultivar Gewurztraminer --rnn_fpath _runs/RTMC/Phenology/ParamMTL10/STL/WA/Roza2/Prosser/Gewurztraminer/param_mtl__1766562884
python3 -m trainers.train_model_rtmc --config rtmc/finetune_pheno/pheno_param_10_10_stl --seed 4 --cultivar Gewurztraminer --rnn_fpath _runs/RTMC/Phenology/ParamMTL10/STL/WA/Roza2/Prosser/Gewurztraminer/param_mtl__1766564253
 
python3 -m trainers.train_model_rtmc --config rtmc/finetune_pheno/pheno_param_10_10_stl --seed 0 --cultivar Green_Veltliner --rnn_fpath _runs/RTMC/Phenology/ParamMTL10/STL/WA/Roza2/Prosser/Green_Veltliner/param_mtl__1766565628
python3 -m trainers.train_model_rtmc --config rtmc/finetune_pheno/pheno_param_10_10_stl --seed 1 --cultivar Green_Veltliner --rnn_fpath _runs/RTMC/Phenology/ParamMTL10/STL/WA/Roza2/Prosser/Green_Veltliner/param_mtl__1766566634
python3 -m trainers.train_model_rtmc --config rtmc/finetune_pheno/pheno_param_10_10_stl --seed 2 --cultivar Green_Veltliner --rnn_fpath _runs/RTMC/Phenology/ParamMTL10/STL/WA/Roza2/Prosser/Green_Veltliner/param_mtl__1766567640
python3 -m trainers.train_model_rtmc --config rtmc/finetune_pheno/pheno_param_10_10_stl --seed 3 --cultivar Green_Veltliner --rnn_fpath _runs/RTMC/Phenology/ParamMTL10/STL/WA/Roza2/Prosser/Green_Veltliner/param_mtl__1766568647
python3 -m trainers.train_model_rtmc --config rtmc/finetune_pheno/pheno_param_10_10_stl --seed 4 --cultivar Green_Veltliner --rnn_fpath _runs/RTMC/Phenology/ParamMTL10/STL/WA/Roza2/Prosser/Green_Veltliner/param_mtl__1766569654
 
python3 -m trainers.train_model_rtmc --config rtmc/finetune_pheno/pheno_param_10_10_stl --seed 0 --cultivar Grenache --rnn_fpath _runs/RTMC/Phenology/ParamMTL10/STL/WA/Roza2/Prosser/Grenache/param_mtl__1766570657
python3 -m trainers.train_model_rtmc --config rtmc/finetune_pheno/pheno_param_10_10_stl --seed 1 --cultivar Grenache --rnn_fpath _runs/RTMC/Phenology/ParamMTL10/STL/WA/Roza2/Prosser/Grenache/param_mtl__1766571996
python3 -m trainers.train_model_rtmc --config rtmc/finetune_pheno/pheno_param_10_10_stl --seed 2 --cultivar Grenache --rnn_fpath _runs/RTMC/Phenology/ParamMTL10/STL/WA/Roza2/Prosser/Grenache/param_mtl__1766573366
python3 -m trainers.train_model_rtmc --config rtmc/finetune_pheno/pheno_param_10_10_stl --seed 3 --cultivar Grenache --rnn_fpath _runs/RTMC/Phenology/ParamMTL10/STL/WA/Roza2/Prosser/Grenache/param_mtl__1766574741
python3 -m trainers.train_model_rtmc --config rtmc/finetune_pheno/pheno_param_10_10_stl --seed 4 --cultivar Grenache --rnn_fpath _runs/RTMC/Phenology/ParamMTL10/STL/WA/Roza2/Prosser/Grenache/param_mtl__1766576118
 
python3 -m trainers.train_model_rtmc --config rtmc/finetune_pheno/pheno_param_10_10_stl --seed 0 --cultivar Lemberger --rnn_fpath _runs/RTMC/Phenology/ParamMTL10/STL/WA/Roza2/Prosser/Lemberger/param_mtl__1766577485
python3 -m trainers.train_model_rtmc --config rtmc/finetune_pheno/pheno_param_10_10_stl --seed 1 --cultivar Lemberger --rnn_fpath _runs/RTMC/Phenology/ParamMTL10/STL/WA/Roza2/Prosser/Lemberger/param_mtl__1766578856
python3 -m trainers.train_model_rtmc --config rtmc/finetune_pheno/pheno_param_10_10_stl --seed 2 --cultivar Lemberger --rnn_fpath _runs/RTMC/Phenology/ParamMTL10/STL/WA/Roza2/Prosser/Lemberger/param_mtl__1766580225
python3 -m trainers.train_model_rtmc --config rtmc/finetune_pheno/pheno_param_10_10_stl --seed 3 --cultivar Lemberger --rnn_fpath _runs/RTMC/Phenology/ParamMTL10/STL/WA/Roza2/Prosser/Lemberger/param_mtl__1766581597
python3 -m trainers.train_model_rtmc --config rtmc/finetune_pheno/pheno_param_10_10_stl --seed 4 --cultivar Lemberger --rnn_fpath _runs/RTMC/Phenology/ParamMTL10/STL/WA/Roza2/Prosser/Lemberger/param_mtl__1766582971
 
python3 -m trainers.train_model_rtmc --config rtmc/finetune_pheno/pheno_param_10_10_stl --seed 0 --cultivar Malbec --rnn_fpath _runs/RTMC/Phenology/ParamMTL10/STL/WA/Roza2/Prosser/Malbec/param_mtl__1766584337
python3 -m trainers.train_model_rtmc --config rtmc/finetune_pheno/pheno_param_10_10_stl --seed 1 --cultivar Malbec --rnn_fpath _runs/RTMC/Phenology/ParamMTL10/STL/WA/Roza2/Prosser/Malbec/param_mtl__1766585712
python3 -m trainers.train_model_rtmc --config rtmc/finetune_pheno/pheno_param_10_10_stl --seed 2 --cultivar Malbec --rnn_fpath _runs/RTMC/Phenology/ParamMTL10/STL/WA/Roza2/Prosser/Malbec/param_mtl__1766587091
python3 -m trainers.train_model_rtmc --config rtmc/finetune_pheno/pheno_param_10_10_stl --seed 3 --cultivar Malbec --rnn_fpath _runs/RTMC/Phenology/ParamMTL10/STL/WA/Roza2/Prosser/Malbec/param_mtl__1766588454
python3 -m trainers.train_model_rtmc --config rtmc/finetune_pheno/pheno_param_10_10_stl --seed 4 --cultivar Malbec --rnn_fpath _runs/RTMC/Phenology/ParamMTL10/STL/WA/Roza2/Prosser/Malbec/param_mtl__1766589784
 
python3 -m trainers.train_model_rtmc --config rtmc/finetune_pheno/pheno_param_10_10_stl --seed 0 --cultivar Melon --rnn_fpath _runs/RTMC/Phenology/ParamMTL10/STL/WA/Roza2/Prosser/Melon/param_mtl__1766591152
python3 -m trainers.train_model_rtmc --config rtmc/finetune_pheno/pheno_param_10_10_stl --seed 1 --cultivar Melon --rnn_fpath _runs/RTMC/Phenology/ParamMTL10/STL/WA/Roza2/Prosser/Melon/param_mtl__1766592158
python3 -m trainers.train_model_rtmc --config rtmc/finetune_pheno/pheno_param_10_10_stl --seed 2 --cultivar Melon --rnn_fpath _runs/RTMC/Phenology/ParamMTL10/STL/WA/Roza2/Prosser/Melon/param_mtl__1766593172
python3 -m trainers.train_model_rtmc --config rtmc/finetune_pheno/pheno_param_10_10_stl --seed 3 --cultivar Melon --rnn_fpath _runs/RTMC/Phenology/ParamMTL10/STL/WA/Roza2/Prosser/Melon/param_mtl__1766594178
python3 -m trainers.train_model_rtmc --config rtmc/finetune_pheno/pheno_param_10_10_stl --seed 4 --cultivar Melon --rnn_fpath _runs/RTMC/Phenology/ParamMTL10/STL/WA/Roza2/Prosser/Melon/param_mtl__1766595185
 
python3 -m trainers.train_model_rtmc --config rtmc/finetune_pheno/pheno_param_10_10_stl --seed 0 --cultivar Merlot --rnn_fpath _runs/RTMC/Phenology/ParamMTL10/STL/WA/Roza2/Prosser/Merlot/param_mtl__1766596192
python3 -m trainers.train_model_rtmc --config rtmc/finetune_pheno/pheno_param_10_10_stl --seed 1 --cultivar Merlot --rnn_fpath _runs/RTMC/Phenology/ParamMTL10/STL/WA/Roza2/Prosser/Merlot/param_mtl__1766597565
python3 -m trainers.train_model_rtmc --config rtmc/finetune_pheno/pheno_param_10_10_stl --seed 2 --cultivar Merlot --rnn_fpath _runs/RTMC/Phenology/ParamMTL10/STL/WA/Roza2/Prosser/Merlot/param_mtl__1766598929
python3 -m trainers.train_model_rtmc --config rtmc/finetune_pheno/pheno_param_10_10_stl --seed 3 --cultivar Merlot --rnn_fpath _runs/RTMC/Phenology/ParamMTL10/STL/WA/Roza2/Prosser/Merlot/param_mtl__1766600299
python3 -m trainers.train_model_rtmc --config rtmc/finetune_pheno/pheno_param_10_10_stl --seed 4 --cultivar Merlot --rnn_fpath _runs/RTMC/Phenology/ParamMTL10/STL/WA/Roza2/Prosser/Merlot/param_mtl__1766601668
 
python3 -m trainers.train_model_rtmc --config rtmc/finetune_pheno/pheno_param_10_10_stl --seed 0 --cultivar Mourvedre --rnn_fpath _runs/RTMC/Phenology/ParamMTL10/STL/WA/Roza2/Prosser/Mourvedre/param_mtl__1766603038
python3 -m trainers.train_model_rtmc --config rtmc/finetune_pheno/pheno_param_10_10_stl --seed 1 --cultivar Mourvedre --rnn_fpath _runs/RTMC/Phenology/ParamMTL10/STL/WA/Roza2/Prosser/Mourvedre/param_mtl__1766604043
python3 -m trainers.train_model_rtmc --config rtmc/finetune_pheno/pheno_param_10_10_stl --seed 2 --cultivar Mourvedre --rnn_fpath _runs/RTMC/Phenology/ParamMTL10/STL/WA/Roza2/Prosser/Mourvedre/param_mtl__1766605058
python3 -m trainers.train_model_rtmc --config rtmc/finetune_pheno/pheno_param_10_10_stl --seed 3 --cultivar Mourvedre --rnn_fpath _runs/RTMC/Phenology/ParamMTL10/STL/WA/Roza2/Prosser/Mourvedre/param_mtl__1766606073
python3 -m trainers.train_model_rtmc --config rtmc/finetune_pheno/pheno_param_10_10_stl --seed 4 --cultivar Mourvedre --rnn_fpath _runs/RTMC/Phenology/ParamMTL10/STL/WA/Roza2/Prosser/Mourvedre/param_mtl__1766607088
 
python3 -m trainers.train_model_rtmc --config rtmc/finetune_pheno/pheno_param_10_10_stl --seed 0 --cultivar Muscat_Blanc --rnn_fpath _runs/RTMC/Phenology/ParamMTL10/STL/WA/Roza2/Prosser/Muscat_Blanc/param_mtl__1766608089
python3 -m trainers.train_model_rtmc --config rtmc/finetune_pheno/pheno_param_10_10_stl --seed 1 --cultivar Muscat_Blanc --rnn_fpath _runs/RTMC/Phenology/ParamMTL10/STL/WA/Roza2/Prosser/Muscat_Blanc/param_mtl__1766609472
python3 -m trainers.train_model_rtmc --config rtmc/finetune_pheno/pheno_param_10_10_stl --seed 2 --cultivar Muscat_Blanc --rnn_fpath _runs/RTMC/Phenology/ParamMTL10/STL/WA/Roza2/Prosser/Muscat_Blanc/param_mtl__1766610853
python3 -m trainers.train_model_rtmc --config rtmc/finetune_pheno/pheno_param_10_10_stl --seed 3 --cultivar Muscat_Blanc --rnn_fpath _runs/RTMC/Phenology/ParamMTL10/STL/WA/Roza2/Prosser/Muscat_Blanc/param_mtl__1766612233
python3 -m trainers.train_model_rtmc --config rtmc/finetune_pheno/pheno_param_10_10_stl --seed 4 --cultivar Muscat_Blanc --rnn_fpath _runs/RTMC/Phenology/ParamMTL10/STL/WA/Roza2/Prosser/Muscat_Blanc/param_mtl__1766613621
 
python3 -m trainers.train_model_rtmc --config rtmc/finetune_pheno/pheno_param_10_10_stl --seed 0 --cultivar Nebbiolo --rnn_fpath _runs/RTMC/Phenology/ParamMTL10/STL/WA/Roza2/Prosser/Nebbiolo/param_mtl__1766614998
python3 -m trainers.train_model_rtmc --config rtmc/finetune_pheno/pheno_param_10_10_stl --seed 1 --cultivar Nebbiolo --rnn_fpath _runs/RTMC/Phenology/ParamMTL10/STL/WA/Roza2/Prosser/Nebbiolo/param_mtl__1766616010
python3 -m trainers.train_model_rtmc --config rtmc/finetune_pheno/pheno_param_10_10_stl --seed 2 --cultivar Nebbiolo --rnn_fpath _runs/RTMC/Phenology/ParamMTL10/STL/WA/Roza2/Prosser/Nebbiolo/param_mtl__1766617018
python3 -m trainers.train_model_rtmc --config rtmc/finetune_pheno/pheno_param_10_10_stl --seed 3 --cultivar Nebbiolo --rnn_fpath _runs/RTMC/Phenology/ParamMTL10/STL/WA/Roza2/Prosser/Nebbiolo/param_mtl__1766618035
python3 -m trainers.train_model_rtmc --config rtmc/finetune_pheno/pheno_param_10_10_stl --seed 4 --cultivar Nebbiolo --rnn_fpath _runs/RTMC/Phenology/ParamMTL10/STL/WA/Roza2/Prosser/Nebbiolo/param_mtl__1766619049
 
python3 -m trainers.train_model_rtmc --config rtmc/finetune_pheno/pheno_param_10_10_stl --seed 0 --cultivar Petit_Verdot --rnn_fpath _runs/RTMC/Phenology/ParamMTL10/STL/WA/Roza2/Prosser/Petit_Verdot/param_mtl__1766620066
python3 -m trainers.train_model_rtmc --config rtmc/finetune_pheno/pheno_param_10_10_stl --seed 1 --cultivar Petit_Verdot --rnn_fpath _runs/RTMC/Phenology/ParamMTL10/STL/WA/Roza2/Prosser/Petit_Verdot/param_mtl__1766621089
python3 -m trainers.train_model_rtmc --config rtmc/finetune_pheno/pheno_param_10_10_stl --seed 2 --cultivar Petit_Verdot --rnn_fpath _runs/RTMC/Phenology/ParamMTL10/STL/WA/Roza2/Prosser/Petit_Verdot/param_mtl__1766622122
python3 -m trainers.train_model_rtmc --config rtmc/finetune_pheno/pheno_param_10_10_stl --seed 3 --cultivar Petit_Verdot --rnn_fpath _runs/RTMC/Phenology/ParamMTL10/STL/WA/Roza2/Prosser/Petit_Verdot/param_mtl__1766623159
python3 -m trainers.train_model_rtmc --config rtmc/finetune_pheno/pheno_param_10_10_stl --seed 4 --cultivar Petit_Verdot --rnn_fpath _runs/RTMC/Phenology/ParamMTL10/STL/WA/Roza2/Prosser/Petit_Verdot/param_mtl__1766624195
 
python3 -m trainers.train_model_rtmc --config rtmc/finetune_pheno/pheno_param_10_10_stl --seed 0 --cultivar Pinot_Blanc --rnn_fpath _runs/RTMC/Phenology/ParamMTL10/STL/WA/Roza2/Prosser/Pinot_Blanc/param_mtl__1766625225
python3 -m trainers.train_model_rtmc --config rtmc/finetune_pheno/pheno_param_10_10_stl --seed 1 --cultivar Pinot_Blanc --rnn_fpath _runs/RTMC/Phenology/ParamMTL10/STL/WA/Roza2/Prosser/Pinot_Blanc/param_mtl__1766626254
python3 -m trainers.train_model_rtmc --config rtmc/finetune_pheno/pheno_param_10_10_stl --seed 2 --cultivar Pinot_Blanc --rnn_fpath _runs/RTMC/Phenology/ParamMTL10/STL/WA/Roza2/Prosser/Pinot_Blanc/param_mtl__1766627275
python3 -m trainers.train_model_rtmc --config rtmc/finetune_pheno/pheno_param_10_10_stl --seed 3 --cultivar Pinot_Blanc --rnn_fpath _runs/RTMC/Phenology/ParamMTL10/STL/WA/Roza2/Prosser/Pinot_Blanc/param_mtl__1766628300
python3 -m trainers.train_model_rtmc --config rtmc/finetune_pheno/pheno_param_10_10_stl --seed 4 --cultivar Pinot_Blanc --rnn_fpath _runs/RTMC/Phenology/ParamMTL10/STL/WA/Roza2/Prosser/Pinot_Blanc/param_mtl__1766629329
 
python3 -m trainers.train_model_rtmc --config rtmc/finetune_pheno/pheno_param_10_10_stl --seed 0 --cultivar Pinot_Gris --rnn_fpath _runs/RTMC/Phenology/ParamMTL10/STL/WA/Roza2/Prosser/Pinot_Gris/param_mtl__1766630353
python3 -m trainers.train_model_rtmc --config rtmc/finetune_pheno/pheno_param_10_10_stl --seed 1 --cultivar Pinot_Gris --rnn_fpath _runs/RTMC/Phenology/ParamMTL10/STL/WA/Roza2/Prosser/Pinot_Gris/param_mtl__1766631725
python3 -m trainers.train_model_rtmc --config rtmc/finetune_pheno/pheno_param_10_10_stl --seed 2 --cultivar Pinot_Gris --rnn_fpath _runs/RTMC/Phenology/ParamMTL10/STL/WA/Roza2/Prosser/Pinot_Gris/param_mtl__1766633106
python3 -m trainers.train_model_rtmc --config rtmc/finetune_pheno/pheno_param_10_10_stl --seed 3 --cultivar Pinot_Gris --rnn_fpath _runs/RTMC/Phenology/ParamMTL10/STL/WA/Roza2/Prosser/Pinot_Gris/param_mtl__1766634511
python3 -m trainers.train_model_rtmc --config rtmc/finetune_pheno/pheno_param_10_10_stl --seed 4 --cultivar Pinot_Gris --rnn_fpath _runs/RTMC/Phenology/ParamMTL10/STL/WA/Roza2/Prosser/Pinot_Gris/param_mtl__1766635913
 
python3 -m trainers.train_model_rtmc --config rtmc/finetune_pheno/pheno_param_10_10_stl --seed 0 --cultivar Pinot_Noir --rnn_fpath _runs/RTMC/Phenology/ParamMTL10/STL/WA/Roza2/Prosser/Pinot_Noir/param_mtl__1766637314
python3 -m trainers.train_model_rtmc --config rtmc/finetune_pheno/pheno_param_10_10_stl --seed 1 --cultivar Pinot_Noir --rnn_fpath _runs/RTMC/Phenology/ParamMTL10/STL/WA/Roza2/Prosser/Pinot_Noir/param_mtl__1766638714
python3 -m trainers.train_model_rtmc --config rtmc/finetune_pheno/pheno_param_10_10_stl --seed 2 --cultivar Pinot_Noir --rnn_fpath _runs/RTMC/Phenology/ParamMTL10/STL/WA/Roza2/Prosser/Pinot_Noir/param_mtl__1766640106
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

