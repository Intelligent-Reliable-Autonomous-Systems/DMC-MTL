#!/bin/bash
#SBATCH -J ch_param_1_1_stl
#SBATCH -o output/ch_param_1_1_stl.out
#SBATCH -e output/ch_param_1_1_stl.err
#SBATCH -p eecs,share,gpu
#SBATCH -t 2-12:00:00
#SBATCH --gres=gpu:1

python3 -m trainers.train_model_rtmc --config rtmc/finetune_ch/ch_param_1_1_stl --seed 0 --cultivar Barbera --rnn_fpath _runs/RTMC/ColdHardiness/ParamMTL1/STL/WA/Roza2/Prosser/Barbera/param_mtl__1766617940
python3 -m trainers.train_model_rtmc --config rtmc/finetune_ch/ch_param_1_1_stl --seed 1 --cultivar Barbera --rnn_fpath _runs/RTMC/ColdHardiness/ParamMTL1/STL/WA/Roza2/Prosser/Barbera/param_mtl__1766618632
python3 -m trainers.train_model_rtmc --config rtmc/finetune_ch/ch_param_1_1_stl --seed 2 --cultivar Barbera --rnn_fpath _runs/RTMC/ColdHardiness/ParamMTL1/STL/WA/Roza2/Prosser/Barbera/param_mtl__1766619327
python3 -m trainers.train_model_rtmc --config rtmc/finetune_ch/ch_param_1_1_stl --seed 3 --cultivar Barbera --rnn_fpath _runs/RTMC/ColdHardiness/ParamMTL1/STL/WA/Roza2/Prosser/Barbera/param_mtl__1766620019
python3 -m trainers.train_model_rtmc --config rtmc/finetune_ch/ch_param_1_1_stl --seed 4 --cultivar Barbera --rnn_fpath _runs/RTMC/ColdHardiness/ParamMTL1/STL/WA/Roza2/Prosser/Barbera/param_mtl__1766620715
 
python3 -m trainers.train_model_rtmc --config rtmc/finetune_ch/ch_param_1_1_stl --seed 0 --cultivar Cabernet_Franc --rnn_fpath _runs/RTMC/ColdHardiness/ParamMTL1/STL/WA/Roza2/Prosser/Cabernet_Franc/param_mtl__1766621408
python3 -m trainers.train_model_rtmc --config rtmc/finetune_ch/ch_param_1_1_stl --seed 1 --cultivar Cabernet_Franc --rnn_fpath _runs/RTMC/ColdHardiness/ParamMTL1/STL/WA/Roza2/Prosser/Cabernet_Franc/param_mtl__1766622114
python3 -m trainers.train_model_rtmc --config rtmc/finetune_ch/ch_param_1_1_stl --seed 2 --cultivar Cabernet_Franc --rnn_fpath _runs/RTMC/ColdHardiness/ParamMTL1/STL/WA/Roza2/Prosser/Cabernet_Franc/param_mtl__1766622822
python3 -m trainers.train_model_rtmc --config rtmc/finetune_ch/ch_param_1_1_stl --seed 3 --cultivar Cabernet_Franc --rnn_fpath _runs/RTMC/ColdHardiness/ParamMTL1/STL/WA/Roza2/Prosser/Cabernet_Franc/param_mtl__1766623528
python3 -m trainers.train_model_rtmc --config rtmc/finetune_ch/ch_param_1_1_stl --seed 4 --cultivar Cabernet_Franc --rnn_fpath _runs/RTMC/ColdHardiness/ParamMTL1/STL/WA/Roza2/Prosser/Cabernet_Franc/param_mtl__1766624235
 
python3 -m trainers.train_model_rtmc --config rtmc/finetune_ch/ch_param_1_1_stl --seed 0 --cultivar Cabernet_Sauvignon --rnn_fpath _runs/RTMC/ColdHardiness/ParamMTL1/STL/WA/Roza2/Prosser/Cabernet_Sauvignon/param_mtl__1766624937
python3 -m trainers.train_model_rtmc --config rtmc/finetune_ch/ch_param_1_1_stl --seed 1 --cultivar Cabernet_Sauvignon --rnn_fpath _runs/RTMC/ColdHardiness/ParamMTL1/STL/WA/Roza2/Prosser/Cabernet_Sauvignon/param_mtl__1766625639
python3 -m trainers.train_model_rtmc --config rtmc/finetune_ch/ch_param_1_1_stl --seed 2 --cultivar Cabernet_Sauvignon --rnn_fpath _runs/RTMC/ColdHardiness/ParamMTL1/STL/WA/Roza2/Prosser/Cabernet_Sauvignon/param_mtl__1766626340
python3 -m trainers.train_model_rtmc --config rtmc/finetune_ch/ch_param_1_1_stl --seed 3 --cultivar Cabernet_Sauvignon --rnn_fpath _runs/RTMC/ColdHardiness/ParamMTL1/STL/WA/Roza2/Prosser/Cabernet_Sauvignon/param_mtl__1766627040
python3 -m trainers.train_model_rtmc --config rtmc/finetune_ch/ch_param_1_1_stl --seed 4 --cultivar Cabernet_Sauvignon --rnn_fpath _runs/RTMC/ColdHardiness/ParamMTL1/STL/WA/Roza2/Prosser/Cabernet_Sauvignon/param_mtl__1766627743
 
python3 -m trainers.train_model_rtmc --config rtmc/finetune_ch/ch_param_1_1_stl --seed 0 --cultivar Chardonnay --rnn_fpath _runs/RTMC/ColdHardiness/ParamMTL1/STL/WA/Roza2/Prosser/Chardonnay/param_mtl__1766628445
python3 -m trainers.train_model_rtmc --config rtmc/finetune_ch/ch_param_1_1_stl --seed 1 --cultivar Chardonnay --rnn_fpath _runs/RTMC/ColdHardiness/ParamMTL1/STL/WA/Roza2/Prosser/Chardonnay/param_mtl__1766629148
python3 -m trainers.train_model_rtmc --config rtmc/finetune_ch/ch_param_1_1_stl --seed 2 --cultivar Chardonnay --rnn_fpath _runs/RTMC/ColdHardiness/ParamMTL1/STL/WA/Roza2/Prosser/Chardonnay/param_mtl__1766629848
python3 -m trainers.train_model_rtmc --config rtmc/finetune_ch/ch_param_1_1_stl --seed 3 --cultivar Chardonnay --rnn_fpath _runs/RTMC/ColdHardiness/ParamMTL1/STL/WA/Roza2/Prosser/Chardonnay/param_mtl__1766630548
python3 -m trainers.train_model_rtmc --config rtmc/finetune_ch/ch_param_1_1_stl --seed 4 --cultivar Chardonnay --rnn_fpath _runs/RTMC/ColdHardiness/ParamMTL1/STL/WA/Roza2/Prosser/Chardonnay/param_mtl__1766631250
 
python3 -m trainers.train_model_rtmc --config rtmc/finetune_ch/ch_param_1_1_stl --seed 0 --cultivar Chenin_Blanc --rnn_fpath _runs/RTMC/ColdHardiness/ParamMTL1/STL/WA/Roza2/Prosser/Chenin_Blanc/param_mtl__1766631951
python3 -m trainers.train_model_rtmc --config rtmc/finetune_ch/ch_param_1_1_stl --seed 1 --cultivar Chenin_Blanc --rnn_fpath _runs/RTMC/ColdHardiness/ParamMTL1/STL/WA/Roza2/Prosser/Chenin_Blanc/param_mtl__1766632649
python3 -m trainers.train_model_rtmc --config rtmc/finetune_ch/ch_param_1_1_stl --seed 2 --cultivar Chenin_Blanc --rnn_fpath _runs/RTMC/ColdHardiness/ParamMTL1/STL/WA/Roza2/Prosser/Chenin_Blanc/param_mtl__1766633353
python3 -m trainers.train_model_rtmc --config rtmc/finetune_ch/ch_param_1_1_stl --seed 3 --cultivar Chenin_Blanc --rnn_fpath _runs/RTMC/ColdHardiness/ParamMTL1/STL/WA/Roza2/Prosser/Chenin_Blanc/param_mtl__1766634050
python3 -m trainers.train_model_rtmc --config rtmc/finetune_ch/ch_param_1_1_stl --seed 4 --cultivar Chenin_Blanc --rnn_fpath _runs/RTMC/ColdHardiness/ParamMTL1/STL/WA/Roza2/Prosser/Chenin_Blanc/param_mtl__1766634749
 
python3 -m trainers.train_model_rtmc --config rtmc/finetune_ch/ch_param_1_1_stl --seed 0 --cultivar Concord --rnn_fpath _runs/RTMC/ColdHardiness/ParamMTL1/STL/WA/Roza2/Prosser/Concord/param_mtl__1766635452
python3 -m trainers.train_model_rtmc --config rtmc/finetune_ch/ch_param_1_1_stl --seed 1 --cultivar Concord --rnn_fpath _runs/RTMC/ColdHardiness/ParamMTL1/STL/WA/Roza2/Prosser/Concord/param_mtl__1766636154
python3 -m trainers.train_model_rtmc --config rtmc/finetune_ch/ch_param_1_1_stl --seed 2 --cultivar Concord --rnn_fpath _runs/RTMC/ColdHardiness/ParamMTL1/STL/WA/Roza2/Prosser/Concord/param_mtl__1766636860
python3 -m trainers.train_model_rtmc --config rtmc/finetune_ch/ch_param_1_1_stl --seed 3 --cultivar Concord --rnn_fpath _runs/RTMC/ColdHardiness/ParamMTL1/STL/WA/Roza2/Prosser/Concord/param_mtl__1766637563
python3 -m trainers.train_model_rtmc --config rtmc/finetune_ch/ch_param_1_1_stl --seed 4 --cultivar Concord --rnn_fpath _runs/RTMC/ColdHardiness/ParamMTL1/STL/WA/Roza2/Prosser/Concord/param_mtl__1766638266
 
python3 -m trainers.train_model_rtmc --config rtmc/finetune_ch/ch_param_1_1_stl --seed 0 --cultivar Gewurztraminer --rnn_fpath _runs/RTMC/ColdHardiness/ParamMTL1/STL/WA/Roza2/Prosser/Gewurztraminer/param_mtl__1766638966
python3 -m trainers.train_model_rtmc --config rtmc/finetune_ch/ch_param_1_1_stl --seed 1 --cultivar Gewurztraminer --rnn_fpath _runs/RTMC/ColdHardiness/ParamMTL1/STL/WA/Roza2/Prosser/Gewurztraminer/param_mtl__1766639664
python3 -m trainers.train_model_rtmc --config rtmc/finetune_ch/ch_param_1_1_stl --seed 2 --cultivar Gewurztraminer --rnn_fpath _runs/RTMC/ColdHardiness/ParamMTL1/STL/WA/Roza2/Prosser/Gewurztraminer/param_mtl__1766640361
python3 -m trainers.train_model_rtmc --config rtmc/finetune_ch/ch_param_1_1_stl --seed 3 --cultivar Gewurztraminer --rnn_fpath _runs/RTMC/ColdHardiness/ParamMTL1/STL/WA/Roza2/Prosser/Gewurztraminer/param_mtl__1766641064
python3 -m trainers.train_model_rtmc --config rtmc/finetune_ch/ch_param_1_1_stl --seed 4 --cultivar Gewurztraminer --rnn_fpath _runs/RTMC/ColdHardiness/ParamMTL1/STL/WA/Roza2/Prosser/Gewurztraminer/param_mtl__1766641768
 
python3 -m trainers.train_model_rtmc --config rtmc/finetune_ch/ch_param_1_1_stl --seed 0 --cultivar Lemberger --rnn_fpath _runs/RTMC/ColdHardiness/ParamMTL1/STL/WA/Roza2/Prosser/Lemberger/param_mtl__1766642467
python3 -m trainers.train_model_rtmc --config rtmc/finetune_ch/ch_param_1_1_stl --seed 1 --cultivar Lemberger --rnn_fpath _runs/RTMC/ColdHardiness/ParamMTL1/STL/WA/Roza2/Prosser/Lemberger/param_mtl__1766643173
python3 -m trainers.train_model_rtmc --config rtmc/finetune_ch/ch_param_1_1_stl --seed 2 --cultivar Lemberger --rnn_fpath _runs/RTMC/ColdHardiness/ParamMTL1/STL/WA/Roza2/Prosser/Lemberger/param_mtl__1766643873
python3 -m trainers.train_model_rtmc --config rtmc/finetune_ch/ch_param_1_1_stl --seed 3 --cultivar Lemberger --rnn_fpath _runs/RTMC/ColdHardiness/ParamMTL1/STL/WA/Roza2/Prosser/Lemberger/param_mtl__1766644577
python3 -m trainers.train_model_rtmc --config rtmc/finetune_ch/ch_param_1_1_stl --seed 4 --cultivar Lemberger --rnn_fpath _runs/RTMC/ColdHardiness/ParamMTL1/STL/WA/Roza2/Prosser/Lemberger/param_mtl__1766645278
 
python3 -m trainers.train_model_rtmc --config rtmc/finetune_ch/ch_param_1_1_stl --seed 0 --cultivar Malbec --rnn_fpath _runs/RTMC/ColdHardiness/ParamMTL1/STL/WA/Roza2/Prosser/Malbec/param_mtl__1766645975
python3 -m trainers.train_model_rtmc --config rtmc/finetune_ch/ch_param_1_1_stl --seed 1 --cultivar Malbec --rnn_fpath _runs/RTMC/ColdHardiness/ParamMTL1/STL/WA/Roza2/Prosser/Malbec/param_mtl__1766646676
python3 -m trainers.train_model_rtmc --config rtmc/finetune_ch/ch_param_1_1_stl --seed 2 --cultivar Malbec --rnn_fpath _runs/RTMC/ColdHardiness/ParamMTL1/STL/WA/Roza2/Prosser/Malbec/param_mtl__1766647377
python3 -m trainers.train_model_rtmc --config rtmc/finetune_ch/ch_param_1_1_stl --seed 3 --cultivar Malbec --rnn_fpath _runs/RTMC/ColdHardiness/ParamMTL1/STL/WA/Roza2/Prosser/Malbec/param_mtl__1766648073
python3 -m trainers.train_model_rtmc --config rtmc/finetune_ch/ch_param_1_1_stl --seed 4 --cultivar Malbec --rnn_fpath _runs/RTMC/ColdHardiness/ParamMTL1/STL/WA/Roza2/Prosser/Malbec/param_mtl__1766648769
 
python3 -m trainers.train_model_rtmc --config rtmc/finetune_ch/ch_param_1_1_stl --seed 0 --cultivar Merlot --rnn_fpath _runs/RTMC/ColdHardiness/ParamMTL1/STL/WA/Roza2/Prosser/Merlot/param_mtl__1766649469
python3 -m trainers.train_model_rtmc --config rtmc/finetune_ch/ch_param_1_1_stl --seed 1 --cultivar Merlot --rnn_fpath _runs/RTMC/ColdHardiness/ParamMTL1/STL/WA/Roza2/Prosser/Merlot/param_mtl__1766650170
python3 -m trainers.train_model_rtmc --config rtmc/finetune_ch/ch_param_1_1_stl --seed 2 --cultivar Merlot --rnn_fpath _runs/RTMC/ColdHardiness/ParamMTL1/STL/WA/Roza2/Prosser/Merlot/param_mtl__1766650874
python3 -m trainers.train_model_rtmc --config rtmc/finetune_ch/ch_param_1_1_stl --seed 3 --cultivar Merlot --rnn_fpath _runs/RTMC/ColdHardiness/ParamMTL1/STL/WA/Roza2/Prosser/Merlot/param_mtl__1766651577
python3 -m trainers.train_model_rtmc --config rtmc/finetune_ch/ch_param_1_1_stl --seed 4 --cultivar Merlot --rnn_fpath _runs/RTMC/ColdHardiness/ParamMTL1/STL/WA/Roza2/Prosser/Merlot/param_mtl__1766652279
 
python3 -m trainers.train_model_rtmc --config rtmc/finetune_ch/ch_param_1_1_stl --seed 0 --cultivar Mourvedre --rnn_fpath _runs/RTMC/ColdHardiness/ParamMTL1/STL/WA/Roza2/Prosser/Mourvedre/param_mtl__1766652976
python3 -m trainers.train_model_rtmc --config rtmc/finetune_ch/ch_param_1_1_stl --seed 1 --cultivar Mourvedre --rnn_fpath _runs/RTMC/ColdHardiness/ParamMTL1/STL/WA/Roza2/Prosser/Mourvedre/param_mtl__1766653670
python3 -m trainers.train_model_rtmc --config rtmc/finetune_ch/ch_param_1_1_stl --seed 2 --cultivar Mourvedre --rnn_fpath _runs/RTMC/ColdHardiness/ParamMTL1/STL/WA/Roza2/Prosser/Mourvedre/param_mtl__1766654365
python3 -m trainers.train_model_rtmc --config rtmc/finetune_ch/ch_param_1_1_stl --seed 3 --cultivar Mourvedre --rnn_fpath _runs/RTMC/ColdHardiness/ParamMTL1/STL/WA/Roza2/Prosser/Mourvedre/param_mtl__1766655058
python3 -m trainers.train_model_rtmc --config rtmc/finetune_ch/ch_param_1_1_stl --seed 4 --cultivar Mourvedre --rnn_fpath _runs/RTMC/ColdHardiness/ParamMTL1/STL/WA/Roza2/Prosser/Mourvedre/param_mtl__1766655749
 
python3 -m trainers.train_model_rtmc --config rtmc/finetune_ch/ch_param_1_1_stl --seed 0 --cultivar Nebbiolo --rnn_fpath _runs/RTMC/ColdHardiness/ParamMTL1/STL/WA/Roza2/Prosser/Nebbiolo/param_mtl__1766656443
python3 -m trainers.train_model_rtmc --config rtmc/finetune_ch/ch_param_1_1_stl --seed 1 --cultivar Nebbiolo --rnn_fpath _runs/RTMC/ColdHardiness/ParamMTL1/STL/WA/Roza2/Prosser/Nebbiolo/param_mtl__1766657140
python3 -m trainers.train_model_rtmc --config rtmc/finetune_ch/ch_param_1_1_stl --seed 2 --cultivar Nebbiolo --rnn_fpath _runs/RTMC/ColdHardiness/ParamMTL1/STL/WA/Roza2/Prosser/Nebbiolo/param_mtl__1766657829
python3 -m trainers.train_model_rtmc --config rtmc/finetune_ch/ch_param_1_1_stl --seed 3 --cultivar Nebbiolo --rnn_fpath _runs/RTMC/ColdHardiness/ParamMTL1/STL/WA/Roza2/Prosser/Nebbiolo/param_mtl__1766658522
python3 -m trainers.train_model_rtmc --config rtmc/finetune_ch/ch_param_1_1_stl --seed 4 --cultivar Nebbiolo --rnn_fpath _runs/RTMC/ColdHardiness/ParamMTL1/STL/WA/Roza2/Prosser/Nebbiolo/param_mtl__1766659217
 
python3 -m trainers.train_model_rtmc --config rtmc/finetune_ch/ch_param_1_1_stl --seed 0 --cultivar Pinot_Gris --rnn_fpath _runs/RTMC/ColdHardiness/ParamMTL1/STL/WA/Roza2/Prosser/Pinot_Gris/param_mtl__1766659907
python3 -m trainers.train_model_rtmc --config rtmc/finetune_ch/ch_param_1_1_stl --seed 1 --cultivar Pinot_Gris --rnn_fpath _runs/RTMC/ColdHardiness/ParamMTL1/STL/WA/Roza2/Prosser/Pinot_Gris/param_mtl__1766660599
python3 -m trainers.train_model_rtmc --config rtmc/finetune_ch/ch_param_1_1_stl --seed 2 --cultivar Pinot_Gris --rnn_fpath _runs/RTMC/ColdHardiness/ParamMTL1/STL/WA/Roza2/Prosser/Pinot_Gris/param_mtl__1766661290
python3 -m trainers.train_model_rtmc --config rtmc/finetune_ch/ch_param_1_1_stl --seed 3 --cultivar Pinot_Gris --rnn_fpath _runs/RTMC/ColdHardiness/ParamMTL1/STL/WA/Roza2/Prosser/Pinot_Gris/param_mtl__1766661979
python3 -m trainers.train_model_rtmc --config rtmc/finetune_ch/ch_param_1_1_stl --seed 4 --cultivar Pinot_Gris --rnn_fpath _runs/RTMC/ColdHardiness/ParamMTL1/STL/WA/Roza2/Prosser/Pinot_Gris/param_mtl__1766662673
 
python3 -m trainers.train_model_rtmc --config rtmc/finetune_ch/ch_param_1_1_stl --seed 0 --cultivar Riesling --rnn_fpath _runs/RTMC/ColdHardiness/ParamMTL1/STL/WA/Roza2/Prosser/Riesling/param_mtl__1766663361
python3 -m trainers.train_model_rtmc --config rtmc/finetune_ch/ch_param_1_1_stl --seed 1 --cultivar Riesling --rnn_fpath _runs/RTMC/ColdHardiness/ParamMTL1/STL/WA/Roza2/Prosser/Riesling/param_mtl__1766664052
python3 -m trainers.train_model_rtmc --config rtmc/finetune_ch/ch_param_1_1_stl --seed 2 --cultivar Riesling --rnn_fpath _runs/RTMC/ColdHardiness/ParamMTL1/STL/WA/Roza2/Prosser/Riesling/param_mtl__1766664744
python3 -m trainers.train_model_rtmc --config rtmc/finetune_ch/ch_param_1_1_stl --seed 3 --cultivar Riesling --rnn_fpath _runs/RTMC/ColdHardiness/ParamMTL1/STL/WA/Roza2/Prosser/Riesling/param_mtl__1766665436
python3 -m trainers.train_model_rtmc --config rtmc/finetune_ch/ch_param_1_1_stl --seed 4 --cultivar Riesling --rnn_fpath _runs/RTMC/ColdHardiness/ParamMTL1/STL/WA/Roza2/Prosser/Riesling/param_mtl__1766666127
 
python3 -m trainers.train_model_rtmc --config rtmc/finetune_ch/ch_param_1_1_stl --seed 0 --cultivar Sangiovese --rnn_fpath _runs/RTMC/ColdHardiness/ParamMTL1/STL/WA/Roza2/Prosser/Sangiovese/param_mtl__1766666818
python3 -m trainers.train_model_rtmc --config rtmc/finetune_ch/ch_param_1_1_stl --seed 1 --cultivar Sangiovese --rnn_fpath _runs/RTMC/ColdHardiness/ParamMTL1/STL/WA/Roza2/Prosser/Sangiovese/param_mtl__1766667511
python3 -m trainers.train_model_rtmc --config rtmc/finetune_ch/ch_param_1_1_stl --seed 2 --cultivar Sangiovese --rnn_fpath _runs/RTMC/ColdHardiness/ParamMTL1/STL/WA/Roza2/Prosser/Sangiovese/param_mtl__1766668203
python3 -m trainers.train_model_rtmc --config rtmc/finetune_ch/ch_param_1_1_stl --seed 3 --cultivar Sangiovese --rnn_fpath _runs/RTMC/ColdHardiness/ParamMTL1/STL/WA/Roza2/Prosser/Sangiovese/param_mtl__1766668891
python3 -m trainers.train_model_rtmc --config rtmc/finetune_ch/ch_param_1_1_stl --seed 4 --cultivar Sangiovese --rnn_fpath _runs/RTMC/ColdHardiness/ParamMTL1/STL/WA/Roza2/Prosser/Sangiovese/param_mtl__1766669583
 
python3 -m trainers.train_model_rtmc --config rtmc/finetune_ch/ch_param_1_1_stl --seed 0 --cultivar Sauvignon_Blanc --rnn_fpath _runs/RTMC/ColdHardiness/ParamMTL1/STL/WA/Roza2/Prosser/Sauvignon_Blanc/param_mtl__1766670277
python3 -m trainers.train_model_rtmc --config rtmc/finetune_ch/ch_param_1_1_stl --seed 1 --cultivar Sauvignon_Blanc --rnn_fpath _runs/RTMC/ColdHardiness/ParamMTL1/STL/WA/Roza2/Prosser/Sauvignon_Blanc/param_mtl__1766670969
python3 -m trainers.train_model_rtmc --config rtmc/finetune_ch/ch_param_1_1_stl --seed 2 --cultivar Sauvignon_Blanc --rnn_fpath _runs/RTMC/ColdHardiness/ParamMTL1/STL/WA/Roza2/Prosser/Sauvignon_Blanc/param_mtl__1766671659
python3 -m trainers.train_model_rtmc --config rtmc/finetune_ch/ch_param_1_1_stl --seed 3 --cultivar Sauvignon_Blanc --rnn_fpath _runs/RTMC/ColdHardiness/ParamMTL1/STL/WA/Roza2/Prosser/Sauvignon_Blanc/param_mtl__1766672350
python3 -m trainers.train_model_rtmc --config rtmc/finetune_ch/ch_param_1_1_stl --seed 4 --cultivar Sauvignon_Blanc --rnn_fpath _runs/RTMC/ColdHardiness/ParamMTL1/STL/WA/Roza2/Prosser/Sauvignon_Blanc/param_mtl__1766673036
 
python3 -m trainers.train_model_rtmc --config rtmc/finetune_ch/ch_param_1_1_stl --seed 0 --cultivar Semillon --rnn_fpath _runs/RTMC/ColdHardiness/ParamMTL1/STL/WA/Roza2/Prosser/Semillon/param_mtl__1766673728
python3 -m trainers.train_model_rtmc --config rtmc/finetune_ch/ch_param_1_1_stl --seed 1 --cultivar Semillon --rnn_fpath _runs/RTMC/ColdHardiness/ParamMTL1/STL/WA/Roza2/Prosser/Semillon/param_mtl__1766674423
python3 -m trainers.train_model_rtmc --config rtmc/finetune_ch/ch_param_1_1_stl --seed 2 --cultivar Semillon --rnn_fpath _runs/RTMC/ColdHardiness/ParamMTL1/STL/WA/Roza2/Prosser/Semillon/param_mtl__1766675114
python3 -m trainers.train_model_rtmc --config rtmc/finetune_ch/ch_param_1_1_stl --seed 3 --cultivar Semillon --rnn_fpath _runs/RTMC/ColdHardiness/ParamMTL1/STL/WA/Roza2/Prosser/Semillon/param_mtl__1766675804
python3 -m trainers.train_model_rtmc --config rtmc/finetune_ch/ch_param_1_1_stl --seed 4 --cultivar Semillon --rnn_fpath _runs/RTMC/ColdHardiness/ParamMTL1/STL/WA/Roza2/Prosser/Semillon/param_mtl__1766676496
 
python3 -m trainers.train_model_rtmc --config rtmc/finetune_ch/ch_param_1_1_stl --seed 0 --cultivar Syrah --rnn_fpath _runs/RTMC/ColdHardiness/ParamMTL1/STL/WA/Roza2/Prosser/Syrah/param_mtl__1766677189
python3 -m trainers.train_model_rtmc --config rtmc/finetune_ch/ch_param_1_1_stl --seed 1 --cultivar Syrah --rnn_fpath _runs/RTMC/ColdHardiness/ParamMTL1/STL/WA/Roza2/Prosser/Syrah/param_mtl__1766677883
python3 -m trainers.train_model_rtmc --config rtmc/finetune_ch/ch_param_1_1_stl --seed 2 --cultivar Syrah --rnn_fpath _runs/RTMC/ColdHardiness/ParamMTL1/STL/WA/Roza2/Prosser/Syrah/param_mtl__1766678576
python3 -m trainers.train_model_rtmc --config rtmc/finetune_ch/ch_param_1_1_stl --seed 3 --cultivar Syrah --rnn_fpath _runs/RTMC/ColdHardiness/ParamMTL1/STL/WA/Roza2/Prosser/Syrah/param_mtl__1766679267
python3 -m trainers.train_model_rtmc --config rtmc/finetune_ch/ch_param_1_1_stl --seed 4 --cultivar Syrah --rnn_fpath _runs/RTMC/ColdHardiness/ParamMTL1/STL/WA/Roza2/Prosser/Syrah/param_mtl__1766679961
 
python3 -m trainers.train_model_rtmc --config rtmc/finetune_ch/ch_param_1_1_stl --seed 0 --cultivar Viognier --rnn_fpath _runs/RTMC/ColdHardiness/ParamMTL1/STL/WA/Roza2/Prosser/Viognier/param_mtl__1766680670
python3 -m trainers.train_model_rtmc --config rtmc/finetune_ch/ch_param_1_1_stl --seed 1 --cultivar Viognier --rnn_fpath _runs/RTMC/ColdHardiness/ParamMTL1/STL/WA/Roza2/Prosser/Viognier/param_mtl__1766681361
python3 -m trainers.train_model_rtmc --config rtmc/finetune_ch/ch_param_1_1_stl --seed 2 --cultivar Viognier --rnn_fpath _runs/RTMC/ColdHardiness/ParamMTL1/STL/WA/Roza2/Prosser/Viognier/param_mtl__1766682054
python3 -m trainers.train_model_rtmc --config rtmc/finetune_ch/ch_param_1_1_stl --seed 3 --cultivar Viognier --rnn_fpath _runs/RTMC/ColdHardiness/ParamMTL1/STL/WA/Roza2/Prosser/Viognier/param_mtl__1766682746
python3 -m trainers.train_model_rtmc --config rtmc/finetune_ch/ch_param_1_1_stl --seed 4 --cultivar Viognier --rnn_fpath _runs/RTMC/ColdHardiness/ParamMTL1/STL/WA/Roza2/Prosser/Viognier/param_mtl__1766683440
 
python3 -m trainers.train_model_rtmc --config rtmc/finetune_ch/ch_param_1_1_stl --seed 0 --cultivar Zinfandel --rnn_fpath _runs/RTMC/ColdHardiness/ParamMTL1/STL/WA/Roza2/Prosser/Zinfandel/param_mtl__1766684134
python3 -m trainers.train_model_rtmc --config rtmc/finetune_ch/ch_param_1_1_stl --seed 1 --cultivar Zinfandel --rnn_fpath _runs/RTMC/ColdHardiness/ParamMTL1/STL/WA/Roza2/Prosser/Zinfandel/param_mtl__1766684826
python3 -m trainers.train_model_rtmc --config rtmc/finetune_ch/ch_param_1_1_stl --seed 2 --cultivar Zinfandel --rnn_fpath _runs/RTMC/ColdHardiness/ParamMTL1/STL/WA/Roza2/Prosser/Zinfandel/param_mtl__1766685524
python3 -m trainers.train_model_rtmc --config rtmc/finetune_ch/ch_param_1_1_stl --seed 3 --cultivar Zinfandel --rnn_fpath _runs/RTMC/ColdHardiness/ParamMTL1/STL/WA/Roza2/Prosser/Zinfandel/param_mtl__1766686223
python3 -m trainers.train_model_rtmc --config rtmc/finetune_ch/ch_param_1_1_stl --seed 4 --cultivar Zinfandel --rnn_fpath _runs/RTMC/ColdHardiness/ParamMTL1/STL/WA/Roza2/Prosser/Zinfandel/param_mtl__1766686904