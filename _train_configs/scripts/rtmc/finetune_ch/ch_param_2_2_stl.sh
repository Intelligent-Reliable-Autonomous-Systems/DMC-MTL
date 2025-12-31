#!/bin/bash
#SBATCH -J ch_param_2_2_stl_stl
#SBATCH -o output/ch_param_2_2_stl.out
#SBATCH -e output/ch_param_2_2_stl.err
#SBATCH -p eecs,share,gpu
#SBATCH -t 2-12:00:00
#SBATCH --gres=gpu:1

python3 -m trainers.train_model_rtmc --config rtmc/finetune_ch/ch_param_2_2_stl --seed 0 --cultivar Barbera --rnn_fpath _runs/RTMC/ColdHardiness/ParamMTL2/STL/WA/Roza2/Prosser/Barbera/param_mtl__1766629604
python3 -m trainers.train_model_rtmc --config rtmc/finetune_ch/ch_param_2_2_stl --seed 1 --cultivar Barbera --rnn_fpath _runs/RTMC/ColdHardiness/ParamMTL2/STL/WA/Roza2/Prosser/Barbera/param_mtl__1766632137
python3 -m trainers.train_model_rtmc --config rtmc/finetune_ch/ch_param_2_2_stl --seed 2 --cultivar Barbera --rnn_fpath _runs/RTMC/ColdHardiness/ParamMTL2/STL/WA/Roza2/Prosser/Barbera/param_mtl__1766634662
python3 -m trainers.train_model_rtmc --config rtmc/finetune_ch/ch_param_2_2_stl --seed 3 --cultivar Barbera --rnn_fpath _runs/RTMC/ColdHardiness/ParamMTL2/STL/WA/Roza2/Prosser/Barbera/param_mtl__1766637179
python3 -m trainers.train_model_rtmc --config rtmc/finetune_ch/ch_param_2_2_stl --seed 4 --cultivar Barbera --rnn_fpath _runs/RTMC/ColdHardiness/ParamMTL2/STL/WA/Roza2/Prosser/Barbera/param_mtl__1766639723
 
python3 -m trainers.train_model_rtmc --config rtmc/finetune_ch/ch_param_2_2_stl --seed 0 --cultivar Cabernet_Franc --rnn_fpath _runs/RTMC/ColdHardiness/ParamMTL2/STL/WA/Roza2/Prosser/Cabernet_Franc/param_mtl__1766642259
python3 -m trainers.train_model_rtmc --config rtmc/finetune_ch/ch_param_2_2_stl --seed 1 --cultivar Cabernet_Franc --rnn_fpath _runs/RTMC/ColdHardiness/ParamMTL2/STL/WA/Roza2/Prosser/Cabernet_Franc/param_mtl__1766644780
python3 -m trainers.train_model_rtmc --config rtmc/finetune_ch/ch_param_2_2_stl --seed 2 --cultivar Cabernet_Franc --rnn_fpath _runs/RTMC/ColdHardiness/ParamMTL2/STL/WA/Roza2/Prosser/Cabernet_Franc/param_mtl__1766647329
python3 -m trainers.train_model_rtmc --config rtmc/finetune_ch/ch_param_2_2_stl --seed 3 --cultivar Cabernet_Franc --rnn_fpath _runs/RTMC/ColdHardiness/ParamMTL2/STL/WA/Roza2/Prosser/Cabernet_Franc/param_mtl__1766649873
python3 -m trainers.train_model_rtmc --config rtmc/finetune_ch/ch_param_2_2_stl --seed 4 --cultivar Cabernet_Franc --rnn_fpath _runs/RTMC/ColdHardiness/ParamMTL2/STL/WA/Roza2/Prosser/Cabernet_Franc/param_mtl__1766652389
 
python3 -m trainers.train_model_rtmc --config rtmc/finetune_ch/ch_param_2_2_stl --seed 0 --cultivar Cabernet_Sauvignon --rnn_fpath _runs/RTMC/ColdHardiness/ParamMTL2/STL/WA/Roza2/Prosser/Cabernet_Sauvignon/param_mtl__1766654936
python3 -m trainers.train_model_rtmc --config rtmc/finetune_ch/ch_param_2_2_stl --seed 1 --cultivar Cabernet_Sauvignon --rnn_fpath _runs/RTMC/ColdHardiness/ParamMTL2/STL/WA/Roza2/Prosser/Cabernet_Sauvignon/param_mtl__1766657484
python3 -m trainers.train_model_rtmc --config rtmc/finetune_ch/ch_param_2_2_stl --seed 2 --cultivar Cabernet_Sauvignon --rnn_fpath _runs/RTMC/ColdHardiness/ParamMTL2/STL/WA/Roza2/Prosser/Cabernet_Sauvignon/param_mtl__1766659996
python3 -m trainers.train_model_rtmc --config rtmc/finetune_ch/ch_param_2_2_stl --seed 3 --cultivar Cabernet_Sauvignon --rnn_fpath _runs/RTMC/ColdHardiness/ParamMTL2/STL/WA/Roza2/Prosser/Cabernet_Sauvignon/param_mtl__1766662535
python3 -m trainers.train_model_rtmc --config rtmc/finetune_ch/ch_param_2_2_stl --seed 4 --cultivar Cabernet_Sauvignon --rnn_fpath _runs/RTMC/ColdHardiness/ParamMTL2/STL/WA/Roza2/Prosser/Cabernet_Sauvignon/param_mtl__1766665120
 
python3 -m trainers.train_model_rtmc --config rtmc/finetune_ch/ch_param_2_2_stl --seed 0 --cultivar Chardonnay --rnn_fpath _runs/RTMC/ColdHardiness/ParamMTL2/STL/WA/Roza2/Prosser/Chardonnay/param_mtl__1766667807
python3 -m trainers.train_model_rtmc --config rtmc/finetune_ch/ch_param_2_2_stl --seed 1 --cultivar Chardonnay --rnn_fpath _runs/RTMC/ColdHardiness/ParamMTL2/STL/WA/Roza2/Prosser/Chardonnay/param_mtl__1766670498
python3 -m trainers.train_model_rtmc --config rtmc/finetune_ch/ch_param_2_2_stl --seed 2 --cultivar Chardonnay --rnn_fpath _runs/RTMC/ColdHardiness/ParamMTL2/STL/WA/Roza2/Prosser/Chardonnay/param_mtl__1766673173
python3 -m trainers.train_model_rtmc --config rtmc/finetune_ch/ch_param_2_2_stl --seed 3 --cultivar Chardonnay --rnn_fpath _runs/RTMC/ColdHardiness/ParamMTL2/STL/WA/Roza2/Prosser/Chardonnay/param_mtl__1766675861
python3 -m trainers.train_model_rtmc --config rtmc/finetune_ch/ch_param_2_2_stl --seed 4 --cultivar Chardonnay --rnn_fpath _runs/RTMC/ColdHardiness/ParamMTL2/STL/WA/Roza2/Prosser/Chardonnay/param_mtl__1766678542
 
python3 -m trainers.train_model_rtmc --config rtmc/finetune_ch/ch_param_2_2_stl --seed 0 --cultivar Chenin_Blanc --rnn_fpath _runs/RTMC/ColdHardiness/ParamMTL2/STL/WA/Roza2/Prosser/Chenin_Blanc/param_mtl__1766681216
python3 -m trainers.train_model_rtmc --config rtmc/finetune_ch/ch_param_2_2_stl --seed 1 --cultivar Chenin_Blanc --rnn_fpath _runs/RTMC/ColdHardiness/ParamMTL2/STL/WA/Roza2/Prosser/Chenin_Blanc/param_mtl__1766683911
python3 -m trainers.train_model_rtmc --config rtmc/finetune_ch/ch_param_2_2_stl --seed 2 --cultivar Chenin_Blanc --rnn_fpath _runs/RTMC/ColdHardiness/ParamMTL2/STL/WA/Roza2/Prosser/Chenin_Blanc/param_mtl__1766686596
python3 -m trainers.train_model_rtmc --config rtmc/finetune_ch/ch_param_2_2_stl --seed 3 --cultivar Chenin_Blanc --rnn_fpath _runs/RTMC/ColdHardiness/ParamMTL2/STL/WA/Roza2/Prosser/Chenin_Blanc/param_mtl__1766689294
python3 -m trainers.train_model_rtmc --config rtmc/finetune_ch/ch_param_2_2_stl --seed 4 --cultivar Chenin_Blanc --rnn_fpath _runs/RTMC/ColdHardiness/ParamMTL2/STL/WA/Roza2/Prosser/Chenin_Blanc/param_mtl__1766691990
 
python3 -m trainers.train_model_rtmc --config rtmc/finetune_ch/ch_param_2_2_stl --seed 0 --cultivar Concord --rnn_fpath _runs/RTMC/ColdHardiness/ParamMTL2/STL/WA/Roza2/Prosser/Concord/param_mtl__1766694659
python3 -m trainers.train_model_rtmc --config rtmc/finetune_ch/ch_param_2_2_stl --seed 1 --cultivar Concord --rnn_fpath _runs/RTMC/ColdHardiness/ParamMTL2/STL/WA/Roza2/Prosser/Concord/param_mtl__1766697355
python3 -m trainers.train_model_rtmc --config rtmc/finetune_ch/ch_param_2_2_stl --seed 2 --cultivar Concord --rnn_fpath _runs/RTMC/ColdHardiness/ParamMTL2/STL/WA/Roza2/Prosser/Concord/param_mtl__1766700018
python3 -m trainers.train_model_rtmc --config rtmc/finetune_ch/ch_param_2_2_stl --seed 3 --cultivar Concord --rnn_fpath _runs/RTMC/ColdHardiness/ParamMTL2/STL/WA/Roza2/Prosser/Concord/param_mtl__1766702700
python3 -m trainers.train_model_rtmc --config rtmc/finetune_ch/ch_param_2_2_stl --seed 4 --cultivar Concord --rnn_fpath _runs/RTMC/ColdHardiness/ParamMTL2/STL/WA/Roza2/Prosser/Concord/param_mtl__1766705243
 
python3 -m trainers.train_model_rtmc --config rtmc/finetune_ch/ch_param_2_2_stl --seed 0 --cultivar Gewurztraminer --rnn_fpath _runs/RTMC/ColdHardiness/ParamMTL2/STL/WA/Roza2/Prosser/Gewurztraminer/param_mtl__1766707739
python3 -m trainers.train_model_rtmc --config rtmc/finetune_ch/ch_param_2_2_stl --seed 1 --cultivar Gewurztraminer --rnn_fpath _runs/RTMC/ColdHardiness/ParamMTL2/STL/WA/Roza2/Prosser/Gewurztraminer/param_mtl__1766710284
python3 -m trainers.train_model_rtmc --config rtmc/finetune_ch/ch_param_2_2_stl --seed 2 --cultivar Gewurztraminer --rnn_fpath _runs/RTMC/ColdHardiness/ParamMTL2/STL/WA/Roza2/Prosser/Gewurztraminer/param_mtl__1766712816
python3 -m trainers.train_model_rtmc --config rtmc/finetune_ch/ch_param_2_2_stl --seed 3 --cultivar Gewurztraminer --rnn_fpath _runs/RTMC/ColdHardiness/ParamMTL2/STL/WA/Roza2/Prosser/Gewurztraminer/param_mtl__1766715357
python3 -m trainers.train_model_rtmc --config rtmc/finetune_ch/ch_param_2_2_stl --seed 4 --cultivar Gewurztraminer --rnn_fpath _runs/RTMC/ColdHardiness/ParamMTL2/STL/WA/Roza2/Prosser/Gewurztraminer/param_mtl__1766717895
 
python3 -m trainers.train_model_rtmc --config rtmc/finetune_ch/ch_param_2_2_stl --seed 0 --cultivar Lemberger --rnn_fpath _runs/RTMC/ColdHardiness/ParamMTL2/STL/WA/Roza2/Prosser/Lemberger/param_mtl__1766720409
python3 -m trainers.train_model_rtmc --config rtmc/finetune_ch/ch_param_2_2_stl --seed 1 --cultivar Lemberger --rnn_fpath _runs/RTMC/ColdHardiness/ParamMTL2/STL/WA/Roza2/Prosser/Lemberger/param_mtl__1766722956
python3 -m trainers.train_model_rtmc --config rtmc/finetune_ch/ch_param_2_2_stl --seed 2 --cultivar Lemberger --rnn_fpath _runs/RTMC/ColdHardiness/ParamMTL2/STL/WA/Roza2/Prosser/Lemberger/param_mtl__1766725480
python3 -m trainers.train_model_rtmc --config rtmc/finetune_ch/ch_param_2_2_stl --seed 3 --cultivar Lemberger --rnn_fpath _runs/RTMC/ColdHardiness/ParamMTL2/STL/WA/Roza2/Prosser/Lemberger/param_mtl__1766728023
python3 -m trainers.train_model_rtmc --config rtmc/finetune_ch/ch_param_2_2_stl --seed 4 --cultivar Lemberger --rnn_fpath _runs/RTMC/ColdHardiness/ParamMTL2/STL/WA/Roza2/Prosser/Lemberger/param_mtl__1766730566
 
python3 -m trainers.train_model_rtmc --config rtmc/finetune_ch/ch_param_2_2_stl --seed 0 --cultivar Malbec --rnn_fpath _runs/RTMC/ColdHardiness/ParamMTL2/STL/WA/Roza2/Prosser/Malbec/param_mtl__1766733106
python3 -m trainers.train_model_rtmc --config rtmc/finetune_ch/ch_param_2_2_stl --seed 1 --cultivar Malbec --rnn_fpath _runs/RTMC/ColdHardiness/ParamMTL2/STL/WA/Roza2/Prosser/Malbec/param_mtl__1766735644
python3 -m trainers.train_model_rtmc --config rtmc/finetune_ch/ch_param_2_2_stl --seed 2 --cultivar Malbec --rnn_fpath _runs/RTMC/ColdHardiness/ParamMTL2/STL/WA/Roza2/Prosser/Malbec/param_mtl__1766738172
python3 -m trainers.train_model_rtmc --config rtmc/finetune_ch/ch_param_2_2_stl --seed 3 --cultivar Malbec --rnn_fpath _runs/RTMC/ColdHardiness/ParamMTL2/STL/WA/Roza2/Prosser/Malbec/param_mtl__1766740718
python3 -m trainers.train_model_rtmc --config rtmc/finetune_ch/ch_param_2_2_stl --seed 4 --cultivar Malbec --rnn_fpath _runs/RTMC/ColdHardiness/ParamMTL2/STL/WA/Roza2/Prosser/Malbec/param_mtl__1766743261
 
python3 -m trainers.train_model_rtmc --config rtmc/finetune_ch/ch_param_2_2_stl --seed 0 --cultivar Merlot --rnn_fpath _runs/RTMC/ColdHardiness/ParamMTL2/STL/WA/Roza2/Prosser/Merlot/param_mtl__1766745785
python3 -m trainers.train_model_rtmc --config rtmc/finetune_ch/ch_param_2_2_stl --seed 1 --cultivar Merlot --rnn_fpath _runs/RTMC/ColdHardiness/ParamMTL2/STL/WA/Roza2/Prosser/Merlot/param_mtl__1766748324
python3 -m trainers.train_model_rtmc --config rtmc/finetune_ch/ch_param_2_2_stl --seed 2 --cultivar Merlot --rnn_fpath _runs/RTMC/ColdHardiness/ParamMTL2/STL/WA/Roza2/Prosser/Merlot/param_mtl__1766750834
python3 -m trainers.train_model_rtmc --config rtmc/finetune_ch/ch_param_2_2_stl --seed 3 --cultivar Merlot --rnn_fpath _runs/RTMC/ColdHardiness/ParamMTL2/STL/WA/Roza2/Prosser/Merlot/param_mtl__1766753246
python3 -m trainers.train_model_rtmc --config rtmc/finetune_ch/ch_param_2_2_stl --seed 4 --cultivar Merlot --rnn_fpath _runs/RTMC/ColdHardiness/ParamMTL2/STL/WA/Roza2/Prosser/Merlot/param_mtl__1766755644
 
python3 -m trainers.train_model_rtmc --config rtmc/finetune_ch/ch_param_2_2_stl --seed 0 --cultivar Mourvedre --rnn_fpath _runs/RTMC/ColdHardiness/ParamMTL2/STL/WA/Roza2/Prosser/Mourvedre/param_mtl__1766758056
python3 -m trainers.train_model_rtmc --config rtmc/finetune_ch/ch_param_2_2_stl --seed 1 --cultivar Mourvedre --rnn_fpath _runs/RTMC/ColdHardiness/ParamMTL2/STL/WA/Roza2/Prosser/Mourvedre/param_mtl__1766760468
python3 -m trainers.train_model_rtmc --config rtmc/finetune_ch/ch_param_2_2_stl --seed 2 --cultivar Mourvedre --rnn_fpath _runs/RTMC/ColdHardiness/ParamMTL2/STL/WA/Roza2/Prosser/Mourvedre/param_mtl__1766762876
python3 -m trainers.train_model_rtmc --config rtmc/finetune_ch/ch_param_2_2_stl --seed 3 --cultivar Mourvedre --rnn_fpath _runs/RTMC/ColdHardiness/ParamMTL2/STL/WA/Roza2/Prosser/Mourvedre/param_mtl__1766765286
python3 -m trainers.train_model_rtmc --config rtmc/finetune_ch/ch_param_2_2_stl --seed 4 --cultivar Mourvedre --rnn_fpath _runs/RTMC/ColdHardiness/ParamMTL2/STL/WA/Roza2/Prosser/Mourvedre/param_mtl__1766767670
 
python3 -m trainers.train_model_rtmc --config rtmc/finetune_ch/ch_param_2_2_stl --seed 0 --cultivar Nebbiolo --rnn_fpath _runs/RTMC/ColdHardiness/ParamMTL2/STL/WA/Roza2/Prosser/Nebbiolo/param_mtl__1766770080
python3 -m trainers.train_model_rtmc --config rtmc/finetune_ch/ch_param_2_2_stl --seed 1 --cultivar Nebbiolo --rnn_fpath _runs/RTMC/ColdHardiness/ParamMTL2/STL/WA/Roza2/Prosser/Nebbiolo/param_mtl__1766772476
python3 -m trainers.train_model_rtmc --config rtmc/finetune_ch/ch_param_2_2_stl --seed 2 --cultivar Nebbiolo --rnn_fpath _runs/RTMC/ColdHardiness/ParamMTL2/STL/WA/Roza2/Prosser/Nebbiolo/param_mtl__1766774885
python3 -m trainers.train_model_rtmc --config rtmc/finetune_ch/ch_param_2_2_stl --seed 3 --cultivar Nebbiolo --rnn_fpath _runs/RTMC/ColdHardiness/ParamMTL2/STL/WA/Roza2/Prosser/Nebbiolo/param_mtl__1766777286
python3 -m trainers.train_model_rtmc --config rtmc/finetune_ch/ch_param_2_2_stl --seed 4 --cultivar Nebbiolo --rnn_fpath _runs/RTMC/ColdHardiness/ParamMTL2/STL/WA/Roza2/Prosser/Nebbiolo/param_mtl__1766779674
 
python3 -m trainers.train_model_rtmc --config rtmc/finetune_ch/ch_param_2_2_stl --seed 0 --cultivar Pinot_Gris --rnn_fpath _runs/RTMC/ColdHardiness/ParamMTL2/STL/WA/Roza2/Prosser/Pinot_Gris/param_mtl__1766782244
python3 -m trainers.train_model_rtmc --config rtmc/finetune_ch/ch_param_2_2_stl --seed 1 --cultivar Pinot_Gris --rnn_fpath _runs/RTMC/ColdHardiness/ParamMTL2/STL/WA/Roza2/Prosser/Pinot_Gris/param_mtl__1766784814
python3 -m trainers.train_model_rtmc --config rtmc/finetune_ch/ch_param_2_2_stl --seed 2 --cultivar Pinot_Gris --rnn_fpath _runs/RTMC/ColdHardiness/ParamMTL2/STL/WA/Roza2/Prosser/Pinot_Gris/param_mtl__1766787361
python3 -m trainers.train_model_rtmc --config rtmc/finetune_ch/ch_param_2_2_stl --seed 3 --cultivar Pinot_Gris --rnn_fpath _runs/RTMC/ColdHardiness/ParamMTL2/STL/WA/Roza2/Prosser/Pinot_Gris/param_mtl__1766789922
python3 -m trainers.train_model_rtmc --config rtmc/finetune_ch/ch_param_2_2_stl --seed 4 --cultivar Pinot_Gris --rnn_fpath _runs/RTMC/ColdHardiness/ParamMTL2/STL/WA/Roza2/Prosser/Pinot_Gris/param_mtl__1766792486
 
python3 -m trainers.train_model_rtmc --config rtmc/finetune_ch/ch_param_2_2_stl --seed 0 --cultivar Riesling --rnn_fpath _runs/RTMC/ColdHardiness/ParamMTL2/STL/WA/Roza2/Prosser/Riesling/param_mtl__1766795035
python3 -m trainers.train_model_rtmc --config rtmc/finetune_ch/ch_param_2_2_stl --seed 1 --cultivar Riesling --rnn_fpath _runs/RTMC/ColdHardiness/ParamMTL2/STL/WA/Roza2/Prosser/Riesling/param_mtl__1766797598
python3 -m trainers.train_model_rtmc --config rtmc/finetune_ch/ch_param_2_2_stl --seed 2 --cultivar Riesling --rnn_fpath _runs/RTMC/ColdHardiness/ParamMTL2/STL/WA/Roza2/Prosser/Riesling/param_mtl__1766800125
python3 -m trainers.train_model_rtmc --config rtmc/finetune_ch/ch_param_2_2_stl --seed 3 --cultivar Riesling --rnn_fpath _runs/RTMC/ColdHardiness/ParamMTL2/STL/WA/Roza2/Prosser/Riesling/param_mtl__1766802545
python3 -m trainers.train_model_rtmc --config rtmc/finetune_ch/ch_param_2_2_stl --seed 4 --cultivar Riesling --rnn_fpath _runs/RTMC/ColdHardiness/ParamMTL2/STL/WA/Roza2/Prosser/Riesling/param_mtl__1766804982
 
python3 -m trainers.train_model_rtmc --config rtmc/finetune_ch/ch_param_2_2_stl --seed 0 --cultivar Sangiovese --rnn_fpath _runs/RTMC/ColdHardiness/ParamMTL2/STL/WA/Roza2/Prosser/Sangiovese/param_mtl__1766807414
python3 -m trainers.train_model_rtmc --config rtmc/finetune_ch/ch_param_2_2_stl --seed 1 --cultivar Sangiovese --rnn_fpath _runs/RTMC/ColdHardiness/ParamMTL2/STL/WA/Roza2/Prosser/Sangiovese/param_mtl__1766809825
python3 -m trainers.train_model_rtmc --config rtmc/finetune_ch/ch_param_2_2_stl --seed 2 --cultivar Sangiovese --rnn_fpath _runs/RTMC/ColdHardiness/ParamMTL2/STL/WA/Roza2/Prosser/Sangiovese/param_mtl__1766812235
python3 -m trainers.train_model_rtmc --config rtmc/finetune_ch/ch_param_2_2_stl --seed 3 --cultivar Sangiovese --rnn_fpath _runs/RTMC/ColdHardiness/ParamMTL2/STL/WA/Roza2/Prosser/Sangiovese/param_mtl__1766814652
python3 -m trainers.train_model_rtmc --config rtmc/finetune_ch/ch_param_2_2_stl --seed 4 --cultivar Sangiovese --rnn_fpath _runs/RTMC/ColdHardiness/ParamMTL2/STL/WA/Roza2/Prosser/Sangiovese/param_mtl__1766817053
 
python3 -m trainers.train_model_rtmc --config rtmc/finetune_ch/ch_param_2_2_stl --seed 0 --cultivar Sauvignon_Blanc --rnn_fpath _runs/RTMC/ColdHardiness/ParamMTL2/STL/WA/Roza2/Prosser/Sauvignon_Blanc/param_mtl__1766819466
python3 -m trainers.train_model_rtmc --config rtmc/finetune_ch/ch_param_2_2_stl --seed 1 --cultivar Sauvignon_Blanc --rnn_fpath _runs/RTMC/ColdHardiness/ParamMTL2/STL/WA/Roza2/Prosser/Sauvignon_Blanc/param_mtl__1766821881
python3 -m trainers.train_model_rtmc --config rtmc/finetune_ch/ch_param_2_2_stl --seed 2 --cultivar Sauvignon_Blanc --rnn_fpath _runs/RTMC/ColdHardiness/ParamMTL2/STL/WA/Roza2/Prosser/Sauvignon_Blanc/param_mtl__1766824291
python3 -m trainers.train_model_rtmc --config rtmc/finetune_ch/ch_param_2_2_stl --seed 3 --cultivar Sauvignon_Blanc --rnn_fpath _runs/RTMC/ColdHardiness/ParamMTL2/STL/WA/Roza2/Prosser/Sauvignon_Blanc/param_mtl__1766826684
python3 -m trainers.train_model_rtmc --config rtmc/finetune_ch/ch_param_2_2_stl --seed 4 --cultivar Sauvignon_Blanc --rnn_fpath _runs/RTMC/ColdHardiness/ParamMTL2/STL/WA/Roza2/Prosser/Sauvignon_Blanc/param_mtl__1766829090
 
python3 -m trainers.train_model_rtmc --config rtmc/finetune_ch/ch_param_2_2_stl --seed 0 --cultivar Semillon --rnn_fpath _runs/RTMC/ColdHardiness/ParamMTL2/STL/WA/Roza2/Prosser/Semillon/param_mtl__1766831491
python3 -m trainers.train_model_rtmc --config rtmc/finetune_ch/ch_param_2_2_stl --seed 1 --cultivar Semillon --rnn_fpath _runs/RTMC/ColdHardiness/ParamMTL2/STL/WA/Roza2/Prosser/Semillon/param_mtl__1766833902
python3 -m trainers.train_model_rtmc --config rtmc/finetune_ch/ch_param_2_2_stl --seed 2 --cultivar Semillon --rnn_fpath _runs/RTMC/ColdHardiness/ParamMTL2/STL/WA/Roza2/Prosser/Semillon/param_mtl__1766836303
python3 -m trainers.train_model_rtmc --config rtmc/finetune_ch/ch_param_2_2_stl --seed 3 --cultivar Semillon --rnn_fpath _runs/RTMC/ColdHardiness/ParamMTL2/STL/WA/Roza2/Prosser/Semillon/param_mtl__1766838713
python3 -m trainers.train_model_rtmc --config rtmc/finetune_ch/ch_param_2_2_stl --seed 4 --cultivar Semillon --rnn_fpath _runs/RTMC/ColdHardiness/ParamMTL2/STL/WA/Roza2/Prosser/Semillon/param_mtl__1766841119
 
python3 -m trainers.train_model_rtmc --config rtmc/finetune_ch/ch_param_2_2_stl --seed 0 --cultivar Syrah --rnn_fpath _runs/RTMC/ColdHardiness/ParamMTL2/STL/WA/Roza2/Prosser/Syrah/param_mtl__1766869029
python3 -m trainers.train_model_rtmc --config rtmc/finetune_ch/ch_param_2_2_stl --seed 1 --cultivar Syrah --rnn_fpath _runs/RTMC/ColdHardiness/ParamMTL2/STL/WA/Roza2/Prosser/Syrah/param_mtl__1766871706
python3 -m trainers.train_model_rtmc --config rtmc/finetune_ch/ch_param_2_2_stl --seed 2 --cultivar Syrah --rnn_fpath _runs/RTMC/ColdHardiness/ParamMTL2/STL/WA/Roza2/Prosser/Syrah/param_mtl__1766874429
python3 -m trainers.train_model_rtmc --config rtmc/finetune_ch/ch_param_2_2_stl --seed 3 --cultivar Syrah --rnn_fpath _runs/RTMC/ColdHardiness/ParamMTL2/STL/WA/Roza2/Prosser/Syrah/param_mtl__1766877124
python3 -m trainers.train_model_rtmc --config rtmc/finetune_ch/ch_param_2_2_stl --seed 4 --cultivar Syrah --rnn_fpath _runs/RTMC/ColdHardiness/ParamMTL2/STL/WA/Roza2/Prosser/Syrah/param_mtl__1766879828
 
python3 -m trainers.train_model_rtmc --config rtmc/finetune_ch/ch_param_2_2_stl --seed 0 --cultivar Viognier --rnn_fpath _runs/RTMC/ColdHardiness/ParamMTL2/STL/WA/Roza2/Prosser/Viognier/param_mtl__1766882435
python3 -m trainers.train_model_rtmc --config rtmc/finetune_ch/ch_param_2_2_stl --seed 1 --cultivar Viognier --rnn_fpath _runs/RTMC/ColdHardiness/ParamMTL2/STL/WA/Roza2/Prosser/Viognier/param_mtl__1766884981
python3 -m trainers.train_model_rtmc --config rtmc/finetune_ch/ch_param_2_2_stl --seed 2 --cultivar Viognier --rnn_fpath _runs/RTMC/ColdHardiness/ParamMTL2/STL/WA/Roza2/Prosser/Viognier/param_mtl__1766887535
python3 -m trainers.train_model_rtmc --config rtmc/finetune_ch/ch_param_2_2_stl --seed 3 --cultivar Viognier --rnn_fpath _runs/RTMC/ColdHardiness/ParamMTL2/STL/WA/Roza2/Prosser/Viognier/param_mtl__1766890078
python3 -m trainers.train_model_rtmc --config rtmc/finetune_ch/ch_param_2_2_stl --seed 4 --cultivar Viognier --rnn_fpath _runs/RTMC/ColdHardiness/ParamMTL2/STL/WA/Roza2/Prosser/Viognier/param_mtl__1766892642
 
python3 -m trainers.train_model_rtmc --config rtmc/finetune_ch/ch_param_2_2_stl --seed 0 --cultivar Zinfandel --rnn_fpath _runs/RTMC/ColdHardiness/ParamMTL2/STL/WA/Roza2/Prosser/Zinfandel/param_mtl__1766895201
python3 -m trainers.train_model_rtmc --config rtmc/finetune_ch/ch_param_2_2_stl --seed 1 --cultivar Zinfandel --rnn_fpath _runs/RTMC/ColdHardiness/ParamMTL2/STL/WA/Roza2/Prosser/Zinfandel/param_mtl__1766897780
python3 -m trainers.train_model_rtmc --config rtmc/finetune_ch/ch_param_2_2_stl --seed 2 --cultivar Zinfandel --rnn_fpath _runs/RTMC/ColdHardiness/ParamMTL2/STL/WA/Roza2/Prosser/Zinfandel/param_mtl__1766900334
python3 -m trainers.train_model_rtmc --config rtmc/finetune_ch/ch_param_2_2_stl --seed 3 --cultivar Zinfandel --rnn_fpath _runs/RTMC/ColdHardiness/ParamMTL2/STL/WA/Roza2/Prosser/Zinfandel/param_mtl__1766902895
python3 -m trainers.train_model_rtmc --config rtmc/finetune_ch/ch_param_2_2_stl --seed 4 --cultivar Zinfandel --rnn_fpath _runs/RTMC/ColdHardiness/ParamMTL2/STL/WA/Roza2/Prosser/Zinfandel/param_mtl__1766905458