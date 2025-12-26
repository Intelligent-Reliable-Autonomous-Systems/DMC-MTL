#!/bin/bash
#SBATCH -J ch_param_5_5_stl
#SBATCH -o output/ch_param_5_5_stl.out
#SBATCH -e output/ch_param_5_5_stl.err
#SBATCH -p eecs,share,gpu
#SBATCH -t 2-12:00:00
#SBATCH --gres=gpu:1

python3 -m trainers.train_model_rtmc --config rtmc/finetune_ch/ch_param_5_5_stl --seed 0 --cultivar Barbera --rnn_fpath _runs/RTMC/ColdHardiness/ParamMTL5/STL/WA/Roza2/Prosser/Barbera/param_mtl__1766636019
python3 -m trainers.train_model_rtmc --config rtmc/finetune_ch/ch_param_5_5_stl --seed 1 --cultivar Barbera --rnn_fpath _runs/RTMC/ColdHardiness/ParamMTL5/STL/WA/Roza2/Prosser/Barbera/param_mtl__1766637026
python3 -m trainers.train_model_rtmc --config rtmc/finetune_ch/ch_param_5_5_stl --seed 2 --cultivar Barbera --rnn_fpath _runs/RTMC/ColdHardiness/ParamMTL5/STL/WA/Roza2/Prosser/Barbera/param_mtl__1766638034
python3 -m trainers.train_model_rtmc --config rtmc/finetune_ch/ch_param_5_5_stl --seed 3 --cultivar Barbera --rnn_fpath _runs/RTMC/ColdHardiness/ParamMTL5/STL/WA/Roza2/Prosser/Barbera/param_mtl__1766639044
python3 -m trainers.train_model_rtmc --config rtmc/finetune_ch/ch_param_5_5_stl --seed 4 --cultivar Barbera --rnn_fpath _runs/RTMC/ColdHardiness/ParamMTL5/STL/WA/Roza2/Prosser/Barbera/param_mtl__1766640056
 
python3 -m trainers.train_model_rtmc --config rtmc/finetune_ch/ch_param_5_5_stl --seed 0 --cultivar Cabernet_Franc --rnn_fpath _runs/RTMC/ColdHardiness/ParamMTL5/STL/WA/Roza2/Prosser/Cabernet_Franc/param_mtl__1766641065
python3 -m trainers.train_model_rtmc --config rtmc/finetune_ch/ch_param_5_5_stl --seed 1 --cultivar Cabernet_Franc --rnn_fpath _runs/RTMC/ColdHardiness/ParamMTL5/STL/WA/Roza2/Prosser/Cabernet_Franc/param_mtl__1766641694
python3 -m trainers.train_model_rtmc --config rtmc/finetune_ch/ch_param_5_5_stl --seed 2 --cultivar Cabernet_Franc --rnn_fpath _runs/RTMC/ColdHardiness/ParamMTL5/STL/WA/Roza2/Prosser/Cabernet_Franc/param_mtl__1766642321
python3 -m trainers.train_model_rtmc --config rtmc/finetune_ch/ch_param_5_5_stl --seed 3 --cultivar Cabernet_Franc --rnn_fpath _runs/RTMC/ColdHardiness/ParamMTL5/STL/WA/Roza2/Prosser/Cabernet_Franc/param_mtl__1766642949
python3 -m trainers.train_model_rtmc --config rtmc/finetune_ch/ch_param_5_5_stl --seed 4 --cultivar Cabernet_Franc --rnn_fpath _runs/RTMC/ColdHardiness/ParamMTL5/STL/WA/Roza2/Prosser/Cabernet_Franc/param_mtl__1766643581
 
python3 -m trainers.train_model_rtmc --config rtmc/finetune_ch/ch_param_5_5_stl --seed 0 --cultivar Cabernet_Sauvignon --rnn_fpath _runs/RTMC/ColdHardiness/ParamMTL5/STL/WA/Roza2/Prosser/Cabernet_Sauvignon/param_mtl__1766644209
python3 -m trainers.train_model_rtmc --config rtmc/finetune_ch/ch_param_5_5_stl --seed 1 --cultivar Cabernet_Sauvignon --rnn_fpath _runs/RTMC/ColdHardiness/ParamMTL5/STL/WA/Roza2/Prosser/Cabernet_Sauvignon/param_mtl__1766645223
python3 -m trainers.train_model_rtmc --config rtmc/finetune_ch/ch_param_5_5_stl --seed 2 --cultivar Cabernet_Sauvignon --rnn_fpath _runs/RTMC/ColdHardiness/ParamMTL5/STL/WA/Roza2/Prosser/Cabernet_Sauvignon/param_mtl__1766646232
python3 -m trainers.train_model_rtmc --config rtmc/finetune_ch/ch_param_5_5_stl --seed 3 --cultivar Cabernet_Sauvignon --rnn_fpath _runs/RTMC/ColdHardiness/ParamMTL5/STL/WA/Roza2/Prosser/Cabernet_Sauvignon/param_mtl__1766647242
python3 -m trainers.train_model_rtmc --config rtmc/finetune_ch/ch_param_5_5_stl --seed 4 --cultivar Cabernet_Sauvignon --rnn_fpath _runs/RTMC/ColdHardiness/ParamMTL5/STL/WA/Roza2/Prosser/Cabernet_Sauvignon/param_mtl__1766648251
 
python3 -m trainers.train_model_rtmc --config rtmc/finetune_ch/ch_param_5_5_stl --seed 0 --cultivar Chardonnay --rnn_fpath _runs/RTMC/ColdHardiness/ParamMTL5/STL/WA/Roza2/Prosser/Chardonnay/param_mtl__1766649253
python3 -m trainers.train_model_rtmc --config rtmc/finetune_ch/ch_param_5_5_stl --seed 1 --cultivar Chardonnay --rnn_fpath _runs/RTMC/ColdHardiness/ParamMTL5/STL/WA/Roza2/Prosser/Chardonnay/param_mtl__1766650243
python3 -m trainers.train_model_rtmc --config rtmc/finetune_ch/ch_param_5_5_stl --seed 2 --cultivar Chardonnay --rnn_fpath _runs/RTMC/ColdHardiness/ParamMTL5/STL/WA/Roza2/Prosser/Chardonnay/param_mtl__1766651250
python3 -m trainers.train_model_rtmc --config rtmc/finetune_ch/ch_param_5_5_stl --seed 3 --cultivar Chardonnay --rnn_fpath _runs/RTMC/ColdHardiness/ParamMTL5/STL/WA/Roza2/Prosser/Chardonnay/param_mtl__1766652256
python3 -m trainers.train_model_rtmc --config rtmc/finetune_ch/ch_param_5_5_stl --seed 4 --cultivar Chardonnay --rnn_fpath _runs/RTMC/ColdHardiness/ParamMTL5/STL/WA/Roza2/Prosser/Chardonnay/param_mtl__1766653268
 
python3 -m trainers.train_model_rtmc --config rtmc/finetune_ch/ch_param_5_5_stl --seed 0 --cultivar Chenin_Blanc --rnn_fpath _runs/RTMC/ColdHardiness/ParamMTL5/STL/WA/Roza2/Prosser/Chenin_Blanc/param_mtl__1766654273
python3 -m trainers.train_model_rtmc --config rtmc/finetune_ch/ch_param_5_5_stl --seed 1 --cultivar Chenin_Blanc --rnn_fpath _runs/RTMC/ColdHardiness/ParamMTL5/STL/WA/Roza2/Prosser/Chenin_Blanc/param_mtl__1766655279
python3 -m trainers.train_model_rtmc --config rtmc/finetune_ch/ch_param_5_5_stl --seed 2 --cultivar Chenin_Blanc --rnn_fpath _runs/RTMC/ColdHardiness/ParamMTL5/STL/WA/Roza2/Prosser/Chenin_Blanc/param_mtl__1766656290
python3 -m trainers.train_model_rtmc --config rtmc/finetune_ch/ch_param_5_5_stl --seed 3 --cultivar Chenin_Blanc --rnn_fpath _runs/RTMC/ColdHardiness/ParamMTL5/STL/WA/Roza2/Prosser/Chenin_Blanc/param_mtl__1766657298
python3 -m trainers.train_model_rtmc --config rtmc/finetune_ch/ch_param_5_5_stl --seed 4 --cultivar Chenin_Blanc --rnn_fpath _runs/RTMC/ColdHardiness/ParamMTL5/STL/WA/Roza2/Prosser/Chenin_Blanc/param_mtl__1766658310
 
python3 -m trainers.train_model_rtmc --config rtmc/finetune_ch/ch_param_5_5_stl --seed 0 --cultivar Concord --rnn_fpath _runs/RTMC/ColdHardiness/ParamMTL5/STL/WA/Roza2/Prosser/Concord/param_mtl__1766659319
python3 -m trainers.train_model_rtmc --config rtmc/finetune_ch/ch_param_5_5_stl --seed 1 --cultivar Concord --rnn_fpath _runs/RTMC/ColdHardiness/ParamMTL5/STL/WA/Roza2/Prosser/Concord/param_mtl__1766660327
python3 -m trainers.train_model_rtmc --config rtmc/finetune_ch/ch_param_5_5_stl --seed 2 --cultivar Concord --rnn_fpath _runs/RTMC/ColdHardiness/ParamMTL5/STL/WA/Roza2/Prosser/Concord/param_mtl__1766661334
python3 -m trainers.train_model_rtmc --config rtmc/finetune_ch/ch_param_5_5_stl --seed 3 --cultivar Concord --rnn_fpath _runs/RTMC/ColdHardiness/ParamMTL5/STL/WA/Roza2/Prosser/Concord/param_mtl__1766662342
python3 -m trainers.train_model_rtmc --config rtmc/finetune_ch/ch_param_5_5_stl --seed 4 --cultivar Concord --rnn_fpath _runs/RTMC/ColdHardiness/ParamMTL5/STL/WA/Roza2/Prosser/Concord/param_mtl__1766663353
 
python3 -m trainers.train_model_rtmc --config rtmc/finetune_ch/ch_param_5_5_stl --seed 0 --cultivar Gewurztraminer --rnn_fpath _runs/RTMC/ColdHardiness/ParamMTL5/STL/WA/Roza2/Prosser/Gewurztraminer/param_mtl__1766664362
python3 -m trainers.train_model_rtmc --config rtmc/finetune_ch/ch_param_5_5_stl --seed 1 --cultivar Gewurztraminer --rnn_fpath _runs/RTMC/ColdHardiness/ParamMTL5/STL/WA/Roza2/Prosser/Gewurztraminer/param_mtl__1766665366
python3 -m trainers.train_model_rtmc --config rtmc/finetune_ch/ch_param_5_5_stl --seed 2 --cultivar Gewurztraminer --rnn_fpath _runs/RTMC/ColdHardiness/ParamMTL5/STL/WA/Roza2/Prosser/Gewurztraminer/param_mtl__1766666346
python3 -m trainers.train_model_rtmc --config rtmc/finetune_ch/ch_param_5_5_stl --seed 3 --cultivar Gewurztraminer --rnn_fpath _runs/RTMC/ColdHardiness/ParamMTL5/STL/WA/Roza2/Prosser/Gewurztraminer/param_mtl__1766667348
python3 -m trainers.train_model_rtmc --config rtmc/finetune_ch/ch_param_5_5_stl --seed 4 --cultivar Gewurztraminer --rnn_fpath _runs/RTMC/ColdHardiness/ParamMTL5/STL/WA/Roza2/Prosser/Gewurztraminer/param_mtl__1766668345
 
python3 -m trainers.train_model_rtmc --config rtmc/finetune_ch/ch_param_5_5_stl --seed 0 --cultivar Lemberger --rnn_fpath _runs/RTMC/ColdHardiness/ParamMTL5/STL/WA/Roza2/Prosser/Lemberger/param_mtl__1766669342
python3 -m trainers.train_model_rtmc --config rtmc/finetune_ch/ch_param_5_5_stl --seed 1 --cultivar Lemberger --rnn_fpath _runs/RTMC/ColdHardiness/ParamMTL5/STL/WA/Roza2/Prosser/Lemberger/param_mtl__1766669966
python3 -m trainers.train_model_rtmc --config rtmc/finetune_ch/ch_param_5_5_stl --seed 2 --cultivar Lemberger --rnn_fpath _runs/RTMC/ColdHardiness/ParamMTL5/STL/WA/Roza2/Prosser/Lemberger/param_mtl__1766670590
python3 -m trainers.train_model_rtmc --config rtmc/finetune_ch/ch_param_5_5_stl --seed 3 --cultivar Lemberger --rnn_fpath _runs/RTMC/ColdHardiness/ParamMTL5/STL/WA/Roza2/Prosser/Lemberger/param_mtl__1766671213
python3 -m trainers.train_model_rtmc --config rtmc/finetune_ch/ch_param_5_5_stl --seed 4 --cultivar Lemberger --rnn_fpath _runs/RTMC/ColdHardiness/ParamMTL5/STL/WA/Roza2/Prosser/Lemberger/param_mtl__1766671839
 
python3 -m trainers.train_model_rtmc --config rtmc/finetune_ch/ch_param_5_5_stl --seed 0 --cultivar Malbec --rnn_fpath _runs/RTMC/ColdHardiness/ParamMTL5/STL/WA/Roza2/Prosser/Malbec/param_mtl__1766672464
python3 -m trainers.train_model_rtmc --config rtmc/finetune_ch/ch_param_5_5_stl --seed 1 --cultivar Malbec --rnn_fpath _runs/RTMC/ColdHardiness/ParamMTL5/STL/WA/Roza2/Prosser/Malbec/param_mtl__1766673465
python3 -m trainers.train_model_rtmc --config rtmc/finetune_ch/ch_param_5_5_stl --seed 2 --cultivar Malbec --rnn_fpath _runs/RTMC/ColdHardiness/ParamMTL5/STL/WA/Roza2/Prosser/Malbec/param_mtl__1766674470
python3 -m trainers.train_model_rtmc --config rtmc/finetune_ch/ch_param_5_5_stl --seed 3 --cultivar Malbec --rnn_fpath _runs/RTMC/ColdHardiness/ParamMTL5/STL/WA/Roza2/Prosser/Malbec/param_mtl__1766675465
python3 -m trainers.train_model_rtmc --config rtmc/finetune_ch/ch_param_5_5_stl --seed 4 --cultivar Malbec --rnn_fpath _runs/RTMC/ColdHardiness/ParamMTL5/STL/WA/Roza2/Prosser/Malbec/param_mtl__1766676466
 
python3 -m trainers.train_model_rtmc --config rtmc/finetune_ch/ch_param_5_5_stl --seed 0 --cultivar Merlot --rnn_fpath _runs/RTMC/ColdHardiness/ParamMTL5/STL/WA/Roza2/Prosser/Merlot/param_mtl__1766677449
python3 -m trainers.train_model_rtmc --config rtmc/finetune_ch/ch_param_5_5_stl --seed 1 --cultivar Merlot --rnn_fpath _runs/RTMC/ColdHardiness/ParamMTL5/STL/WA/Roza2/Prosser/Merlot/param_mtl__1766678439
python3 -m trainers.train_model_rtmc --config rtmc/finetune_ch/ch_param_5_5_stl --seed 2 --cultivar Merlot --rnn_fpath _runs/RTMC/ColdHardiness/ParamMTL5/STL/WA/Roza2/Prosser/Merlot/param_mtl__1766679438
python3 -m trainers.train_model_rtmc --config rtmc/finetune_ch/ch_param_5_5_stl --seed 3 --cultivar Merlot --rnn_fpath _runs/RTMC/ColdHardiness/ParamMTL5/STL/WA/Roza2/Prosser/Merlot/param_mtl__1766680436
python3 -m trainers.train_model_rtmc --config rtmc/finetune_ch/ch_param_5_5_stl --seed 4 --cultivar Merlot --rnn_fpath _runs/RTMC/ColdHardiness/ParamMTL5/STL/WA/Roza2/Prosser/Merlot/param_mtl__1766681425
 
python3 -m trainers.train_model_rtmc --config rtmc/finetune_ch/ch_param_5_5_stl --seed 0 --cultivar Mourvedre --rnn_fpath _runs/RTMC/ColdHardiness/ParamMTL5/STL/WA/Roza2/Prosser/Mourvedre/param_mtl__1766682425
python3 -m trainers.train_model_rtmc --config rtmc/finetune_ch/ch_param_5_5_stl --seed 1 --cultivar Mourvedre --rnn_fpath _runs/RTMC/ColdHardiness/ParamMTL5/STL/WA/Roza2/Prosser/Mourvedre/param_mtl__1766683419
python3 -m trainers.train_model_rtmc --config rtmc/finetune_ch/ch_param_5_5_stl --seed 2 --cultivar Mourvedre --rnn_fpath _runs/RTMC/ColdHardiness/ParamMTL5/STL/WA/Roza2/Prosser/Mourvedre/param_mtl__1766684422
python3 -m trainers.train_model_rtmc --config rtmc/finetune_ch/ch_param_5_5_stl --seed 3 --cultivar Mourvedre --rnn_fpath _runs/RTMC/ColdHardiness/ParamMTL5/STL/WA/Roza2/Prosser/Mourvedre/param_mtl__1766685418
python3 -m trainers.train_model_rtmc --config rtmc/finetune_ch/ch_param_5_5_stl --seed 4 --cultivar Mourvedre --rnn_fpath _runs/RTMC/ColdHardiness/ParamMTL5/STL/WA/Roza2/Prosser/Mourvedre/param_mtl__1766686414
 
python3 -m trainers.train_model_rtmc --config rtmc/finetune_ch/ch_param_5_5_stl --seed 0 --cultivar Nebbiolo --rnn_fpath _runs/RTMC/ColdHardiness/ParamMTL5/STL/WA/Roza2/Prosser/Nebbiolo/param_mtl__1766687414
python3 -m trainers.train_model_rtmc --config rtmc/finetune_ch/ch_param_5_5_stl --seed 1 --cultivar Nebbiolo --rnn_fpath _runs/RTMC/ColdHardiness/ParamMTL5/STL/WA/Roza2/Prosser/Nebbiolo/param_mtl__1766688415
python3 -m trainers.train_model_rtmc --config rtmc/finetune_ch/ch_param_5_5_stl --seed 2 --cultivar Nebbiolo --rnn_fpath _runs/RTMC/ColdHardiness/ParamMTL5/STL/WA/Roza2/Prosser/Nebbiolo/param_mtl__1766689410
python3 -m trainers.train_model_rtmc --config rtmc/finetune_ch/ch_param_5_5_stl --seed 3 --cultivar Nebbiolo --rnn_fpath _runs/RTMC/ColdHardiness/ParamMTL5/STL/WA/Roza2/Prosser/Nebbiolo/param_mtl__1766690387
python3 -m trainers.train_model_rtmc --config rtmc/finetune_ch/ch_param_5_5_stl --seed 4 --cultivar Nebbiolo --rnn_fpath _runs/RTMC/ColdHardiness/ParamMTL5/STL/WA/Roza2/Prosser/Nebbiolo/param_mtl__1766691382
 
python3 -m trainers.train_model_rtmc --config rtmc/finetune_ch/ch_param_5_5_stl --seed 0 --cultivar Pinot_Gris --rnn_fpath _runs/RTMC/ColdHardiness/ParamMTL5/STL/WA/Roza2/Prosser/Pinot_Gris/param_mtl__1766692379
python3 -m trainers.train_model_rtmc --config rtmc/finetune_ch/ch_param_5_5_stl --seed 1 --cultivar Pinot_Gris --rnn_fpath _runs/RTMC/ColdHardiness/ParamMTL5/STL/WA/Roza2/Prosser/Pinot_Gris/param_mtl__1766693380
python3 -m trainers.train_model_rtmc --config rtmc/finetune_ch/ch_param_5_5_stl --seed 2 --cultivar Pinot_Gris --rnn_fpath _runs/RTMC/ColdHardiness/ParamMTL5/STL/WA/Roza2/Prosser/Pinot_Gris/param_mtl__1766694379
python3 -m trainers.train_model_rtmc --config rtmc/finetune_ch/ch_param_5_5_stl --seed 3 --cultivar Pinot_Gris --rnn_fpath _runs/RTMC/ColdHardiness/ParamMTL5/STL/WA/Roza2/Prosser/Pinot_Gris/param_mtl__1766695384
python3 -m trainers.train_model_rtmc --config rtmc/finetune_ch/ch_param_5_5_stl --seed 4 --cultivar Pinot_Gris --rnn_fpath _runs/RTMC/ColdHardiness/ParamMTL5/STL/WA/Roza2/Prosser/Pinot_Gris/param_mtl__1766696387
 
python3 -m trainers.train_model_rtmc --config rtmc/finetune_ch/ch_param_5_5_stl --seed 0 --cultivar Riesling --rnn_fpath _runs/RTMC/ColdHardiness/ParamMTL5/STL/WA/Roza2/Prosser/Riesling/param_mtl__1766697384
python3 -m trainers.train_model_rtmc --config rtmc/finetune_ch/ch_param_5_5_stl --seed 1 --cultivar Riesling --rnn_fpath _runs/RTMC/ColdHardiness/ParamMTL5/STL/WA/Roza2/Prosser/Riesling/param_mtl__1766698380
python3 -m trainers.train_model_rtmc --config rtmc/finetune_ch/ch_param_5_5_stl --seed 2 --cultivar Riesling --rnn_fpath _runs/RTMC/ColdHardiness/ParamMTL5/STL/WA/Roza2/Prosser/Riesling/param_mtl__1766699359
python3 -m trainers.train_model_rtmc --config rtmc/finetune_ch/ch_param_5_5_stl --seed 3 --cultivar Riesling --rnn_fpath _runs/RTMC/ColdHardiness/ParamMTL5/STL/WA/Roza2/Prosser/Riesling/param_mtl__1766700359
python3 -m trainers.train_model_rtmc --config rtmc/finetune_ch/ch_param_5_5_stl --seed 4 --cultivar Riesling --rnn_fpath _runs/RTMC/ColdHardiness/ParamMTL5/STL/WA/Roza2/Prosser/Riesling/param_mtl__1766701359
 
python3 -m trainers.train_model_rtmc --config rtmc/finetune_ch/ch_param_5_5_stl --seed 0 --cultivar Sangiovese --rnn_fpath _runs/RTMC/ColdHardiness/ParamMTL5/STL/WA/Roza2/Prosser/Sangiovese/param_mtl__1766702362
python3 -m trainers.train_model_rtmc --config rtmc/finetune_ch/ch_param_5_5_stl --seed 1 --cultivar Sangiovese --rnn_fpath _runs/RTMC/ColdHardiness/ParamMTL5/STL/WA/Roza2/Prosser/Sangiovese/param_mtl__1766703368
python3 -m trainers.train_model_rtmc --config rtmc/finetune_ch/ch_param_5_5_stl --seed 2 --cultivar Sangiovese --rnn_fpath _runs/RTMC/ColdHardiness/ParamMTL5/STL/WA/Roza2/Prosser/Sangiovese/param_mtl__1766704373
python3 -m trainers.train_model_rtmc --config rtmc/finetune_ch/ch_param_5_5_stl --seed 3 --cultivar Sangiovese --rnn_fpath _runs/RTMC/ColdHardiness/ParamMTL5/STL/WA/Roza2/Prosser/Sangiovese/param_mtl__1766705376
python3 -m trainers.train_model_rtmc --config rtmc/finetune_ch/ch_param_5_5_stl --seed 4 --cultivar Sangiovese --rnn_fpath _runs/RTMC/ColdHardiness/ParamMTL5/STL/WA/Roza2/Prosser/Sangiovese/param_mtl__1766706358
 
python3 -m trainers.train_model_rtmc --config rtmc/finetune_ch/ch_param_5_5_stl --seed 0 --cultivar Sauvignon_Blanc --rnn_fpath _runs/RTMC/ColdHardiness/ParamMTL5/STL/WA/Roza2/Prosser/Sauvignon_Blanc/param_mtl__1766707346
python3 -m trainers.train_model_rtmc --config rtmc/finetune_ch/ch_param_5_5_stl --seed 1 --cultivar Sauvignon_Blanc --rnn_fpath _runs/RTMC/ColdHardiness/ParamMTL5/STL/WA/Roza2/Prosser/Sauvignon_Blanc/param_mtl__1766708325
python3 -m trainers.train_model_rtmc --config rtmc/finetune_ch/ch_param_5_5_stl --seed 2 --cultivar Sauvignon_Blanc --rnn_fpath _runs/RTMC/ColdHardiness/ParamMTL5/STL/WA/Roza2/Prosser/Sauvignon_Blanc/param_mtl__1766709326
python3 -m trainers.train_model_rtmc --config rtmc/finetune_ch/ch_param_5_5_stl --seed 3 --cultivar Sauvignon_Blanc --rnn_fpath _runs/RTMC/ColdHardiness/ParamMTL5/STL/WA/Roza2/Prosser/Sauvignon_Blanc/param_mtl__1766710322
python3 -m trainers.train_model_rtmc --config rtmc/finetune_ch/ch_param_5_5_stl --seed 4 --cultivar Sauvignon_Blanc --rnn_fpath _runs/RTMC/ColdHardiness/ParamMTL5/STL/WA/Roza2/Prosser/Sauvignon_Blanc/param_mtl__1766711322
 
python3 -m trainers.train_model_rtmc --config rtmc/finetune_ch/ch_param_5_5_stl --seed 0 --cultivar Semillon --rnn_fpath _runs/RTMC/ColdHardiness/ParamMTL5/STL/WA/Roza2/Prosser/Semillon/param_mtl__1766712321
python3 -m trainers.train_model_rtmc --config rtmc/finetune_ch/ch_param_5_5_stl --seed 1 --cultivar Semillon --rnn_fpath _runs/RTMC/ColdHardiness/ParamMTL5/STL/WA/Roza2/Prosser/Semillon/param_mtl__1766713314
python3 -m trainers.train_model_rtmc --config rtmc/finetune_ch/ch_param_5_5_stl --seed 2 --cultivar Semillon --rnn_fpath _runs/RTMC/ColdHardiness/ParamMTL5/STL/WA/Roza2/Prosser/Semillon/param_mtl__1766714293
python3 -m trainers.train_model_rtmc --config rtmc/finetune_ch/ch_param_5_5_stl --seed 3 --cultivar Semillon --rnn_fpath _runs/RTMC/ColdHardiness/ParamMTL5/STL/WA/Roza2/Prosser/Semillon/param_mtl__1766715295
python3 -m trainers.train_model_rtmc --config rtmc/finetune_ch/ch_param_5_5_stl --seed 4 --cultivar Semillon --rnn_fpath _runs/RTMC/ColdHardiness/ParamMTL5/STL/WA/Roza2/Prosser/Semillon/param_mtl__1766716295
 
python3 -m trainers.train_model_rtmc --config rtmc/finetune_ch/ch_param_5_5_stl --seed 0 --cultivar Syrah --rnn_fpath _runs/RTMC/ColdHardiness/ParamMTL5/STL/WA/Roza2/Prosser/Syrah/param_mtl__1766717293
python3 -m trainers.train_model_rtmc --config rtmc/finetune_ch/ch_param_5_5_stl --seed 1 --cultivar Syrah --rnn_fpath _runs/RTMC/ColdHardiness/ParamMTL5/STL/WA/Roza2/Prosser/Syrah/param_mtl__1766718275
python3 -m trainers.train_model_rtmc --config rtmc/finetune_ch/ch_param_5_5_stl --seed 2 --cultivar Syrah --rnn_fpath _runs/RTMC/ColdHardiness/ParamMTL5/STL/WA/Roza2/Prosser/Syrah/param_mtl__1766719273
python3 -m trainers.train_model_rtmc --config rtmc/finetune_ch/ch_param_5_5_stl --seed 3 --cultivar Syrah --rnn_fpath _runs/RTMC/ColdHardiness/ParamMTL5/STL/WA/Roza2/Prosser/Syrah/param_mtl__1766720270
python3 -m trainers.train_model_rtmc --config rtmc/finetune_ch/ch_param_5_5_stl --seed 4 --cultivar Syrah --rnn_fpath _runs/RTMC/ColdHardiness/ParamMTL5/STL/WA/Roza2/Prosser/Syrah/param_mtl__1766721270
 
python3 -m trainers.train_model_rtmc --config rtmc/finetune_ch/ch_param_5_5_stl --seed 0 --cultivar Viognier --rnn_fpath _runs/RTMC/ColdHardiness/ParamMTL5/STL/WA/Roza2/Prosser/Viognier/param_mtl__1766722287
python3 -m trainers.train_model_rtmc --config rtmc/finetune_ch/ch_param_5_5_stl --seed 1 --cultivar Viognier --rnn_fpath _runs/RTMC/ColdHardiness/ParamMTL5/STL/WA/Roza2/Prosser/Viognier/param_mtl__1766723267
python3 -m trainers.train_model_rtmc --config rtmc/finetune_ch/ch_param_5_5_stl --seed 2 --cultivar Viognier --rnn_fpath _runs/RTMC/ColdHardiness/ParamMTL5/STL/WA/Roza2/Prosser/Viognier/param_mtl__1766724267
python3 -m trainers.train_model_rtmc --config rtmc/finetune_ch/ch_param_5_5_stl --seed 3 --cultivar Viognier --rnn_fpath _runs/RTMC/ColdHardiness/ParamMTL5/STL/WA/Roza2/Prosser/Viognier/param_mtl__1766725264
python3 -m trainers.train_model_rtmc --config rtmc/finetune_ch/ch_param_5_5_stl --seed 4 --cultivar Viognier --rnn_fpath _runs/RTMC/ColdHardiness/ParamMTL5/STL/WA/Roza2/Prosser/Viognier/param_mtl__1766726261
 
python3 -m trainers.train_model_rtmc --config rtmc/finetune_ch/ch_param_5_5_stl --seed 0 --cultivar Zinfandel --rnn_fpath _runs/RTMC/ColdHardiness/ParamMTL5/STL/WA/Roza2/Prosser/Zinfandel/param_mtl__1766727250
python3 -m trainers.train_model_rtmc --config rtmc/finetune_ch/ch_param_5_5_stl --seed 1 --cultivar Zinfandel --rnn_fpath _runs/RTMC/ColdHardiness/ParamMTL5/STL/WA/Roza2/Prosser/Zinfandel/param_mtl__1766728230
python3 -m trainers.train_model_rtmc --config rtmc/finetune_ch/ch_param_5_5_stl --seed 2 --cultivar Zinfandel --rnn_fpath _runs/RTMC/ColdHardiness/ParamMTL5/STL/WA/Roza2/Prosser/Zinfandel/param_mtl__1766729225
python3 -m trainers.train_model_rtmc --config rtmc/finetune_ch/ch_param_5_5_stl --seed 3 --cultivar Zinfandel --rnn_fpath _runs/RTMC/ColdHardiness/ParamMTL5/STL/WA/Roza2/Prosser/Zinfandel/param_mtl__1766730226
python3 -m trainers.train_model_rtmc --config rtmc/finetune_ch/ch_param_5_5_stl --seed 4 --cultivar Zinfandel --rnn_fpath _runs/RTMC/ColdHardiness/ParamMTL5/STL/WA/Roza2/Prosser/Zinfandel/param_mtl__1766731228