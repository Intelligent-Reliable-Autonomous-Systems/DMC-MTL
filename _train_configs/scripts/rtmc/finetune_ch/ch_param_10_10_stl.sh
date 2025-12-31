#!/bin/bash
#SBATCH -J ch_param_10_10_stl
#SBATCH -o output/ch_param_10_10_stl.out
#SBATCH -e output/ch_param_10_10_stl.err
#SBATCH -p eecs,share,gpu
#SBATCH -t 2-12:00:00
#SBATCH --gres=gpu:1

python3 -m trainers.train_model_rtmc --config rtmc/finetune_ch/ch_param_10_10_stl --seed 0 --cultivar Barbera --rnn_fpath _runs/RTMC/ColdHardiness/ParamMTL10/STL/WA/Roza2/Prosser/Barbera/param_mtl__1766664148
python3 -m trainers.train_model_rtmc --config rtmc/finetune_ch/ch_param_10_10_stl --seed 1 --cultivar Barbera --rnn_fpath _runs/RTMC/ColdHardiness/ParamMTL10/STL/WA/Roza2/Prosser/Barbera/param_mtl__1766670433
python3 -m trainers.train_model_rtmc --config rtmc/finetune_ch/ch_param_10_10_stl --seed 2 --cultivar Barbera --rnn_fpath _runs/RTMC/ColdHardiness/ParamMTL10/STL/WA/Roza2/Prosser/Barbera/param_mtl__1766676726
python3 -m trainers.train_model_rtmc --config rtmc/finetune_ch/ch_param_10_10_stl --seed 3 --cultivar Barbera  --rnn_fpath _runs/RTMC/ColdHardiness/ParamMTL10/STL/WA/Roza2/Prosser/Barbera/param_mtl__1766683036
python3 -m trainers.train_model_rtmc --config rtmc/finetune_ch/ch_param_10_10_stl --seed 4 --cultivar Barbera --rnn_fpath _runs/RTMC/ColdHardiness/ParamMTL10/STL/WA/Roza2/Prosser/Barbera/param_mtl__1766689381
 
python3 -m trainers.train_model_rtmc --config rtmc/finetune_ch/ch_param_10_10_stl --seed 0 --cultivar Cabernet_Franc --rnn_fpath _runs/RTMC/ColdHardiness/ParamMTL10/STL/WA/Roza2/Prosser/Cabernet_Franc/param_mtl__1766695741
python3 -m trainers.train_model_rtmc --config rtmc/finetune_ch/ch_param_10_10_stl --seed 1 --cultivar Cabernet_Franc --rnn_fpath _runs/RTMC/ColdHardiness/ParamMTL10/STL/WA/Roza2/Prosser/Cabernet_Franc/param_mtl__1766698430
python3 -m trainers.train_model_rtmc --config rtmc/finetune_ch/ch_param_10_10_stl --seed 2 --cultivar Cabernet_Franc --rnn_fpath _runs/RTMC/ColdHardiness/ParamMTL10/STL/WA/Roza2/Prosser/Cabernet_Franc/param_mtl__1766701097
python3 -m trainers.train_model_rtmc --config rtmc/finetune_ch/ch_param_10_10_stl --seed 3 --cultivar Cabernet_Franc  --rnn_fpath _runs/RTMC/ColdHardiness/ParamMTL10/STL/WA/Roza2/Prosser/Cabernet_Franc/param_mtl__1766703725
python3 -m trainers.train_model_rtmc --config rtmc/finetune_ch/ch_param_10_10_stl --seed 4 --cultivar Cabernet_Franc --rnn_fpath _runs/RTMC/ColdHardiness/ParamMTL10/STL/WA/Roza2/Prosser/Cabernet_Franc/param_mtl__1766706256
 
python3 -m trainers.train_model_rtmc --config rtmc/finetune_ch/ch_param_10_10_stl --seed 0 --cultivar Cabernet_Sauvignon --rnn_fpath _runs/RTMC/ColdHardiness/ParamMTL10/STL/WA/Roza2/Prosser/Cabernet_Sauvignon/param_mtl__1766708790
python3 -m trainers.train_model_rtmc --config rtmc/finetune_ch/ch_param_10_10_stl --seed 1 --cultivar Cabernet_Sauvignon --rnn_fpath _runs/RTMC/ColdHardiness/ParamMTL10/STL/WA/Roza2/Prosser/Cabernet_Sauvignon/param_mtl__1766714792
python3 -m trainers.train_model_rtmc --config rtmc/finetune_ch/ch_param_10_10_stl --seed 2 --cultivar Cabernet_Sauvignon --rnn_fpath _runs/RTMC/ColdHardiness/ParamMTL10/STL/WA/Roza2/Prosser/Cabernet_Sauvignon/param_mtl__1766720782
python3 -m trainers.train_model_rtmc --config rtmc/finetune_ch/ch_param_10_10_stl --seed 3 --cultivar Cabernet_Sauvignon  --rnn_fpath _runs/RTMC/ColdHardiness/ParamMTL10/STL/WA/Roza2/Prosser/Cabernet_Sauvignon/param_mtl__1766726784
python3 -m trainers.train_model_rtmc --config rtmc/finetune_ch/ch_param_10_10_stl --seed 4 --cultivar Cabernet_Sauvignon --rnn_fpath _runs/RTMC/ColdHardiness/ParamMTL10/STL/WA/Roza2/Prosser/Cabernet_Sauvignon/param_mtl__1766732791
 
python3 -m trainers.train_model_rtmc --config rtmc/finetune_ch/ch_param_10_10_stl --seed 0 --cultivar Chardonnay --rnn_fpath _runs/RTMC/ColdHardiness/ParamMTL10/STL/WA/Roza2/Prosser/Chardonnay/param_mtl__1766738804
python3 -m trainers.train_model_rtmc --config rtmc/finetune_ch/ch_param_10_10_stl --seed 1 --cultivar Chardonnay --rnn_fpath _runs/RTMC/ColdHardiness/ParamMTL10/STL/WA/Roza2/Prosser/Chardonnay/param_mtl__1766744831
python3 -m trainers.train_model_rtmc --config rtmc/finetune_ch/ch_param_10_10_stl --seed 2 --cultivar Chardonnay --rnn_fpath _runs/RTMC/ColdHardiness/ParamMTL10/STL/WA/Roza2/Prosser/Chardonnay/param_mtl__1766750851
python3 -m trainers.train_model_rtmc --config rtmc/finetune_ch/ch_param_10_10_stl --seed 3 --cultivar Chardonnay  --rnn_fpath _runs/RTMC/ColdHardiness/ParamMTL10/STL/WA/Roza2/Prosser/Chardonnay/param_mtl__1766756560
python3 -m trainers.train_model_rtmc --config rtmc/finetune_ch/ch_param_10_10_stl --seed 4 --cultivar Chardonnay --rnn_fpath _runs/RTMC/ColdHardiness/ParamMTL10/STL/WA/Roza2/Prosser/Chardonnay/param_mtl__1766762262
 
python3 -m trainers.train_model_rtmc --config rtmc/finetune_ch/ch_param_10_10_stl --seed 0 --cultivar Chenin_Blanc --rnn_fpath _runs/RTMC/ColdHardiness/ParamMTL10/STL/WA/Roza2/Prosser/Chenin_Blanc/param_mtl__1766767962
python3 -m trainers.train_model_rtmc --config rtmc/finetune_ch/ch_param_10_10_stl --seed 1 --cultivar Chenin_Blanc --rnn_fpath _runs/RTMC/ColdHardiness/ParamMTL10/STL/WA/Roza2/Prosser/Chenin_Blanc/param_mtl__1766773654
python3 -m trainers.train_model_rtmc --config rtmc/finetune_ch/ch_param_10_10_stl --seed 2 --cultivar Chenin_Blanc --rnn_fpath _runs/RTMC/ColdHardiness/ParamMTL10/STL/WA/Roza2/Prosser/Chenin_Blanc/param_mtl__1766779363
python3 -m trainers.train_model_rtmc --config rtmc/finetune_ch/ch_param_10_10_stl --seed 3 --cultivar Chenin_Blanc  --rnn_fpath _runs/RTMC/ColdHardiness/ParamMTL10/STL/WA/Roza2/Prosser/Chenin_Blanc/param_mtl__1766785448
python3 -m trainers.train_model_rtmc --config rtmc/finetune_ch/ch_param_10_10_stl --seed 4 --cultivar Chenin_Blanc --rnn_fpath _runs/RTMC/ColdHardiness/ParamMTL10/STL/WA/Roza2/Prosser/Chenin_Blanc/param_mtl__1766791463
 
python3 -m trainers.train_model_rtmc --config rtmc/finetune_ch/ch_param_10_10_stl --seed 0 --cultivar Concord --rnn_fpath _runs/RTMC/ColdHardiness/ParamMTL10/STL/WA/Roza2/Prosser/Concord/param_mtl__1766797497
python3 -m trainers.train_model_rtmc --config rtmc/finetune_ch/ch_param_10_10_stl --seed 1 --cultivar Concord --rnn_fpath _runs/RTMC/ColdHardiness/ParamMTL10/STL/WA/Roza2/Prosser/Concord/param_mtl__1766803332
python3 -m trainers.train_model_rtmc --config rtmc/finetune_ch/ch_param_10_10_stl --seed 2 --cultivar Concord --rnn_fpath _runs/RTMC/ColdHardiness/ParamMTL10/STL/WA/Roza2/Prosser/Concord/param_mtl__1766809089
python3 -m trainers.train_model_rtmc --config rtmc/finetune_ch/ch_param_10_10_stl --seed 3 --cultivar Concord  --rnn_fpath _runs/RTMC/ColdHardiness/ParamMTL10/STL/WA/Roza2/Prosser/Concord/param_mtl__1766814790
python3 -m trainers.train_model_rtmc --config rtmc/finetune_ch/ch_param_10_10_stl --seed 4 --cultivar Concord --rnn_fpath _runs/RTMC/ColdHardiness/ParamMTL10/STL/WA/Roza2/Prosser/Concord/param_mtl__1766820486
 
python3 -m trainers.train_model_rtmc --config rtmc/finetune_ch/ch_param_10_10_stl --seed 0 --cultivar Gewurztraminer --rnn_fpath _runs/RTMC/ColdHardiness/ParamMTL10/STL/WA/Roza2/Prosser/Gewurztraminer/param_mtl__1766826196
python3 -m trainers.train_model_rtmc --config rtmc/finetune_ch/ch_param_10_10_stl --seed 1 --cultivar Gewurztraminer --rnn_fpath _runs/RTMC/ColdHardiness/ParamMTL10/STL/WA/Roza2/Prosser/Gewurztraminer/param_mtl__1766830242
python3 -m trainers.train_model_rtmc --config rtmc/finetune_ch/ch_param_10_10_stl --seed 2 --cultivar Gewurztraminer --rnn_fpath _runs/RTMC/ColdHardiness/ParamMTL10/STL/WA/Roza2/Prosser/Gewurztraminer/param_mtl__1766834281
python3 -m trainers.train_model_rtmc --config rtmc/finetune_ch/ch_param_10_10_stl --seed 3 --cultivar Gewurztraminer  --rnn_fpath _runs/RTMC/ColdHardiness/ParamMTL10/STL/WA/Roza2/Prosser/Gewurztraminer/param_mtl__1766838338
python3 -m trainers.train_model_rtmc --config rtmc/finetune_ch/ch_param_10_10_stl --seed 4 --cultivar Gewurztraminer --rnn_fpath _runs/RTMC/ColdHardiness/ParamMTL10/STL/WA/Roza2/Prosser/Gewurztraminer/param_mtl__1766842387
 
python3 -m trainers.train_model_rtmc --config rtmc/finetune_ch/ch_param_10_10_stl --seed 0 --cultivar Lemberger --rnn_fpath _runs/RTMC/ColdHardiness/ParamMTL10/STL/WA/Roza2/Prosser/Lemberger/param_mtl__1766846380
python3 -m trainers.train_model_rtmc --config rtmc/finetune_ch/ch_param_10_10_stl --seed 1 --cultivar Lemberger --rnn_fpath _runs/RTMC/ColdHardiness/ParamMTL10/STL/WA/Roza2/Prosser/Lemberger/param_mtl__1766848607
python3 -m trainers.train_model_rtmc --config rtmc/finetune_ch/ch_param_10_10_stl --seed 2 --cultivar Lemberger --rnn_fpath _runs/RTMC/ColdHardiness/ParamMTL10/STL/WA/Roza2/Prosser/Lemberger/param_mtl__1766850839
python3 -m trainers.train_model_rtmc --config rtmc/finetune_ch/ch_param_10_10_stl --seed 3 --cultivar Lemberger  --rnn_fpath _runs/RTMC/ColdHardiness/ParamMTL10/STL/WA/Roza2/Prosser/Lemberger/param_mtl__1766853074
python3 -m trainers.train_model_rtmc --config rtmc/finetune_ch/ch_param_10_10_stl --seed 4 --cultivar Lemberger --rnn_fpath _runs/RTMC/ColdHardiness/ParamMTL10/STL/WA/Roza2/Prosser/Lemberger/param_mtl__1766855313
 
python3 -m trainers.train_model_rtmc --config rtmc/finetune_ch/ch_param_10_10_stl --seed 0 --cultivar Malbec --rnn_fpath _runs/RTMC/ColdHardiness/ParamMTL10/STL/WA/Roza2/Prosser/Malbec/param_mtl__1766857566
python3 -m trainers.train_model_rtmc --config rtmc/finetune_ch/ch_param_10_10_stl --seed 1 --cultivar Malbec --rnn_fpath _runs/RTMC/ColdHardiness/ParamMTL10/STL/WA/Roza2/Prosser/Malbec/param_mtl__1766862905
python3 -m trainers.train_model_rtmc --config rtmc/finetune_ch/ch_param_10_10_stl --seed 2 --cultivar Malbec --rnn_fpath _runs/RTMC/ColdHardiness/ParamMTL10/STL/WA/Roza2/Prosser/Malbec/param_mtl__1766869150
python3 -m trainers.train_model_rtmc --config rtmc/finetune_ch/ch_param_10_10_stl --seed 3 --cultivar Malbec  --rnn_fpath _runs/RTMC/ColdHardiness/ParamMTL10/STL/WA/Roza2/Prosser/Malbec/param_mtl__1766875531
python3 -m trainers.train_model_rtmc --config rtmc/finetune_ch/ch_param_10_10_stl --seed 4 --cultivar Malbec --rnn_fpath _runs/RTMC/ColdHardiness/ParamMTL10/STL/WA/Roza2/Prosser/Malbec/param_mtl__1766881834
 
python3 -m trainers.train_model_rtmc --config rtmc/finetune_ch/ch_param_10_10_stl --seed 0 --cultivar Merlot --rnn_fpath _runs/RTMC/ColdHardiness/ParamMTL10/STL/WA/Roza2/Prosser/Merlot/param_mtl__1766887870
python3 -m trainers.train_model_rtmc --config rtmc/finetune_ch/ch_param_10_10_stl --seed 1 --cultivar Merlot --rnn_fpath _runs/RTMC/ColdHardiness/ParamMTL10/STL/WA/Roza2/Prosser/Merlot/param_mtl__1766893932
python3 -m trainers.train_model_rtmc --config rtmc/finetune_ch/ch_param_10_10_stl --seed 2 --cultivar Merlot --rnn_fpath _runs/RTMC/ColdHardiness/ParamMTL10/STL/WA/Roza2/Prosser/Merlot/param_mtl__1766899974
python3 -m trainers.train_model_rtmc --config rtmc/finetune_ch/ch_param_10_10_stl --seed 3 --cultivar Merlot  --rnn_fpath _runs/RTMC/ColdHardiness/ParamMTL10/STL/WA/Roza2/Prosser/Merlot/param_mtl__1766906017
python3 -m trainers.train_model_rtmc --config rtmc/finetune_ch/ch_param_10_10_stl --seed 4 --cultivar Merlot --rnn_fpath _runs/RTMC/ColdHardiness/ParamMTL10/STL/WA/Roza2/Prosser/Merlot/param_mtl__1766911881
 
python3 -m trainers.train_model_rtmc --config rtmc/finetune_ch/ch_param_10_10_stl --seed 0 --cultivar Mourvedre --rnn_fpath _runs/RTMC/ColdHardiness/ParamMTL10/STL/WA/Roza2/Prosser/Mourvedre/param_mtl__1766917630
python3 -m trainers.train_model_rtmc --config rtmc/finetune_ch/ch_param_10_10_stl --seed 1 --cultivar Mourvedre --rnn_fpath _runs/RTMC/ColdHardiness/ParamMTL10/STL/WA/Roza2/Prosser/Mourvedre/param_mtl__1766922824
python3 -m trainers.train_model_rtmc --config rtmc/finetune_ch/ch_param_10_10_stl --seed 2 --cultivar Mourvedre --rnn_fpath _runs/RTMC/ColdHardiness/ParamMTL10/STL/WA/Roza2/Prosser/Mourvedre/param_mtl__1766928017
python3 -m trainers.train_model_rtmc --config rtmc/finetune_ch/ch_param_10_10_stl --seed 3 --cultivar Mourvedre  --rnn_fpath _runs/RTMC/ColdHardiness/ParamMTL10/STL/WA/Roza2/Prosser/Mourvedre/param_mtl__1766933209
python3 -m trainers.train_model_rtmc --config rtmc/finetune_ch/ch_param_10_10_stl --seed 4 --cultivar Mourvedre --rnn_fpath _runs/RTMC/ColdHardiness/ParamMTL10/STL/WA/Roza2/Prosser/Mourvedre/param_mtl__1766938436
 
python3 -m trainers.train_model_rtmc --config rtmc/finetune_ch/ch_param_10_10_stl --seed 0 --cultivar Nebbiolo --rnn_fpath _runs/RTMC/ColdHardiness/ParamMTL10/STL/WA/Roza2/Prosser/Nebbiolo/param_mtl__1766943632
python3 -m trainers.train_model_rtmc --config rtmc/finetune_ch/ch_param_10_10_stl --seed 1 --cultivar Nebbiolo --rnn_fpath _runs/RTMC/ColdHardiness/ParamMTL10/STL/WA/Roza2/Prosser/Nebbiolo/param_mtl__1766949372
python3 -m trainers.train_model_rtmc --config rtmc/finetune_ch/ch_param_10_10_stl --seed 2 --cultivar Nebbiolo --rnn_fpath _runs/RTMC/ColdHardiness/ParamMTL10/STL/WA/Roza2/Prosser/Nebbiolo/param_mtl__1766955104
python3 -m trainers.train_model_rtmc --config rtmc/finetune_ch/ch_param_10_10_stl --seed 3 --cultivar Nebbiolo  --rnn_fpath _runs/RTMC/ColdHardiness/ParamMTL10/STL/WA/Roza2/Prosser/Nebbiolo/param_mtl__1766960865
python3 -m trainers.train_model_rtmc --config rtmc/finetune_ch/ch_param_10_10_stl --seed 4 --cultivar Nebbiolo --rnn_fpath _runs/RTMC/ColdHardiness/ParamMTL10/STL/WA/Roza2/Prosser/Nebbiolo/param_mtl__1766972443
 
python3 -m trainers.train_model_rtmc --config rtmc/finetune_ch/ch_param_10_10_stl --seed 0 --cultivar Pinot_Gris --rnn_fpath _runs/RTMC/ColdHardiness/ParamMTL10/STL/WA/Roza2/Prosser/Pinot_Gris/param_mtl__1766974544
python3 -m trainers.train_model_rtmc --config rtmc/finetune_ch/ch_param_10_10_stl --seed 1 --cultivar Pinot_Gris --rnn_fpath _runs/RTMC/ColdHardiness/ParamMTL10/STL/WA/Roza2/Prosser/Pinot_Gris/param_mtl__1766976612
python3 -m trainers.train_model_rtmc --config rtmc/finetune_ch/ch_param_10_10_stl --seed 2 --cultivar Pinot_Gris --rnn_fpath _runs/RTMC/ColdHardiness/ParamMTL10/STL/WA/Roza2/Prosser/Pinot_Gris/param_mtl__1766978695
python3 -m trainers.train_model_rtmc --config rtmc/finetune_ch/ch_param_10_10_stl --seed 3 --cultivar Pinot_Gris  --rnn_fpath _runs/RTMC/ColdHardiness/ParamMTL10/STL/WA/Roza2/Prosser/Pinot_Gris/param_mtl__1766980751
python3 -m trainers.train_model_rtmc --config rtmc/finetune_ch/ch_param_10_10_stl --seed 4 --cultivar Pinot_Gris --rnn_fpath _runs/RTMC/ColdHardiness/ParamMTL10/STL/WA/Roza2/Prosser/Pinot_Gris/param_mtl__1766982829
 
python3 -m trainers.train_model_rtmc --config rtmc/finetune_ch/ch_param_10_10_stl --seed 0 --cultivar Riesling --rnn_fpath _runs/RTMC/ColdHardiness/ParamMTL10/STL/WA/Roza2/Prosser/Riesling/param_mtl__1766984902
python3 -m trainers.train_model_rtmc --config rtmc/finetune_ch/ch_param_10_10_stl --seed 1 --cultivar Riesling --rnn_fpath _runs/RTMC/ColdHardiness/ParamMTL10/STL/WA/Roza2/Prosser/Riesling/param_mtl__1766986974
python3 -m trainers.train_model_rtmc --config rtmc/finetune_ch/ch_param_10_10_stl --seed 2 --cultivar Riesling --rnn_fpath _runs/RTMC/ColdHardiness/ParamMTL10/STL/WA/Roza2/Prosser/Riesling/param_mtl__1766989079
python3 -m trainers.train_model_rtmc --config rtmc/finetune_ch/ch_param_10_10_stl --seed 3 --cultivar Riesling  --rnn_fpath _runs/RTMC/ColdHardiness/ParamMTL10/STL/WA/Roza2/Prosser/Riesling/param_mtl__1766991178
python3 -m trainers.train_model_rtmc --config rtmc/finetune_ch/ch_param_10_10_stl --seed 4 --cultivar Riesling --rnn_fpath _runs/RTMC/ColdHardiness/ParamMTL10/STL/WA/Roza2/Prosser/Riesling/param_mtl__1766993268
 
python3 -m trainers.train_model_rtmc --config rtmc/finetune_ch/ch_param_10_10_stl --seed 0 --cultivar Sangiovese --rnn_fpath _runs/RTMC/ColdHardiness/ParamMTL10/STL/WA/Roza2/Prosser/Sangiovese/param_mtl__1766995356
python3 -m trainers.train_model_rtmc --config rtmc/finetune_ch/ch_param_10_10_stl --seed 1 --cultivar Sangiovese --rnn_fpath _runs/RTMC/ColdHardiness/ParamMTL10/STL/WA/Roza2/Prosser/Sangiovese/param_mtl__1766997435
python3 -m trainers.train_model_rtmc --config rtmc/finetune_ch/ch_param_10_10_stl --seed 2 --cultivar Sangiovese --rnn_fpath _runs/RTMC/ColdHardiness/ParamMTL10/STL/WA/Roza2/Prosser/Sangiovese/param_mtl__1766999509
python3 -m trainers.train_model_rtmc --config rtmc/finetune_ch/ch_param_10_10_stl --seed 3 --cultivar Sangiovese  --rnn_fpath _runs/RTMC/ColdHardiness/ParamMTL10/STL/WA/Roza2/Prosser/Sangiovese/param_mtl__1767001593
python3 -m trainers.train_model_rtmc --config rtmc/finetune_ch/ch_param_10_10_stl --seed 4 --cultivar Sangiovese --rnn_fpath _runs/RTMC/ColdHardiness/ParamMTL10/STL/WA/Roza2/Prosser/Sangiovese/param_mtl__1767003666
 
python3 -m trainers.train_model_rtmc --config rtmc/finetune_ch/ch_param_10_10_stl --seed 0 --cultivar Sauvignon_Blanc --rnn_fpath _runs/RTMC/ColdHardiness/ParamMTL10/STL/WA/Roza2/Prosser/Sauvignon_Blanc/param_mtl__1767005740
python3 -m trainers.train_model_rtmc --config rtmc/finetune_ch/ch_param_10_10_stl --seed 1 --cultivar Sauvignon_Blanc --rnn_fpath _runs/RTMC/ColdHardiness/ParamMTL10/STL/WA/Roza2/Prosser/Sauvignon_Blanc/param_mtl__1767007798
python3 -m trainers.train_model_rtmc --config rtmc/finetune_ch/ch_param_10_10_stl --seed 2 --cultivar Sauvignon_Blanc --rnn_fpath _runs/RTMC/ColdHardiness/ParamMTL10/STL/WA/Roza2/Prosser/Sauvignon_Blanc/param_mtl__1767009875
python3 -m trainers.train_model_rtmc --config rtmc/finetune_ch/ch_param_10_10_stl --seed 3 --cultivar Sauvignon_Blanc  --rnn_fpath _runs/RTMC/ColdHardiness/ParamMTL10/STL/WA/Roza2/Prosser/Sauvignon_Blanc/param_mtl__1767011960
python3 -m trainers.train_model_rtmc --config rtmc/finetune_ch/ch_param_10_10_stl --seed 4 --cultivar Sauvignon_Blanc --rnn_fpath _runs/RTMC/ColdHardiness/ParamMTL10/STL/WA/Roza2/Prosser/Sauvignon_Blanc/param_mtl__1767014047
 
python3 -m trainers.train_model_rtmc --config rtmc/finetune_ch/ch_param_10_10_stl --seed 0 --cultivar Semillon --rnn_fpath _runs/RTMC/ColdHardiness/ParamMTL10/STL/WA/Roza2/Prosser/Semillon/param_mtl__1767016118
python3 -m trainers.train_model_rtmc --config rtmc/finetune_ch/ch_param_10_10_stl --seed 1 --cultivar Semillon --rnn_fpath _runs/RTMC/ColdHardiness/ParamMTL10/STL/WA/Roza2/Prosser/Semillon/param_mtl__1767018201
python3 -m trainers.train_model_rtmc --config rtmc/finetune_ch/ch_param_10_10_stl --seed 2 --cultivar Semillon --rnn_fpath _runs/RTMC/ColdHardiness/ParamMTL10/STL/WA/Roza2/Prosser/Semillon/param_mtl__1767020275
python3 -m trainers.train_model_rtmc --config rtmc/finetune_ch/ch_param_10_10_stl --seed 3 --cultivar Semillon  --rnn_fpath _runs/RTMC/ColdHardiness/ParamMTL10/STL/WA/Roza2/Prosser/Semillon/param_mtl__1767022363
python3 -m trainers.train_model_rtmc --config rtmc/finetune_ch/ch_param_10_10_stl --seed 4 --cultivar Semillon --rnn_fpath _runs/RTMC/ColdHardiness/ParamMTL10/STL/WA/Roza2/Prosser/Semillon/param_mtl__1767024480
 
python3 -m trainers.train_model_rtmc --config rtmc/finetune_ch/ch_param_10_10_stl --seed 0 --cultivar Syrah --rnn_fpath _runs/RTMC/ColdHardiness/ParamMTL10/STL/WA/Roza2/Prosser/Syrah/param_mtl__1767026578
python3 -m trainers.train_model_rtmc --config rtmc/finetune_ch/ch_param_10_10_stl --seed 1 --cultivar Syrah --rnn_fpath _runs/RTMC/ColdHardiness/ParamMTL10/STL/WA/Roza2/Prosser/Syrah/param_mtl__1767028697
python3 -m trainers.train_model_rtmc --config rtmc/finetune_ch/ch_param_10_10_stl --seed 2 --cultivar Syrah --rnn_fpath _runs/RTMC/ColdHardiness/ParamMTL10/STL/WA/Roza2/Prosser/Syrah/param_mtl__1767030787
python3 -m trainers.train_model_rtmc --config rtmc/finetune_ch/ch_param_10_10_stl --seed 3 --cultivar Syrah  --rnn_fpath _runs/RTMC/ColdHardiness/ParamMTL10/STL/WA/Roza2/Prosser/Syrah/param_mtl__1767032885
python3 -m trainers.train_model_rtmc --config rtmc/finetune_ch/ch_param_10_10_stl --seed 4 --cultivar Syrah --rnn_fpath _runs/RTMC/ColdHardiness/ParamMTL10/STL/WA/Roza2/Prosser/Syrah/param_mtl__1767034950
 
python3 -m trainers.train_model_rtmc --config rtmc/finetune_ch/ch_param_10_10_stl --seed 0 --cultivar Viognier --rnn_fpath _runs/RTMC/ColdHardiness/ParamMTL10/STL/WA/Roza2/Prosser/Viognier/param_mtl__1767037067
python3 -m trainers.train_model_rtmc --config rtmc/finetune_ch/ch_param_10_10_stl --seed 1 --cultivar Viognier --rnn_fpath _runs/RTMC/ColdHardiness/ParamMTL10/STL/WA/Roza2/Prosser/Viognier/param_mtl__1767039162
python3 -m trainers.train_model_rtmc --config rtmc/finetune_ch/ch_param_10_10_stl --seed 2 --cultivar Viognier --rnn_fpath _runs/RTMC/ColdHardiness/ParamMTL10/STL/WA/Roza2/Prosser/Viognier/param_mtl__1767041247
python3 -m trainers.train_model_rtmc --config rtmc/finetune_ch/ch_param_10_10_stl --seed 3 --cultivar Viognier  --rnn_fpath _runs/RTMC/ColdHardiness/ParamMTL10/STL/WA/Roza2/Prosser/Viognier/param_mtl__1767043319
python3 -m trainers.train_model_rtmc --config rtmc/finetune_ch/ch_param_10_10_stl --seed 4 --cultivar Viognier --rnn_fpath _runs/RTMC/ColdHardiness/ParamMTL10/STL/WA/Roza2/Prosser/Viognier/param_mtl__1767045394
 
python3 -m trainers.train_model_rtmc --config rtmc/finetune_ch/ch_param_10_10_stl --seed 0 --cultivar Zinfandel --rnn_fpath _runs/RTMC/ColdHardiness/ParamMTL10/STL/WA/Roza2/Prosser/Zinfandel/param_mtl__1767047479
python3 -m trainers.train_model_rtmc --config rtmc/finetune_ch/ch_param_10_10_stl --seed 1 --cultivar Zinfandel --rnn_fpath _runs/RTMC/ColdHardiness/ParamMTL10/STL/WA/Roza2/Prosser/Zinfandel/param_mtl__1767049563
python3 -m trainers.train_model_rtmc --config rtmc/finetune_ch/ch_param_10_10_stl --seed 2 --cultivar Zinfandel --rnn_fpath _runs/RTMC/ColdHardiness/ParamMTL10/STL/WA/Roza2/Prosser/Zinfandel/param_mtl__1767051653
python3 -m trainers.train_model_rtmc --config rtmc/finetune_ch/ch_param_10_10_stl --seed 3 --cultivar Zinfandel  --rnn_fpath _runs/RTMC/ColdHardiness/ParamMTL10/STL/WA/Roza2/Prosser/Zinfandel/param_mtl__1767053725
python3 -m trainers.train_model_rtmc --config rtmc/finetune_ch/ch_param_10_10_stl --seed 4 --cultivar Zinfandel --rnn_fpath _runs/RTMC/ColdHardiness/ParamMTL10/STL/WA/Roza2/Prosser/Zinfandel/param_mtl__1767055798