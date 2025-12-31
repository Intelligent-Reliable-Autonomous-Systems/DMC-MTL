#!/bin/bash
#SBATCH -J pheno_param_5_5_stl
#SBATCH -o output/pheno_param_5_5_stl.out
#SBATCH -e output/pheno_param_5_5_stl.err
#SBATCH -p eecs,share,gpu
#SBATCH -t 2-12:00:00
#SBATCH --gres=gpu:1

python3 -m trainers.train_model_rtmc --config rtmc/finetune_pheno/pheno_param_5_5_stl --seed 0 --cultivar Aligote --rnn_fpath _runs/RTMC/Phenology/ParamMTL5/STL/WA/Roza2/Prosser/Aligote/param_mtl__1766486859
python3 -m trainers.train_model_rtmc --config rtmc/finetune_pheno/pheno_param_5_5_stl --seed 1 --cultivar Aligote --rnn_fpath _runs/RTMC/Phenology/ParamMTL5/STL/WA/Roza2/Prosser/Aligote/param_mtl__1766489759
python3 -m trainers.train_model_rtmc --config rtmc/finetune_pheno/pheno_param_5_5_stl --seed 2 --cultivar Aligote --rnn_fpath _runs/RTMC/Phenology/ParamMTL5/STL/WA/Roza2/Prosser/Aligote/param_mtl__1766492632
python3 -m trainers.train_model_rtmc --config rtmc/finetune_pheno/pheno_param_5_5_stl --seed 3 --cultivar Aligote --rnn_fpath _runs/RTMC/Phenology/ParamMTL5/STL/WA/Roza2/Prosser/Aligote/param_mtl__1766495514
python3 -m trainers.train_model_rtmc --config rtmc/finetune_pheno/pheno_param_5_5_stl --seed 4 --cultivar Aligote --rnn_fpath _runs/RTMC/Phenology/ParamMTL5/STL/WA/Roza2/Prosser/Aligote/param_mtl__1766498392
 
python3 -m trainers.train_model_rtmc --config rtmc/finetune_pheno/pheno_param_5_5_stl --seed 0 --cultivar Alvarinho --rnn_fpath _runs/RTMC/Phenology/ParamMTL5/STL/WA/Roza2/Prosser/Alvarinho/param_mtl__1766501272
python3 -m trainers.train_model_rtmc --config rtmc/finetune_pheno/pheno_param_5_5_stl --seed 1 --cultivar Alvarinho --rnn_fpath _runs/RTMC/Phenology/ParamMTL5/STL/WA/Roza2/Prosser/Alvarinho/param_mtl__1766504164
python3 -m trainers.train_model_rtmc --config rtmc/finetune_pheno/pheno_param_5_5_stl --seed 2 --cultivar Alvarinho --rnn_fpath _runs/RTMC/Phenology/ParamMTL5/STL/WA/Roza2/Prosser/Alvarinho/param_mtl__1766507056
python3 -m trainers.train_model_rtmc --config rtmc/finetune_pheno/pheno_param_5_5_stl --seed 3 --cultivar Alvarinho --rnn_fpath _runs/RTMC/Phenology/ParamMTL5/STL/WA/Roza2/Prosser/Alvarinho/param_mtl__1766509933
python3 -m trainers.train_model_rtmc --config rtmc/finetune_pheno/pheno_param_5_5_stl --seed 4 --cultivar Alvarinho --rnn_fpath _runs/RTMC/Phenology/ParamMTL5/STL/WA/Roza2/Prosser/Alvarinho/param_mtl__1766512829
 
python3 -m trainers.train_model_rtmc --config rtmc/finetune_pheno/pheno_param_5_5_stl --seed 0 --cultivar Auxerrois --rnn_fpath _runs/RTMC/Phenology/ParamMTL5/STL/WA/Roza2/Prosser/Auxerrois/param_mtl__1766515717
python3 -m trainers.train_model_rtmc --config rtmc/finetune_pheno/pheno_param_5_5_stl --seed 1 --cultivar Auxerrois --rnn_fpath _runs/RTMC/Phenology/ParamMTL5/STL/WA/Roza2/Prosser/Auxerrois/param_mtl__1766518622
python3 -m trainers.train_model_rtmc --config rtmc/finetune_pheno/pheno_param_5_5_stl --seed 2 --cultivar Auxerrois --rnn_fpath _runs/RTMC/Phenology/ParamMTL5/STL/WA/Roza2/Prosser/Auxerrois/param_mtl__1766521524
python3 -m trainers.train_model_rtmc --config rtmc/finetune_pheno/pheno_param_5_5_stl --seed 3 --cultivar Auxerrois --rnn_fpath _runs/RTMC/Phenology/ParamMTL5/STL/WA/Roza2/Prosser/Auxerrois/param_mtl__1766524638
python3 -m trainers.train_model_rtmc --config rtmc/finetune_pheno/pheno_param_5_5_stl --seed 4 --cultivar Auxerrois --rnn_fpath _runs/RTMC/Phenology/ParamMTL5/STL/WA/Roza2/Prosser/Auxerrois/param_mtl__1766527972
 
python3 -m trainers.train_model_rtmc --config rtmc/finetune_pheno/pheno_param_5_5_stl --seed 0 --cultivar Barbera --rnn_fpath _runs/RTMC/Phenology/ParamMTL5/STL/WA/Roza2/Prosser/Barbera/param_mtl__1766531308
python3 -m trainers.train_model_rtmc --config rtmc/finetune_pheno/pheno_param_5_5_stl --seed 1 --cultivar Barbera --rnn_fpath _runs/RTMC/Phenology/ParamMTL5/STL/WA/Roza2/Prosser/Barbera/param_mtl__1766534647
python3 -m trainers.train_model_rtmc --config rtmc/finetune_pheno/pheno_param_5_5_stl --seed 2 --cultivar Barbera --rnn_fpath _runs/RTMC/Phenology/ParamMTL5/STL/WA/Roza2/Prosser/Barbera/param_mtl__1766537997
python3 -m trainers.train_model_rtmc --config rtmc/finetune_pheno/pheno_param_5_5_stl --seed 3 --cultivar Barbera --rnn_fpath _runs/RTMC/Phenology/ParamMTL5/STL/WA/Roza2/Prosser/Barbera/param_mtl__1766541332
python3 -m trainers.train_model_rtmc --config rtmc/finetune_pheno/pheno_param_5_5_stl --seed 4 --cultivar Barbera --rnn_fpath _runs/RTMC/Phenology/ParamMTL5/STL/WA/Roza2/Prosser/Barbera/param_mtl__1766544649
 
python3 -m trainers.train_model_rtmc --config rtmc/finetune_pheno/pheno_param_5_5_stl --seed 0 --cultivar Cabernet_Franc --rnn_fpath _runs/RTMC/Phenology/ParamMTL5/STL/WA/Roza2/Prosser/Cabernet_Franc/param_mtl__1766547976
python3 -m trainers.train_model_rtmc --config rtmc/finetune_pheno/pheno_param_5_5_stl --seed 1 --cultivar Cabernet_Franc --rnn_fpath _runs/RTMC/Phenology/ParamMTL5/STL/WA/Roza2/Prosser/Cabernet_Franc/param_mtl__1766551298
python3 -m trainers.train_model_rtmc --config rtmc/finetune_pheno/pheno_param_5_5_stl --seed 2 --cultivar Cabernet_Franc --rnn_fpath _runs/RTMC/Phenology/ParamMTL5/STL/WA/Roza2/Prosser/Cabernet_Franc/param_mtl__1766554643
python3 -m trainers.train_model_rtmc --config rtmc/finetune_pheno/pheno_param_5_5_stl --seed 3 --cultivar Cabernet_Franc --rnn_fpath _runs/RTMC/Phenology/ParamMTL5/STL/WA/Roza2/Prosser/Cabernet_Franc/param_mtl__1766557989
python3 -m trainers.train_model_rtmc --config rtmc/finetune_pheno/pheno_param_5_5_stl --seed 4 --cultivar Cabernet_Franc --rnn_fpath _runs/RTMC/Phenology/ParamMTL5/STL/WA/Roza2/Prosser/Cabernet_Franc/param_mtl__1766561338
 
python3 -m trainers.train_model_rtmc --config rtmc/finetune_pheno/pheno_param_5_5_stl --seed 0 --cultivar Cabernet_Sauvignon --rnn_fpath _runs/RTMC/Phenology/ParamMTL5/STL/WA/Roza2/Prosser/Cabernet_Sauvignon/param_mtl__1766564677
python3 -m trainers.train_model_rtmc --config rtmc/finetune_pheno/pheno_param_5_5_stl --seed 1 --cultivar Cabernet_Sauvignon --rnn_fpath _runs/RTMC/Phenology/ParamMTL5/STL/WA/Roza2/Prosser/Cabernet_Sauvignon/param_mtl__1766568024
python3 -m trainers.train_model_rtmc --config rtmc/finetune_pheno/pheno_param_5_5_stl --seed 2 --cultivar Cabernet_Sauvignon --rnn_fpath _runs/RTMC/Phenology/ParamMTL5/STL/WA/Roza2/Prosser/Cabernet_Sauvignon/param_mtl__1766571606
python3 -m trainers.train_model_rtmc --config rtmc/finetune_pheno/pheno_param_5_5_stl --seed 3 --cultivar Cabernet_Sauvignon --rnn_fpath _runs/RTMC/Phenology/ParamMTL5/STL/WA/Roza2/Prosser/Cabernet_Sauvignon/param_mtl__1766575178
python3 -m trainers.train_model_rtmc --config rtmc/finetune_pheno/pheno_param_5_5_stl --seed 4 --cultivar Cabernet_Sauvignon --rnn_fpath _runs/RTMC/Phenology/ParamMTL5/STL/WA/Roza2/Prosser/Cabernet_Sauvignon/param_mtl__1766578769
 
python3 -m trainers.train_model_rtmc --config rtmc/finetune_pheno/pheno_param_5_5_stl --seed 0 --cultivar Chardonnay --rnn_fpath _runs/RTMC/Phenology/ParamMTL5/STL/WA/Roza2/Prosser/Chardonnay/param_mtl__1766582350
python3 -m trainers.train_model_rtmc --config rtmc/finetune_pheno/pheno_param_5_5_stl --seed 1 --cultivar Chardonnay --rnn_fpath _runs/RTMC/Phenology/ParamMTL5/STL/WA/Roza2/Prosser/Chardonnay/param_mtl__1766586068
python3 -m trainers.train_model_rtmc --config rtmc/finetune_pheno/pheno_param_5_5_stl --seed 2 --cultivar Chardonnay --rnn_fpath _runs/RTMC/Phenology/ParamMTL5/STL/WA/Roza2/Prosser/Chardonnay/param_mtl__1766589847
python3 -m trainers.train_model_rtmc --config rtmc/finetune_pheno/pheno_param_5_5_stl --seed 3 --cultivar Chardonnay --rnn_fpath _runs/RTMC/Phenology/ParamMTL5/STL/WA/Roza2/Prosser/Chardonnay/param_mtl__1766593602
python3 -m trainers.train_model_rtmc --config rtmc/finetune_pheno/pheno_param_5_5_stl --seed 4 --cultivar Chardonnay --rnn_fpath _runs/RTMC/Phenology/ParamMTL5/STL/WA/Roza2/Prosser/Chardonnay/param_mtl__1766597404
 
python3 -m trainers.train_model_rtmc --config rtmc/finetune_pheno/pheno_param_5_5_stl --seed 0 --cultivar Chenin_Blanc --rnn_fpath _runs/RTMC/Phenology/ParamMTL5/STL/WA/Roza2/Prosser/Chenin_Blanc/param_mtl__1766601175
python3 -m trainers.train_model_rtmc --config rtmc/finetune_pheno/pheno_param_5_5_stl --seed 1 --cultivar Chenin_Blanc --rnn_fpath _runs/RTMC/Phenology/ParamMTL5/STL/WA/Roza2/Prosser/Chenin_Blanc/param_mtl__1766604936
python3 -m trainers.train_model_rtmc --config rtmc/finetune_pheno/pheno_param_5_5_stl --seed 2 --cultivar Chenin_Blanc --rnn_fpath _runs/RTMC/Phenology/ParamMTL5/STL/WA/Roza2/Prosser/Chenin_Blanc/param_mtl__1766608743
python3 -m trainers.train_model_rtmc --config rtmc/finetune_pheno/pheno_param_5_5_stl --seed 3 --cultivar Chenin_Blanc --rnn_fpath _runs/RTMC/Phenology/ParamMTL5/STL/WA/Roza2/Prosser/Chenin_Blanc/param_mtl__1766612535
python3 -m trainers.train_model_rtmc --config rtmc/finetune_pheno/pheno_param_5_5_stl --seed 4 --cultivar Chenin_Blanc --rnn_fpath _runs/RTMC/Phenology/ParamMTL5/STL/WA/Roza2/Prosser/Chenin_Blanc/param_mtl__1766616307
 
python3 -m trainers.train_model_rtmc --config rtmc/finetune_pheno/pheno_param_5_5_stl --seed 0 --cultivar Concord --rnn_fpath _runs/RTMC/Phenology/ParamMTL5/STL/WA/Roza2/Prosser/Concord/param_mtl__1766620096
python3 -m trainers.train_model_rtmc --config rtmc/finetune_pheno/pheno_param_5_5_stl --seed 1 --cultivar Concord --rnn_fpath _runs/RTMC/Phenology/ParamMTL5/STL/WA/Roza2/Prosser/Concord/param_mtl__1766623876
python3 -m trainers.train_model_rtmc --config rtmc/finetune_pheno/pheno_param_5_5_stl --seed 2 --cultivar Concord --rnn_fpath _runs/RTMC/Phenology/ParamMTL5/STL/WA/Roza2/Prosser/Concord/param_mtl__1766627682
python3 -m trainers.train_model_rtmc --config rtmc/finetune_pheno/pheno_param_5_5_stl --seed 3 --cultivar Concord --rnn_fpath _runs/RTMC/Phenology/ParamMTL5/STL/WA/Roza2/Prosser/Concord/param_mtl__1766631457
python3 -m trainers.train_model_rtmc --config rtmc/finetune_pheno/pheno_param_5_5_stl --seed 4 --cultivar Concord --rnn_fpath _runs/RTMC/Phenology/ParamMTL5/STL/WA/Roza2/Prosser/Concord/param_mtl__1766635247
 
python3 -m trainers.train_model_rtmc --config rtmc/finetune_pheno/pheno_param_5_5_stl --seed 0 --cultivar Durif --rnn_fpath _runs/RTMC/Phenology/ParamMTL5/STL/WA/Roza2/Prosser/Durif/param_mtl__1766639018
python3 -m trainers.train_model_rtmc --config rtmc/finetune_pheno/pheno_param_5_5_stl --seed 1 --cultivar Durif --rnn_fpath _runs/RTMC/Phenology/ParamMTL5/STL/WA/Roza2/Prosser/Durif/param_mtl__1766642814
python3 -m trainers.train_model_rtmc --config rtmc/finetune_pheno/pheno_param_5_5_stl --seed 2 --cultivar Durif --rnn_fpath _runs/RTMC/Phenology/ParamMTL5/STL/WA/Roza2/Prosser/Durif/param_mtl__1766646594
python3 -m trainers.train_model_rtmc --config rtmc/finetune_pheno/pheno_param_5_5_stl --seed 3 --cultivar Durif --rnn_fpath _runs/RTMC/Phenology/ParamMTL5/STL/WA/Roza2/Prosser/Durif/param_mtl__1766650393
python3 -m trainers.train_model_rtmc --config rtmc/finetune_pheno/pheno_param_5_5_stl --seed 4 --cultivar Durif --rnn_fpath _runs/RTMC/Phenology/ParamMTL5/STL/WA/Roza2/Prosser/Durif/param_mtl__1766654166
 
python3 -m trainers.train_model_rtmc --config rtmc/finetune_pheno/pheno_param_5_5_stl --seed 0 --cultivar Gewurztraminer --rnn_fpath _runs/RTMC/Phenology/ParamMTL5/STL/WA/Roza2/Prosser/Gewurztraminer/param_mtl__1766657976
python3 -m trainers.train_model_rtmc --config rtmc/finetune_pheno/pheno_param_5_5_stl --seed 1 --cultivar Gewurztraminer --rnn_fpath _runs/RTMC/Phenology/ParamMTL5/STL/WA/Roza2/Prosser/Gewurztraminer/param_mtl__1766661803
python3 -m trainers.train_model_rtmc --config rtmc/finetune_pheno/pheno_param_5_5_stl --seed 2 --cultivar Gewurztraminer --rnn_fpath _runs/RTMC/Phenology/ParamMTL5/STL/WA/Roza2/Prosser/Gewurztraminer/param_mtl__1766665684
python3 -m trainers.train_model_rtmc --config rtmc/finetune_pheno/pheno_param_5_5_stl --seed 3 --cultivar Gewurztraminer --rnn_fpath _runs/RTMC/Phenology/ParamMTL5/STL/WA/Roza2/Prosser/Gewurztraminer/param_mtl__1766669720
python3 -m trainers.train_model_rtmc --config rtmc/finetune_pheno/pheno_param_5_5_stl --seed 4 --cultivar Gewurztraminer --rnn_fpath _runs/RTMC/Phenology/ParamMTL5/STL/WA/Roza2/Prosser/Gewurztraminer/param_mtl__1766673711
 
python3 -m trainers.train_model_rtmc --config rtmc/finetune_pheno/pheno_param_5_5_stl --seed 0 --cultivar Green_Veltliner --rnn_fpath _runs/RTMC/Phenology/ParamMTL5/STL/WA/Roza2/Prosser/Green_Veltliner/param_mtl__1766677735
python3 -m trainers.train_model_rtmc --config rtmc/finetune_pheno/pheno_param_5_5_stl --seed 1 --cultivar Green_Veltliner --rnn_fpath _runs/RTMC/Phenology/ParamMTL5/STL/WA/Roza2/Prosser/Green_Veltliner/param_mtl__1766681739
python3 -m trainers.train_model_rtmc --config rtmc/finetune_pheno/pheno_param_5_5_stl --seed 2 --cultivar Green_Veltliner --rnn_fpath _runs/RTMC/Phenology/ParamMTL5/STL/WA/Roza2/Prosser/Green_Veltliner/param_mtl__1766685763
python3 -m trainers.train_model_rtmc --config rtmc/finetune_pheno/pheno_param_5_5_stl --seed 3 --cultivar Green_Veltliner --rnn_fpath _runs/RTMC/Phenology/ParamMTL5/STL/WA/Roza2/Prosser/Green_Veltliner/param_mtl__1766689752
python3 -m trainers.train_model_rtmc --config rtmc/finetune_pheno/pheno_param_5_5_stl --seed 4 --cultivar Green_Veltliner --rnn_fpath _runs/RTMC/Phenology/ParamMTL5/STL/WA/Roza2/Prosser/Green_Veltliner/param_mtl__1766693774
 
python3 -m trainers.train_model_rtmc --config rtmc/finetune_pheno/pheno_param_5_5_stl --seed 0 --cultivar Grenache --rnn_fpath _runs/RTMC/Phenology/ParamMTL5/STL/WA/Roza2/Prosser/Grenache/param_mtl__1766697806
python3 -m trainers.train_model_rtmc --config rtmc/finetune_pheno/pheno_param_5_5_stl --seed 1 --cultivar Grenache --rnn_fpath _runs/RTMC/Phenology/ParamMTL5/STL/WA/Roza2/Prosser/Grenache/param_mtl__1766869347
python3 -m trainers.train_model_rtmc --config rtmc/finetune_pheno/pheno_param_5_5_stl --seed 2 --cultivar Grenache --rnn_fpath _runs/RTMC/Phenology/ParamMTL5/STL/WA/Roza2/Prosser/Grenache/param_mtl__1766873401
python3 -m trainers.train_model_rtmc --config rtmc/finetune_pheno/pheno_param_5_5_stl --seed 3 --cultivar Grenache --rnn_fpath _runs/RTMC/Phenology/ParamMTL5/STL/WA/Roza2/Prosser/Grenache/param_mtl__1766877448
python3 -m trainers.train_model_rtmc --config rtmc/finetune_pheno/pheno_param_5_5_stl --seed 4 --cultivar Grenache --rnn_fpath _runs/RTMC/Phenology/ParamMTL5/STL/WA/Roza2/Prosser/Grenache/param_mtl__1766881423
 
python3 -m trainers.train_model_rtmc --config rtmc/finetune_pheno/pheno_param_5_5_stl --seed 0 --cultivar Lemberger --rnn_fpath _runs/RTMC/Phenology/ParamMTL5/STL/WA/Roza2/Prosser/Lemberger/param_mtl__1766885253
python3 -m trainers.train_model_rtmc --config rtmc/finetune_pheno/pheno_param_5_5_stl --seed 1 --cultivar Lemberger --rnn_fpath _runs/RTMC/Phenology/ParamMTL5/STL/WA/Roza2/Prosser/Lemberger/param_mtl__1766889071
python3 -m trainers.train_model_rtmc --config rtmc/finetune_pheno/pheno_param_5_5_stl --seed 2 --cultivar Lemberger --rnn_fpath _runs/RTMC/Phenology/ParamMTL5/STL/WA/Roza2/Prosser/Lemberger/param_mtl__1766892918
python3 -m trainers.train_model_rtmc --config rtmc/finetune_pheno/pheno_param_5_5_stl --seed 3 --cultivar Lemberger --rnn_fpath _runs/RTMC/Phenology/ParamMTL5/STL/WA/Roza2/Prosser/Lemberger/param_mtl__1766896754
python3 -m trainers.train_model_rtmc --config rtmc/finetune_pheno/pheno_param_5_5_stl --seed 4 --cultivar Lemberger --rnn_fpath _runs/RTMC/Phenology/ParamMTL5/STL/WA/Roza2/Prosser/Lemberger/param_mtl__1766900585
 
python3 -m trainers.train_model_rtmc --config rtmc/finetune_pheno/pheno_param_5_5_stl --seed 0 --cultivar Malbec --rnn_fpath _runs/RTMC/Phenology/ParamMTL5/STL/WA/Roza2/Prosser/Malbec/param_mtl__1766904417
python3 -m trainers.train_model_rtmc --config rtmc/finetune_pheno/pheno_param_5_5_stl --seed 1 --cultivar Malbec --rnn_fpath _runs/RTMC/Phenology/ParamMTL5/STL/WA/Roza2/Prosser/Malbec/param_mtl__1766908245
python3 -m trainers.train_model_rtmc --config rtmc/finetune_pheno/pheno_param_5_5_stl --seed 2 --cultivar Malbec --rnn_fpath _runs/RTMC/Phenology/ParamMTL5/STL/WA/Roza2/Prosser/Malbec/param_mtl__1766911898
python3 -m trainers.train_model_rtmc --config rtmc/finetune_pheno/pheno_param_5_5_stl --seed 3 --cultivar Malbec --rnn_fpath _runs/RTMC/Phenology/ParamMTL5/STL/WA/Roza2/Prosser/Malbec/param_mtl__1766915531
python3 -m trainers.train_model_rtmc --config rtmc/finetune_pheno/pheno_param_5_5_stl --seed 4 --cultivar Malbec --rnn_fpath _runs/RTMC/Phenology/ParamMTL5/STL/WA/Roza2/Prosser/Malbec/param_mtl__1766919177
 
python3 -m trainers.train_model_rtmc --config rtmc/finetune_pheno/pheno_param_5_5_stl --seed 0 --cultivar Melon --rnn_fpath _runs/RTMC/Phenology/ParamMTL5/STL/WA/Roza2/Prosser/Melon/param_mtl__1766922846
python3 -m trainers.train_model_rtmc --config rtmc/finetune_pheno/pheno_param_5_5_stl --seed 1 --cultivar Melon --rnn_fpath _runs/RTMC/Phenology/ParamMTL5/STL/WA/Roza2/Prosser/Melon/param_mtl__1766926508
python3 -m trainers.train_model_rtmc --config rtmc/finetune_pheno/pheno_param_5_5_stl --seed 2 --cultivar Melon --rnn_fpath _runs/RTMC/Phenology/ParamMTL5/STL/WA/Roza2/Prosser/Melon/param_mtl__1766930166
python3 -m trainers.train_model_rtmc --config rtmc/finetune_pheno/pheno_param_5_5_stl --seed 3 --cultivar Melon --rnn_fpath _runs/RTMC/Phenology/ParamMTL5/STL/WA/Roza2/Prosser/Melon/param_mtl__1766933823
python3 -m trainers.train_model_rtmc --config rtmc/finetune_pheno/pheno_param_5_5_stl --seed 4 --cultivar Melon --rnn_fpath _runs/RTMC/Phenology/ParamMTL5/STL/WA/Roza2/Prosser/Melon/param_mtl__1766937474
 
python3 -m trainers.train_model_rtmc --config rtmc/finetune_pheno/pheno_param_5_5_stl --seed 0 --cultivar Merlot --rnn_fpath _runs/RTMC/Phenology/ParamMTL5/STL/WA/Roza2/Prosser/Merlot/param_mtl__1766941109
python3 -m trainers.train_model_rtmc --config rtmc/finetune_pheno/pheno_param_5_5_stl --seed 1 --cultivar Merlot --rnn_fpath _runs/RTMC/Phenology/ParamMTL5/STL/WA/Roza2/Prosser/Merlot/param_mtl__1766944760
python3 -m trainers.train_model_rtmc --config rtmc/finetune_pheno/pheno_param_5_5_stl --seed 2 --cultivar Merlot --rnn_fpath _runs/RTMC/Phenology/ParamMTL5/STL/WA/Roza2/Prosser/Merlot/param_mtl__1766948404
python3 -m trainers.train_model_rtmc --config rtmc/finetune_pheno/pheno_param_5_5_stl --seed 3 --cultivar Merlot --rnn_fpath _runs/RTMC/Phenology/ParamMTL5/STL/WA/Roza2/Prosser/Merlot/param_mtl__1766952042
python3 -m trainers.train_model_rtmc --config rtmc/finetune_pheno/pheno_param_5_5_stl --seed 4 --cultivar Merlot --rnn_fpath _runs/RTMC/Phenology/ParamMTL5/STL/WA/Roza2/Prosser/Merlot/param_mtl__1766955690
 
python3 -m trainers.train_model_rtmc --config rtmc/finetune_pheno/pheno_param_5_5_stl --seed 0 --cultivar Mourvedre --rnn_fpath _runs/RTMC/Phenology/ParamMTL5/STL/WA/Roza2/Prosser/Mourvedre/param_mtl__1766959341
python3 -m trainers.train_model_rtmc --config rtmc/finetune_pheno/pheno_param_5_5_stl --seed 1 --cultivar Mourvedre --rnn_fpath _runs/RTMC/Phenology/ParamMTL5/STL/WA/Roza2/Prosser/Mourvedre/param_mtl__1766962979
python3 -m trainers.train_model_rtmc --config rtmc/finetune_pheno/pheno_param_5_5_stl --seed 2 --cultivar Mourvedre --rnn_fpath _runs/RTMC/Phenology/ParamMTL5/STL/WA/Roza2/Prosser/Mourvedre/param_mtl__1766972287
python3 -m trainers.train_model_rtmc --config rtmc/finetune_pheno/pheno_param_5_5_stl --seed 3 --cultivar Mourvedre --rnn_fpath _runs/RTMC/Phenology/ParamMTL5/STL/WA/Roza2/Prosser/Mourvedre/param_mtl__1766973561
python3 -m trainers.train_model_rtmc --config rtmc/finetune_pheno/pheno_param_5_5_stl --seed 4 --cultivar Mourvedre --rnn_fpath _runs/RTMC/Phenology/ParamMTL5/STL/WA/Roza2/Prosser/Mourvedre/param_mtl__1766974857
 
python3 -m trainers.train_model_rtmc --config rtmc/finetune_pheno/pheno_param_5_5_stl --seed 0 --cultivar Muscat_Blanc --rnn_fpath _runs/RTMC/Phenology/ParamMTL5/STL/WA/Roza2/Prosser/Muscat_Blanc/param_mtl__1766976159
python3 -m trainers.train_model_rtmc --config rtmc/finetune_pheno/pheno_param_5_5_stl --seed 1 --cultivar Muscat_Blanc --rnn_fpath _runs/RTMC/Phenology/ParamMTL5/STL/WA/Roza2/Prosser/Muscat_Blanc/param_mtl__1766977433
python3 -m trainers.train_model_rtmc --config rtmc/finetune_pheno/pheno_param_5_5_stl --seed 2 --cultivar Muscat_Blanc --rnn_fpath _runs/RTMC/Phenology/ParamMTL5/STL/WA/Roza2/Prosser/Muscat_Blanc/param_mtl__1766978727
python3 -m trainers.train_model_rtmc --config rtmc/finetune_pheno/pheno_param_5_5_stl --seed 3 --cultivar Muscat_Blanc --rnn_fpath _runs/RTMC/Phenology/ParamMTL5/STL/WA/Roza2/Prosser/Muscat_Blanc/param_mtl__1766980018
python3 -m trainers.train_model_rtmc --config rtmc/finetune_pheno/pheno_param_5_5_stl --seed 4 --cultivar Muscat_Blanc --rnn_fpath _runs/RTMC/Phenology/ParamMTL5/STL/WA/Roza2/Prosser/Muscat_Blanc/param_mtl__1766981304
 
python3 -m trainers.train_model_rtmc --config rtmc/finetune_pheno/pheno_param_5_5_stl --seed 0 --cultivar Nebbiolo --rnn_fpath _runs/RTMC/Phenology/ParamMTL5/STL/WA/Roza2/Prosser/Nebbiolo/param_mtl__1766982597
python3 -m trainers.train_model_rtmc --config rtmc/finetune_pheno/pheno_param_5_5_stl --seed 1 --cultivar Nebbiolo --rnn_fpath _runs/RTMC/Phenology/ParamMTL5/STL/WA/Roza2/Prosser/Nebbiolo/param_mtl__1766983876
python3 -m trainers.train_model_rtmc --config rtmc/finetune_pheno/pheno_param_5_5_stl --seed 2 --cultivar Nebbiolo --rnn_fpath _runs/RTMC/Phenology/ParamMTL5/STL/WA/Roza2/Prosser/Nebbiolo/param_mtl__1766985169
python3 -m trainers.train_model_rtmc --config rtmc/finetune_pheno/pheno_param_5_5_stl --seed 3 --cultivar Nebbiolo --rnn_fpath _runs/RTMC/Phenology/ParamMTL5/STL/WA/Roza2/Prosser/Nebbiolo/param_mtl__1766986456
python3 -m trainers.train_model_rtmc --config rtmc/finetune_pheno/pheno_param_5_5_stl --seed 4 --cultivar Nebbiolo --rnn_fpath _runs/RTMC/Phenology/ParamMTL5/STL/WA/Roza2/Prosser/Nebbiolo/param_mtl__1766987739
 
python3 -m trainers.train_model_rtmc --config rtmc/finetune_pheno/pheno_param_5_5_stl --seed 0 --cultivar Petit_Verdot --rnn_fpath _runs/RTMC/Phenology/ParamMTL5/STL/WA/Roza2/Prosser/Petit_Verdot/param_mtl__1766989041
python3 -m trainers.train_model_rtmc --config rtmc/finetune_pheno/pheno_param_5_5_stl --seed 1 --cultivar Petit_Verdot --rnn_fpath _runs/RTMC/Phenology/ParamMTL5/STL/WA/Roza2/Prosser/Petit_Verdot/param_mtl__1766990326
python3 -m trainers.train_model_rtmc --config rtmc/finetune_pheno/pheno_param_5_5_stl --seed 2 --cultivar Petit_Verdot --rnn_fpath _runs/RTMC/Phenology/ParamMTL5/STL/WA/Roza2/Prosser/Petit_Verdot/param_mtl__1766991620
python3 -m trainers.train_model_rtmc --config rtmc/finetune_pheno/pheno_param_5_5_stl --seed 3 --cultivar Petit_Verdot --rnn_fpath _runs/RTMC/Phenology/ParamMTL5/STL/WA/Roza2/Prosser/Petit_Verdot/param_mtl__1766992912
python3 -m trainers.train_model_rtmc --config rtmc/finetune_pheno/pheno_param_5_5_stl --seed 4 --cultivar Petit_Verdot --rnn_fpath _runs/RTMC/Phenology/ParamMTL5/STL/WA/Roza2/Prosser/Petit_Verdot/param_mtl__1766994195
 
python3 -m trainers.train_model_rtmc --config rtmc/finetune_pheno/pheno_param_5_5_stl --seed 0 --cultivar Pinot_Blanc --rnn_fpath _runs/RTMC/Phenology/ParamMTL5/STL/WA/Roza2/Prosser/Pinot_Blanc/param_mtl__1766995488
python3 -m trainers.train_model_rtmc --config rtmc/finetune_pheno/pheno_param_5_5_stl --seed 1 --cultivar Pinot_Blanc --rnn_fpath _runs/RTMC/Phenology/ParamMTL5/STL/WA/Roza2/Prosser/Pinot_Blanc/param_mtl__1766996782
python3 -m trainers.train_model_rtmc --config rtmc/finetune_pheno/pheno_param_5_5_stl --seed 2 --cultivar Pinot_Blanc --rnn_fpath _runs/RTMC/Phenology/ParamMTL5/STL/WA/Roza2/Prosser/Pinot_Blanc/param_mtl__1766998073
python3 -m trainers.train_model_rtmc --config rtmc/finetune_pheno/pheno_param_5_5_stl --seed 3 --cultivar Pinot_Blanc --rnn_fpath _runs/RTMC/Phenology/ParamMTL5/STL/WA/Roza2/Prosser/Pinot_Blanc/param_mtl__1766999362
python3 -m trainers.train_model_rtmc --config rtmc/finetune_pheno/pheno_param_5_5_stl --seed 4 --cultivar Pinot_Blanc --rnn_fpath _runs/RTMC/Phenology/ParamMTL5/STL/WA/Roza2/Prosser/Pinot_Blanc/param_mtl__1767000663
 
python3 -m trainers.train_model_rtmc --config rtmc/finetune_pheno/pheno_param_5_5_stl --seed 0 --cultivar Pinot_Gris --rnn_fpath _runs/RTMC/Phenology/ParamMTL5/STL/WA/Roza2/Prosser/Pinot_Gris/param_mtl__1767001956
python3 -m trainers.train_model_rtmc --config rtmc/finetune_pheno/pheno_param_5_5_stl --seed 1 --cultivar Pinot_Gris --rnn_fpath _runs/RTMC/Phenology/ParamMTL5/STL/WA/Roza2/Prosser/Pinot_Gris/param_mtl__1767003246
python3 -m trainers.train_model_rtmc --config rtmc/finetune_pheno/pheno_param_5_5_stl --seed 2 --cultivar Pinot_Gris --rnn_fpath _runs/RTMC/Phenology/ParamMTL5/STL/WA/Roza2/Prosser/Pinot_Gris/param_mtl__1767004545
python3 -m trainers.train_model_rtmc --config rtmc/finetune_pheno/pheno_param_5_5_stl --seed 3 --cultivar Pinot_Gris --rnn_fpath _runs/RTMC/Phenology/ParamMTL5/STL/WA/Roza2/Prosser/Pinot_Gris/param_mtl__1767005838
python3 -m trainers.train_model_rtmc --config rtmc/finetune_pheno/pheno_param_5_5_stl --seed 4 --cultivar Pinot_Gris --rnn_fpath _runs/RTMC/Phenology/ParamMTL5/STL/WA/Roza2/Prosser/Pinot_Gris/param_mtl__1767007134
 
python3 -m trainers.train_model_rtmc --config rtmc/finetune_pheno/pheno_param_5_5_stl --seed 0 --cultivar Pinot_Noir --rnn_fpath _runs/RTMC/Phenology/ParamMTL5/STL/WA/Roza2/Prosser/Pinot_Noir/param_mtl__1767008422
python3 -m trainers.train_model_rtmc --config rtmc/finetune_pheno/pheno_param_5_5_stl --seed 1 --cultivar Pinot_Noir --rnn_fpath _runs/RTMC/Phenology/ParamMTL5/STL/WA/Roza2/Prosser/Pinot_Noir/param_mtl__1767009718
python3 -m trainers.train_model_rtmc --config rtmc/finetune_pheno/pheno_param_5_5_stl --seed 2 --cultivar Pinot_Noir --rnn_fpath _runs/RTMC/Phenology/ParamMTL5/STL/WA/Roza2/Prosser/Pinot_Noir/param_mtl__1767011011
python3 -m trainers.train_model_rtmc --config rtmc/finetune_pheno/pheno_param_5_5_stl --seed 3 --cultivar Pinot_Noir --rnn_fpath _runs/RTMC/Phenology/ParamMTL5/STL/WA/Roza2/Prosser/Pinot_Noir/param_mtl__1767012312
python3 -m trainers.train_model_rtmc --config rtmc/finetune_pheno/pheno_param_5_5_stl --seed 4 --cultivar Pinot_Noir --rnn_fpath _runs/RTMC/Phenology/ParamMTL5/STL/WA/Roza2/Prosser/Pinot_Noir/param_mtl__1767013606
 
python3 -m trainers.train_model_rtmc --config rtmc/finetune_pheno/pheno_param_5_5_stl --seed 0 --cultivar Riesling --rnn_fpath _runs/RTMC/Phenology/ParamMTL5/STL/WA/Roza2/Prosser/Riesling/param_mtl__1767014900
python3 -m trainers.train_model_rtmc --config rtmc/finetune_pheno/pheno_param_5_5_stl --seed 1 --cultivar Riesling --rnn_fpath _runs/RTMC/Phenology/ParamMTL5/STL/WA/Roza2/Prosser/Riesling/param_mtl__1767016203
python3 -m trainers.train_model_rtmc --config rtmc/finetune_pheno/pheno_param_5_5_stl --seed 2 --cultivar Riesling --rnn_fpath _runs/RTMC/Phenology/ParamMTL5/STL/WA/Roza2/Prosser/Riesling/param_mtl__1767017496
python3 -m trainers.train_model_rtmc --config rtmc/finetune_pheno/pheno_param_5_5_stl --seed 3 --cultivar Riesling --rnn_fpath _runs/RTMC/Phenology/ParamMTL5/STL/WA/Roza2/Prosser/Riesling/param_mtl__1767018787
python3 -m trainers.train_model_rtmc --config rtmc/finetune_pheno/pheno_param_5_5_stl --seed 4 --cultivar Riesling --rnn_fpath _runs/RTMC/Phenology/ParamMTL5/STL/WA/Roza2/Prosser/Riesling/param_mtl__1767020082
 
python3 -m trainers.train_model_rtmc --config rtmc/finetune_pheno/pheno_param_5_5_stl --seed 0 --cultivar Sangiovese --rnn_fpath _runs/RTMC/Phenology/ParamMTL5/STL/WA/Roza2/Prosser/Sangiovese/param_mtl__1767021363
python3 -m trainers.train_model_rtmc --config rtmc/finetune_pheno/pheno_param_5_5_stl --seed 1 --cultivar Sangiovese --rnn_fpath _runs/RTMC/Phenology/ParamMTL5/STL/WA/Roza2/Prosser/Sangiovese/param_mtl__1767022667
python3 -m trainers.train_model_rtmc --config rtmc/finetune_pheno/pheno_param_5_5_stl --seed 2 --cultivar Sangiovese --rnn_fpath _runs/RTMC/Phenology/ParamMTL5/STL/WA/Roza2/Prosser/Sangiovese/param_mtl__1767023977
python3 -m trainers.train_model_rtmc --config rtmc/finetune_pheno/pheno_param_5_5_stl --seed 3 --cultivar Sangiovese --rnn_fpath _runs/RTMC/Phenology/ParamMTL5/STL/WA/Roza2/Prosser/Sangiovese/param_mtl__1767025272
python3 -m trainers.train_model_rtmc --config rtmc/finetune_pheno/pheno_param_5_5_stl --seed 4 --cultivar Sangiovese --rnn_fpath _runs/RTMC/Phenology/ParamMTL5/STL/WA/Roza2/Prosser/Sangiovese/param_mtl__1767026566
 
python3 -m trainers.train_model_rtmc --config rtmc/finetune_pheno/pheno_param_5_5_stl --seed 0 --cultivar Sauvignon_Blanc --rnn_fpath _runs/RTMC/Phenology/ParamMTL5/STL/WA/Roza2/Prosser/Sauvignon_Blanc/param_mtl__1767027877
python3 -m trainers.train_model_rtmc --config rtmc/finetune_pheno/pheno_param_5_5_stl --seed 1 --cultivar Sauvignon_Blanc --rnn_fpath _runs/RTMC/Phenology/ParamMTL5/STL/WA/Roza2/Prosser/Sauvignon_Blanc/param_mtl__1767029181
python3 -m trainers.train_model_rtmc --config rtmc/finetune_pheno/pheno_param_5_5_stl --seed 2 --cultivar Sauvignon_Blanc --rnn_fpath _runs/RTMC/Phenology/ParamMTL5/STL/WA/Roza2/Prosser/Sauvignon_Blanc/param_mtl__1767030475
python3 -m trainers.train_model_rtmc --config rtmc/finetune_pheno/pheno_param_5_5_stl --seed 3 --cultivar Sauvignon_Blanc --rnn_fpath _runs/RTMC/Phenology/ParamMTL5/STL/WA/Roza2/Prosser/Sauvignon_Blanc/param_mtl__1767031766
python3 -m trainers.train_model_rtmc --config rtmc/finetune_pheno/pheno_param_5_5_stl --seed 4 --cultivar Sauvignon_Blanc --rnn_fpath _runs/RTMC/Phenology/ParamMTL5/STL/WA/Roza2/Prosser/Sauvignon_Blanc/param_mtl__1767033062
 
python3 -m trainers.train_model_rtmc --config rtmc/finetune_pheno/pheno_param_5_5_stl --seed 0 --cultivar Semillon --rnn_fpath _runs/RTMC/Phenology/ParamMTL5/STL/WA/Roza2/Prosser/Semillon/param_mtl__1767034356
python3 -m trainers.train_model_rtmc --config rtmc/finetune_pheno/pheno_param_5_5_stl --seed 1 --cultivar Semillon --rnn_fpath _runs/RTMC/Phenology/ParamMTL5/STL/WA/Roza2/Prosser/Semillon/param_mtl__1767035643
python3 -m trainers.train_model_rtmc --config rtmc/finetune_pheno/pheno_param_5_5_stl --seed 2 --cultivar Semillon --rnn_fpath _runs/RTMC/Phenology/ParamMTL5/STL/WA/Roza2/Prosser/Semillon/param_mtl__1767036941
python3 -m trainers.train_model_rtmc --config rtmc/finetune_pheno/pheno_param_5_5_stl --seed 3 --cultivar Semillon --rnn_fpath _runs/RTMC/Phenology/ParamMTL5/STL/WA/Roza2/Prosser/Semillon/param_mtl__1767038239
python3 -m trainers.train_model_rtmc --config rtmc/finetune_pheno/pheno_param_5_5_stl --seed 4 --cultivar Semillon --rnn_fpath _runs/RTMC/Phenology/ParamMTL5/STL/WA/Roza2/Prosser/Semillon/param_mtl__1767039530
 
python3 -m trainers.train_model_rtmc --config rtmc/finetune_pheno/pheno_param_5_5_stl --seed 0 --cultivar Tempranillo --rnn_fpath _runs/RTMC/Phenology/ParamMTL5/STL/WA/Roza2/Prosser/Tempranillo/param_mtl__1767040823
python3 -m trainers.train_model_rtmc --config rtmc/finetune_pheno/pheno_param_5_5_stl --seed 1 --cultivar Tempranillo --rnn_fpath _runs/RTMC/Phenology/ParamMTL5/STL/WA/Roza2/Prosser/Tempranillo/param_mtl__1767042122
python3 -m trainers.train_model_rtmc --config rtmc/finetune_pheno/pheno_param_5_5_stl --seed 2 --cultivar Tempranillo --rnn_fpath _runs/RTMC/Phenology/ParamMTL5/STL/WA/Roza2/Prosser/Tempranillo/param_mtl__1767043413
python3 -m trainers.train_model_rtmc --config rtmc/finetune_pheno/pheno_param_5_5_stl --seed 3 --cultivar Tempranillo --rnn_fpath _runs/RTMC/Phenology/ParamMTL5/STL/WA/Roza2/Prosser/Tempranillo/param_mtl__1767044717
python3 -m trainers.train_model_rtmc --config rtmc/finetune_pheno/pheno_param_5_5_stl --seed 4 --cultivar Tempranillo --rnn_fpath _runs/RTMC/Phenology/ParamMTL5/STL/WA/Roza2/Prosser/Tempranillo/param_mtl__1767046008
 
python3 -m trainers.train_model_rtmc --config rtmc/finetune_pheno/pheno_param_5_5_stl --seed 0 --cultivar Viognier --rnn_fpath _runs/RTMC/Phenology/ParamMTL5/STL/WA/Roza2/Prosser/Viognier/param_mtl__1767047304
python3 -m trainers.train_model_rtmc --config rtmc/finetune_pheno/pheno_param_5_5_stl --seed 1 --cultivar Viognier --rnn_fpath _runs/RTMC/Phenology/ParamMTL5/STL/WA/Roza2/Prosser/Viognier/param_mtl__1767048595
python3 -m trainers.train_model_rtmc --config rtmc/finetune_pheno/pheno_param_5_5_stl --seed 2 --cultivar Viognier --rnn_fpath _runs/RTMC/Phenology/ParamMTL5/STL/WA/Roza2/Prosser/Viognier/param_mtl__1767049888
python3 -m trainers.train_model_rtmc --config rtmc/finetune_pheno/pheno_param_5_5_stl --seed 3 --cultivar Viognier --rnn_fpath _runs/RTMC/Phenology/ParamMTL5/STL/WA/Roza2/Prosser/Viognier/param_mtl__1767051179
python3 -m trainers.train_model_rtmc --config rtmc/finetune_pheno/pheno_param_5_5_stl --seed 4 --cultivar Viognier --rnn_fpath _runs/RTMC/Phenology/ParamMTL5/STL/WA/Roza2/Prosser/Viognier/param_mtl__1767052475
 
python3 -m trainers.train_model_rtmc --config rtmc/finetune_pheno/pheno_param_5_5_stl --seed 0 --cultivar Zinfandel --rnn_fpath _runs/RTMC/Phenology/ParamMTL5/STL/WA/Roza2/Prosser/Zinfandel/param_mtl__1767053766
python3 -m trainers.train_model_rtmc --config rtmc/finetune_pheno/pheno_param_5_5_stl --seed 1 --cultivar Zinfandel --rnn_fpath _runs/RTMC/Phenology/ParamMTL5/STL/WA/Roza2/Prosser/Zinfandel/param_mtl__1767055057
python3 -m trainers.train_model_rtmc --config rtmc/finetune_pheno/pheno_param_5_5_stl --seed 2 --cultivar Zinfandel --rnn_fpath _runs/RTMC/Phenology/ParamMTL5/STL/WA/Roza2/Prosser/Zinfandel/param_mtl__1767056348
python3 -m trainers.train_model_rtmc --config rtmc/finetune_pheno/pheno_param_5_5_stl --seed 3 --cultivar Zinfandel --rnn_fpath _runs/RTMC/Phenology/ParamMTL5/STL/WA/Roza2/Prosser/Zinfandel/param_mtl__1767057647
python3 -m trainers.train_model_rtmc --config rtmc/finetune_pheno/pheno_param_5_5_stl --seed 4 --cultivar Zinfandel --rnn_fpath _runs/RTMC/Phenology/ParamMTL5/STL/WA/Roza2/Prosser/Zinfandel/param_mtl__1767058892