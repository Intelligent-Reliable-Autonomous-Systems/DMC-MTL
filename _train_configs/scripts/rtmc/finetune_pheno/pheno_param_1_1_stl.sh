#!/bin/bash
#SBATCH -J pheno_param_1_1_stl
#SBATCH -o output/pheno_param_1_1_stl.out
#SBATCH -e output/pheno_param_1_1_stl.err
#SBATCH -p eecs,share,gpu
#SBATCH -t 2-12:00:00
#SBATCH --gres=gpu:1

python3 -m trainers.train_model --config rtmc/finetune_pheno/pheno_param_1_1_stl --seed 0 --cultivar Aligote --rnn_fpath _runs/RTMC/Phenology/ParamMTL1/STL/WA/Roza2/Prosser/Aligote/param_mtl__1766458278
python3 -m trainers.train_model --config rtmc/finetune_pheno/pheno_param_1_1_stl --seed 1 --cultivar Aligote --rnn_fpath _runs/RTMC/Phenology/ParamMTL1/STL/WA/Roza2/Prosser/Aligote/param_mtl__1766458864
python3 -m trainers.train_model --config rtmc/finetune_pheno/pheno_param_1_1_stl --seed 2 --cultivar Aligote --rnn_fpath _runs/RTMC/Phenology/ParamMTL1/STL/WA/Roza2/Prosser/Aligote/param_mtl__1766459455
python3 -m trainers.train_model --config rtmc/finetune_pheno/pheno_param_1_1_stl --seed 3 --cultivar Aligote --rnn_fpath _runs/RTMC/Phenology/ParamMTL1/STL/WA/Roza2/Prosser/Aligote/param_mtl__1766460043
python3 -m trainers.train_model --config rtmc/finetune_pheno/pheno_param_1_1_stl --seed 4 --cultivar Aligote --rnn_fpath _runs/RTMC/Phenology/ParamMTL1/STL/WA/Roza2/Prosser/Aligote/param_mtl__1766460627
 
python3 -m trainers.train_model --config rtmc/finetune_pheno/pheno_param_1_1_stl --seed 0 --cultivar Alvarinho --rnn_fpath _runs/RTMC/Phenology/ParamMTL1/STL/WA/Roza2/Prosser/Alvarinho/param_mtl__1766461214
python3 -m trainers.train_model --config rtmc/finetune_pheno/pheno_param_1_1_stl --seed 1 --cultivar Alvarinho --rnn_fpath _runs/RTMC/Phenology/ParamMTL1/STL/WA/Roza2/Prosser/Alvarinho/param_mtl__1766461802
python3 -m trainers.train_model --config rtmc/finetune_pheno/pheno_param_1_1_stl --seed 2 --cultivar Alvarinho --rnn_fpath _runs/RTMC/Phenology/ParamMTL1/STL/WA/Roza2/Prosser/Alvarinho/param_mtl__1766462393
python3 -m trainers.train_model --config rtmc/finetune_pheno/pheno_param_1_1_stl --seed 3 --cultivar Alvarinho --rnn_fpath _runs/RTMC/Phenology/ParamMTL1/STL/WA/Roza2/Prosser/Alvarinho/param_mtl__1766462979
python3 -m trainers.train_model --config rtmc/finetune_pheno/pheno_param_1_1_stl --seed 4 --cultivar Alvarinho --rnn_fpath _runs/RTMC/Phenology/ParamMTL1/STL/WA/Roza2/Prosser/Alvarinho/param_mtl__1766463565
 
python3 -m trainers.train_model --config rtmc/finetune_pheno/pheno_param_1_1_stl --seed 0 --cultivar Auxerrois --rnn_fpath _runs/RTMC/Phenology/ParamMTL1/STL/WA/Roza2/Prosser/Auxerrois/param_mtl__1766464151
python3 -m trainers.train_model --config rtmc/finetune_pheno/pheno_param_1_1_stl --seed 1 --cultivar Auxerrois --rnn_fpath _runs/RTMC/Phenology/ParamMTL1/STL/WA/Roza2/Prosser/Auxerrois/param_mtl__1766464743
python3 -m trainers.train_model --config rtmc/finetune_pheno/pheno_param_1_1_stl --seed 2 --cultivar Auxerrois --rnn_fpath _runs/RTMC/Phenology/ParamMTL1/STL/WA/Roza2/Prosser/Auxerrois/param_mtl__1766465333
python3 -m trainers.train_model --config rtmc/finetune_pheno/pheno_param_1_1_stl --seed 3 --cultivar Auxerrois --rnn_fpath _runs/RTMC/Phenology/ParamMTL1/STL/WA/Roza2/Prosser/Auxerrois/param_mtl__1766465920
python3 -m trainers.train_model --config rtmc/finetune_pheno/pheno_param_1_1_stl --seed 4 --cultivar Auxerrois --rnn_fpath _runs/RTMC/Phenology/ParamMTL1/STL/WA/Roza2/Prosser/Auxerrois/param_mtl__1766466508
 
python3 -m trainers.train_model --config rtmc/finetune_pheno/pheno_param_1_1_stl --seed 0 --cultivar Barbera --rnn_fpath _runs/RTMC/Phenology/ParamMTL1/STL/WA/Roza2/Prosser/Barbera/param_mtl__1766467095
python3 -m trainers.train_model --config rtmc/finetune_pheno/pheno_param_1_1_stl --seed 1 --cultivar Barbera --rnn_fpath _runs/RTMC/Phenology/ParamMTL1/STL/WA/Roza2/Prosser/Barbera/param_mtl__1766467688
python3 -m trainers.train_model --config rtmc/finetune_pheno/pheno_param_1_1_stl --seed 2 --cultivar Barbera --rnn_fpath _runs/RTMC/Phenology/ParamMTL1/STL/WA/Roza2/Prosser/Barbera/param_mtl__1766468277
python3 -m trainers.train_model --config rtmc/finetune_pheno/pheno_param_1_1_stl --seed 3 --cultivar Barbera --rnn_fpath _runs/RTMC/Phenology/ParamMTL1/STL/WA/Roza2/Prosser/Barbera/param_mtl__1766468867
python3 -m trainers.train_model --config rtmc/finetune_pheno/pheno_param_1_1_stl --seed 4 --cultivar Barbera --rnn_fpath _runs/RTMC/Phenology/ParamMTL1/STL/WA/Roza2/Prosser/Barbera/param_mtl__1766469456
 
python3 -m trainers.train_model --config rtmc/finetune_pheno/pheno_param_1_1_stl --seed 0 --cultivar Cabernet_Franc --rnn_fpath _runs/RTMC/Phenology/ParamMTL1/STL/WA/Roza2/Prosser/Cabernet_Franc/param_mtl__1766470048
python3 -m trainers.train_model --config rtmc/finetune_pheno/pheno_param_1_1_stl --seed 1 --cultivar Cabernet_Franc --rnn_fpath _runs/RTMC/Phenology/ParamMTL1/STL/WA/Roza2/Prosser/Cabernet_Franc/param_mtl__1766470642
python3 -m trainers.train_model --config rtmc/finetune_pheno/pheno_param_1_1_stl --seed 2 --cultivar Cabernet_Franc --rnn_fpath _runs/RTMC/Phenology/ParamMTL1/STL/WA/Roza2/Prosser/Cabernet_Franc/param_mtl__1766471233
python3 -m trainers.train_model --config rtmc/finetune_pheno/pheno_param_1_1_stl --seed 3 --cultivar Cabernet_Franc --rnn_fpath _runs/RTMC/Phenology/ParamMTL1/STL/WA/Roza2/Prosser/Cabernet_Franc/param_mtl__1766471829
python3 -m trainers.train_model --config rtmc/finetune_pheno/pheno_param_1_1_stl --seed 4 --cultivar Cabernet_Franc --rnn_fpath _runs/RTMC/Phenology/ParamMTL1/STL/WA/Roza2/Prosser/Cabernet_Franc/param_mtl__1766472421
 
python3 -m trainers.train_model --config rtmc/finetune_pheno/pheno_param_1_1_stl --seed 0 --cultivar Cabernet_Sauvignon --rnn_fpath _runs/RTMC/Phenology/ParamMTL1/STL/WA/Roza2/Prosser/Cabernet_Sauvignon/param_mtl__1766473014
python3 -m trainers.train_model --config rtmc/finetune_pheno/pheno_param_1_1_stl --seed 1 --cultivar Cabernet_Sauvignon --rnn_fpath _runs/RTMC/Phenology/ParamMTL1/STL/WA/Roza2/Prosser/Cabernet_Sauvignon/param_mtl__1766473608
python3 -m trainers.train_model --config rtmc/finetune_pheno/pheno_param_1_1_stl --seed 2 --cultivar Cabernet_Sauvignon --rnn_fpath _runs/RTMC/Phenology/ParamMTL1/STL/WA/Roza2/Prosser/Cabernet_Sauvignon/param_mtl__1766474204
python3 -m trainers.train_model --config rtmc/finetune_pheno/pheno_param_1_1_stl --seed 3 --cultivar Cabernet_Sauvignon --rnn_fpath _runs/RTMC/Phenology/ParamMTL1/STL/WA/Roza2/Prosser/Cabernet_Sauvignon/param_mtl__1766474799
python3 -m trainers.train_model --config rtmc/finetune_pheno/pheno_param_1_1_stl --seed 4 --cultivar Cabernet_Sauvignon --rnn_fpath _runs/RTMC/Phenology/ParamMTL1/STL/WA/Roza2/Prosser/Cabernet_Sauvignon/param_mtl__1766475390
 
python3 -m trainers.train_model --config rtmc/finetune_pheno/pheno_param_1_1_stl --seed 0 --cultivar Chardonnay --rnn_fpath _runs/RTMC/Phenology/ParamMTL1/STL/WA/Roza2/Prosser/Chardonnay/param_mtl__1766475982
python3 -m trainers.train_model --config rtmc/finetune_pheno/pheno_param_1_1_stl --seed 1 --cultivar Chardonnay --rnn_fpath _runs/RTMC/Phenology/ParamMTL1/STL/WA/Roza2/Prosser/Chardonnay/param_mtl__1766476574
python3 -m trainers.train_model --config rtmc/finetune_pheno/pheno_param_1_1_stl --seed 2 --cultivar Chardonnay --rnn_fpath _runs/RTMC/Phenology/ParamMTL1/STL/WA/Roza2/Prosser/Chardonnay/param_mtl__1766477166
python3 -m trainers.train_model --config rtmc/finetune_pheno/pheno_param_1_1_stl --seed 3 --cultivar Chardonnay --rnn_fpath _runs/RTMC/Phenology/ParamMTL1/STL/WA/Roza2/Prosser/Chardonnay/param_mtl__1766477756
python3 -m trainers.train_model --config rtmc/finetune_pheno/pheno_param_1_1_stl --seed 4 --cultivar Chardonnay --rnn_fpath _runs/RTMC/Phenology/ParamMTL1/STL/WA/Roza2/Prosser/Chardonnay/param_mtl__1766478350
 
python3 -m trainers.train_model --config rtmc/finetune_pheno/pheno_param_1_1_stl --seed 0 --cultivar Chenin_Blanc --rnn_fpath _runs/RTMC/Phenology/ParamMTL1/STL/WA/Roza2/Prosser/Chenin_Blanc/param_mtl__1766478944
python3 -m trainers.train_model --config rtmc/finetune_pheno/pheno_param_1_1_stl --seed 1 --cultivar Chenin_Blanc --rnn_fpath _runs/RTMC/Phenology/ParamMTL1/STL/WA/Roza2/Prosser/Chenin_Blanc/param_mtl__1766479536
python3 -m trainers.train_model --config rtmc/finetune_pheno/pheno_param_1_1_stl --seed 2 --cultivar Chenin_Blanc --rnn_fpath _runs/RTMC/Phenology/ParamMTL1/STL/WA/Roza2/Prosser/Chenin_Blanc/param_mtl__1766480129
python3 -m trainers.train_model --config rtmc/finetune_pheno/pheno_param_1_1_stl --seed 3 --cultivar Chenin_Blanc --rnn_fpath _runs/RTMC/Phenology/ParamMTL1/STL/WA/Roza2/Prosser/Chenin_Blanc/param_mtl__1766480723
python3 -m trainers.train_model --config rtmc/finetune_pheno/pheno_param_1_1_stl --seed 4 --cultivar Chenin_Blanc --rnn_fpath _runs/RTMC/Phenology/ParamMTL1/STL/WA/Roza2/Prosser/Chenin_Blanc/param_mtl__1766481316
 
python3 -m trainers.train_model --config rtmc/finetune_pheno/pheno_param_1_1_stl --seed 0 --cultivar Concord --rnn_fpath _runs/RTMC/Phenology/ParamMTL1/STL/WA/Roza2/Prosser/Concord/param_mtl__1766481902
python3 -m trainers.train_model --config rtmc/finetune_pheno/pheno_param_1_1_stl --seed 1 --cultivar Concord --rnn_fpath _runs/RTMC/Phenology/ParamMTL1/STL/WA/Roza2/Prosser/Concord/param_mtl__1766482489
python3 -m trainers.train_model --config rtmc/finetune_pheno/pheno_param_1_1_stl --seed 2 --cultivar Concord --rnn_fpath _runs/RTMC/Phenology/ParamMTL1/STL/WA/Roza2/Prosser/Concord/param_mtl__1766483080
python3 -m trainers.train_model --config rtmc/finetune_pheno/pheno_param_1_1_stl --seed 3 --cultivar Concord --rnn_fpath _runs/RTMC/Phenology/ParamMTL1/STL/WA/Roza2/Prosser/Concord/param_mtl__1766483670
python3 -m trainers.train_model --config rtmc/finetune_pheno/pheno_param_1_1_stl --seed 4 --cultivar Concord --rnn_fpath _runs/RTMC/Phenology/ParamMTL1/STL/WA/Roza2/Prosser/Concord/param_mtl__1766484261
 
python3 -m trainers.train_model --config rtmc/finetune_pheno/pheno_param_1_1_stl --seed 0 --cultivar Durif --rnn_fpath _runs/RTMC/Phenology/ParamMTL1/STL/WA/Roza2/Prosser/Durif/param_mtl__1766484855
python3 -m trainers.train_model --config rtmc/finetune_pheno/pheno_param_1_1_stl --seed 1 --cultivar Durif --rnn_fpath _runs/RTMC/Phenology/ParamMTL1/STL/WA/Roza2/Prosser/Durif/param_mtl__1766485448
python3 -m trainers.train_model --config rtmc/finetune_pheno/pheno_param_1_1_stl --seed 2 --cultivar Durif --rnn_fpath _runs/RTMC/Phenology/ParamMTL1/STL/WA/Roza2/Prosser/Durif/param_mtl__1766486043
python3 -m trainers.train_model --config rtmc/finetune_pheno/pheno_param_1_1_stl --seed 3 --cultivar Durif --rnn_fpath _runs/RTMC/Phenology/ParamMTL1/STL/WA/Roza2/Prosser/Durif/param_mtl__1766486632
python3 -m trainers.train_model --config rtmc/finetune_pheno/pheno_param_1_1_stl --seed 4 --cultivar Durif --rnn_fpath _runs/RTMC/Phenology/ParamMTL1/STL/WA/Roza2/Prosser/Durif/param_mtl__1766487218
 
python3 -m trainers.train_model --config rtmc/finetune_pheno/pheno_param_1_1_stl --seed 0 --cultivar Gewurztraminer --rnn_fpath _runs/RTMC/Phenology/ParamMTL1/STL/WA/Roza2/Prosser/Gewurztraminer/param_mtl__1766487807
python3 -m trainers.train_model --config rtmc/finetune_pheno/pheno_param_1_1_stl --seed 1 --cultivar Gewurztraminer --rnn_fpath _runs/RTMC/Phenology/ParamMTL1/STL/WA/Roza2/Prosser/Gewurztraminer/param_mtl__1766488400
python3 -m trainers.train_model --config rtmc/finetune_pheno/pheno_param_1_1_stl --seed 2 --cultivar Gewurztraminer --rnn_fpath _runs/RTMC/Phenology/ParamMTL1/STL/WA/Roza2/Prosser/Gewurztraminer/param_mtl__1766488990
python3 -m trainers.train_model --config rtmc/finetune_pheno/pheno_param_1_1_stl --seed 3 --cultivar Gewurztraminer --rnn_fpath _runs/RTMC/Phenology/ParamMTL1/STL/WA/Roza2/Prosser/Gewurztraminer/param_mtl__1766489579
python3 -m trainers.train_model --config rtmc/finetune_pheno/pheno_param_1_1_stl --seed 4 --cultivar Gewurztraminer --rnn_fpath _runs/RTMC/Phenology/ParamMTL1/STL/WA/Roza2/Prosser/Gewurztraminer/param_mtl__1766490169
 
python3 -m trainers.train_model --config rtmc/finetune_pheno/pheno_param_1_1_stl --seed 0 --cultivar Green_Veltliner --rnn_fpath _runs/RTMC/Phenology/ParamMTL1/STL/WA/Roza2/Prosser/Green_Veltliner/param_mtl__1766490758
python3 -m trainers.train_model --config rtmc/finetune_pheno/pheno_param_1_1_stl --seed 1 --cultivar Green_Veltliner --rnn_fpath _runs/RTMC/Phenology/ParamMTL1/STL/WA/Roza2/Prosser/Green_Veltliner/param_mtl__1766491345
python3 -m trainers.train_model --config rtmc/finetune_pheno/pheno_param_1_1_stl --seed 2 --cultivar Green_Veltliner --rnn_fpath _runs/RTMC/Phenology/ParamMTL1/STL/WA/Roza2/Prosser/Green_Veltliner/param_mtl__1766491934
python3 -m trainers.train_model --config rtmc/finetune_pheno/pheno_param_1_1_stl --seed 3 --cultivar Green_Veltliner --rnn_fpath _runs/RTMC/Phenology/ParamMTL1/STL/WA/Roza2/Prosser/Green_Veltliner/param_mtl__1766492523
python3 -m trainers.train_model --config rtmc/finetune_pheno/pheno_param_1_1_stl --seed 4 --cultivar Green_Veltliner --rnn_fpath _runs/RTMC/Phenology/ParamMTL1/STL/WA/Roza2/Prosser/Green_Veltliner/param_mtl__1766493111
 
python3 -m trainers.train_model --config rtmc/finetune_pheno/pheno_param_1_1_stl --seed 0 --cultivar Grenache --rnn_fpath _runs/RTMC/Phenology/ParamMTL1/STL/WA/Roza2/Prosser/Grenache/param_mtl__1766493701
python3 -m trainers.train_model --config rtmc/finetune_pheno/pheno_param_1_1_stl --seed 1 --cultivar Grenache --rnn_fpath _runs/RTMC/Phenology/ParamMTL1/STL/WA/Roza2/Prosser/Grenache/param_mtl__1766494289
python3 -m trainers.train_model --config rtmc/finetune_pheno/pheno_param_1_1_stl --seed 2 --cultivar Grenache --rnn_fpath _runs/RTMC/Phenology/ParamMTL1/STL/WA/Roza2/Prosser/Grenache/param_mtl__1766494884
python3 -m trainers.train_model --config rtmc/finetune_pheno/pheno_param_1_1_stl --seed 3 --cultivar Grenache --rnn_fpath _runs/RTMC/Phenology/ParamMTL1/STL/WA/Roza2/Prosser/Grenache/param_mtl__1766495475
python3 -m trainers.train_model --config rtmc/finetune_pheno/pheno_param_1_1_stl --seed 4 --cultivar Grenache --rnn_fpath _runs/RTMC/Phenology/ParamMTL1/STL/WA/Roza2/Prosser/Grenache/param_mtl__1766496057
 
python3 -m trainers.train_model --config rtmc/finetune_pheno/pheno_param_1_1_stl --seed 0 --cultivar Lemberger --rnn_fpath _runs/RTMC/Phenology/ParamMTL1/STL/WA/Roza2/Prosser/Lemberger/param_mtl__1766496644
python3 -m trainers.train_model --config rtmc/finetune_pheno/pheno_param_1_1_stl --seed 1 --cultivar Lemberger --rnn_fpath _runs/RTMC/Phenology/ParamMTL1/STL/WA/Roza2/Prosser/Lemberger/param_mtl__1766497234
python3 -m trainers.train_model --config rtmc/finetune_pheno/pheno_param_1_1_stl --seed 2 --cultivar Lemberger --rnn_fpath _runs/RTMC/Phenology/ParamMTL1/STL/WA/Roza2/Prosser/Lemberger/param_mtl__1766497822
python3 -m trainers.train_model --config rtmc/finetune_pheno/pheno_param_1_1_stl --seed 3 --cultivar Lemberger --rnn_fpath _runs/RTMC/Phenology/ParamMTL1/STL/WA/Roza2/Prosser/Lemberger/param_mtl__1766498408
python3 -m trainers.train_model --config rtmc/finetune_pheno/pheno_param_1_1_stl --seed 4 --cultivar Lemberger --rnn_fpath _runs/RTMC/Phenology/ParamMTL1/STL/WA/Roza2/Prosser/Lemberger/param_mtl__1766498997
 
python3 -m trainers.train_model --config rtmc/finetune_pheno/pheno_param_1_1_stl --seed 0 --cultivar Malbec --rnn_fpath _runs/RTMC/Phenology/ParamMTL1/STL/WA/Roza2/Prosser/Malbec/param_mtl__1766499587
python3 -m trainers.train_model --config rtmc/finetune_pheno/pheno_param_1_1_stl --seed 1 --cultivar Malbec --rnn_fpath _runs/RTMC/Phenology/ParamMTL1/STL/WA/Roza2/Prosser/Malbec/param_mtl__1766500178
python3 -m trainers.train_model --config rtmc/finetune_pheno/pheno_param_1_1_stl --seed 2 --cultivar Malbec --rnn_fpath _runs/RTMC/Phenology/ParamMTL1/STL/WA/Roza2/Prosser/Malbec/param_mtl__1766500768
python3 -m trainers.train_model --config rtmc/finetune_pheno/pheno_param_1_1_stl --seed 3 --cultivar Malbec --rnn_fpath _runs/RTMC/Phenology/ParamMTL1/STL/WA/Roza2/Prosser/Malbec/param_mtl__1766501360
python3 -m trainers.train_model --config rtmc/finetune_pheno/pheno_param_1_1_stl --seed 4 --cultivar Malbec --rnn_fpath _runs/RTMC/Phenology/ParamMTL1/STL/WA/Roza2/Prosser/Malbec/param_mtl__1766501948
 
python3 -m trainers.train_model --config rtmc/finetune_pheno/pheno_param_1_1_stl --seed 0 --cultivar Melon --rnn_fpath _runs/RTMC/Phenology/ParamMTL1/STL/WA/Roza2/Prosser/Melon/param_mtl__1766502539
python3 -m trainers.train_model --config rtmc/finetune_pheno/pheno_param_1_1_stl --seed 1 --cultivar Melon --rnn_fpath _runs/RTMC/Phenology/ParamMTL1/STL/WA/Roza2/Prosser/Melon/param_mtl__1766503131
python3 -m trainers.train_model --config rtmc/finetune_pheno/pheno_param_1_1_stl --seed 2 --cultivar Melon --rnn_fpath _runs/RTMC/Phenology/ParamMTL1/STL/WA/Roza2/Prosser/Melon/param_mtl__1766503723
python3 -m trainers.train_model --config rtmc/finetune_pheno/pheno_param_1_1_stl --seed 3 --cultivar Melon --rnn_fpath _runs/RTMC/Phenology/ParamMTL1/STL/WA/Roza2/Prosser/Melon/param_mtl__1766504316
python3 -m trainers.train_model --config rtmc/finetune_pheno/pheno_param_1_1_stl --seed 4 --cultivar Melon --rnn_fpath _runs/RTMC/Phenology/ParamMTL1/STL/WA/Roza2/Prosser/Melon/param_mtl__1766504901
 
python3 -m trainers.train_model --config rtmc/finetune_pheno/pheno_param_1_1_stl --seed 0 --cultivar Merlot --rnn_fpath _runs/RTMC/Phenology/ParamMTL1/STL/WA/Roza2/Prosser/Merlot/param_mtl__1766505495
python3 -m trainers.train_model --config rtmc/finetune_pheno/pheno_param_1_1_stl --seed 1 --cultivar Merlot --rnn_fpath _runs/RTMC/Phenology/ParamMTL1/STL/WA/Roza2/Prosser/Merlot/param_mtl__1766506083
python3 -m trainers.train_model --config rtmc/finetune_pheno/pheno_param_1_1_stl --seed 2 --cultivar Merlot --rnn_fpath _runs/RTMC/Phenology/ParamMTL1/STL/WA/Roza2/Prosser/Merlot/param_mtl__1766506673
python3 -m trainers.train_model --config rtmc/finetune_pheno/pheno_param_1_1_stl --seed 3 --cultivar Merlot --rnn_fpath _runs/RTMC/Phenology/ParamMTL1/STL/WA/Roza2/Prosser/Merlot/param_mtl__1766507262
python3 -m trainers.train_model --config rtmc/finetune_pheno/pheno_param_1_1_stl --seed 4 --cultivar Merlot --rnn_fpath _runs/RTMC/Phenology/ParamMTL1/STL/WA/Roza2/Prosser/Merlot/param_mtl__1766507849
 
python3 -m trainers.train_model --config rtmc/finetune_pheno/pheno_param_1_1_stl --seed 0 --cultivar Mourvedre --rnn_fpath _runs/RTMC/Phenology/ParamMTL1/STL/WA/Roza2/Prosser/Mourvedre/param_mtl__1766508436
python3 -m trainers.train_model --config rtmc/finetune_pheno/pheno_param_1_1_stl --seed 1 --cultivar Mourvedre --rnn_fpath _runs/RTMC/Phenology/ParamMTL1/STL/WA/Roza2/Prosser/Mourvedre/param_mtl__1766509025
python3 -m trainers.train_model --config rtmc/finetune_pheno/pheno_param_1_1_stl --seed 2 --cultivar Mourvedre --rnn_fpath _runs/RTMC/Phenology/ParamMTL1/STL/WA/Roza2/Prosser/Mourvedre/param_mtl__1766509616
python3 -m trainers.train_model --config rtmc/finetune_pheno/pheno_param_1_1_stl --seed 3 --cultivar Mourvedre --rnn_fpath _runs/RTMC/Phenology/ParamMTL1/STL/WA/Roza2/Prosser/Mourvedre/param_mtl__1766510207
python3 -m trainers.train_model --config rtmc/finetune_pheno/pheno_param_1_1_stl --seed 4 --cultivar Mourvedre --rnn_fpath _runs/RTMC/Phenology/ParamMTL1/STL/WA/Roza2/Prosser/Mourvedre/param_mtl__1766510796
 
python3 -m trainers.train_model --config rtmc/finetune_pheno/pheno_param_1_1_stl --seed 0 --cultivar Muscat_Blanc --rnn_fpath _runs/RTMC/Phenology/ParamMTL1/STL/WA/Roza2/Prosser/Muscat_Blanc/param_mtl__1766511385
python3 -m trainers.train_model --config rtmc/finetune_pheno/pheno_param_1_1_stl --seed 1 --cultivar Muscat_Blanc --rnn_fpath _runs/RTMC/Phenology/ParamMTL1/STL/WA/Roza2/Prosser/Muscat_Blanc/param_mtl__1766511981
python3 -m trainers.train_model --config rtmc/finetune_pheno/pheno_param_1_1_stl --seed 2 --cultivar Muscat_Blanc --rnn_fpath _runs/RTMC/Phenology/ParamMTL1/STL/WA/Roza2/Prosser/Muscat_Blanc/param_mtl__1766512574
python3 -m trainers.train_model --config rtmc/finetune_pheno/pheno_param_1_1_stl --seed 3 --cultivar Muscat_Blanc --rnn_fpath _runs/RTMC/Phenology/ParamMTL1/STL/WA/Roza2/Prosser/Muscat_Blanc/param_mtl__1766513164
python3 -m trainers.train_model --config rtmc/finetune_pheno/pheno_param_1_1_stl --seed 4 --cultivar Muscat_Blanc --rnn_fpath _runs/RTMC/Phenology/ParamMTL1/STL/WA/Roza2/Prosser/Muscat_Blanc/param_mtl__1766513758
 
python3 -m trainers.train_model --config rtmc/finetune_pheno/pheno_param_1_1_stl --seed 0 --cultivar Nebbiolo --rnn_fpath _runs/RTMC/Phenology/ParamMTL1/STL/WA/Roza2/Prosser/Nebbiolo/param_mtl__1766514350
python3 -m trainers.train_model --config rtmc/finetune_pheno/pheno_param_1_1_stl --seed 1 --cultivar Nebbiolo --rnn_fpath _runs/RTMC/Phenology/ParamMTL1/STL/WA/Roza2/Prosser/Nebbiolo/param_mtl__1766514941
python3 -m trainers.train_model --config rtmc/finetune_pheno/pheno_param_1_1_stl --seed 2 --cultivar Nebbiolo --rnn_fpath _runs/RTMC/Phenology/ParamMTL1/STL/WA/Roza2/Prosser/Nebbiolo/param_mtl__1766515536
python3 -m trainers.train_model --config rtmc/finetune_pheno/pheno_param_1_1_stl --seed 3 --cultivar Nebbiolo --rnn_fpath _runs/RTMC/Phenology/ParamMTL1/STL/WA/Roza2/Prosser/Nebbiolo/param_mtl__1766516130
python3 -m trainers.train_model --config rtmc/finetune_pheno/pheno_param_1_1_stl --seed 4 --cultivar Nebbiolo --rnn_fpath _runs/RTMC/Phenology/ParamMTL1/STL/WA/Roza2/Prosser/Nebbiolo/param_mtl__1766516721
 
python3 -m trainers.train_model --config rtmc/finetune_pheno/pheno_param_1_1_stl --seed 0 --cultivar Petit_Verdot --rnn_fpath _runs/RTMC/Phenology/ParamMTL1/STL/WA/Roza2/Prosser/Petit_Verdot/param_mtl__1766517312
python3 -m trainers.train_model --config rtmc/finetune_pheno/pheno_param_1_1_stl --seed 1 --cultivar Petit_Verdot --rnn_fpath _runs/RTMC/Phenology/ParamMTL1/STL/WA/Roza2/Prosser/Petit_Verdot/param_mtl__1766517904
python3 -m trainers.train_model --config rtmc/finetune_pheno/pheno_param_1_1_stl --seed 2 --cultivar Petit_Verdot --rnn_fpath _runs/RTMC/Phenology/ParamMTL1/STL/WA/Roza2/Prosser/Petit_Verdot/param_mtl__1766518495
python3 -m trainers.train_model --config rtmc/finetune_pheno/pheno_param_1_1_stl --seed 3 --cultivar Petit_Verdot --rnn_fpath _runs/RTMC/Phenology/ParamMTL1/STL/WA/Roza2/Prosser/Petit_Verdot/param_mtl__1766519086
python3 -m trainers.train_model --config rtmc/finetune_pheno/pheno_param_1_1_stl --seed 4 --cultivar Petit_Verdot --rnn_fpath _runs/RTMC/Phenology/ParamMTL1/STL/WA/Roza2/Prosser/Petit_Verdot/param_mtl__1766519676
 
python3 -m trainers.train_model --config rtmc/finetune_pheno/pheno_param_1_1_stl --seed 0 --cultivar Pinot_Blanc --rnn_fpath _runs/RTMC/Phenology/ParamMTL1/STL/WA/Roza2/Prosser/Pinot_Blanc/param_mtl__1766520268
python3 -m trainers.train_model --config rtmc/finetune_pheno/pheno_param_1_1_stl --seed 1 --cultivar Pinot_Blanc --rnn_fpath _runs/RTMC/Phenology/ParamMTL1/STL/WA/Roza2/Prosser/Pinot_Blanc/param_mtl__1766520863
python3 -m trainers.train_model --config rtmc/finetune_pheno/pheno_param_1_1_stl --seed 2 --cultivar Pinot_Blanc --rnn_fpath _runs/RTMC/Phenology/ParamMTL1/STL/WA/Roza2/Prosser/Pinot_Blanc/param_mtl__1766521459
python3 -m trainers.train_model --config rtmc/finetune_pheno/pheno_param_1_1_stl --seed 3 --cultivar Pinot_Blanc --rnn_fpath _runs/RTMC/Phenology/ParamMTL1/STL/WA/Roza2/Prosser/Pinot_Blanc/param_mtl__1766522056
python3 -m trainers.train_model --config rtmc/finetune_pheno/pheno_param_1_1_stl --seed 4 --cultivar Pinot_Blanc --rnn_fpath _runs/RTMC/Phenology/ParamMTL1/STL/WA/Roza2/Prosser/Pinot_Blanc/param_mtl__1766522651
 
python3 -m trainers.train_model --config rtmc/finetune_pheno/pheno_param_1_1_stl --seed 0 --cultivar Pinot_Gris --rnn_fpath _runs/RTMC/Phenology/ParamMTL1/STL/WA/Roza2/Prosser/Pinot_Gris/param_mtl__1766523242
python3 -m trainers.train_model --config rtmc/finetune_pheno/pheno_param_1_1_stl --seed 1 --cultivar Pinot_Gris --rnn_fpath _runs/RTMC/Phenology/ParamMTL1/STL/WA/Roza2/Prosser/Pinot_Gris/param_mtl__1766523834
python3 -m trainers.train_model --config rtmc/finetune_pheno/pheno_param_1_1_stl --seed 2 --cultivar Pinot_Gris --rnn_fpath _runs/RTMC/Phenology/ParamMTL1/STL/WA/Roza2/Prosser/Pinot_Gris/param_mtl__1766524428
python3 -m trainers.train_model --config rtmc/finetune_pheno/pheno_param_1_1_stl --seed 3 --cultivar Pinot_Gris --rnn_fpath _runs/RTMC/Phenology/ParamMTL1/STL/WA/Roza2/Prosser/Pinot_Gris/param_mtl__1766525021
python3 -m trainers.train_model --config rtmc/finetune_pheno/pheno_param_1_1_stl --seed 4 --cultivar Pinot_Gris --rnn_fpath _runs/RTMC/Phenology/ParamMTL1/STL/WA/Roza2/Prosser/Pinot_Gris/param_mtl__1766525614
 
python3 -m trainers.train_model --config rtmc/finetune_pheno/pheno_param_1_1_stl --seed 0 --cultivar Pinot_Noir --rnn_fpath _runs/RTMC/Phenology/ParamMTL1/STL/WA/Roza2/Prosser/Pinot_Noir/param_mtl__1766526210
python3 -m trainers.train_model --config rtmc/finetune_pheno/pheno_param_1_1_stl --seed 1 --cultivar Pinot_Noir --rnn_fpath _runs/RTMC/Phenology/ParamMTL1/STL/WA/Roza2/Prosser/Pinot_Noir/param_mtl__1766526805
python3 -m trainers.train_model --config rtmc/finetune_pheno/pheno_param_1_1_stl --seed 2 --cultivar Pinot_Noir --rnn_fpath _runs/RTMC/Phenology/ParamMTL1/STL/WA/Roza2/Prosser/Pinot_Noir/param_mtl__1766527400
python3 -m trainers.train_model --config rtmc/finetune_pheno/pheno_param_1_1_stl --seed 3 --cultivar Pinot_Noir --rnn_fpath _runs/RTMC/Phenology/ParamMTL1/STL/WA/Roza2/Prosser/Pinot_Noir/param_mtl__1766527998
python3 -m trainers.train_model --config rtmc/finetune_pheno/pheno_param_1_1_stl --seed 4 --cultivar Pinot_Noir --rnn_fpath _runs/RTMC/Phenology/ParamMTL1/STL/WA/Roza2/Prosser/Pinot_Noir/param_mtl__1766528592
 
python3 -m trainers.train_model --config rtmc/finetune_pheno/pheno_param_1_1_stl --seed 0 --cultivar Riesling --rnn_fpath _runs/RTMC/Phenology/ParamMTL1/STL/WA/Roza2/Prosser/Riesling/param_mtl__1766529204
python3 -m trainers.train_model --config rtmc/finetune_pheno/pheno_param_1_1_stl --seed 1 --cultivar Riesling --rnn_fpath _runs/RTMC/Phenology/ParamMTL1/STL/WA/Roza2/Prosser/Riesling/param_mtl__1766529821
python3 -m trainers.train_model --config rtmc/finetune_pheno/pheno_param_1_1_stl --seed 2 --cultivar Riesling --rnn_fpath _runs/RTMC/Phenology/ParamMTL1/STL/WA/Roza2/Prosser/Riesling/param_mtl__1766530419
python3 -m trainers.train_model --config rtmc/finetune_pheno/pheno_param_1_1_stl --seed 3 --cultivar Riesling --rnn_fpath _runs/RTMC/Phenology/ParamMTL1/STL/WA/Roza2/Prosser/Riesling/param_mtl__1766531013
python3 -m trainers.train_model --config rtmc/finetune_pheno/pheno_param_1_1_stl --seed 4 --cultivar Riesling --rnn_fpath _runs/RTMC/Phenology/ParamMTL1/STL/WA/Roza2/Prosser/Riesling/param_mtl__1766531604
 
python3 -m trainers.train_model --config rtmc/finetune_pheno/pheno_param_1_1_stl --seed 0 --cultivar Sangiovese --rnn_fpath _runs/RTMC/Phenology/ParamMTL1/STL/WA/Roza2/Prosser/Sangiovese/param_mtl__1766532198
python3 -m trainers.train_model --config rtmc/finetune_pheno/pheno_param_1_1_stl --seed 1 --cultivar Sangiovese --rnn_fpath _runs/RTMC/Phenology/ParamMTL1/STL/WA/Roza2/Prosser/Sangiovese/param_mtl__1766532789
python3 -m trainers.train_model --config rtmc/finetune_pheno/pheno_param_1_1_stl --seed 2 --cultivar Sangiovese --rnn_fpath _runs/RTMC/Phenology/ParamMTL1/STL/WA/Roza2/Prosser/Sangiovese/param_mtl__1766533387
python3 -m trainers.train_model --config rtmc/finetune_pheno/pheno_param_1_1_stl --seed 3 --cultivar Sangiovese --rnn_fpath _runs/RTMC/Phenology/ParamMTL1/STL/WA/Roza2/Prosser/Sangiovese/param_mtl__1766533983
python3 -m trainers.train_model --config rtmc/finetune_pheno/pheno_param_1_1_stl --seed 4 --cultivar Sangiovese --rnn_fpath _runs/RTMC/Phenology/ParamMTL1/STL/WA/Roza2/Prosser/Sangiovese/param_mtl__1766534580
 
python3 -m trainers.train_model --config rtmc/finetune_pheno/pheno_param_1_1_stl --seed 0 --cultivar Sauvignon_Blanc --rnn_fpath _runs/RTMC/Phenology/ParamMTL1/STL/WA/Roza2/Prosser/Sauvignon_Blanc/param_mtl__1766535176
python3 -m trainers.train_model --config rtmc/finetune_pheno/pheno_param_1_1_stl --seed 1 --cultivar Sauvignon_Blanc --rnn_fpath _runs/RTMC/Phenology/ParamMTL1/STL/WA/Roza2/Prosser/Sauvignon_Blanc/param_mtl__1766535775
python3 -m trainers.train_model --config rtmc/finetune_pheno/pheno_param_1_1_stl --seed 2 --cultivar Sauvignon_Blanc --rnn_fpath _runs/RTMC/Phenology/ParamMTL1/STL/WA/Roza2/Prosser/Sauvignon_Blanc/param_mtl__1766536376
python3 -m trainers.train_model --config rtmc/finetune_pheno/pheno_param_1_1_stl --seed 3 --cultivar Sauvignon_Blanc --rnn_fpath _runs/RTMC/Phenology/ParamMTL1/STL/WA/Roza2/Prosser/Sauvignon_Blanc/param_mtl__1766536978
python3 -m trainers.train_model --config rtmc/finetune_pheno/pheno_param_1_1_stl --seed 4 --cultivar Sauvignon_Blanc --rnn_fpath _runs/RTMC/Phenology/ParamMTL1/STL/WA/Roza2/Prosser/Sauvignon_Blanc/param_mtl__1766537575
 
python3 -m trainers.train_model --config rtmc/finetune_pheno/pheno_param_1_1_stl --seed 0 --cultivar Semillon --rnn_fpath _runs/RTMC/Phenology/ParamMTL1/STL/WA/Roza2/Prosser/Semillon/param_mtl__1766538172
python3 -m trainers.train_model --config rtmc/finetune_pheno/pheno_param_1_1_stl --seed 1 --cultivar Semillon --rnn_fpath _runs/RTMC/Phenology/ParamMTL1/STL/WA/Roza2/Prosser/Semillon/param_mtl__1766538772
python3 -m trainers.train_model --config rtmc/finetune_pheno/pheno_param_1_1_stl --seed 2 --cultivar Semillon --rnn_fpath _runs/RTMC/Phenology/ParamMTL1/STL/WA/Roza2/Prosser/Semillon/param_mtl__1766539374
python3 -m trainers.train_model --config rtmc/finetune_pheno/pheno_param_1_1_stl --seed 3 --cultivar Semillon --rnn_fpath _runs/RTMC/Phenology/ParamMTL1/STL/WA/Roza2/Prosser/Semillon/param_mtl__1766539976
python3 -m trainers.train_model --config rtmc/finetune_pheno/pheno_param_1_1_stl --seed 4 --cultivar Semillon --rnn_fpath _runs/RTMC/Phenology/ParamMTL1/STL/WA/Roza2/Prosser/Semillon/param_mtl__1766540578
 
python3 -m trainers.train_model --config rtmc/finetune_pheno/pheno_param_1_1_stl --seed 0 --cultivar Tempranillo --rnn_fpath _runs/RTMC/Phenology/ParamMTL1/STL/WA/Roza2/Prosser/Tempranillo/param_mtl__1766541177
python3 -m trainers.train_model --config rtmc/finetune_pheno/pheno_param_1_1_stl --seed 1 --cultivar Tempranillo --rnn_fpath _runs/RTMC/Phenology/ParamMTL1/STL/WA/Roza2/Prosser/Tempranillo/param_mtl__1766541770
python3 -m trainers.train_model --config rtmc/finetune_pheno/pheno_param_1_1_stl --seed 2 --cultivar Tempranillo --rnn_fpath _runs/RTMC/Phenology/ParamMTL1/STL/WA/Roza2/Prosser/Tempranillo/param_mtl__1766542367
python3 -m trainers.train_model --config rtmc/finetune_pheno/pheno_param_1_1_stl --seed 3 --cultivar Tempranillo --rnn_fpath _runs/RTMC/Phenology/ParamMTL1/STL/WA/Roza2/Prosser/Tempranillo/param_mtl__1766542965
python3 -m trainers.train_model --config rtmc/finetune_pheno/pheno_param_1_1_stl --seed 4 --cultivar Tempranillo --rnn_fpath _runs/RTMC/Phenology/ParamMTL1/STL/WA/Roza2/Prosser/Tempranillo/param_mtl__1766543556
 
python3 -m trainers.train_model --config rtmc/finetune_pheno/pheno_param_1_1_stl --seed 0 --cultivar Viognier --rnn_fpath _runs/RTMC/Phenology/ParamMTL1/STL/WA/Roza2/Prosser/Viognier/param_mtl__1766544153
python3 -m trainers.train_model --config rtmc/finetune_pheno/pheno_param_1_1_stl --seed 1 --cultivar Viognier --rnn_fpath _runs/RTMC/Phenology/ParamMTL1/STL/WA/Roza2/Prosser/Viognier/param_mtl__1766544745
python3 -m trainers.train_model --config rtmc/finetune_pheno/pheno_param_1_1_stl --seed 2 --cultivar Viognier --rnn_fpath _runs/RTMC/Phenology/ParamMTL1/STL/WA/Roza2/Prosser/Viognier/param_mtl__1766545341
python3 -m trainers.train_model --config rtmc/finetune_pheno/pheno_param_1_1_stl --seed 3 --cultivar Viognier --rnn_fpath _runs/RTMC/Phenology/ParamMTL1/STL/WA/Roza2/Prosser/Viognier/param_mtl__1766545934
python3 -m trainers.train_model --config rtmc/finetune_pheno/pheno_param_1_1_stl --seed 4 --cultivar Viognier --rnn_fpath _runs/RTMC/Phenology/ParamMTL1/STL/WA/Roza2/Prosser/Viognier/param_mtl__1766546529
 
python3 -m trainers.train_model --config rtmc/finetune_pheno/pheno_param_1_1_stl --seed 0 --cultivar Zinfandel --rnn_fpath _runs/RTMC/Phenology/ParamMTL1/STL/WA/Roza2/Prosser/Zinfandel/param_mtl__1766547124
python3 -m trainers.train_model --config rtmc/finetune_pheno/pheno_param_1_1_stl --seed 1 --cultivar Zinfandel --rnn_fpath _runs/RTMC/Phenology/ParamMTL1/STL/WA/Roza2/Prosser/Zinfandel/param_mtl__1766547717
python3 -m trainers.train_model --config rtmc/finetune_pheno/pheno_param_1_1_stl --seed 2 --cultivar Zinfandel --rnn_fpath _runs/RTMC/Phenology/ParamMTL1/STL/WA/Roza2/Prosser/Zinfandel/param_mtl__1766548310
python3 -m trainers.train_model --config rtmc/finetune_pheno/pheno_param_1_1_stl --seed 3 --cultivar Zinfandel --rnn_fpath _runs/RTMC/Phenology/ParamMTL1/STL/WA/Roza2/Prosser/Zinfandel/param_mtl__1766548903
python3 -m trainers.train_model --config rtmc/finetune_pheno/pheno_param_1_1_stl --seed 4 --cultivar Zinfandel --rnn_fpath _runs/RTMC/Phenology/ParamMTL1/STL/WA/Roza2/Prosser/Zinfandel/param_mtl__1766549498