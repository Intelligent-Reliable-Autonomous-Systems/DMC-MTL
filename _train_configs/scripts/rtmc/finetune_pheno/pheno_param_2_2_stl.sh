#!/bin/bash
#SBATCH -J pheno_param_2_2_stl
#SBATCH -o output/pheno_param_2_2_stl.out
#SBATCH -e output/pheno_param_2_2_stl.err
#SBATCH -p eecs,share,gpu
#SBATCH -t 2-12:00:00
#SBATCH --gres=gpu:1

python3 -m trainers.train_model_rtmc --config rtmc/finetune_pheno/pheno_param_2_2_stl --seed 0 --cultivar Aligote _runs/RTMC/Phenology/ParamMTL2/STL/WA/Roza2/Prosser/Aligote/param_mtl__1766460142
python3 -m trainers.train_model_rtmc --config rtmc/finetune_pheno/pheno_param_2_2_stl --seed 1 --cultivar Aligote _runs/RTMC/Phenology/ParamMTL2/STL/WA/Roza2/Prosser/Aligote/param_mtl__1766460966
python3 -m trainers.train_model_rtmc --config rtmc/finetune_pheno/pheno_param_2_2_stl --seed 2 --cultivar Aligote _runs/RTMC/Phenology/ParamMTL2/STL/WA/Roza2/Prosser/Aligote/param_mtl__1766461780
python3 -m trainers.train_model_rtmc --config rtmc/finetune_pheno/pheno_param_2_2_stl --seed 3 --cultivar Aligote _runs/RTMC/Phenology/ParamMTL2/STL/WA/Roza2/Prosser/Aligote/param_mtl__1766462586
python3 -m trainers.train_model_rtmc --config rtmc/finetune_pheno/pheno_param_2_2_stl --seed 4 --cultivar Aligote _runs/RTMC/Phenology/ParamMTL2/STL/WA/Roza2/Prosser/Aligote/param_mtl__1766463405
 
python3 -m trainers.train_model_rtmc --config rtmc/finetune_pheno/pheno_param_2_2_stl --seed 0 --cultivar Alvarinho _runs/RTMC/Phenology/ParamMTL2/STL/WA/Roza2/Prosser/Alvarinho/param_mtl__1766464209
python3 -m trainers.train_model_rtmc --config rtmc/finetune_pheno/pheno_param_2_2_stl --seed 1 --cultivar Alvarinho _runs/RTMC/Phenology/ParamMTL2/STL/WA/Roza2/Prosser/Alvarinho/param_mtl__1766465027
python3 -m trainers.train_model_rtmc --config rtmc/finetune_pheno/pheno_param_2_2_stl --seed 2 --cultivar Alvarinho _runs/RTMC/Phenology/ParamMTL2/STL/WA/Roza2/Prosser/Alvarinho/param_mtl__1766465850
python3 -m trainers.train_model_rtmc --config rtmc/finetune_pheno/pheno_param_2_2_stl --seed 3 --cultivar Alvarinho _runs/RTMC/Phenology/ParamMTL2/STL/WA/Roza2/Prosser/Alvarinho/param_mtl__1766466664
python3 -m trainers.train_model_rtmc --config rtmc/finetune_pheno/pheno_param_2_2_stl --seed 4 --cultivar Alvarinho _runs/RTMC/Phenology/ParamMTL2/STL/WA/Roza2/Prosser/Alvarinho/param_mtl__1766467467
 
python3 -m trainers.train_model_rtmc --config rtmc/finetune_pheno/pheno_param_2_2_stl --seed 0 --cultivar Auxerrois _runs/RTMC/Phenology/ParamMTL2/STL/WA/Roza2/Prosser/Auxerrois/param_mtl__1766468287
python3 -m trainers.train_model_rtmc --config rtmc/finetune_pheno/pheno_param_2_2_stl --seed 1 --cultivar Auxerrois _runs/RTMC/Phenology/ParamMTL2/STL/WA/Roza2/Prosser/Auxerrois/param_mtl__1766469109
python3 -m trainers.train_model_rtmc --config rtmc/finetune_pheno/pheno_param_2_2_stl --seed 2 --cultivar Auxerrois _runs/RTMC/Phenology/ParamMTL2/STL/WA/Roza2/Prosser/Auxerrois/param_mtl__1766469921
python3 -m trainers.train_model_rtmc --config rtmc/finetune_pheno/pheno_param_2_2_stl --seed 3 --cultivar Auxerrois _runs/RTMC/Phenology/ParamMTL2/STL/WA/Roza2/Prosser/Auxerrois/param_mtl__1766470727
python3 -m trainers.train_model_rtmc --config rtmc/finetune_pheno/pheno_param_2_2_stl --seed 4 --cultivar Auxerrois _runs/RTMC/Phenology/ParamMTL2/STL/WA/Roza2/Prosser/Auxerrois/param_mtl__1766471548
 
python3 -m trainers.train_model_rtmc --config rtmc/finetune_pheno/pheno_param_2_2_stl --seed 0 --cultivar Barbera _runs/RTMC/Phenology/ParamMTL2/STL/WA/Roza2/Prosser/Barbera/param_mtl__1766472372
python3 -m trainers.train_model_rtmc --config rtmc/finetune_pheno/pheno_param_2_2_stl --seed 1 --cultivar Barbera _runs/RTMC/Phenology/ParamMTL2/STL/WA/Roza2/Prosser/Barbera/param_mtl__1766473186
python3 -m trainers.train_model_rtmc --config rtmc/finetune_pheno/pheno_param_2_2_stl --seed 2 --cultivar Barbera _runs/RTMC/Phenology/ParamMTL2/STL/WA/Roza2/Prosser/Barbera/param_mtl__1766473994
python3 -m trainers.train_model_rtmc --config rtmc/finetune_pheno/pheno_param_2_2_stl --seed 3 --cultivar Barbera _runs/RTMC/Phenology/ParamMTL2/STL/WA/Roza2/Prosser/Barbera/param_mtl__1766474814
python3 -m trainers.train_model_rtmc --config rtmc/finetune_pheno/pheno_param_2_2_stl --seed 4 --cultivar Barbera _runs/RTMC/Phenology/ParamMTL2/STL/WA/Roza2/Prosser/Barbera/param_mtl__1766475631
 
python3 -m trainers.train_model_rtmc --config rtmc/finetune_pheno/pheno_param_2_2_stl --seed 0 --cultivar Cabernet_Franc _runs/RTMC/Phenology/ParamMTL2/STL/WA/Roza2/Prosser/Cabernet_Franc/param_mtl__1766476450
python3 -m trainers.train_model_rtmc --config rtmc/finetune_pheno/pheno_param_2_2_stl --seed 1 --cultivar Cabernet_Franc _runs/RTMC/Phenology/ParamMTL2/STL/WA/Roza2/Prosser/Cabernet_Franc/param_mtl__1766477257
python3 -m trainers.train_model_rtmc --config rtmc/finetune_pheno/pheno_param_2_2_stl --seed 2 --cultivar Cabernet_Franc _runs/RTMC/Phenology/ParamMTL2/STL/WA/Roza2/Prosser/Cabernet_Franc/param_mtl__1766478081
python3 -m trainers.train_model_rtmc --config rtmc/finetune_pheno/pheno_param_2_2_stl --seed 3 --cultivar Cabernet_Franc _runs/RTMC/Phenology/ParamMTL2/STL/WA/Roza2/Prosser/Cabernet_Franc/param_mtl__1766478906
python3 -m trainers.train_model_rtmc --config rtmc/finetune_pheno/pheno_param_2_2_stl --seed 4 --cultivar Cabernet_Franc _runs/RTMC/Phenology/ParamMTL2/STL/WA/Roza2/Prosser/Cabernet_Franc/param_mtl__1766479722
 
python3 -m trainers.train_model_rtmc --config rtmc/finetune_pheno/pheno_param_2_2_stl --seed 0 --cultivar Cabernet_Sauvignon _runs/RTMC/Phenology/ParamMTL2/STL/WA/Roza2/Prosser/Cabernet_Sauvignon/param_mtl__1766480530
python3 -m trainers.train_model_rtmc --config rtmc/finetune_pheno/pheno_param_2_2_stl --seed 1 --cultivar Cabernet_Sauvignon _runs/RTMC/Phenology/ParamMTL2/STL/WA/Roza2/Prosser/Cabernet_Sauvignon/param_mtl__1766481349
python3 -m trainers.train_model_rtmc --config rtmc/finetune_pheno/pheno_param_2_2_stl --seed 2 --cultivar Cabernet_Sauvignon _runs/RTMC/Phenology/ParamMTL2/STL/WA/Roza2/Prosser/Cabernet_Sauvignon/param_mtl__1766482169
python3 -m trainers.train_model_rtmc --config rtmc/finetune_pheno/pheno_param_2_2_stl --seed 3 --cultivar Cabernet_Sauvignon _runs/RTMC/Phenology/ParamMTL2/STL/WA/Roza2/Prosser/Cabernet_Sauvignon/param_mtl__1766482984
python3 -m trainers.train_model_rtmc --config rtmc/finetune_pheno/pheno_param_2_2_stl --seed 4 --cultivar Cabernet_Sauvignon _runs/RTMC/Phenology/ParamMTL2/STL/WA/Roza2/Prosser/Cabernet_Sauvignon/param_mtl__1766483791
 
python3 -m trainers.train_model_rtmc --config rtmc/finetune_pheno/pheno_param_2_2_stl --seed 0 --cultivar Chardonnay _runs/RTMC/Phenology/ParamMTL2/STL/WA/Roza2/Prosser/Chardonnay/param_mtl__1766484611
python3 -m trainers.train_model_rtmc --config rtmc/finetune_pheno/pheno_param_2_2_stl --seed 1 --cultivar Chardonnay _runs/RTMC/Phenology/ParamMTL2/STL/WA/Roza2/Prosser/Chardonnay/param_mtl__1766485432
python3 -m trainers.train_model_rtmc --config rtmc/finetune_pheno/pheno_param_2_2_stl --seed 2 --cultivar Chardonnay _runs/RTMC/Phenology/ParamMTL2/STL/WA/Roza2/Prosser/Chardonnay/param_mtl__1766486248
python3 -m trainers.train_model_rtmc --config rtmc/finetune_pheno/pheno_param_2_2_stl --seed 3 --cultivar Chardonnay _runs/RTMC/Phenology/ParamMTL2/STL/WA/Roza2/Prosser/Chardonnay/param_mtl__1766487047
python3 -m trainers.train_model_rtmc --config rtmc/finetune_pheno/pheno_param_2_2_stl --seed 4 --cultivar Chardonnay _runs/RTMC/Phenology/ParamMTL2/STL/WA/Roza2/Prosser/Chardonnay/param_mtl__1766487829
 
python3 -m trainers.train_model_rtmc --config rtmc/finetune_pheno/pheno_param_2_2_stl --seed 0 --cultivar Chenin_Blanc _runs/RTMC/Phenology/ParamMTL2/STL/WA/Roza2/Prosser/Chenin_Blanc/param_mtl__1766488609
python3 -m trainers.train_model_rtmc --config rtmc/finetune_pheno/pheno_param_2_2_stl --seed 1 --cultivar Chenin_Blanc _runs/RTMC/Phenology/ParamMTL2/STL/WA/Roza2/Prosser/Chenin_Blanc/param_mtl__1766489379
python3 -m trainers.train_model_rtmc --config rtmc/finetune_pheno/pheno_param_2_2_stl --seed 2 --cultivar Chenin_Blanc _runs/RTMC/Phenology/ParamMTL2/STL/WA/Roza2/Prosser/Chenin_Blanc/param_mtl__1766490165
python3 -m trainers.train_model_rtmc --config rtmc/finetune_pheno/pheno_param_2_2_stl --seed 3 --cultivar Chenin_Blanc _runs/RTMC/Phenology/ParamMTL2/STL/WA/Roza2/Prosser/Chenin_Blanc/param_mtl__1766490951
python3 -m trainers.train_model_rtmc --config rtmc/finetune_pheno/pheno_param_2_2_stl --seed 4 --cultivar Chenin_Blanc _runs/RTMC/Phenology/ParamMTL2/STL/WA/Roza2/Prosser/Chenin_Blanc/param_mtl__1766491723
 
python3 -m trainers.train_model_rtmc --config rtmc/finetune_pheno/pheno_param_2_2_stl --seed 0 --cultivar Concord _runs/RTMC/Phenology/ParamMTL2/STL/WA/Roza2/Prosser/Concord/param_mtl__1766492507
python3 -m trainers.train_model_rtmc --config rtmc/finetune_pheno/pheno_param_2_2_stl --seed 1 --cultivar Concord _runs/RTMC/Phenology/ParamMTL2/STL/WA/Roza2/Prosser/Concord/param_mtl__1766493287
python3 -m trainers.train_model_rtmc --config rtmc/finetune_pheno/pheno_param_2_2_stl --seed 2 --cultivar Concord _runs/RTMC/Phenology/ParamMTL2/STL/WA/Roza2/Prosser/Concord/param_mtl__1766494062
python3 -m trainers.train_model_rtmc --config rtmc/finetune_pheno/pheno_param_2_2_stl --seed 3 --cultivar Concord _runs/RTMC/Phenology/ParamMTL2/STL/WA/Roza2/Prosser/Concord/param_mtl__1766494847
python3 -m trainers.train_model_rtmc --config rtmc/finetune_pheno/pheno_param_2_2_stl --seed 4 --cultivar Concord _runs/RTMC/Phenology/ParamMTL2/STL/WA/Roza2/Prosser/Concord/param_mtl__1766495619
 
python3 -m trainers.train_model_rtmc --config rtmc/finetune_pheno/pheno_param_2_2_stl --seed 0 --cultivar Durif _runs/RTMC/Phenology/ParamMTL2/STL/WA/Roza2/Prosser/Durif/param_mtl__1766496408
python3 -m trainers.train_model_rtmc --config rtmc/finetune_pheno/pheno_param_2_2_stl --seed 1 --cultivar Durif _runs/RTMC/Phenology/ParamMTL2/STL/WA/Roza2/Prosser/Durif/param_mtl__1766497181
python3 -m trainers.train_model_rtmc --config rtmc/finetune_pheno/pheno_param_2_2_stl --seed 2 --cultivar Durif _runs/RTMC/Phenology/ParamMTL2/STL/WA/Roza2/Prosser/Durif/param_mtl__1766497970
python3 -m trainers.train_model_rtmc --config rtmc/finetune_pheno/pheno_param_2_2_stl --seed 3 --cultivar Durif _runs/RTMC/Phenology/ParamMTL2/STL/WA/Roza2/Prosser/Durif/param_mtl__1766498758
python3 -m trainers.train_model_rtmc --config rtmc/finetune_pheno/pheno_param_2_2_stl --seed 4 --cultivar Durif _runs/RTMC/Phenology/ParamMTL2/STL/WA/Roza2/Prosser/Durif/param_mtl__1766499530
 
python3 -m trainers.train_model_rtmc --config rtmc/finetune_pheno/pheno_param_2_2_stl --seed 0 --cultivar Gewurztraminer _runs/RTMC/Phenology/ParamMTL2/STL/WA/Roza2/Prosser/Gewurztraminer/param_mtl__1766500319
python3 -m trainers.train_model_rtmc --config rtmc/finetune_pheno/pheno_param_2_2_stl --seed 1 --cultivar Gewurztraminer _runs/RTMC/Phenology/ParamMTL2/STL/WA/Roza2/Prosser/Gewurztraminer/param_mtl__1766501090
python3 -m trainers.train_model_rtmc --config rtmc/finetune_pheno/pheno_param_2_2_stl --seed 2 --cultivar Gewurztraminer _runs/RTMC/Phenology/ParamMTL2/STL/WA/Roza2/Prosser/Gewurztraminer/param_mtl__1766501876
python3 -m trainers.train_model_rtmc --config rtmc/finetune_pheno/pheno_param_2_2_stl --seed 3 --cultivar Gewurztraminer _runs/RTMC/Phenology/ParamMTL2/STL/WA/Roza2/Prosser/Gewurztraminer/param_mtl__1766502649
python3 -m trainers.train_model_rtmc --config rtmc/finetune_pheno/pheno_param_2_2_stl --seed 4 --cultivar Gewurztraminer _runs/RTMC/Phenology/ParamMTL2/STL/WA/Roza2/Prosser/Gewurztraminer/param_mtl__1766503435
 
python3 -m trainers.train_model_rtmc --config rtmc/finetune_pheno/pheno_param_2_2_stl --seed 0 --cultivar Green_Veltliner _runs/RTMC/Phenology/ParamMTL2/STL/WA/Roza2/Prosser/Green_Veltliner/param_mtl__1766504205
python3 -m trainers.train_model_rtmc --config rtmc/finetune_pheno/pheno_param_2_2_stl --seed 1 --cultivar Green_Veltliner _runs/RTMC/Phenology/ParamMTL2/STL/WA/Roza2/Prosser/Green_Veltliner/param_mtl__1766504994
python3 -m trainers.train_model_rtmc --config rtmc/finetune_pheno/pheno_param_2_2_stl --seed 2 --cultivar Green_Veltliner _runs/RTMC/Phenology/ParamMTL2/STL/WA/Roza2/Prosser/Green_Veltliner/param_mtl__1766505771
python3 -m trainers.train_model_rtmc --config rtmc/finetune_pheno/pheno_param_2_2_stl --seed 3 --cultivar Green_Veltliner _runs/RTMC/Phenology/ParamMTL2/STL/WA/Roza2/Prosser/Green_Veltliner/param_mtl__1766506559
python3 -m trainers.train_model_rtmc --config rtmc/finetune_pheno/pheno_param_2_2_stl --seed 4 --cultivar Green_Veltliner _runs/RTMC/Phenology/ParamMTL2/STL/WA/Roza2/Prosser/Green_Veltliner/param_mtl__1766507337
 
python3 -m trainers.train_model_rtmc --config rtmc/finetune_pheno/pheno_param_2_2_stl --seed 0 --cultivar Grenache _runs/RTMC/Phenology/ParamMTL2/STL/WA/Roza2/Prosser/Grenache/param_mtl__1766508110
python3 -m trainers.train_model_rtmc --config rtmc/finetune_pheno/pheno_param_2_2_stl --seed 1 --cultivar Grenache _runs/RTMC/Phenology/ParamMTL2/STL/WA/Roza2/Prosser/Grenache/param_mtl__1766508898
python3 -m trainers.train_model_rtmc --config rtmc/finetune_pheno/pheno_param_2_2_stl --seed 2 --cultivar Grenache _runs/RTMC/Phenology/ParamMTL2/STL/WA/Roza2/Prosser/Grenache/param_mtl__1766509672
python3 -m trainers.train_model_rtmc --config rtmc/finetune_pheno/pheno_param_2_2_stl --seed 3 --cultivar Grenache _runs/RTMC/Phenology/ParamMTL2/STL/WA/Roza2/Prosser/Grenache/param_mtl__1766510458
python3 -m trainers.train_model_rtmc --config rtmc/finetune_pheno/pheno_param_2_2_stl --seed 4 --cultivar Grenache _runs/RTMC/Phenology/ParamMTL2/STL/WA/Roza2/Prosser/Grenache/param_mtl__1766511234
 
python3 -m trainers.train_model_rtmc --config rtmc/finetune_pheno/pheno_param_2_2_stl --seed 0 --cultivar Lemberger _runs/RTMC/Phenology/ParamMTL2/STL/WA/Roza2/Prosser/Lemberger/param_mtl__1766512007
python3 -m trainers.train_model_rtmc --config rtmc/finetune_pheno/pheno_param_2_2_stl --seed 1 --cultivar Lemberger _runs/RTMC/Phenology/ParamMTL2/STL/WA/Roza2/Prosser/Lemberger/param_mtl__1766512794
python3 -m trainers.train_model_rtmc --config rtmc/finetune_pheno/pheno_param_2_2_stl --seed 2 --cultivar Lemberger _runs/RTMC/Phenology/ParamMTL2/STL/WA/Roza2/Prosser/Lemberger/param_mtl__1766513567
python3 -m trainers.train_model_rtmc --config rtmc/finetune_pheno/pheno_param_2_2_stl --seed 3 --cultivar Lemberger _runs/RTMC/Phenology/ParamMTL2/STL/WA/Roza2/Prosser/Lemberger/param_mtl__1766514353
python3 -m trainers.train_model_rtmc --config rtmc/finetune_pheno/pheno_param_2_2_stl --seed 4 --cultivar Lemberger _runs/RTMC/Phenology/ParamMTL2/STL/WA/Roza2/Prosser/Lemberger/param_mtl__1766515129
 
python3 -m trainers.train_model_rtmc --config rtmc/finetune_pheno/pheno_param_2_2_stl --seed 0 --cultivar Malbec _runs/RTMC/Phenology/ParamMTL2/STL/WA/Roza2/Prosser/Malbec/param_mtl__1766515916
python3 -m trainers.train_model_rtmc --config rtmc/finetune_pheno/pheno_param_2_2_stl --seed 1 --cultivar Malbec _runs/RTMC/Phenology/ParamMTL2/STL/WA/Roza2/Prosser/Malbec/param_mtl__1766516694
python3 -m trainers.train_model_rtmc --config rtmc/finetune_pheno/pheno_param_2_2_stl --seed 2 --cultivar Malbec _runs/RTMC/Phenology/ParamMTL2/STL/WA/Roza2/Prosser/Malbec/param_mtl__1766517482
python3 -m trainers.train_model_rtmc --config rtmc/finetune_pheno/pheno_param_2_2_stl --seed 3 --cultivar Malbec _runs/RTMC/Phenology/ParamMTL2/STL/WA/Roza2/Prosser/Malbec/param_mtl__1766518261
python3 -m trainers.train_model_rtmc --config rtmc/finetune_pheno/pheno_param_2_2_stl --seed 4 --cultivar Malbec _runs/RTMC/Phenology/ParamMTL2/STL/WA/Roza2/Prosser/Malbec/param_mtl__1766519049
 
python3 -m trainers.train_model_rtmc --config rtmc/finetune_pheno/pheno_param_2_2_stl --seed 0 --cultivar Melon _runs/RTMC/Phenology/ParamMTL2/STL/WA/Roza2/Prosser/Melon/param_mtl__1766519829
python3 -m trainers.train_model_rtmc --config rtmc/finetune_pheno/pheno_param_2_2_stl --seed 1 --cultivar Melon _runs/RTMC/Phenology/ParamMTL2/STL/WA/Roza2/Prosser/Melon/param_mtl__1766520601
python3 -m trainers.train_model_rtmc --config rtmc/finetune_pheno/pheno_param_2_2_stl --seed 2 --cultivar Melon _runs/RTMC/Phenology/ParamMTL2/STL/WA/Roza2/Prosser/Melon/param_mtl__1766521390
python3 -m trainers.train_model_rtmc --config rtmc/finetune_pheno/pheno_param_2_2_stl --seed 3 --cultivar Melon _runs/RTMC/Phenology/ParamMTL2/STL/WA/Roza2/Prosser/Melon/param_mtl__1766522181
python3 -m trainers.train_model_rtmc --config rtmc/finetune_pheno/pheno_param_2_2_stl --seed 4 --cultivar Melon _runs/RTMC/Phenology/ParamMTL2/STL/WA/Roza2/Prosser/Melon/param_mtl__1766522954
 
python3 -m trainers.train_model_rtmc --config rtmc/finetune_pheno/pheno_param_2_2_stl --seed 0 --cultivar Merlot _runs/RTMC/Phenology/ParamMTL2/STL/WA/Roza2/Prosser/Merlot/param_mtl__1766523744
python3 -m trainers.train_model_rtmc --config rtmc/finetune_pheno/pheno_param_2_2_stl --seed 1 --cultivar Merlot _runs/RTMC/Phenology/ParamMTL2/STL/WA/Roza2/Prosser/Merlot/param_mtl__1766524516
python3 -m trainers.train_model_rtmc --config rtmc/finetune_pheno/pheno_param_2_2_stl --seed 2 --cultivar Merlot _runs/RTMC/Phenology/ParamMTL2/STL/WA/Roza2/Prosser/Merlot/param_mtl__1766525302
python3 -m trainers.train_model_rtmc --config rtmc/finetune_pheno/pheno_param_2_2_stl --seed 3 --cultivar Merlot _runs/RTMC/Phenology/ParamMTL2/STL/WA/Roza2/Prosser/Merlot/param_mtl__1766526078
python3 -m trainers.train_model_rtmc --config rtmc/finetune_pheno/pheno_param_2_2_stl --seed 4 --cultivar Merlot _runs/RTMC/Phenology/ParamMTL2/STL/WA/Roza2/Prosser/Merlot/param_mtl__1766526866
 
python3 -m trainers.train_model_rtmc --config rtmc/finetune_pheno/pheno_param_2_2_stl --seed 0 --cultivar Mourvedre _runs/RTMC/Phenology/ParamMTL2/STL/WA/Roza2/Prosser/Mourvedre/param_mtl__1766527640
python3 -m trainers.train_model_rtmc --config rtmc/finetune_pheno/pheno_param_2_2_stl --seed 1 --cultivar Mourvedre _runs/RTMC/Phenology/ParamMTL2/STL/WA/Roza2/Prosser/Mourvedre/param_mtl__1766528426
python3 -m trainers.train_model_rtmc --config rtmc/finetune_pheno/pheno_param_2_2_stl --seed 2 --cultivar Mourvedre _runs/RTMC/Phenology/ParamMTL2/STL/WA/Roza2/Prosser/Mourvedre/param_mtl__1766529196
python3 -m trainers.train_model_rtmc --config rtmc/finetune_pheno/pheno_param_2_2_stl --seed 3 --cultivar Mourvedre _runs/RTMC/Phenology/ParamMTL2/STL/WA/Roza2/Prosser/Mourvedre/param_mtl__1766529980
python3 -m trainers.train_model_rtmc --config rtmc/finetune_pheno/pheno_param_2_2_stl --seed 4 --cultivar Mourvedre _runs/RTMC/Phenology/ParamMTL2/STL/WA/Roza2/Prosser/Mourvedre/param_mtl__1766530751
 
python3 -m trainers.train_model_rtmc --config rtmc/finetune_pheno/pheno_param_2_2_stl --seed 0 --cultivar Muscat_Blanc _runs/RTMC/Phenology/ParamMTL2/STL/WA/Roza2/Prosser/Muscat_Blanc/param_mtl__1766531535
python3 -m trainers.train_model_rtmc --config rtmc/finetune_pheno/pheno_param_2_2_stl --seed 1 --cultivar Muscat_Blanc _runs/RTMC/Phenology/ParamMTL2/STL/WA/Roza2/Prosser/Muscat_Blanc/param_mtl__1766532307
python3 -m trainers.train_model_rtmc --config rtmc/finetune_pheno/pheno_param_2_2_stl --seed 2 --cultivar Muscat_Blanc _runs/RTMC/Phenology/ParamMTL2/STL/WA/Roza2/Prosser/Muscat_Blanc/param_mtl__1766533096
python3 -m trainers.train_model_rtmc --config rtmc/finetune_pheno/pheno_param_2_2_stl --seed 3 --cultivar Muscat_Blanc _runs/RTMC/Phenology/ParamMTL2/STL/WA/Roza2/Prosser/Muscat_Blanc/param_mtl__1766533870
python3 -m trainers.train_model_rtmc --config rtmc/finetune_pheno/pheno_param_2_2_stl --seed 4 --cultivar Muscat_Blanc _runs/RTMC/Phenology/ParamMTL2/STL/WA/Roza2/Prosser/Muscat_Blanc/param_mtl__1766534641
 
python3 -m trainers.train_model_rtmc --config rtmc/finetune_pheno/pheno_param_2_2_stl --seed 0 --cultivar Nebbiolo _runs/RTMC/Phenology/ParamMTL2/STL/WA/Roza2/Prosser/Nebbiolo/param_mtl__1766535427
python3 -m trainers.train_model_rtmc --config rtmc/finetune_pheno/pheno_param_2_2_stl --seed 1 --cultivar Nebbiolo _runs/RTMC/Phenology/ParamMTL2/STL/WA/Roza2/Prosser/Nebbiolo/param_mtl__1766536200
python3 -m trainers.train_model_rtmc --config rtmc/finetune_pheno/pheno_param_2_2_stl --seed 2 --cultivar Nebbiolo _runs/RTMC/Phenology/ParamMTL2/STL/WA/Roza2/Prosser/Nebbiolo/param_mtl__1766536990
python3 -m trainers.train_model_rtmc --config rtmc/finetune_pheno/pheno_param_2_2_stl --seed 3 --cultivar Nebbiolo _runs/RTMC/Phenology/ParamMTL2/STL/WA/Roza2/Prosser/Nebbiolo/param_mtl__1766537768
python3 -m trainers.train_model_rtmc --config rtmc/finetune_pheno/pheno_param_2_2_stl --seed 4 --cultivar Nebbiolo _runs/RTMC/Phenology/ParamMTL2/STL/WA/Roza2/Prosser/Nebbiolo/param_mtl__1766538551
 
python3 -m trainers.train_model_rtmc --config rtmc/finetune_pheno/pheno_param_2_2_stl --seed 0 --cultivar Petit_Verdot _runs/RTMC/Phenology/ParamMTL2/STL/WA/Roza2/Prosser/Petit_Verdot/param_mtl__1766539332
python3 -m trainers.train_model_rtmc --config rtmc/finetune_pheno/pheno_param_2_2_stl --seed 1 --cultivar Petit_Verdot _runs/RTMC/Phenology/ParamMTL2/STL/WA/Roza2/Prosser/Petit_Verdot/param_mtl__1766540108
python3 -m trainers.train_model_rtmc --config rtmc/finetune_pheno/pheno_param_2_2_stl --seed 2 --cultivar Petit_Verdot _runs/RTMC/Phenology/ParamMTL2/STL/WA/Roza2/Prosser/Petit_Verdot/param_mtl__1766540894
python3 -m trainers.train_model_rtmc --config rtmc/finetune_pheno/pheno_param_2_2_stl --seed 3 --cultivar Petit_Verdot _runs/RTMC/Phenology/ParamMTL2/STL/WA/Roza2/Prosser/Petit_Verdot/param_mtl__1766541666
python3 -m trainers.train_model_rtmc --config rtmc/finetune_pheno/pheno_param_2_2_stl --seed 4 --cultivar Petit_Verdot _runs/RTMC/Phenology/ParamMTL2/STL/WA/Roza2/Prosser/Petit_Verdot/param_mtl__1766542457
 
python3 -m trainers.train_model_rtmc --config rtmc/finetune_pheno/pheno_param_2_2_stl --seed 0 --cultivar Pinot_Blanc _runs/RTMC/Phenology/ParamMTL2/STL/WA/Roza2/Prosser/Pinot_Blanc/param_mtl__1766543234
python3 -m trainers.train_model_rtmc --config rtmc/finetune_pheno/pheno_param_2_2_stl --seed 1 --cultivar Pinot_Blanc _runs/RTMC/Phenology/ParamMTL2/STL/WA/Roza2/Prosser/Pinot_Blanc/param_mtl__1766544023
python3 -m trainers.train_model_rtmc --config rtmc/finetune_pheno/pheno_param_2_2_stl --seed 2 --cultivar Pinot_Blanc _runs/RTMC/Phenology/ParamMTL2/STL/WA/Roza2/Prosser/Pinot_Blanc/param_mtl__1766544800
python3 -m trainers.train_model_rtmc --config rtmc/finetune_pheno/pheno_param_2_2_stl --seed 3 --cultivar Pinot_Blanc _runs/RTMC/Phenology/ParamMTL2/STL/WA/Roza2/Prosser/Pinot_Blanc/param_mtl__1766545589
python3 -m trainers.train_model_rtmc --config rtmc/finetune_pheno/pheno_param_2_2_stl --seed 4 --cultivar Pinot_Blanc _runs/RTMC/Phenology/ParamMTL2/STL/WA/Roza2/Prosser/Pinot_Blanc/param_mtl__1766546361
 
python3 -m trainers.train_model_rtmc --config rtmc/finetune_pheno/pheno_param_2_2_stl --seed 0 --cultivar Pinot_Gris _runs/RTMC/Phenology/ParamMTL2/STL/WA/Roza2/Prosser/Pinot_Gris/param_mtl__1766547151
python3 -m trainers.train_model_rtmc --config rtmc/finetune_pheno/pheno_param_2_2_stl --seed 1 --cultivar Pinot_Gris _runs/RTMC/Phenology/ParamMTL2/STL/WA/Roza2/Prosser/Pinot_Gris/param_mtl__1766547928
python3 -m trainers.train_model_rtmc --config rtmc/finetune_pheno/pheno_param_2_2_stl --seed 2 --cultivar Pinot_Gris _runs/RTMC/Phenology/ParamMTL2/STL/WA/Roza2/Prosser/Pinot_Gris/param_mtl__1766548718
python3 -m trainers.train_model_rtmc --config rtmc/finetune_pheno/pheno_param_2_2_stl --seed 3 --cultivar Pinot_Gris _runs/RTMC/Phenology/ParamMTL2/STL/WA/Roza2/Prosser/Pinot_Gris/param_mtl__1766549493
python3 -m trainers.train_model_rtmc --config rtmc/finetune_pheno/pheno_param_2_2_stl --seed 4 --cultivar Pinot_Gris _runs/RTMC/Phenology/ParamMTL2/STL/WA/Roza2/Prosser/Pinot_Gris/param_mtl__1766550270
 
python3 -m trainers.train_model_rtmc --config rtmc/finetune_pheno/pheno_param_2_2_stl --seed 0 --cultivar Pinot_Noir _runs/RTMC/Phenology/ParamMTL2/STL/WA/Roza2/Prosser/Pinot_Noir/param_mtl__1766551058
python3 -m trainers.train_model_rtmc --config rtmc/finetune_pheno/pheno_param_2_2_stl --seed 1 --cultivar Pinot_Noir _runs/RTMC/Phenology/ParamMTL2/STL/WA/Roza2/Prosser/Pinot_Noir/param_mtl__1766551833
python3 -m trainers.train_model_rtmc --config rtmc/finetune_pheno/pheno_param_2_2_stl --seed 2 --cultivar Pinot_Noir _runs/RTMC/Phenology/ParamMTL2/STL/WA/Roza2/Prosser/Pinot_Noir/param_mtl__1766552618
python3 -m trainers.train_model_rtmc --config rtmc/finetune_pheno/pheno_param_2_2_stl --seed 3 --cultivar Pinot_Noir _runs/RTMC/Phenology/ParamMTL2/STL/WA/Roza2/Prosser/Pinot_Noir/param_mtl__1766553396
python3 -m trainers.train_model_rtmc --config rtmc/finetune_pheno/pheno_param_2_2_stl --seed 4 --cultivar Pinot_Noir _runs/RTMC/Phenology/ParamMTL2/STL/WA/Roza2/Prosser/Pinot_Noir/param_mtl__1766554182
 
python3 -m trainers.train_model_rtmc --config rtmc/finetune_pheno/pheno_param_2_2_stl --seed 0 --cultivar Riesling _runs/RTMC/Phenology/ParamMTL2/STL/WA/Roza2/Prosser/Riesling/param_mtl__1766554966
python3 -m trainers.train_model_rtmc --config rtmc/finetune_pheno/pheno_param_2_2_stl --seed 1 --cultivar Riesling _runs/RTMC/Phenology/ParamMTL2/STL/WA/Roza2/Prosser/Riesling/param_mtl__1766555751
python3 -m trainers.train_model_rtmc --config rtmc/finetune_pheno/pheno_param_2_2_stl --seed 2 --cultivar Riesling _runs/RTMC/Phenology/ParamMTL2/STL/WA/Roza2/Prosser/Riesling/param_mtl__1766556526
python3 -m trainers.train_model_rtmc --config rtmc/finetune_pheno/pheno_param_2_2_stl --seed 3 --cultivar Riesling _runs/RTMC/Phenology/ParamMTL2/STL/WA/Roza2/Prosser/Riesling/param_mtl__1766557317
python3 -m trainers.train_model_rtmc --config rtmc/finetune_pheno/pheno_param_2_2_stl --seed 4 --cultivar Riesling _runs/RTMC/Phenology/ParamMTL2/STL/WA/Roza2/Prosser/Riesling/param_mtl__1766558098
 
python3 -m trainers.train_model_rtmc --config rtmc/finetune_pheno/pheno_param_2_2_stl --seed 0 --cultivar Sangiovese _runs/RTMC/Phenology/ParamMTL2/STL/WA/Roza2/Prosser/Sangiovese/param_mtl__1766558869
python3 -m trainers.train_model_rtmc --config rtmc/finetune_pheno/pheno_param_2_2_stl --seed 1 --cultivar Sangiovese _runs/RTMC/Phenology/ParamMTL2/STL/WA/Roza2/Prosser/Sangiovese/param_mtl__1766559658
python3 -m trainers.train_model_rtmc --config rtmc/finetune_pheno/pheno_param_2_2_stl --seed 2 --cultivar Sangiovese _runs/RTMC/Phenology/ParamMTL2/STL/WA/Roza2/Prosser/Sangiovese/param_mtl__1766560444
python3 -m trainers.train_model_rtmc --config rtmc/finetune_pheno/pheno_param_2_2_stl --seed 3 --cultivar Sangiovese _runs/RTMC/Phenology/ParamMTL2/STL/WA/Roza2/Prosser/Sangiovese/param_mtl__1766561218
python3 -m trainers.train_model_rtmc --config rtmc/finetune_pheno/pheno_param_2_2_stl --seed 4 --cultivar Sangiovese _runs/RTMC/Phenology/ParamMTL2/STL/WA/Roza2/Prosser/Sangiovese/param_mtl__1766562004
 
python3 -m trainers.train_model_rtmc --config rtmc/finetune_pheno/pheno_param_2_2_stl --seed 0 --cultivar Sauvignon_Blanc _runs/RTMC/Phenology/ParamMTL2/STL/WA/Roza2/Prosser/Sauvignon_Blanc/param_mtl__1766562792
python3 -m trainers.train_model_rtmc --config rtmc/finetune_pheno/pheno_param_2_2_stl --seed 1 --cultivar Sauvignon_Blanc _runs/RTMC/Phenology/ParamMTL2/STL/WA/Roza2/Prosser/Sauvignon_Blanc/param_mtl__1766563567
python3 -m trainers.train_model_rtmc --config rtmc/finetune_pheno/pheno_param_2_2_stl --seed 2 --cultivar Sauvignon_Blanc _runs/RTMC/Phenology/ParamMTL2/STL/WA/Roza2/Prosser/Sauvignon_Blanc/param_mtl__1766564359
python3 -m trainers.train_model_rtmc --config rtmc/finetune_pheno/pheno_param_2_2_stl --seed 3 --cultivar Sauvignon_Blanc _runs/RTMC/Phenology/ParamMTL2/STL/WA/Roza2/Prosser/Sauvignon_Blanc/param_mtl__1766565143
python3 -m trainers.train_model_rtmc --config rtmc/finetune_pheno/pheno_param_2_2_stl --seed 4 --cultivar Sauvignon_Blanc _runs/RTMC/Phenology/ParamMTL2/STL/WA/Roza2/Prosser/Sauvignon_Blanc/param_mtl__1766565917
 
python3 -m trainers.train_model_rtmc --config rtmc/finetune_pheno/pheno_param_2_2_stl --seed 0 --cultivar Semillon _runs/RTMC/Phenology/ParamMTL2/STL/WA/Roza2/Prosser/Semillon/param_mtl__1766566708
python3 -m trainers.train_model_rtmc --config rtmc/finetune_pheno/pheno_param_2_2_stl --seed 1 --cultivar Semillon _runs/RTMC/Phenology/ParamMTL2/STL/WA/Roza2/Prosser/Semillon/param_mtl__1766567491
python3 -m trainers.train_model_rtmc --config rtmc/finetune_pheno/pheno_param_2_2_stl --seed 2 --cultivar Semillon _runs/RTMC/Phenology/ParamMTL2/STL/WA/Roza2/Prosser/Semillon/param_mtl__1766568259
python3 -m trainers.train_model_rtmc --config rtmc/finetune_pheno/pheno_param_2_2_stl --seed 3 --cultivar Semillon _runs/RTMC/Phenology/ParamMTL2/STL/WA/Roza2/Prosser/Semillon/param_mtl__1766569033
python3 -m trainers.train_model_rtmc --config rtmc/finetune_pheno/pheno_param_2_2_stl --seed 4 --cultivar Semillon _runs/RTMC/Phenology/ParamMTL2/STL/WA/Roza2/Prosser/Semillon/param_mtl__1766569801
 
python3 -m trainers.train_model_rtmc --config rtmc/finetune_pheno/pheno_param_2_2_stl --seed 0 --cultivar Tempranillo _runs/RTMC/Phenology/ParamMTL2/STL/WA/Roza2/Prosser/Tempranillo/param_mtl__1766570578
python3 -m trainers.train_model_rtmc --config rtmc/finetune_pheno/pheno_param_2_2_stl --seed 1 --cultivar Tempranillo _runs/RTMC/Phenology/ParamMTL2/STL/WA/Roza2/Prosser/Tempranillo/param_mtl__1766571343
python3 -m trainers.train_model_rtmc --config rtmc/finetune_pheno/pheno_param_2_2_stl --seed 2 --cultivar Tempranillo _runs/RTMC/Phenology/ParamMTL2/STL/WA/Roza2/Prosser/Tempranillo/param_mtl__1766572119
python3 -m trainers.train_model_rtmc --config rtmc/finetune_pheno/pheno_param_2_2_stl --seed 3 --cultivar Tempranillo _runs/RTMC/Phenology/ParamMTL2/STL/WA/Roza2/Prosser/Tempranillo/param_mtl__1766572886
python3 -m trainers.train_model_rtmc --config rtmc/finetune_pheno/pheno_param_2_2_stl --seed 4 --cultivar Tempranillo _runs/RTMC/Phenology/ParamMTL2/STL/WA/Roza2/Prosser/Tempranillo/param_mtl__1766573658
 
python3 -m trainers.train_model_rtmc --config rtmc/finetune_pheno/pheno_param_2_2_stl --seed 0 --cultivar Viognier _runs/RTMC/Phenology/ParamMTL2/STL/WA/Roza2/Prosser/Viognier/param_mtl__1766574423
python3 -m trainers.train_model_rtmc --config rtmc/finetune_pheno/pheno_param_2_2_stl --seed 1 --cultivar Viognier _runs/RTMC/Phenology/ParamMTL2/STL/WA/Roza2/Prosser/Viognier/param_mtl__1766575195
python3 -m trainers.train_model_rtmc --config rtmc/finetune_pheno/pheno_param_2_2_stl --seed 2 --cultivar Viognier _runs/RTMC/Phenology/ParamMTL2/STL/WA/Roza2/Prosser/Viognier/param_mtl__1766575962
python3 -m trainers.train_model_rtmc --config rtmc/finetune_pheno/pheno_param_2_2_stl --seed 3 --cultivar Viognier _runs/RTMC/Phenology/ParamMTL2/STL/WA/Roza2/Prosser/Viognier/param_mtl__1766576726
python3 -m trainers.train_model_rtmc --config rtmc/finetune_pheno/pheno_param_2_2_stl --seed 4 --cultivar Viognier _runs/RTMC/Phenology/ParamMTL2/STL/WA/Roza2/Prosser/Viognier/param_mtl__1766577488
 
python3 -m trainers.train_model_rtmc --config rtmc/finetune_pheno/pheno_param_2_2_stl --seed 0 --cultivar Zinfandel _runs/RTMC/Phenology/ParamMTL2/STL/WA/Roza2/Prosser/Zinfandel/param_mtl__1766578257
python3 -m trainers.train_model_rtmc --config rtmc/finetune_pheno/pheno_param_2_2_stl --seed 1 --cultivar Zinfandel _runs/RTMC/Phenology/ParamMTL2/STL/WA/Roza2/Prosser/Zinfandel/param_mtl__1766579020
python3 -m trainers.train_model_rtmc --config rtmc/finetune_pheno/pheno_param_2_2_stl --seed 2 --cultivar Zinfandel _runs/RTMC/Phenology/ParamMTL2/STL/WA/Roza2/Prosser/Zinfandel/param_mtl__1766579785
python3 -m trainers.train_model_rtmc --config rtmc/finetune_pheno/pheno_param_2_2_stl --seed 3 --cultivar Zinfandel _runs/RTMC/Phenology/ParamMTL2/STL/WA/Roza2/Prosser/Zinfandel/param_mtl__1766580553
python3 -m trainers.train_model_rtmc --config rtmc/finetune_pheno/pheno_param_2_2_stl --seed 4 --cultivar Zinfandel _runs/RTMC/Phenology/ParamMTL2/STL/WA/Roza2/Prosser/Zinfandel/param_mtl__1766581324