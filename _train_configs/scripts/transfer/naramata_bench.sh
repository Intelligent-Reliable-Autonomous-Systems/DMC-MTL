#!/bin/bash
#SBATCH -J naramata_bench
#SBATCH -o output/naramata_bench.out
#SBATCH -e output/naramata_bench.err
#SBATCH -p eecs,share,gpu
#SBATCH -t 2-00:00:00
#SBATCH --gres=gpu:1

python3 -m trainers.train_model --config mthl/transfer/ch_deep_withold_mtl_station --seed 0 --region BCOV --withold "{Site: [NaramataBench], Cultivar: [Cabernet_Franc]}"
python3 -m trainers.train_model --config mthl/transfer/ch_deep_withold_mtl_station --seed 1 --region BCOV --withold "{Site: [NaramataBench], Cultivar: [Cabernet_Franc]}"
python3 -m trainers.train_model --config mthl/transfer/ch_deep_withold_mtl_station --seed 2 --region BCOV --withold "{Site: [NaramataBench], Cultivar: [Cabernet_Franc]}"
python3 -m trainers.train_model --config mthl/transfer/ch_deep_withold_mtl_station --seed 3 --region BCOV --withold "{Site: [NaramataBench], Cultivar: [Cabernet_Franc]}"
python3 -m trainers.train_model --config mthl/transfer/ch_deep_withold_mtl_station --seed 4 --region BCOV --withold "{Site: [NaramataBench], Cultivar: [Cabernet_Franc]}"

python3 -m trainers.train_model --config mthl/transfer/ch_deep_withold_mtl_station --seed 0 --region BCOV --withold "{Site: [NaramataBench], Cultivar: [Chardonnay]}"
python3 -m trainers.train_model --config mthl/transfer/ch_deep_withold_mtl_station --seed 1 --region BCOV --withold "{Site: [NaramataBench], Cultivar: [Chardonnay]}"
python3 -m trainers.train_model --config mthl/transfer/ch_deep_withold_mtl_station --seed 2 --region BCOV --withold "{Site: [NaramataBench], Cultivar: [Chardonnay]}"
python3 -m trainers.train_model --config mthl/transfer/ch_deep_withold_mtl_station --seed 3 --region BCOV --withold "{Site: [NaramataBench], Cultivar: [Chardonnay]}"
python3 -m trainers.train_model --config mthl/transfer/ch_deep_withold_mtl_station --seed 4 --region BCOV --withold "{Site: [NaramataBench], Cultivar: [Chardonnay]}"

python3 -m trainers.train_model --config mthl/transfer/ch_deep_withold_mtl_station --seed 0 --region BCOV --withold "{Site: [NaramataBench], Cultivar: [Merlot]}"
python3 -m trainers.train_model --config mthl/transfer/ch_deep_withold_mtl_station --seed 1 --region BCOV --withold "{Site: [NaramataBench], Cultivar: [Merlot]}"
python3 -m trainers.train_model --config mthl/transfer/ch_deep_withold_mtl_station --seed 2 --region BCOV --withold "{Site: [NaramataBench], Cultivar: [Merlot]}"
python3 -m trainers.train_model --config mthl/transfer/ch_deep_withold_mtl_station --seed 3 --region BCOV --withold "{Site: [NaramataBench], Cultivar: [Merlot]}"
python3 -m trainers.train_model --config mthl/transfer/ch_deep_withold_mtl_station --seed 4 --region BCOV --withold "{Site: [NaramataBench], Cultivar: [Merlot]}"

python3 -m trainers.train_model --config mthl/transfer/ch_deep_withold_mtl_station --seed 0 --region BCOV --withold "{Site: [NaramataBench], Cultivar: [Pinot_Gris]}"
python3 -m trainers.train_model --config mthl/transfer/ch_deep_withold_mtl_station --seed 1 --region BCOV --withold "{Site: [NaramataBench], Cultivar: [Pinot_Gris]}"
python3 -m trainers.train_model --config mthl/transfer/ch_deep_withold_mtl_station --seed 2 --region BCOV --withold "{Site: [NaramataBench], Cultivar: [Pinot_Gris]}"
python3 -m trainers.train_model --config mthl/transfer/ch_deep_withold_mtl_station --seed 3 --region BCOV --withold "{Site: [NaramataBench], Cultivar: [Pinot_Gris]}"
python3 -m trainers.train_model --config mthl/transfer/ch_deep_withold_mtl_station --seed 4 --region BCOV --withold "{Site: [NaramataBench], Cultivar: [Pinot_Gris]}"

python3 -m trainers.train_model --config mthl/transfer/ch_deep_withold_mtl_station --seed 0 --region BCOV --withold "{Site: [NaramataBench], Cultivar: [Pinot_Noir]}"
python3 -m trainers.train_model --config mthl/transfer/ch_deep_withold_mtl_station --seed 1 --region BCOV --withold "{Site: [NaramataBench], Cultivar: [Pinot_Noir]}"
python3 -m trainers.train_model --config mthl/transfer/ch_deep_withold_mtl_station --seed 2 --region BCOV --withold "{Site: [NaramataBench], Cultivar: [Pinot_Noir]}"
python3 -m trainers.train_model --config mthl/transfer/ch_deep_withold_mtl_station --seed 3 --region BCOV --withold "{Site: [NaramataBench], Cultivar: [Pinot_Noir]}"
python3 -m trainers.train_model --config mthl/transfer/ch_deep_withold_mtl_station --seed 4 --region BCOV --withold "{Site: [NaramataBench], Cultivar: [Pinot_Noir]}"

python3 -m trainers.train_model --config mthl/transfer/ch_deep_withold_mtl_station --seed 0 --region BCOV --withold "{Site: [NaramataBench], Cultivar: [Riesling]}"
python3 -m trainers.train_model --config mthl/transfer/ch_deep_withold_mtl_station --seed 1 --region BCOV --withold "{Site: [NaramataBench], Cultivar: [Riesling]}"
python3 -m trainers.train_model --config mthl/transfer/ch_deep_withold_mtl_station --seed 2 --region BCOV --withold "{Site: [NaramataBench], Cultivar: [Riesling]}"
python3 -m trainers.train_model --config mthl/transfer/ch_deep_withold_mtl_station --seed 3 --region BCOV --withold "{Site: [NaramataBench], Cultivar: [Riesling]}"
python3 -m trainers.train_model --config mthl/transfer/ch_deep_withold_mtl_station --seed 4 --region BCOV --withold "{Site: [NaramataBench], Cultivar: [Riesling]}"