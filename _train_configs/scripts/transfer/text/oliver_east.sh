#!/bin/bash
#SBATCH -J oliver_east
#SBATCH -o output/oliver_east.out
#SBATCH -e output/oliver_east.err
#SBATCH -p eecs,gpu
#SBATCH -t 2-00:00:00
#SBATCH --gres=gpu:1

python3 -m trainers.train_model --config mthl/transfer/ch_deep_withold_mtl_station_text --seed 0 --region BCOV --withold "{Site: [OliverEast], Cultivar: [Cabernet_Franc]}"
python3 -m trainers.train_model --config mthl/transfer/ch_deep_withold_mtl_station_text --seed 1 --region BCOV --withold "{Site: [OliverEast], Cultivar: [Cabernet_Franc]}"
python3 -m trainers.train_model --config mthl/transfer/ch_deep_withold_mtl_station_text --seed 2 --region BCOV --withold "{Site: [OliverEast], Cultivar: [Cabernet_Franc]}"
python3 -m trainers.train_model --config mthl/transfer/ch_deep_withold_mtl_station_text --seed 3 --region BCOV --withold "{Site: [OliverEast], Cultivar: [Cabernet_Franc]}"
python3 -m trainers.train_model --config mthl/transfer/ch_deep_withold_mtl_station_text --seed 4 --region BCOV --withold "{Site: [OliverEast], Cultivar: [Cabernet_Franc]}"

python3 -m trainers.train_model --config mthl/transfer/ch_deep_withold_mtl_station_text --seed 0 --region BCOV --withold "{Site: [OliverEast], Cultivar: [Chardonnay]}"
python3 -m trainers.train_model --config mthl/transfer/ch_deep_withold_mtl_station_text --seed 1 --region BCOV --withold "{Site: [OliverEast], Cultivar: [Chardonnay]}"
python3 -m trainers.train_model --config mthl/transfer/ch_deep_withold_mtl_station_text --seed 2 --region BCOV --withold "{Site: [OliverEast], Cultivar: [Chardonnay]}"
python3 -m trainers.train_model --config mthl/transfer/ch_deep_withold_mtl_station_text --seed 3 --region BCOV --withold "{Site: [OliverEast], Cultivar: [Chardonnay]}"
python3 -m trainers.train_model --config mthl/transfer/ch_deep_withold_mtl_station_text --seed 4 --region BCOV --withold "{Site: [OliverEast], Cultivar: [Chardonnay]}"

python3 -m trainers.train_model --config mthl/transfer/ch_deep_withold_mtl_station_text --seed 0 --region BCOV --withold "{Site: [OliverEast], Cultivar: [Gewurztraminer]}"
python3 -m trainers.train_model --config mthl/transfer/ch_deep_withold_mtl_station_text --seed 1 --region BCOV --withold "{Site: [OliverEast], Cultivar: [Gewurztraminer]}"
python3 -m trainers.train_model --config mthl/transfer/ch_deep_withold_mtl_station_text --seed 2 --region BCOV --withold "{Site: [OliverEast], Cultivar: [Gewurztraminer]}"
python3 -m trainers.train_model --config mthl/transfer/ch_deep_withold_mtl_station_text --seed 3 --region BCOV --withold "{Site: [OliverEast], Cultivar: [Gewurztraminer]}"
python3 -m trainers.train_model --config mthl/transfer/ch_deep_withold_mtl_station_text --seed 4 --region BCOV --withold "{Site: [OliverEast], Cultivar: [Gewurztraminer]}"

python3 -m trainers.train_model --config mthl/transfer/ch_deep_withold_mtl_station_text --seed 0 --region BCOV --withold "{Site: [OliverEast], Cultivar: [Merlot]}"
python3 -m trainers.train_model --config mthl/transfer/ch_deep_withold_mtl_station_text --seed 1 --region BCOV --withold "{Site: [OliverEast], Cultivar: [Merlot]}"
python3 -m trainers.train_model --config mthl/transfer/ch_deep_withold_mtl_station_text --seed 2 --region BCOV --withold "{Site: [OliverEast], Cultivar: [Merlot]}"
python3 -m trainers.train_model --config mthl/transfer/ch_deep_withold_mtl_station_text --seed 3 --region BCOV --withold "{Site: [OliverEast], Cultivar: [Merlot]}"
python3 -m trainers.train_model --config mthl/transfer/ch_deep_withold_mtl_station_text --seed 4 --region BCOV --withold "{Site: [OliverEast], Cultivar: [Merlot]}"

python3 -m trainers.train_model --config mthl/transfer/ch_deep_withold_mtl_station_text --seed 0 --region BCOV --withold "{Site: [OliverEast], Cultivar: [Pinot_Blanc]}"
python3 -m trainers.train_model --config mthl/transfer/ch_deep_withold_mtl_station_text --seed 1 --region BCOV --withold "{Site: [OliverEast], Cultivar: [Pinot_Blanc]}"
python3 -m trainers.train_model --config mthl/transfer/ch_deep_withold_mtl_station_text --seed 2 --region BCOV --withold "{Site: [OliverEast], Cultivar: [Pinot_Blanc]}"
python3 -m trainers.train_model --config mthl/transfer/ch_deep_withold_mtl_station_text --seed 3 --region BCOV --withold "{Site: [OliverEast], Cultivar: [Pinot_Blanc]}"
python3 -m trainers.train_model --config mthl/transfer/ch_deep_withold_mtl_station_text --seed 4 --region BCOV --withold "{Site: [OliverEast], Cultivar: [Pinot_Blanc]}"

python3 -m trainers.train_model --config mthl/transfer/ch_deep_withold_mtl_station_text --seed 0 --region BCOV --withold "{Site: [OliverEast], Cultivar: [Pinot_Gris]}"
python3 -m trainers.train_model --config mthl/transfer/ch_deep_withold_mtl_station_text --seed 1 --region BCOV --withold "{Site: [OliverEast], Cultivar: [Pinot_Gris]}"
python3 -m trainers.train_model --config mthl/transfer/ch_deep_withold_mtl_station_text --seed 2 --region BCOV --withold "{Site: [OliverEast], Cultivar: [Pinot_Gris]}"
python3 -m trainers.train_model --config mthl/transfer/ch_deep_withold_mtl_station_text --seed 3 --region BCOV --withold "{Site: [OliverEast], Cultivar: [Pinot_Gris]}"
python3 -m trainers.train_model --config mthl/transfer/ch_deep_withold_mtl_station_text --seed 4 --region BCOV --withold "{Site: [OliverEast], Cultivar: [Pinot_Gris]}"

python3 -m trainers.train_model --config mthl/transfer/ch_deep_withold_mtl_station_text --seed 0 --region BCOV --withold "{Site: [OliverEast], Cultivar: [Riesling]}"
python3 -m trainers.train_model --config mthl/transfer/ch_deep_withold_mtl_station_text --seed 1 --region BCOV --withold "{Site: [OliverEast], Cultivar: [Riesling]}"
python3 -m trainers.train_model --config mthl/transfer/ch_deep_withold_mtl_station_text --seed 2 --region BCOV --withold "{Site: [OliverEast], Cultivar: [Riesling]}"
python3 -m trainers.train_model --config mthl/transfer/ch_deep_withold_mtl_station_text --seed 3 --region BCOV --withold "{Site: [OliverEast], Cultivar: [Riesling]}"
python3 -m trainers.train_model --config mthl/transfer/ch_deep_withold_mtl_station_text --seed 4 --region BCOV --withold "{Site: [OliverEast], Cultivar: [Riesling]}"

python3 -m trainers.train_model --config mthl/transfer/ch_deep_withold_mtl_station_text --seed 0 --region BCOV --withold "{Site: [OliverEast], Cultivar: [Sauvignon_Blanc]}"
python3 -m trainers.train_model --config mthl/transfer/ch_deep_withold_mtl_station_text --seed 1 --region BCOV --withold "{Site: [OliverEast], Cultivar: [Sauvignon_Blanc]}"
python3 -m trainers.train_model --config mthl/transfer/ch_deep_withold_mtl_station_text --seed 2 --region BCOV --withold "{Site: [OliverEast], Cultivar: [Sauvignon_Blanc]}"
python3 -m trainers.train_model --config mthl/transfer/ch_deep_withold_mtl_station_text --seed 3 --region BCOV --withold "{Site: [OliverEast], Cultivar: [Sauvignon_Blanc]}"
python3 -m trainers.train_model --config mthl/transfer/ch_deep_withold_mtl_station_text --seed 4 --region BCOV --withold "{Site: [OliverEast], Cultivar: [Sauvignon_Blanc]}"
