#!/bin/bash
#SBATCH -J extra_4
#SBATCH -o output/extra_4.out
#SBATCH -e output/extra_4.err
#SBATCH -p eecs,gpu
#SBATCH -t 2-00:00:00
#SBATCH --gres=gpu:1


python3 -m trainers.train_model --config mthl/transfer/ch_deep_withold_mtl_station_text --seed 0 --region BCOV --withold "{Site: [OsoyoosSoutheast], Cultivar: [Chardonnay]}"
python3 -m trainers.train_model --config mthl/transfer/ch_deep_withold_mtl_station_text --seed 1 --region BCOV --withold "{Site: [OsoyoosSoutheast], Cultivar: [Chardonnay]}"
python3 -m trainers.train_model --config mthl/transfer/ch_deep_withold_mtl_station_text --seed 2 --region BCOV --withold "{Site: [OsoyoosSoutheast], Cultivar: [Chardonnay]}"
python3 -m trainers.train_model --config mthl/transfer/ch_deep_withold_mtl_station_text --seed 3 --region BCOV --withold "{Site: [OsoyoosSoutheast], Cultivar: [Chardonnay]}"
python3 -m trainers.train_model --config mthl/transfer/ch_deep_withold_mtl_station_text --seed 4 --region BCOV --withold "{Site: [OsoyoosSoutheast], Cultivar: [Chardonnay]}"

python3 -m trainers.train_model --config mthl/transfer/ch_deep_withold_mtl_station_text --seed 0 --region BCOV --withold "{Site: [OsoyoosSoutheast], Cultivar: [Syrah]}"
python3 -m trainers.train_model --config mthl/transfer/ch_deep_withold_mtl_station_text --seed 1 --region BCOV --withold "{Site: [OsoyoosSoutheast], Cultivar: [Syrah]}"
python3 -m trainers.train_model --config mthl/transfer/ch_deep_withold_mtl_station_text --seed 2 --region BCOV --withold "{Site: [OsoyoosSoutheast], Cultivar: [Syrah]}"
python3 -m trainers.train_model --config mthl/transfer/ch_deep_withold_mtl_station_text --seed 3 --region BCOV --withold "{Site: [OsoyoosSoutheast], Cultivar: [Syrah]}"
python3 -m trainers.train_model --config mthl/transfer/ch_deep_withold_mtl_station_text --seed 4 --region BCOV --withold "{Site: [OsoyoosSoutheast], Cultivar: [Syrah]}"

python3 -m trainers.train_model --config mthl/transfer/ch_deep_withold_mtl_station_text --seed 0 --region BCOV --withold "{Site: [OKFallsWest], Cultivar: [Gewurztraminer]}"
python3 -m trainers.train_model --config mthl/transfer/ch_deep_withold_mtl_station_text --seed 1 --region BCOV --withold "{Site: [OKFallsWest], Cultivar: [Gewurztraminer]}"
python3 -m trainers.train_model --config mthl/transfer/ch_deep_withold_mtl_station_text --seed 2 --region BCOV --withold "{Site: [OKFallsWest], Cultivar: [Gewurztraminer]}"
python3 -m trainers.train_model --config mthl/transfer/ch_deep_withold_mtl_station_text --seed 3 --region BCOV --withold "{Site: [OKFallsWest], Cultivar: [Gewurztraminer]}"
python3 -m trainers.train_model --config mthl/transfer/ch_deep_withold_mtl_station_text --seed 4 --region BCOV --withold "{Site: [OKFallsWest], Cultivar: [Gewurztraminer]}"
