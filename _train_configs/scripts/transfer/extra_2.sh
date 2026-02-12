#!/bin/bash
#SBATCH -J extra_2
#SBATCH -o output/extra_2.out
#SBATCH -e output/extra_2.err
#SBATCH -p eecs,gpu
#SBATCH -t 2-00:00:00
#SBATCH --gres=gpu:1

python3 -m trainers.train_model --config mthl/transfer/ch_deep_withold_mtl_station --seed 0 --region BCOV --withold "{Site: [OsoyoosSoutheast], Cultivar: [Cabernet_Franc]}"
python3 -m trainers.train_model --config mthl/transfer/ch_deep_withold_mtl_station --seed 1 --region BCOV --withold "{Site: [OsoyoosSoutheast], Cultivar: [Cabernet_Franc]}"
python3 -m trainers.train_model --config mthl/transfer/ch_deep_withold_mtl_station --seed 2 --region BCOV --withold "{Site: [OsoyoosSoutheast], Cultivar: [Cabernet_Franc]}"
python3 -m trainers.train_model --config mthl/transfer/ch_deep_withold_mtl_station --seed 3 --region BCOV --withold "{Site: [OsoyoosSoutheast], Cultivar: [Cabernet_Franc]}"
python3 -m trainers.train_model --config mthl/transfer/ch_deep_withold_mtl_station --seed 4 --region BCOV --withold "{Site: [OsoyoosSoutheast], Cultivar: [Cabernet_Franc]}"

python3 -m trainers.train_model --config mthl/transfer/ch_deep_withold_mtl_station --seed 0 --region BCOV --withold "{Site: [OsoyoosSoutheast], Cultivar: [Cabernet_Sauvignon]}"
python3 -m trainers.train_model --config mthl/transfer/ch_deep_withold_mtl_station --seed 1 --region BCOV --withold "{Site: [OsoyoosSoutheast], Cultivar: [Cabernet_Sauvignon]}"
python3 -m trainers.train_model --config mthl/transfer/ch_deep_withold_mtl_station --seed 2 --region BCOV --withold "{Site: [OsoyoosSoutheast], Cultivar: [Cabernet_Sauvignon]}"
python3 -m trainers.train_model --config mthl/transfer/ch_deep_withold_mtl_station --seed 3 --region BCOV --withold "{Site: [OsoyoosSoutheast], Cultivar: [Cabernet_Sauvignon]}"
python3 -m trainers.train_model --config mthl/transfer/ch_deep_withold_mtl_station --seed 4 --region BCOV --withold "{Site: [OsoyoosSoutheast], Cultivar: [Cabernet_Sauvignon]}"

python3 -m trainers.train_model --config mthl/transfer/ch_deep_withold_mtl_station --seed 0 --region BCOV --withold "{Site: [OsoyoosSoutheast], Cultivar: [Chardonnay]}"
python3 -m trainers.train_model --config mthl/transfer/ch_deep_withold_mtl_station --seed 1 --region BCOV --withold "{Site: [OsoyoosSoutheast], Cultivar: [Chardonnay]}"
python3 -m trainers.train_model --config mthl/transfer/ch_deep_withold_mtl_station --seed 2 --region BCOV --withold "{Site: [OsoyoosSoutheast], Cultivar: [Chardonnay]}"
python3 -m trainers.train_model --config mthl/transfer/ch_deep_withold_mtl_station --seed 3 --region BCOV --withold "{Site: [OsoyoosSoutheast], Cultivar: [Chardonnay]}"
python3 -m trainers.train_model --config mthl/transfer/ch_deep_withold_mtl_station --seed 4 --region BCOV --withold "{Site: [OsoyoosSoutheast], Cultivar: [Chardonnay]}"

python3 -m trainers.train_model --config mthl/transfer/ch_deep_withold_mtl_station --seed 0 --region BCOV --withold "{Site: [OsoyoosSoutheast], Cultivar: [Syrah]}"
python3 -m trainers.train_model --config mthl/transfer/ch_deep_withold_mtl_station --seed 1 --region BCOV --withold "{Site: [OsoyoosSoutheast], Cultivar: [Syrah]}"
python3 -m trainers.train_model --config mthl/transfer/ch_deep_withold_mtl_station --seed 2 --region BCOV --withold "{Site: [OsoyoosSoutheast], Cultivar: [Syrah]}"
python3 -m trainers.train_model --config mthl/transfer/ch_deep_withold_mtl_station --seed 3 --region BCOV --withold "{Site: [OsoyoosSoutheast], Cultivar: [Syrah]}"
python3 -m trainers.train_model --config mthl/transfer/ch_deep_withold_mtl_station --seed 4 --region BCOV --withold "{Site: [OsoyoosSoutheast], Cultivar: [Syrah]}"
