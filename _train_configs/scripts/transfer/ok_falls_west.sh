#!/bin/bash
#SBATCH -J ok_falls_west
#SBATCH -o output/ok_falls_west.out
#SBATCH -e output/ok_falls_west.err
#SBATCH -p eecs,share,gpu
#SBATCH -t 2-00:00:00
#SBATCH --gres=gpu:1

python3 -m trainers.train_model --config mthl/transfer/ch_deep_withold_mtl_station --seed 0 --region BCOV --withold "{Site: [OkFallsWest], Cultivar: [Gewurztraminer]}"
python3 -m trainers.train_model --config mthl/transfer/ch_deep_withold_mtl_station --seed 1 --region BCOV --withold "{Site: [OkFallsWest], Cultivar: [Gewurztraminer]}"
python3 -m trainers.train_model --config mthl/transfer/ch_deep_withold_mtl_station --seed 2 --region BCOV --withold "{Site: [OkFallsWest], Cultivar: [Gewurztraminer]}"
python3 -m trainers.train_model --config mthl/transfer/ch_deep_withold_mtl_station --seed 3 --region BCOV --withold "{Site: [OkFallsWest], Cultivar: [Gewurztraminer]}"
python3 -m trainers.train_model --config mthl/transfer/ch_deep_withold_mtl_station --seed 4 --region BCOV --withold "{Site: [OkFallsWest], Cultivar: [Gewurztraminer]}"

python3 -m trainers.train_model --config mthl/transfer/ch_deep_withold_mtl_station --seed 0 --region BCOV --withold "{Site: [OliverSouth], Cultivar: [Chardonnay]}"
python3 -m trainers.train_model --config mthl/transfer/ch_deep_withold_mtl_station --seed 1 --region BCOV --withold "{Site: [OliverSouth], Cultivar: [Chardonnay]}"
python3 -m trainers.train_model --config mthl/transfer/ch_deep_withold_mtl_station --seed 2 --region BCOV --withold "{Site: [OliverSouth], Cultivar: [Chardonnay]}"
python3 -m trainers.train_model --config mthl/transfer/ch_deep_withold_mtl_station --seed 3 --region BCOV --withold "{Site: [OliverSouth], Cultivar: [Chardonnay]}"
python3 -m trainers.train_model --config mthl/transfer/ch_deep_withold_mtl_station --seed 4 --region BCOV --withold "{Site: [OliverSouth], Cultivar: [Chardonnay]}"

python3 -m trainers.train_model --config mthl/transfer/ch_deep_withold_mtl_station --seed 0 --region BCOV --withold "{Site: [OliverWest], Cultivar: [Merlot]}"
python3 -m trainers.train_model --config mthl/transfer/ch_deep_withold_mtl_station --seed 1 --region BCOV --withold "{Site: [OliverWest], Cultivar: [Merlot]}"
python3 -m trainers.train_model --config mthl/transfer/ch_deep_withold_mtl_station --seed 2 --region BCOV --withold "{Site: [OliverWest], Cultivar: [Merlot]}"
python3 -m trainers.train_model --config mthl/transfer/ch_deep_withold_mtl_station --seed 3 --region BCOV --withold "{Site: [OliverWest], Cultivar: [Merlot]}"
python3 -m trainers.train_model --config mthl/transfer/ch_deep_withold_mtl_station --seed 4 --region BCOV --withold "{Site: [OliverWest], Cultivar: [Merlot]}"

python3 -m trainers.train_model --config mthl/transfer/ch_deep_withold_mtl_station --seed 0 --region BCOV --withold "{Site: [OsoyoosWest], Cultivar: [Merlot]}"
python3 -m trainers.train_model --config mthl/transfer/ch_deep_withold_mtl_station --seed 1 --region BCOV --withold "{Site: [OsoyoosWest], Cultivar: [Merlot]}"
python3 -m trainers.train_model --config mthl/transfer/ch_deep_withold_mtl_station --seed 2 --region BCOV --withold "{Site: [OsoyoosWest], Cultivar: [Merlot]}"
python3 -m trainers.train_model --config mthl/transfer/ch_deep_withold_mtl_station --seed 3 --region BCOV --withold "{Site: [OsoyoosWest], Cultivar: [Merlot]}"
python3 -m trainers.train_model --config mthl/transfer/ch_deep_withold_mtl_station --seed 4 --region BCOV --withold "{Site: [OsoyoosWest], Cultivar: [Merlot]}"
