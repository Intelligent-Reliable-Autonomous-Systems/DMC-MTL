#!/bin/bash
#SBATCH -J blacksage
#SBATCH -o output/blacksage.out
#SBATCH -e output/blacksage.err
#SBATCH -p eecs,share,gpu
#SBATCH -t 2-00:00:00
#SBATCH --gres=gpu:1

python3 -m trainers.train_model --config mthl/transfer/ch_deep_withold_mtl_station --seed 0 --region BCOV --withold "{Site: [BlackSage], Cultivar: [Cabernet_Franc]}"
python3 -m trainers.train_model --config mthl/transfer/ch_deep_withold_mtl_station --seed 1 --region BCOV --withold "{Site: [BlackSage], Cultivar: [Cabernet_Franc]}"
python3 -m trainers.train_model --config mthl/transfer/ch_deep_withold_mtl_station --seed 2 --region BCOV --withold "{Site: [BlackSage], Cultivar: [Cabernet_Franc]}"
python3 -m trainers.train_model --config mthl/transfer/ch_deep_withold_mtl_station --seed 3 --region BCOV --withold "{Site: [BlackSage], Cultivar: [Cabernet_Franc]}"
python3 -m trainers.train_model --config mthl/transfer/ch_deep_withold_mtl_station --seed 4 --region BCOV --withold "{Site: [BlackSage], Cultivar: [Cabernet_Franc]}"

python3 -m trainers.train_model --config mthl/transfer/ch_deep_withold_mtl_station --seed 0 --region BCOV --withold "{Site: [BlackSage], Cultivar: [Cabernet_Sauvignon]}"
python3 -m trainers.train_model --config mthl/transfer/ch_deep_withold_mtl_station --seed 1 --region BCOV --withold "{Site: [BlackSage], Cultivar: [Cabernet_Sauvignon]}"
python3 -m trainers.train_model --config mthl/transfer/ch_deep_withold_mtl_station --seed 2 --region BCOV --withold "{Site: [BlackSage], Cultivar: [Cabernet_Sauvignon]}"
python3 -m trainers.train_model --config mthl/transfer/ch_deep_withold_mtl_station --seed 3 --region BCOV --withold "{Site: [BlackSage], Cultivar: [Cabernet_Sauvignon]}"
python3 -m trainers.train_model --config mthl/transfer/ch_deep_withold_mtl_station --seed 4 --region BCOV --withold "{Site: [BlackSage], Cultivar: [Cabernet_Sauvignon]}"

python3 -m trainers.train_model --config mthl/transfer/ch_deep_withold_mtl_station --seed 0 --region BCOV --withold "{Site: [BlackSage], Cultivar: [Chardonnay]}"
python3 -m trainers.train_model --config mthl/transfer/ch_deep_withold_mtl_station --seed 1 --region BCOV --withold "{Site: [BlackSage], Cultivar: [Chardonnay]}"
python3 -m trainers.train_model --config mthl/transfer/ch_deep_withold_mtl_station --seed 2 --region BCOV --withold "{Site: [BlackSage], Cultivar: [Chardonnay]}"
python3 -m trainers.train_model --config mthl/transfer/ch_deep_withold_mtl_station --seed 3 --region BCOV --withold "{Site: [BlackSage], Cultivar: [Chardonnay]}"
python3 -m trainers.train_model --config mthl/transfer/ch_deep_withold_mtl_station --seed 4 --region BCOV --withold "{Site: [BlackSage], Cultivar: [Chardonnay]}"

python3 -m trainers.train_model --config mthl/transfer/ch_deep_withold_mtl_station --seed 0 --region BCOV --withold "{Site: [BlackSage], Cultivar: [Merlot]}"
python3 -m trainers.train_model --config mthl/transfer/ch_deep_withold_mtl_station --seed 1 --region BCOV --withold "{Site: [BlackSage], Cultivar: [Merlot]}"
python3 -m trainers.train_model --config mthl/transfer/ch_deep_withold_mtl_station --seed 2 --region BCOV --withold "{Site: [BlackSage], Cultivar: [Merlot]}"
python3 -m trainers.train_model --config mthl/transfer/ch_deep_withold_mtl_station --seed 3 --region BCOV --withold "{Site: [BlackSage], Cultivar: [Merlot]}"
python3 -m trainers.train_model --config mthl/transfer/ch_deep_withold_mtl_station --seed 4 --region BCOV --withold "{Site: [BlackSage], Cultivar: [Merlot]}"

python3 -m trainers.train_model --config mthl/transfer/ch_deep_withold_mtl_station --seed 0 --region BCOV --withold "{Site: [BlackSage], Cultivar: [Pinot_Blanc]}"
python3 -m trainers.train_model --config mthl/transfer/ch_deep_withold_mtl_station --seed 1 --region BCOV --withold "{Site: [BlackSage], Cultivar: [Pinot_Blanc]}"
python3 -m trainers.train_model --config mthl/transfer/ch_deep_withold_mtl_station --seed 2 --region BCOV --withold "{Site: [BlackSage], Cultivar: [Pinot_Blanc]}"
python3 -m trainers.train_model --config mthl/transfer/ch_deep_withold_mtl_station --seed 3 --region BCOV --withold "{Site: [BlackSage], Cultivar: [Pinot_Blanc]}"
python3 -m trainers.train_model --config mthl/transfer/ch_deep_withold_mtl_station --seed 4 --region BCOV --withold "{Site: [BlackSage], Cultivar: [Pinot_Blanc]}"

python3 -m trainers.train_model --config mthl/transfer/ch_deep_withold_mtl_station --seed 0 --region BCOV --withold "{Site: [BlackSage], Cultivar: [Pinot_Gris]}"
python3 -m trainers.train_model --config mthl/transfer/ch_deep_withold_mtl_station --seed 1 --region BCOV --withold "{Site: [BlackSage], Cultivar: [Pinot_Gris]}"
python3 -m trainers.train_model --config mthl/transfer/ch_deep_withold_mtl_station --seed 2 --region BCOV --withold "{Site: [BlackSage], Cultivar: [Pinot_Gris]}"
python3 -m trainers.train_model --config mthl/transfer/ch_deep_withold_mtl_station --seed 3 --region BCOV --withold "{Site: [BlackSage], Cultivar: [Pinot_Gris]}"
python3 -m trainers.train_model --config mthl/transfer/ch_deep_withold_mtl_station --seed 4 --region BCOV --withold "{Site: [BlackSage], Cultivar: [Pinot_Gris]}"

python3 -m trainers.train_model --config mthl/transfer/ch_deep_withold_mtl_station --seed 0 --region BCOV --withold "{Site: [BlackSage], Cultivar: [Pinot_Noir]}"
python3 -m trainers.train_model --config mthl/transfer/ch_deep_withold_mtl_station --seed 1 --region BCOV --withold "{Site: [BlackSage], Cultivar: [Pinot_Noir]}"
python3 -m trainers.train_model --config mthl/transfer/ch_deep_withold_mtl_station --seed 2 --region BCOV --withold "{Site: [BlackSage], Cultivar: [Pinot_Noir]}"
python3 -m trainers.train_model --config mthl/transfer/ch_deep_withold_mtl_station --seed 3 --region BCOV --withold "{Site: [BlackSage], Cultivar: [Pinot_Noir]}"
python3 -m trainers.train_model --config mthl/transfer/ch_deep_withold_mtl_station --seed 4 --region BCOV --withold "{Site: [BlackSage], Cultivar: [Pinot_Noir]}"

python3 -m trainers.train_model --config mthl/transfer/ch_deep_withold_mtl_station --seed 0 --region BCOV --withold "{Site: [BlackSage], Cultivar: [Riesling]}"
python3 -m trainers.train_model --config mthl/transfer/ch_deep_withold_mtl_station --seed 1 --region BCOV --withold "{Site: [BlackSage], Cultivar: [Riesling]}"
python3 -m trainers.train_model --config mthl/transfer/ch_deep_withold_mtl_station --seed 2 --region BCOV --withold "{Site: [BlackSage], Cultivar: [Riesling]}"
python3 -m trainers.train_model --config mthl/transfer/ch_deep_withold_mtl_station --seed 3 --region BCOV --withold "{Site: [BlackSage], Cultivar: [Riesling]}"
python3 -m trainers.train_model --config mthl/transfer/ch_deep_withold_mtl_station --seed 4 --region BCOV --withold "{Site: [BlackSage], Cultivar: [Riesling]}"

python3 -m trainers.train_model --config mthl/transfer/ch_deep_withold_mtl_station --seed 0 --region BCOV --withold "{Site: [BlackSage], Cultivar: [Sauvignon_Blanc]}"
python3 -m trainers.train_model --config mthl/transfer/ch_deep_withold_mtl_station --seed 1 --region BCOV --withold "{Site: [BlackSage], Cultivar: [Sauvignon_Blanc]}"
python3 -m trainers.train_model --config mthl/transfer/ch_deep_withold_mtl_station --seed 2 --region BCOV --withold "{Site: [BlackSage], Cultivar: [Sauvignon_Blanc]}"
python3 -m trainers.train_model --config mthl/transfer/ch_deep_withold_mtl_station --seed 3 --region BCOV --withold "{Site: [BlackSage], Cultivar: [Sauvignon_Blanc]}"
python3 -m trainers.train_model --config mthl/transfer/ch_deep_withold_mtl_station --seed 4 --region BCOV --withold "{Site: [BlackSage], Cultivar: [Sauvignon_Blanc]}"

python3 -m trainers.train_model --config mthl/transfer/ch_deep_withold_mtl_station --seed 0 --region BCOV --withold "{Site: [BlackSage], Cultivar: [Syrah]}"
python3 -m trainers.train_model --config mthl/transfer/ch_deep_withold_mtl_station --seed 1 --region BCOV --withold "{Site: [BlackSage], Cultivar: [Syrah]}"
python3 -m trainers.train_model --config mthl/transfer/ch_deep_withold_mtl_station --seed 2 --region BCOV --withold "{Site: [BlackSage], Cultivar: [Syrah]}"
python3 -m trainers.train_model --config mthl/transfer/ch_deep_withold_mtl_station --seed 3 --region BCOV --withold "{Site: [BlackSage], Cultivar: [Syrah]}"
python3 -m trainers.train_model --config mthl/transfer/ch_deep_withold_mtl_station --seed 4 --region BCOV --withold "{Site: [BlackSage], Cultivar: [Syrah]}"