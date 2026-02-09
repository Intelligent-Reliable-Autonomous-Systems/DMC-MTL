#!/bin/bash
#SBATCH -J creek_shores
#SBATCH -o output/creek_shores.out
#SBATCH -e output/creek_shores.err
#SBATCH -p eecs,share,gpu
#SBATCH -t 2-00:00:00
#SBATCH --gres=gpu:1

python3 -m trainers.train_model --config mthl/transfer/ch_deep_withold_mtl_station --seed 0 --region ONNP --withold "{Site: [CreekShores], Cultivar: [Cabernet_Franc]}"
python3 -m trainers.train_model --config mthl/transfer/ch_deep_withold_mtl_station --seed 1 --region ONNP --withold "{Site: [CreekShores], Cultivar: [Cabernet_Franc]}"
python3 -m trainers.train_model --config mthl/transfer/ch_deep_withold_mtl_station --seed 2 --region ONNP --withold "{Site: [CreekShores], Cultivar: [Cabernet_Franc]}"
python3 -m trainers.train_model --config mthl/transfer/ch_deep_withold_mtl_station --seed 3 --region ONNP --withold "{Site: [CreekShores], Cultivar: [Cabernet_Franc]}"
python3 -m trainers.train_model --config mthl/transfer/ch_deep_withold_mtl_station --seed 4 --region ONNP --withold "{Site: [CreekShores], Cultivar: [Cabernet_Franc]}"

python3 -m trainers.train_model --config mthl/transfer/ch_deep_withold_mtl_station --seed 0 --region ONNP --withold "{Site: [CreekShores], Cultivar: [Cabernet_Sauvignon]}"
python3 -m trainers.train_model --config mthl/transfer/ch_deep_withold_mtl_station --seed 1 --region ONNP --withold "{Site: [CreekShores], Cultivar: [Cabernet_Sauvignon]}"
python3 -m trainers.train_model --config mthl/transfer/ch_deep_withold_mtl_station --seed 2 --region ONNP --withold "{Site: [CreekShores], Cultivar: [Cabernet_Sauvignon]}"
python3 -m trainers.train_model --config mthl/transfer/ch_deep_withold_mtl_station --seed 3 --region ONNP --withold "{Site: [CreekShores], Cultivar: [Cabernet_Sauvignon]}"
python3 -m trainers.train_model --config mthl/transfer/ch_deep_withold_mtl_station --seed 4 --region ONNP --withold "{Site: [CreekShores], Cultivar: [Cabernet_Sauvignon]}"

python3 -m trainers.train_model --config mthl/transfer/ch_deep_withold_mtl_station --seed 0 --region ONNP --withold "{Site: [CreekShores], Cultivar: [Chardonnay]}"
python3 -m trainers.train_model --config mthl/transfer/ch_deep_withold_mtl_station --seed 1 --region ONNP --withold "{Site: [CreekShores], Cultivar: [Chardonnay]}"
python3 -m trainers.train_model --config mthl/transfer/ch_deep_withold_mtl_station --seed 2 --region ONNP --withold "{Site: [CreekShores], Cultivar: [Chardonnay]}"
python3 -m trainers.train_model --config mthl/transfer/ch_deep_withold_mtl_station --seed 3 --region ONNP --withold "{Site: [CreekShores], Cultivar: [Chardonnay]}"
python3 -m trainers.train_model --config mthl/transfer/ch_deep_withold_mtl_station --seed 4 --region ONNP --withold "{Site: [CreekShores], Cultivar: [Chardonnay]}"

python3 -m trainers.train_model --config mthl/transfer/ch_deep_withold_mtl_station --seed 0 --region ONNP --withold "{Site: [CreekShores], Cultivar: [Merlot]}"
python3 -m trainers.train_model --config mthl/transfer/ch_deep_withold_mtl_station --seed 1 --region ONNP --withold "{Site: [CreekShores], Cultivar: [Merlot]}"
python3 -m trainers.train_model --config mthl/transfer/ch_deep_withold_mtl_station --seed 2 --region ONNP --withold "{Site: [CreekShores], Cultivar: [Merlot]}"
python3 -m trainers.train_model --config mthl/transfer/ch_deep_withold_mtl_station --seed 3 --region ONNP --withold "{Site: [CreekShores], Cultivar: [Merlot]}"
python3 -m trainers.train_model --config mthl/transfer/ch_deep_withold_mtl_station --seed 4 --region ONNP --withold "{Site: [CreekShores], Cultivar: [Merlot]}"

python3 -m trainers.train_model --config mthl/transfer/ch_deep_withold_mtl_station --seed 0 --region ONNP --withold "{Site: [CreekShores], Cultivar: [Pinot_Noir]}"
python3 -m trainers.train_model --config mthl/transfer/ch_deep_withold_mtl_station --seed 1 --region ONNP --withold "{Site: [CreekShores], Cultivar: [Pinot_Noir]}"
python3 -m trainers.train_model --config mthl/transfer/ch_deep_withold_mtl_station --seed 2 --region ONNP --withold "{Site: [CreekShores], Cultivar: [Pinot_Noir]}"
python3 -m trainers.train_model --config mthl/transfer/ch_deep_withold_mtl_station --seed 3 --region ONNP --withold "{Site: [CreekShores], Cultivar: [Pinot_Noir]}"
python3 -m trainers.train_model --config mthl/transfer/ch_deep_withold_mtl_station --seed 4 --region ONNP --withold "{Site: [CreekShores], Cultivar: [Pinot_Noir]}"

python3 -m trainers.train_model --config mthl/transfer/ch_deep_withold_mtl_station --seed 0 --region ONNP --withold "{Site: [CreekShores], Cultivar: [Riesling]}"
python3 -m trainers.train_model --config mthl/transfer/ch_deep_withold_mtl_station --seed 1 --region ONNP --withold "{Site: [CreekShores], Cultivar: [Riesling]}"
python3 -m trainers.train_model --config mthl/transfer/ch_deep_withold_mtl_station --seed 2 --region ONNP --withold "{Site: [CreekShores], Cultivar: [Riesling]}"
python3 -m trainers.train_model --config mthl/transfer/ch_deep_withold_mtl_station --seed 3 --region ONNP --withold "{Site: [CreekShores], Cultivar: [Riesling]}"
python3 -m trainers.train_model --config mthl/transfer/ch_deep_withold_mtl_station --seed 4 --region ONNP --withold "{Site: [CreekShores], Cultivar: [Riesling]}"

python3 -m trainers.train_model --config mthl/transfer/ch_deep_withold_mtl_station --seed 0 --region ONNP --withold "{Site: [CreekShores], Cultivar: [Syrah]}"
python3 -m trainers.train_model --config mthl/transfer/ch_deep_withold_mtl_station --seed 1 --region ONNP --withold "{Site: [CreekShores], Cultivar: [Syrah]}"
python3 -m trainers.train_model --config mthl/transfer/ch_deep_withold_mtl_station --seed 2 --region ONNP --withold "{Site: [CreekShores], Cultivar: [Syrah]}"
python3 -m trainers.train_model --config mthl/transfer/ch_deep_withold_mtl_station --seed 3 --region ONNP --withold "{Site: [CreekShores], Cultivar: [Syrah]}"
python3 -m trainers.train_model --config mthl/transfer/ch_deep_withold_mtl_station --seed 4 --region ONNP --withold "{Site: [CreekShores], Cultivar: [Syrah]}"