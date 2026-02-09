#!/bin/bash
#SBATCH -J lincoln_lakeshore
#SBATCH -o output/lincoln_lakeshore.out
#SBATCH -e output/lincoln_lakeshore.err
#SBATCH -p eecs,share,gpu
#SBATCH -t 2-00:00:00
#SBATCH --gres=gpu:1

python3 -m trainers.train_model --config mthl/transfer/ch_deep_withold_mtl_station --seed 0 --region ONNP --withold "{Site: [LincolnLakeshore], Cultivar: [Cabernet_Franc]}"
python3 -m trainers.train_model --config mthl/transfer/ch_deep_withold_mtl_station --seed 1 --region ONNP --withold "{Site: [LincolnLakeshore], Cultivar: [Cabernet_Franc]}"
python3 -m trainers.train_model --config mthl/transfer/ch_deep_withold_mtl_station --seed 2 --region ONNP --withold "{Site: [LincolnLakeshore], Cultivar: [Cabernet_Franc]}"
python3 -m trainers.train_model --config mthl/transfer/ch_deep_withold_mtl_station --seed 3 --region ONNP --withold "{Site: [LincolnLakeshore], Cultivar: [Cabernet_Franc]}"
python3 -m trainers.train_model --config mthl/transfer/ch_deep_withold_mtl_station --seed 4 --region ONNP --withold "{Site: [LincolnLakeshore], Cultivar: [Cabernet_Franc]}"

python3 -m trainers.train_model --config mthl/transfer/ch_deep_withold_mtl_station --seed 0 --region ONNP --withold "{Site: [LincolnLakeshore], Cultivar: [Cabernet_Sauvignon]}"
python3 -m trainers.train_model --config mthl/transfer/ch_deep_withold_mtl_station --seed 1 --region ONNP --withold "{Site: [LincolnLakeshore], Cultivar: [Cabernet_Sauvignon]}"
python3 -m trainers.train_model --config mthl/transfer/ch_deep_withold_mtl_station --seed 2 --region ONNP --withold "{Site: [LincolnLakeshore], Cultivar: [Cabernet_Sauvignon]}"
python3 -m trainers.train_model --config mthl/transfer/ch_deep_withold_mtl_station --seed 3 --region ONNP --withold "{Site: [LincolnLakeshore], Cultivar: [Cabernet_Sauvignon]}"
python3 -m trainers.train_model --config mthl/transfer/ch_deep_withold_mtl_station --seed 4 --region ONNP --withold "{Site: [LincolnLakeshore], Cultivar: [Cabernet_Sauvignon]}"

python3 -m trainers.train_model --config mthl/transfer/ch_deep_withold_mtl_station --seed 0 --region ONNP --withold "{Site: [LincolnLakeshore], Cultivar: [Chardonnay]}"
python3 -m trainers.train_model --config mthl/transfer/ch_deep_withold_mtl_station --seed 1 --region ONNP --withold "{Site: [LincolnLakeshore], Cultivar: [Chardonnay]}"
python3 -m trainers.train_model --config mthl/transfer/ch_deep_withold_mtl_station --seed 2 --region ONNP --withold "{Site: [LincolnLakeshore], Cultivar: [Chardonnay]}"
python3 -m trainers.train_model --config mthl/transfer/ch_deep_withold_mtl_station --seed 3 --region ONNP --withold "{Site: [LincolnLakeshore], Cultivar: [Chardonnay]}"
python3 -m trainers.train_model --config mthl/transfer/ch_deep_withold_mtl_station --seed 4 --region ONNP --withold "{Site: [LincolnLakeshore], Cultivar: [Chardonnay]}"

python3 -m trainers.train_model --config mthl/transfer/ch_deep_withold_mtl_station --seed 0 --region ONNP --withold "{Site: [LincolnLakeshore], Cultivar: [Merlot]}"
python3 -m trainers.train_model --config mthl/transfer/ch_deep_withold_mtl_station --seed 1 --region ONNP --withold "{Site: [LincolnLakeshore], Cultivar: [Merlot]}"
python3 -m trainers.train_model --config mthl/transfer/ch_deep_withold_mtl_station --seed 2 --region ONNP --withold "{Site: [LincolnLakeshore], Cultivar: [Merlot]}"
python3 -m trainers.train_model --config mthl/transfer/ch_deep_withold_mtl_station --seed 3 --region ONNP --withold "{Site: [LincolnLakeshore], Cultivar: [Merlot]}"
python3 -m trainers.train_model --config mthl/transfer/ch_deep_withold_mtl_station --seed 4 --region ONNP --withold "{Site: [LincolnLakeshore], Cultivar: [Merlot]}"

python3 -m trainers.train_model --config mthl/transfer/ch_deep_withold_mtl_station --seed 0 --region ONNP --withold "{Site: [LincolnLakeshore], Cultivar: [Pinot_Noir]}"
python3 -m trainers.train_model --config mthl/transfer/ch_deep_withold_mtl_station --seed 1 --region ONNP --withold "{Site: [LincolnLakeshore], Cultivar: [Pinot_Noir]}"
python3 -m trainers.train_model --config mthl/transfer/ch_deep_withold_mtl_station --seed 2 --region ONNP --withold "{Site: [LincolnLakeshore], Cultivar: [Pinot_Noir]}"
python3 -m trainers.train_model --config mthl/transfer/ch_deep_withold_mtl_station --seed 3 --region ONNP --withold "{Site: [LincolnLakeshore], Cultivar: [Pinot_Noir]}"
python3 -m trainers.train_model --config mthl/transfer/ch_deep_withold_mtl_station --seed 4 --region ONNP --withold "{Site: [LincolnLakeshore], Cultivar: [Pinot_Noir]}"

python3 -m trainers.train_model --config mthl/transfer/ch_deep_withold_mtl_station --seed 0 --region ONNP --withold "{Site: [LincolnLakeshore], Cultivar: [Riesling]}"
python3 -m trainers.train_model --config mthl/transfer/ch_deep_withold_mtl_station --seed 1 --region ONNP --withold "{Site: [LincolnLakeshore], Cultivar: [Riesling]}"
python3 -m trainers.train_model --config mthl/transfer/ch_deep_withold_mtl_station --seed 2 --region ONNP --withold "{Site: [LincolnLakeshore], Cultivar: [Riesling]}"
python3 -m trainers.train_model --config mthl/transfer/ch_deep_withold_mtl_station --seed 3 --region ONNP --withold "{Site: [LincolnLakeshore], Cultivar: [Riesling]}"
python3 -m trainers.train_model --config mthl/transfer/ch_deep_withold_mtl_station --seed 4 --region ONNP --withold "{Site: [LincolnLakeshore], Cultivar: [Riesling]}"

python3 -m trainers.train_model --config mthl/transfer/ch_deep_withold_mtl_station --seed 0 --region ONNP --withold "{Site: [LincolnLakeshore], Cultivar: [Sauvignon_Blanc]}"
python3 -m trainers.train_model --config mthl/transfer/ch_deep_withold_mtl_station --seed 1 --region ONNP --withold "{Site: [LincolnLakeshore], Cultivar: [Sauvignon_Blanc]}"
python3 -m trainers.train_model --config mthl/transfer/ch_deep_withold_mtl_station --seed 2 --region ONNP --withold "{Site: [LincolnLakeshore], Cultivar: [Sauvignon_Blanc]}"
python3 -m trainers.train_model --config mthl/transfer/ch_deep_withold_mtl_station --seed 3 --region ONNP --withold "{Site: [LincolnLakeshore], Cultivar: [Sauvignon_Blanc]}"
python3 -m trainers.train_model --config mthl/transfer/ch_deep_withold_mtl_station --seed 4 --region ONNP --withold "{Site: [LincolnLakeshore], Cultivar: [Sauvignon_Blanc]}"