#!/bin/bash
#SBATCH -J text_four_mile_creek
#SBATCH -o output/text_four_mile_creek.out
#SBATCH -e output/text_four_mile_creek.err
#SBATCH -p eecs,gpu
#SBATCH -t 2-00:00:00
#SBATCH --gres=gpu:1

python3 -m trainers.train_model --config mthl/transfer/ch_deep_withold_mtl_station_text --seed 0 --region ONNP --withold "{Site: [FourMileCreek], Cultivar: [Cabernet_Franc]}"
python3 -m trainers.train_model --config mthl/transfer/ch_deep_withold_mtl_station_text --seed 1 --region ONNP --withold "{Site: [FourMileCreek], Cultivar: [Cabernet_Franc]}"
python3 -m trainers.train_model --config mthl/transfer/ch_deep_withold_mtl_station_text --seed 2 --region ONNP --withold "{Site: [FourMileCreek], Cultivar: [Cabernet_Franc]}"
python3 -m trainers.train_model --config mthl/transfer/ch_deep_withold_mtl_station_text --seed 3 --region ONNP --withold "{Site: [FourMileCreek], Cultivar: [Cabernet_Franc]}"
python3 -m trainers.train_model --config mthl/transfer/ch_deep_withold_mtl_station_text --seed 4 --region ONNP --withold "{Site: [FourMileCreek], Cultivar: [Cabernet_Franc]}"

python3 -m trainers.train_model --config mthl/transfer/ch_deep_withold_mtl_station_text --seed 0 --region ONNP --withold "{Site: [FourMileCreek], Cultivar: [Cabernet_Sauvignon]}"
python3 -m trainers.train_model --config mthl/transfer/ch_deep_withold_mtl_station_text --seed 1 --region ONNP --withold "{Site: [FourMileCreek], Cultivar: [Cabernet_Sauvignon]}"
python3 -m trainers.train_model --config mthl/transfer/ch_deep_withold_mtl_station_text --seed 2 --region ONNP --withold "{Site: [FourMileCreek], Cultivar: [Cabernet_Sauvignon]}"
python3 -m trainers.train_model --config mthl/transfer/ch_deep_withold_mtl_station_text --seed 3 --region ONNP --withold "{Site: [FourMileCreek], Cultivar: [Cabernet_Sauvignon]}"
python3 -m trainers.train_model --config mthl/transfer/ch_deep_withold_mtl_station_text --seed 4 --region ONNP --withold "{Site: [FourMileCreek], Cultivar: [Cabernet_Sauvignon]}"

python3 -m trainers.train_model --config mthl/transfer/ch_deep_withold_mtl_station_text --seed 0 --region ONNP --withold "{Site: [FourMileCreek], Cultivar: [Chardonnay]}"
python3 -m trainers.train_model --config mthl/transfer/ch_deep_withold_mtl_station_text --seed 1 --region ONNP --withold "{Site: [FourMileCreek], Cultivar: [Chardonnay]}"
python3 -m trainers.train_model --config mthl/transfer/ch_deep_withold_mtl_station_text --seed 2 --region ONNP --withold "{Site: [FourMileCreek], Cultivar: [Chardonnay]}"
python3 -m trainers.train_model --config mthl/transfer/ch_deep_withold_mtl_station_text --seed 3 --region ONNP --withold "{Site: [FourMileCreek], Cultivar: [Chardonnay]}"
python3 -m trainers.train_model --config mthl/transfer/ch_deep_withold_mtl_station_text --seed 4 --region ONNP --withold "{Site: [FourMileCreek], Cultivar: [Chardonnay]}"

python3 -m trainers.train_model --config mthl/transfer/ch_deep_withold_mtl_station_text --seed 0 --region ONNP --withold "{Site: [FourMileCreek], Cultivar: [Merlot]}"
python3 -m trainers.train_model --config mthl/transfer/ch_deep_withold_mtl_station_text --seed 1 --region ONNP --withold "{Site: [FourMileCreek], Cultivar: [Merlot]}"
python3 -m trainers.train_model --config mthl/transfer/ch_deep_withold_mtl_station_text --seed 2 --region ONNP --withold "{Site: [FourMileCreek], Cultivar: [Merlot]}"
python3 -m trainers.train_model --config mthl/transfer/ch_deep_withold_mtl_station_text --seed 3 --region ONNP --withold "{Site: [FourMileCreek], Cultivar: [Merlot]}"
python3 -m trainers.train_model --config mthl/transfer/ch_deep_withold_mtl_station_text --seed 4 --region ONNP --withold "{Site: [FourMileCreek], Cultivar: [Merlot]}"

python3 -m trainers.train_model --config mthl/transfer/ch_deep_withold_mtl_station_text --seed 0 --region ONNP --withold "{Site: [FourMileCreek], Cultivar: [Pinot_Noir]}"
python3 -m trainers.train_model --config mthl/transfer/ch_deep_withold_mtl_station_text --seed 1 --region ONNP --withold "{Site: [FourMileCreek], Cultivar: [Pinot_Noir]}"
python3 -m trainers.train_model --config mthl/transfer/ch_deep_withold_mtl_station_text --seed 2 --region ONNP --withold "{Site: [FourMileCreek], Cultivar: [Pinot_Noir]}"
python3 -m trainers.train_model --config mthl/transfer/ch_deep_withold_mtl_station_text --seed 3 --region ONNP --withold "{Site: [FourMileCreek], Cultivar: [Pinot_Noir]}"
python3 -m trainers.train_model --config mthl/transfer/ch_deep_withold_mtl_station_text --seed 4 --region ONNP --withold "{Site: [FourMileCreek], Cultivar: [Pinot_Noir]}"

python3 -m trainers.train_model --config mthl/transfer/ch_deep_withold_mtl_station_text --seed 0 --region ONNP --withold "{Site: [FourMileCreek], Cultivar: [Riesling]}"
python3 -m trainers.train_model --config mthl/transfer/ch_deep_withold_mtl_station_text --seed 1 --region ONNP --withold "{Site: [FourMileCreek], Cultivar: [Riesling]}"
python3 -m trainers.train_model --config mthl/transfer/ch_deep_withold_mtl_station_text --seed 2 --region ONNP --withold "{Site: [FourMileCreek], Cultivar: [Riesling]}"
python3 -m trainers.train_model --config mthl/transfer/ch_deep_withold_mtl_station_text --seed 3 --region ONNP --withold "{Site: [FourMileCreek], Cultivar: [Riesling]}"
python3 -m trainers.train_model --config mthl/transfer/ch_deep_withold_mtl_station_text --seed 4 --region ONNP --withold "{Site: [FourMileCreek], Cultivar: [Riesling]}"

python3 -m trainers.train_model --config mthl/transfer/ch_deep_withold_mtl_station_text --seed 0 --region ONNP --withold "{Site: [FourMileCreek], Cultivar: [Sauvignon_Blanc]}"
python3 -m trainers.train_model --config mthl/transfer/ch_deep_withold_mtl_station_text --seed 1 --region ONNP --withold "{Site: [FourMileCreek], Cultivar: [Sauvignon_Blanc]}"
python3 -m trainers.train_model --config mthl/transfer/ch_deep_withold_mtl_station_text --seed 2 --region ONNP --withold "{Site: [FourMileCreek], Cultivar: [Sauvignon_Blanc]}"
python3 -m trainers.train_model --config mthl/transfer/ch_deep_withold_mtl_station_text --seed 3 --region ONNP --withold "{Site: [FourMileCreek], Cultivar: [Sauvignon_Blanc]}"
python3 -m trainers.train_model --config mthl/transfer/ch_deep_withold_mtl_station_text --seed 4 --region ONNP --withold "{Site: [FourMileCreek], Cultivar: [Sauvignon_Blanc]}"