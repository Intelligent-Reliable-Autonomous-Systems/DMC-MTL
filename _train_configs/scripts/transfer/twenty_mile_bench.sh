#!/bin/bash
#SBATCH -J twenty_mile_bench
#SBATCH -o output/twenty_mile_bench.out
#SBATCH -e output/twenty_mile_bench.err
#SBATCH -p eecs,gpu
#SBATCH -t 2-00:00:00
#SBATCH --gres=gpu:1


python3 -m trainers.train_model --config mthl/transfer/ch_deep_withold_mtl_station --seed 0 --region ONNP --withold "{Site: [TwentyMileBench], Cultivar: [Cabernet_Franc]}"
python3 -m trainers.train_model --config mthl/transfer/ch_deep_withold_mtl_station --seed 1 --region ONNP --withold "{Site: [TwentyMileBench], Cultivar: [Cabernet_Franc]}"
python3 -m trainers.train_model --config mthl/transfer/ch_deep_withold_mtl_station --seed 2 --region ONNP --withold "{Site: [TwentyMileBench], Cultivar: [Cabernet_Franc]}"
python3 -m trainers.train_model --config mthl/transfer/ch_deep_withold_mtl_station --seed 3 --region ONNP --withold "{Site: [TwentyMileBench], Cultivar: [Cabernet_Franc]}"
python3 -m trainers.train_model --config mthl/transfer/ch_deep_withold_mtl_station --seed 4 --region ONNP --withold "{Site: [TwentyMileBench], Cultivar: [Cabernet_Franc]}"

python3 -m trainers.train_model --config mthl/transfer/ch_deep_withold_mtl_station --seed 0 --region ONNP --withold "{Site: [TwentyMileBench], Cultivar: [Chardonnay]}"
python3 -m trainers.train_model --config mthl/transfer/ch_deep_withold_mtl_station --seed 1 --region ONNP --withold "{Site: [TwentyMileBench], Cultivar: [Chardonnay]}"
python3 -m trainers.train_model --config mthl/transfer/ch_deep_withold_mtl_station --seed 2 --region ONNP --withold "{Site: [TwentyMileBench], Cultivar: [Chardonnay]}"
python3 -m trainers.train_model --config mthl/transfer/ch_deep_withold_mtl_station --seed 3 --region ONNP --withold "{Site: [TwentyMileBench], Cultivar: [Chardonnay]}"
python3 -m trainers.train_model --config mthl/transfer/ch_deep_withold_mtl_station --seed 4 --region ONNP --withold "{Site: [TwentyMileBench], Cultivar: [Chardonnay]}"

python3 -m trainers.train_model --config mthl/transfer/ch_deep_withold_mtl_station --seed 0 --region ONNP --withold "{Site: [TwentyMileBench], Cultivar: [Sauvignon_Blanc]}"
python3 -m trainers.train_model --config mthl/transfer/ch_deep_withold_mtl_station --seed 1 --region ONNP --withold "{Site: [TwentyMileBench], Cultivar: [Sauvignon_Blanc]}"
python3 -m trainers.train_model --config mthl/transfer/ch_deep_withold_mtl_station --seed 2 --region ONNP --withold "{Site: [TwentyMileBench], Cultivar: [Sauvignon_Blanc]}"
python3 -m trainers.train_model --config mthl/transfer/ch_deep_withold_mtl_station --seed 3 --region ONNP --withold "{Site: [TwentyMileBench], Cultivar: [Sauvignon_Blanc]}"
python3 -m trainers.train_model --config mthl/transfer/ch_deep_withold_mtl_station --seed 4 --region ONNP --withold "{Site: [TwentyMileBench], Cultivar: [Sauvignon_Blanc]}"
