#!/bin/bash
#SBATCH -J stdavidsbench_chardonnay
#SBATCH -o output/stdavidsbench_chardonnay.out
#SBATCH -e output/stdavidsbench_chardonnay.err
#SBATCH -p eecs,share,gpu
#SBATCH -t 2-12:00:00
#SBATCH --gres=gpu:1


python3 -m trainers.train_model --config hierarchical_mtl/transfer/ch_deep_withold_mtl_station --seed 0 --region ONNP --withold "{Site: [StDavidsBench], Cultivar: [Chardonnay]}"
python3 -m trainers.train_model --config hierarchical_mtl/transfer/ch_deep_withold_mtl_station --seed 1 --region ONNP --withold "{Site: [StDavidsBench], Cultivar: [Chardonnay]}"
python3 -m trainers.train_model --config hierarchical_mtl/transfer/ch_deep_withold_mtl_station --seed 2 --region ONNP --withold "{Site: [StDavidsBench], Cultivar: [Chardonnay]}"
python3 -m trainers.train_model --config hierarchical_mtl/transfer/ch_deep_withold_mtl_station --seed 3 --region ONNP --withold "{Site: [StDavidsBench], Cultivar: [Chardonnay]}"
python3 -m trainers.train_model --config hierarchical_mtl/transfer/ch_deep_withold_mtl_station --seed 4 --region ONNP --withold "{Site: [StDavidsBench], Cultivar: [Chardonnay]}"
