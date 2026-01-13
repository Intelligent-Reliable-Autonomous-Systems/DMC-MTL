#!/bin/bash
#SBATCH -J shorthillsbench_sauvignon_blanc
#SBATCH -o output/shorthillsbench_sauvignon_blanc.out
#SBATCH -e output/shorthillsbench_sauvignon_blanc.err
#SBATCH -p eecs,share,gpu
#SBATCH -t 2-12:00:00
#SBATCH --gres=gpu:1

python3 -m trainers.train_model --config hierarchical_mtl/transfer/ch_deep_withold_mtl_station --seed 0 --region ONNP --withold "{Site: [ShortHillsBench], Cultivar: [Sauvignon_Blanc]}"
python3 -m trainers.train_model --config hierarchical_mtl/transfer/ch_deep_withold_mtl_station --seed 1 --region ONNP --withold "{Site: [ShortHillsBench], Cultivar: [Sauvignon_Blanc]}"
python3 -m trainers.train_model --config hierarchical_mtl/transfer/ch_deep_withold_mtl_station --seed 2 --region ONNP --withold "{Site: [ShortHillsBench], Cultivar: [Sauvignon_Blanc]}"
python3 -m trainers.train_model --config hierarchical_mtl/transfer/ch_deep_withold_mtl_station --seed 3 --region ONNP --withold "{Site: [ShortHillsBench], Cultivar: [Sauvignon_Blanc]}"
python3 -m trainers.train_model --config hierarchical_mtl/transfer/ch_deep_withold_mtl_station --seed 4 --region ONNP --withold "{Site: [ShortHillsBench], Cultivar: [Sauvignon_Blanc]}"


