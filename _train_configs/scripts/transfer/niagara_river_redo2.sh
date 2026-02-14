#!/bin/bash
#SBATCH -J text_niagara_river2
#SBATCH -o output/text_niagara_river2.out
#SBATCH -e output/text_niagara_river2.err
#SBATCH -p eecs,gpu
#SBATCH -t 2-00:00:00
#SBATCH --gres=gpu:1


python3 -m trainers.train_model --config mthl/transfer/ch_deep_withold_mtl_station_text --seed 3 --region ONNP --withold "{Site: [NiagaraRiver], Cultivar: [Cabernet_Franc]}"
python3 -m trainers.train_model --config mthl/transfer/ch_deep_withold_mtl_station_text --seed 4 --region ONNP --withold "{Site: [NiagaraRiver], Cultivar: [Cabernet_Franc]}"
