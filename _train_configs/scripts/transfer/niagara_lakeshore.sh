#!/bin/bash
#SBATCH -J niagara_lakeshore
#SBATCH -o output/niagara_lakeshore.out
#SBATCH -e output/niagara_lakeshore.err
#SBATCH -p eecs,share,gpu
#SBATCH -t 2-00:00:00
#SBATCH --gres=gpu:1

python3 -m trainers.train_model --config mthl/transfer/ch_deep_withold_mtl_station --seed 0 --region ONNP --withold "{Site: [NiagaraLakeshores], Cultivar: [Cabernet_Franc]}"
python3 -m trainers.train_model --config mthl/transfer/ch_deep_withold_mtl_station --seed 1 --region ONNP --withold "{Site: [NiagaraLakeshores], Cultivar: [Cabernet_Franc]}"
python3 -m trainers.train_model --config mthl/transfer/ch_deep_withold_mtl_station --seed 2 --region ONNP --withold "{Site: [NiagaraLakeshores], Cultivar: [Cabernet_Franc]}"
python3 -m trainers.train_model --config mthl/transfer/ch_deep_withold_mtl_station --seed 3 --region ONNP --withold "{Site: [NiagaraLakeshores], Cultivar: [Cabernet_Franc]}"
python3 -m trainers.train_model --config mthl/transfer/ch_deep_withold_mtl_station --seed 4 --region ONNP --withold "{Site: [NiagaraLakeshores], Cultivar: [Cabernet_Franc]}"

python3 -m trainers.train_model --config mthl/transfer/ch_deep_withold_mtl_station --seed 0 --region ONNP --withold "{Site: [NiagaraLakeshores], Cultivar: [Chardonnay]}"
python3 -m trainers.train_model --config mthl/transfer/ch_deep_withold_mtl_station --seed 1 --region ONNP --withold "{Site: [NiagaraLakeshores], Cultivar: [Chardonnay]}"
python3 -m trainers.train_model --config mthl/transfer/ch_deep_withold_mtl_station --seed 2 --region ONNP --withold "{Site: [NiagaraLakeshores], Cultivar: [Chardonnay]}"
python3 -m trainers.train_model --config mthl/transfer/ch_deep_withold_mtl_station --seed 3 --region ONNP --withold "{Site: [NiagaraLakeshores], Cultivar: [Chardonnay]}"
python3 -m trainers.train_model --config mthl/transfer/ch_deep_withold_mtl_station --seed 4 --region ONNP --withold "{Site: [NiagaraLakeshores], Cultivar: [Chardonnay]}"

python3 -m trainers.train_model --config mthl/transfer/ch_deep_withold_mtl_station --seed 0 --region ONNP --withold "{Site: [NiagaraLakeshores], Cultivar: [Merlot]}"
python3 -m trainers.train_model --config mthl/transfer/ch_deep_withold_mtl_station --seed 1 --region ONNP --withold "{Site: [NiagaraLakeshores], Cultivar: [Merlot]}"
python3 -m trainers.train_model --config mthl/transfer/ch_deep_withold_mtl_station --seed 2 --region ONNP --withold "{Site: [NiagaraLakeshores], Cultivar: [Merlot]}"
python3 -m trainers.train_model --config mthl/transfer/ch_deep_withold_mtl_station --seed 3 --region ONNP --withold "{Site: [NiagaraLakeshores], Cultivar: [Merlot]}"
python3 -m trainers.train_model --config mthl/transfer/ch_deep_withold_mtl_station --seed 4 --region ONNP --withold "{Site: [NiagaraLakeshores], Cultivar: [Merlot]}"

python3 -m trainers.train_model --config mthl/transfer/ch_deep_withold_mtl_station --seed 0 --region ONNP --withold "{Site: [NiagaraLakeshores], Cultivar: [Pinot_Noir]}"
python3 -m trainers.train_model --config mthl/transfer/ch_deep_withold_mtl_station --seed 1 --region ONNP --withold "{Site: [NiagaraLakeshores], Cultivar: [Pinot_Noir]}"
python3 -m trainers.train_model --config mthl/transfer/ch_deep_withold_mtl_station --seed 2 --region ONNP --withold "{Site: [NiagaraLakeshores], Cultivar: [Pinot_Noir]}"
python3 -m trainers.train_model --config mthl/transfer/ch_deep_withold_mtl_station --seed 3 --region ONNP --withold "{Site: [NiagaraLakeshores], Cultivar: [Pinot_Noir]}"
python3 -m trainers.train_model --config mthl/transfer/ch_deep_withold_mtl_station --seed 4 --region ONNP --withold "{Site: [NiagaraLakeshores], Cultivar: [Pinot_Noir]}"