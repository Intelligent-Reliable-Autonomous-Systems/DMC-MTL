#!/bin/bash
#SBATCH -J ch_deep
#SBATCH -o output/ch_deep.out
#SBATCH -e output/ch_deep.err
#SBATCH -p eecs,share,gpu
#SBATCH -t 2-12:00:00
#SBATCH --gres=gpu:1

python3 -m plotters.compile_runs_error --start_dir _runs/RTMC/FineTuneColdHardiness/DeepMTL1_1/None/None/None/All
python3 -m plotters.compile_runs_error --start_dir _runs/RTMC/FineTuneColdHardiness/DeepMTL1_2/None/None/None/All
python3 -m plotters.compile_runs_error --start_dir _runs/RTMC/FineTuneColdHardiness/DeepMTL1_5/None/None/None/All
python3 -m plotters.compile_runs_error --start_dir _runs/RTMC/FineTuneColdHardiness/DeepMTL1_10/None/None/None/All

python3 -m plotters.compile_runs_error --start_dir _runs/RTMC/FineTuneColdHardiness/DeepMTL2_1/None/None/None/All
python3 -m plotters.compile_runs_error --start_dir _runs/RTMC/FineTuneColdHardiness/DeepMTL2_2/None/None/None/All
python3 -m plotters.compile_runs_error --start_dir _runs/RTMC/FineTuneColdHardiness/DeepMTL2_5/None/None/None/All
python3 -m plotters.compile_runs_error --start_dir _runs/RTMC/FineTuneColdHardiness/DeepMTL2_10/None/None/None/All

python3 -m plotters.compile_runs_error --start_dir _runs/RTMC/FineTuneColdHardiness/DeepMTL5_1/None/None/None/All
python3 -m plotters.compile_runs_error --start_dir _runs/RTMC/FineTuneColdHardiness/DeepMTL5_2/None/None/None/All
python3 -m plotters.compile_runs_error --start_dir _runs/RTMC/FineTuneColdHardiness/DeepMTL5_5/None/None/None/All
python3 -m plotters.compile_runs_error --start_dir _runs/RTMC/FineTuneColdHardiness/DeepMTL5_10/None/None/None/All

python3 -m plotters.compile_runs_error --start_dir _runs/RTMC/FineTuneColdHardiness/DeepMTL10_1/None/None/None/All
python3 -m plotters.compile_runs_error --start_dir _runs/RTMC/FineTuneColdHardiness/DeepMTL10_2/None/None/None/All
python3 -m plotters.compile_runs_error --start_dir _runs/RTMC/FineTuneColdHardiness/DeepMTL10_5/None/None/None/All
python3 -m plotters.compile_runs_error --start_dir _runs/RTMC/FineTuneColdHardiness/DeepMTL10_10/None/None/None/All


python3 -m plotters.compile_runs_error --start_dir _runs/RTMC/FineTuneColdHardiness/ParamMTL1_1/None/None/None/All
python3 -m plotters.compile_runs_error --start_dir _runs/RTMC/FineTuneColdHardiness/ParamMTL1_2/None/None/None/All
python3 -m plotters.compile_runs_error --start_dir _runs/RTMC/FineTuneColdHardiness/ParamMTL1_5/None/None/None/All
python3 -m plotters.compile_runs_error --start_dir _runs/RTMC/FineTuneColdHardiness/ParamMTL1_10/None/None/None/All

python3 -m plotters.compile_runs_error --start_dir _runs/RTMC/FineTuneColdHardiness/ParamMTL2_1/None/None/None/All
python3 -m plotters.compile_runs_error --start_dir _runs/RTMC/FineTuneColdHardiness/ParamMTL2_2/None/None/None/All
python3 -m plotters.compile_runs_error --start_dir _runs/RTMC/FineTuneColdHardiness/ParamMTL2_5/None/None/None/All
python3 -m plotters.compile_runs_error --start_dir _runs/RTMC/FineTuneColdHardiness/ParamMTL2_10/None/None/None/All

python3 -m plotters.compile_runs_error --start_dir _runs/RTMC/FineTuneColdHardiness/ParamMTL5_1/None/None/None/All
python3 -m plotters.compile_runs_error --start_dir _runs/RTMC/FineTuneColdHardiness/ParamMTL5_2/None/None/None/All
python3 -m plotters.compile_runs_error --start_dir _runs/RTMC/FineTuneColdHardiness/ParamMTL5_5/None/None/None/All
python3 -m plotters.compile_runs_error --start_dir _runs/RTMC/FineTuneColdHardiness/ParamMTL5_10/None/None/None/All

python3 -m plotters.compile_runs_error --start_dir _runs/RTMC/FineTuneColdHardiness/ParamMTL10_1/None/None/None/All
python3 -m plotters.compile_runs_error --start_dir _runs/RTMC/FineTuneColdHardiness/ParamMTL10_2/None/None/None/All
python3 -m plotters.compile_runs_error --start_dir _runs/RTMC/FineTuneColdHardiness/ParamMTL10_5/None/None/None/All
python3 -m plotters.compile_runs_error --start_dir _runs/RTMC/FineTuneColdHardiness/ParamMTL10_10/None/None/None/All


python3 -m plotters.compile_runs_error --start_dir _runs/RTMC/FineTunePhenology/DeepMTL1_1/None/None/None/All
python3 -m plotters.compile_runs_error --start_dir _runs/RTMC/FineTunePhenology/DeepMTL1_2/None/None/None/All
python3 -m plotters.compile_runs_error --start_dir _runs/RTMC/FineTunePhenology/DeepMTL1_5/None/None/None/All
python3 -m plotters.compile_runs_error --start_dir _runs/RTMC/FineTunePhenology/DeepMTL1_10/None/None/None/All

python3 -m plotters.compile_runs_error --start_dir _runs/RTMC/FineTunePhenology/DeepMTL2_1/None/None/None/All
python3 -m plotters.compile_runs_error --start_dir _runs/RTMC/FineTunePhenology/DeepMTL2_2/None/None/None/All
python3 -m plotters.compile_runs_error --start_dir _runs/RTMC/FineTunePhenology/DeepMTL2_5/None/None/None/All
python3 -m plotters.compile_runs_error --start_dir _runs/RTMC/FineTunePhenology/DeepMTL2_10/None/None/None/All

python3 -m plotters.compile_runs_error --start_dir _runs/RTMC/FineTunePhenology/DeepMTL5_1/None/None/None/All
python3 -m plotters.compile_runs_error --start_dir _runs/RTMC/FineTunePhenology/DeepMTL5_2/None/None/None/All
python3 -m plotters.compile_runs_error --start_dir _runs/RTMC/FineTunePhenology/DeepMTL5_5/None/None/None/All
python3 -m plotters.compile_runs_error --start_dir _runs/RTMC/FineTunePhenology/DeepMTL5_10/None/None/None/All

python3 -m plotters.compile_runs_error --start_dir _runs/RTMC/FineTunePhenology/DeepMTL10_1/None/None/None/All
python3 -m plotters.compile_runs_error --start_dir _runs/RTMC/FineTunePhenology/DeepMTL10_2/None/None/None/All
python3 -m plotters.compile_runs_error --start_dir _runs/RTMC/FineTunePhenology/DeepMTL10_5/None/None/None/All
python3 -m plotters.compile_runs_error --start_dir _runs/RTMC/FineTunePhenology/DeepMTL10_10/None/None/None/All


python3 -m plotters.compile_runs_error --start_dir _runs/RTMC/FineTunePhenology/ParamMTL1_1/None/None/None/All
python3 -m plotters.compile_runs_error --start_dir _runs/RTMC/FineTunePhenology/ParamMTL1_2/None/None/None/All
python3 -m plotters.compile_runs_error --start_dir _runs/RTMC/FineTunePhenology/ParamMTL1_5/None/None/None/All
python3 -m plotters.compile_runs_error --start_dir _runs/RTMC/FineTunePhenology/ParamMTL1_10/None/None/None/All

python3 -m plotters.compile_runs_error --start_dir _runs/RTMC/FineTunePhenology/ParamMTL2_1/None/None/None/All
python3 -m plotters.compile_runs_error --start_dir _runs/RTMC/FineTunePhenology/ParamMTL2_2/None/None/None/All
python3 -m plotters.compile_runs_error --start_dir _runs/RTMC/FineTunePhenology/ParamMTL2_5/None/None/None/All
python3 -m plotters.compile_runs_error --start_dir _runs/RTMC/FineTunePhenology/ParamMTL2_10/None/None/None/All

python3 -m plotters.compile_runs_error --start_dir _runs/RTMC/FineTunePhenology/ParamMTL5_1/None/None/None/All
python3 -m plotters.compile_runs_error --start_dir _runs/RTMC/FineTunePhenology/ParamMTL5_2/None/None/None/All
python3 -m plotters.compile_runs_error --start_dir _runs/RTMC/FineTunePhenology/ParamMTL5_5/None/None/None/All
python3 -m plotters.compile_runs_error --start_dir _runs/RTMC/FineTunePhenology/ParamMTL5_10/None/None/None/All

python3 -m plotters.compile_runs_error --start_dir _runs/RTMC/FineTunePhenology/ParamMTL10_1/None/None/None/All
python3 -m plotters.compile_runs_error --start_dir _runs/RTMC/FineTunePhenology/ParamMTL10_2/None/None/None/All
python3 -m plotters.compile_runs_error --start_dir _runs/RTMC/FineTunePhenology/ParamMTL10_5/None/None/None/All
python3 -m plotters.compile_runs_error --start_dir _runs/RTMC/FineTunePhenology/ParamMTL10_10/None/None/None/All



