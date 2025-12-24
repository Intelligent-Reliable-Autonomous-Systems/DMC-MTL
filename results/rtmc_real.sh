#!/bin/bash
#SBATCH -J rtmc_results
#SBATCH -o output/rtmc_results.out
#SBATCH -e output/rtmc_results.err
#SBATCH -p eecs,share,gpu
#SBATCH -t 2-12:00:00
#SBATCH --gres=gpu:1

python3 -m results.compile_runs --start_dir _runs/PaperExperiments/RTMC/ColdHardiness/ParamMTL1/WA/Roza2/Prosser/All
python3 -m results.compile_runs --start_dir _runs/PaperExperiments/RTMC/ColdHardiness/ParamMTL2/WA/Roza2/Prosser/All
python3 -m results.compile_runs --start_dir _runs/PaperExperiments/RTMC/ColdHardiness/ParamMTL5/WA/Roza2/Prosser/All
python3 -m results.compile_runs --start_dir _runs/PaperExperiments/RTMC/ColdHardiness/ParamMTL10/WA/Roza2/Prosser/All

python3 -m results.compile_runs --start_dir _runs/PaperExperiments/RTMC/Phenology/ParamMTL1/WA/Roza2/Prosser/All
python3 -m results.compile_runs --start_dir _runs/PaperExperiments/RTMC/Phenology/ParamMTL2/WA/Roza2/Prosser/All
python3 -m results.compile_runs --start_dir _runs/PaperExperiments/RTMC/Phenology/ParamMTL5/WA/Roza2/Prosser/All
python3 -m results.compile_runs --start_dir _runs/PaperExperiments/RTMC/Phenology/ParamMTL10/WA/Roza2/Prosser/All

python3 -m plotters.compile_runs_error --start_dir _runs/RTMC/FineTunePhenology/ParamMTL1_1/WA/Roza2/Prosser/All
python3 -m plotters.compile_runs_error --start_dir _runs/RTMC/FineTunePhenology/ParamMTL2_2/WA/Roza2/Prosser/All
python3 -m plotters.compile_runs_error --start_dir _runs/RTMC/FineTunePhenology/ParamMTL5_5/WA/Roza2/Prosser/All
python3 -m plotters.compile_runs_error --start_dir _runs/RTMC/FineTunePhenology/ParamMTL10_10/WA/Roza2/Prosser/All

python3 -m plotters.compile_runs_error --start_dir _runs/RTMC/FineTuneColdHardiness/ParamMTL1_1/WA/Roza2/Prosser/All
python3 -m plotters.compile_runs_error --start_dir _runs/RTMC/FineTuneColdHardiness/ParamMTL2_2/WA/Roza2/Prosser/All
python3 -m plotters.compile_runs_error --start_dir _runs/RTMC/FineTuneColdHardiness/ParamMTL5_5/WA/Roza2/Prosser/All
python3 -m plotters.compile_runs_error --start_dir _runs/RTMC/FineTuneColdHardiness/ParamMTL10_10/WA/Roza2/Prosser/All



