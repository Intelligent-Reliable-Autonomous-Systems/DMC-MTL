#!/bin/bash
#SBATCH -J pinn_results
#SBATCH -o output/ch_deep.out
#SBATCH -e output/ch_deep.err
#SBATCH -p eecs,share,gpu
#SBATCH -t 2-12:00:00
#SBATCH --gres=gpu:1

python3 -m results.compile_runs --start_dir _runs/PaperExperiments/ColdHardinessLimited/PINN_2/WA/Roza2/Prosser/All
python3 -m results.compile_runs --start_dir _runs/PaperExperiments/ColdHardinessLimited/PINN_3/WA/Roza2/Prosser/All
python3 -m results.compile_runs --start_dir _runs/PaperExperiments/ColdHardinessLimited/PINN_5/WA/Roza2/Prosser/All
python3 -m results.compile_runs --start_dir _runs/PaperExperiments/ColdHardinessLimited/PINN_10/WA/Roza2/Prosser/All
python3 -m results.compile_runs --start_dir _runs/PaperExperiments/ColdHardinessLimited/PINN_15/WA/Roza2/Prosser/All

python3 -m results.compile_runs --start_dir _runs/PaperExperiments/PhenologyLimited/PINN_1/WA/Roza2/Prosser/All

python3 -m results.compile_runs --start_dir _runs/PaperExperiments/ColdHardinessLimited/PINN_1/WA/Roza2/Prosser/All

python3 -m results.compile_runs --start_dir _runs/PaperExperiments/PhenologyLimited/PINN_2/WA/Roza2/Prosser/All
python3 -m results.compile_runs --start_dir _runs/PaperExperiments/PhenologyLimited/PINN_3/WA/Roza2/Prosser/All
python3 -m results.compile_runs --start_dir _runs/PaperExperiments/PhenologyLimited/PINN_5/WA/Roza2/Prosser/All
python3 -m results.compile_runs --start_dir _runs/PaperExperiments/PhenologyLimited/PINN_10/WA/Roza2/Prosser/All
python3 -m results.compile_runs --start_dir _runs/PaperExperiments/PhenologyLimited/PINN_15/WA/Roza2/Prosser/All
