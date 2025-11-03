#!/bin/bash

python3 -m plotters.compile_runs --print --start_dir _runs/JournalExperiments/Pred/Phenology/ParamMTL/Multi --prefix param_mtl
python3 -m plotters.compile_runs --print --start_dir _runs/JournalExperiments/Pred/Phenology/ClassMTL/Multi --prefix class_mtl

python3 -m plotters.compile_runs_error --start_dir _runs/JournalExperiments/RTMC/Phenology/ErrorFCGRU/Multi 
python3 -m plotters.compile_runs_error --start_dir _runs/JournalExperiments/RTMC/Phenology/ErrorTransformer/Multi 
python3 -m plotters.compile_runs_error --start_dir _runs/JournalExperiments/RTMC/Phenology/VecFCGRU/Multi 
python3 -m plotters.compile_runs_error --start_dir _runs/JournalExperiments/RTMC/Phenology/VecTransformer/Multi 

python3 -m plotters.compile_runs_error_ch --start_dir _runs/JournalExperiments/RTMC/ColdHardiness/Synth/ErrorFCGRU/Multi 
python3 -m plotters.compile_runs_error --start_dir _runs/JournalExperiments/RTMC/Phenology/Synth/GatedDecayError/Multi 
python3 -m plotters.compile_runs_error --start_dir _runs/JournalExperiments/RTMC/Phenology/Synth/GatedError/Multi 
python3 -m plotters.compile_runs_error --start_dir _runs/JournalExperiments/RTMC/Phenology/Synth/IterativeScaledVecFCGRU/Multi 
python3 -m plotters.compile_runs_error --start_dir _runs/JournalExperiments/RTMC/Phenology/Synth/ScaledVecFCGRU/Multi 





