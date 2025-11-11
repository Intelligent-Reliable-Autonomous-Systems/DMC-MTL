#!/bin/bash

python3 -m plotters.compile_runs --start_dir _runs/PaperExperiments/PhenologyLimited/ParamMTL1_1
python3 -m plotters.compile_runs --start_dir _runs/PaperExperiments/PhenologyLimited/ParamMTL1_2
python3 -m plotters.compile_runs --start_dir _runs/PaperExperiments/PhenologyLimited/ParamMTL2_1

python3 -m plotters.compile_runs --start_dir _runs/PaperExperiments/PhenologyLimited/ParamMTL2_2
python3 -m plotters.compile_runs --start_dir _runs/PaperExperiments/PhenologyLimited/ParamMTL1
python3 -m plotters.compile_runs --start_dir _runs/PaperExperiments/PhenologyLimited/ParamMTL2
python3 -m plotters.compile_runs --start_dir _runs/PaperExperiments/PhenologyLimited/ParamMTL3
python3 -m plotters.compile_runs --start_dir _runs/PaperExperiments/PhenologyLimited/ParamMTL5
python3 -m plotters.compile_runs --start_dir _runs/PaperExperiments/PhenologyLimited/ParamMTL10
python3 -m plotters.compile_runs --start_dir _runs/PaperExperiments/PhenologyLimited/ParamMTL15

python3 -m plotters.compile_runs --start_dir _runs/PaperExperiments/PhenologySmoothing/ParamMTL0.2
python3 -m plotters.compile_runs --start_dir _runs/PaperExperiments/PhenologySmoothing/ParamMTL0.1
python3 -m plotters.compile_runs --start_dir _runs/PaperExperiments/PhenologySmoothing/ParamMTL0.01
python3 -m plotters.compile_runs --start_dir _runs/PaperExperiments/PhenologySmoothing/ParamMTL0.001


python3 -m plotters.load_results --config _runs/PaperExperiments/PhenologyLimited/ParamMTL1_1
python3 -m plotters.load_results --config _runs/PaperExperiments/PhenologyLimited/ParamMTL1_2
python3 -m plotters.load_results --config _runs/PaperExperiments/PhenologyLimited/ParamMTL2_1







