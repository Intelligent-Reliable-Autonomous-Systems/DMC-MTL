#!/bin/bash
# gen_wofost_data.sh
# Generates data for all listed wheat varieties
# Written by Will Solow, 2025

# Wheats to run
wheats=("Winter_Wheat_101" "Winter_Wheat_102" "Winter_Wheat_103" "Winter_Wheat_104" 
                 "Winter_Wheat_105" "Winter_Wheat_106" "Winter_Wheat_107" "Bermude"
                 "Apache")

for wheat in "${wheats[@]}"; do
    python3 -m _data.gen_synth_data --crop_variety "$wheat" 
    #python3 -m _data.compile_data_counts --cultivar "$wheat"
done

echo "Submitted jobs with varying inputs: ${cultivars[*]}"
