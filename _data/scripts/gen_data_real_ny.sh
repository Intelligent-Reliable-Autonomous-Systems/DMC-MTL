#!/bin/bash
# gen_cultivar_data.sh
# Generates data for all listed cultivars
# Written by Will Solow, 2025

# Cultivars to run
cultivars=("Cabernet Franc" 
                   "Cabernet Sauvignon" "Chardonnay" "Chenin Blanc" "Concord" 
                   "Gewurztraminer" 
                   "Lemberger" "Malbec" "Merlot" 
                  "Pinot Blanc" "Pinot Gris" "Pinot Noir" "Riesling" 
                   "Sangiovese" "Sauvignon Blanc"  "Syrah" 
                   )

for cultivar in "${cultivars[@]}"; do
    python3 -m _data.data_real --cultivar "$cultivar" --name grape_coldhardiness --region NY --station Bejo --site Bejo
    python3 -m _data.data_real --cultivar "$cultivar" --name grape_coldhardiness --region NY --station Hudson --site Hudson
    python3 -m _data.data_real --cultivar "$cultivar" --name grape_coldhardiness --region NY --station Portland --site Portland
    python3 -m _data.data_real --cultivar "$cultivar" --name grape_coldhardiness --region NY --station Geneva --site Geneva
done

echo "Submitted jobs with varying inputs: ${cultivars[*]}"
