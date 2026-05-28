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
    python3 -m _data.data_real --cultivar "$cultivar" --name grape_coldhardiness --region WI --station Madison --site Madison
    python3 -m _data.data_real --cultivar "$cultivar" --name grape_coldhardiness --region QC --station Montreal --site Montreal
    python3 -m _data.data_real --cultivar "$cultivar" --name grape_coldhardiness --region TX --station Brownfield --site Brownfield
    python3 -m _data.data_real --cultivar "$cultivar" --name grape_coldhardiness --region PA --station PennState --site PennState
done

echo "Submitted jobs with varying inputs: ${cultivars[*]}"
