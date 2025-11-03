#!/bin/bash
# gen_cultivar_data.sh
# Generates data for all listed cultivars
# Written by Will Solow, 2025

# Cultivars to run
cultivars=("Aligote" "Alvarinho" "Auxerrois" "Barbera" "Cabernet_Franc" 
                   "Cabernet_Sauvignon" "Chardonnay" "Chenin_Blanc" "Concord" 
                  "Dolcetto" "Durif" "Gewurztraminer" "Green_Veltliner" "Grenache" 
                   "Lemberger" "Malbec" "Melon" "Merlot" "Mourvedre" "Muscat_Blanc" "Nebbiolo" 
                   "Petit_Verdot" "Pinot_Blanc" "Pinot_Gris" "Pinot_Noir" "Riesling" 
                   "Sangiovese" "Sauvignon_Blanc" "Semillon" "Syrah" "Tempranillo" 
                   "Viognier" "Zinfandel")

for cultivar in "${cultivars[@]}"; do
    #python3 -m _data.data_real --cultivar "$cultivar" --name grape_phenology
    python3 -m _data.compile_data_counts --cultivar "$cultivar"
done

echo "Submitted jobs with varying inputs: ${cultivars[*]}"
