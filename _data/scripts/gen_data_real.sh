#!/bin/bash
# gen_cultivar_data.sh
# Generates data for all listed cultivars
# Written by Will Solow, 2025

# Cultivars to run
cultivars=("Aligote" "Alvarinho" "Auxerrois" "Barbera" "Cabernet Franc" 
                   "Cabernet Sauvignon" "Chardonnay" "Chenin Blanc" "Concord" 
                  "Dolcetto" "Durif" "Gewurztraminer" "Green Veltliner" "Grenache" 
                   "Lemberger" "Malbec" "Melon" "Merlot" "Mourvedre" "Muscat Blanc" "Nebbiolo" 
                   "Petit Verdot" "Pinot Blanc" "Pinot Gris" "Pinot Noir" "Riesling" 
                   "Sangiovese" "Sauvignon Blanc" "Semillon" "Syrah" 
                   "Viognier" "Zinfandel")

for cultivar in "${cultivars[@]}"; do
    python3 -m _data.data_real --cultivar "$cultivar" --name "$1" --region "$2" --station "$3" --site "$4"
    #python3 -m _data.data_real --cultivar "$cultivar"
done

echo "Submitted jobs with varying inputs: ${cultivars[*]}"
