#!/bin/bash
# gen_wofost_data.sh
# Generates data for all listed wheat varieties
# Written by Will Solow, 2025

# Wheats to run
cultivars=("Aligote" "Alvarinho" "Auxerrois" "Barbera" "Cabernet_Franc" 
                   "Cabernet_Sauvignon" "Chardonnay" "Chenin_Blanc" "Concord" 
                  "Dolcetto" "Durif" "Gewurztraminer" "Green_Veltliner" "Grenache" 
                   "Lemberger" "Malbec" "Melon" "Merlot" "Mourvedre" "Muscat_Blanc" "Nebbiolo" 
                   "Petit_Verdot" "Pinot_Blanc" "Pinot_Gris" "Pinot_Noir" "Riesling" 
                   "Sangiovese" "Sauvignon_Blanc" "Semillon" "Syrah" "Tempranillo" 
                   "Viognier" "Zinfandel")

for cultivar in "${cultivars[@]}"; do
    echo "$cultivar"
    python3 -m _data.gen_synth_data --crop_variety "$cultivar" --config pheno_synth_config --model pheno
done

echo "Submitted jobs with varying inputs: ${cultivars[*]}"
