#!/bin/bash
# gen_cultivar_data.sh
# Generates data for all listed cultivars
# Written by Will Solow, 2025

# Cultivars to run
cultivars=("Attika" "Benton" "Bing" "Black Gold" "Black Pearl" "Black Tartarian" "Chelan" "Early Robin" "Early Star"
           "Emperor Francis" "Hedelfingen" "Index" "Kristen" "Lambert" "Lapins" "Napoleon" "Rainier" "Regina" "Sandra Rose" 
           "Santina" "Schmidt" "Schneider" "Selah" "Skeena" "Stella" "Sunburst" "Sweetheart" "Tieton" "Ulster" "Van" "Vega")

for cultivar in "${cultivars[@]}"; do
    #python3 -m _data.data_real --cultivar "$cultivar" --name "$1" --region "$2" --station "$3" --site "$4"
    python3 -m _data.data_real --cultivar "$cultivar" --name cherry_coldhardiness --region WA --fruit Cherry
done

echo "Submitted jobs with varying inputs: ${cultivars[*]}"
