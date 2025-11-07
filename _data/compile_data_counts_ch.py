"""
compile_data_counts.py

Compiles the data for phenology and cold hardiness and
nicely prints to string

Written by Will Solow, 2025
"""

import argparse
import _data.process_data_real as ld


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--cultivar", default="Aligote", type=str, help="Path to Config")
    parser.add_argument("--region", default="WA", type=str)
    parser.add_argument("--site", default="Prosser", type=str)
    parser.add_argument("--station", default="Roza2", type=str)
    args = parser.parse_args()

    region = ["WA", "BCOV", "ONNP"]
    station = {
        "WA": ["Roza2"],
        "BCOV": ["Osoyoos_CS", "Summerland", "Kelowna", "Kelowna_UBC"],
        "ONNP": ["Vineland_Station", "Grimbsy_Mtn"],
    }
    site = {
        "Roza2": ["Prosser"],
        "Osoyoos_CS": ["OsoyoosNorth", "OsoyoosNortheast", "OsoyoosSoutheast", "OsoyoosWest", "BlackSage"],
        "Kelowna": ["WestKelowna"],
        "Summerland": ["NaramataBench"],
        "Kelowna_UBC": ["OliverEast", "OliverSouth", "OliverWest", "OKFallsEast", "OKFallsWest"],
        "Vineland_Station": [
            "CreekShores",
            "FourMileCreek",
            "NiagaraLakeshores",
            "NiagaraRiver",
            "ShortHillsBench",
            "StDavidsBench",
        ],
        "Grimbsy_Mtn": ["BeamsvilleBench", "LincolnLakeshore", "TwentyMileBench", "VinemountRidge"],
    }

    # cultivars = ["Barbera","Cabernet_Franc", "Cabernet_Sauvignon","Chardonnay","Chenin_Blanc","Concord","Gewurztraminer","Grenache", "Lemberger","Malbec","Merlot","Mourvedre","Nebbiolo",  "Pinot_Gris","Riesling", "Sangiovese","Sauvignon_Blanc","Semillon","Syrah", "Viognier","Zinfandel",]
    cultivars = [
        "Aligote",
        "Alvarinho",
        "Auxerrois",
        "Barbera",
        "Cabernet Franc",
        "Cabernet Sauvignon",
        "Chardonnay",
        "Chenin Blanc",
        "Concord",
        "Durif",
        "Gewurztraminer",
        "Green Veltliner",
        "Grenache",
        "Lemberger",
        "Malbec",
        "Melon",
        "Merlot",
        "Mourvedre",
        "Muscat Blanc",
        "Nebbiolo",
        "Petit Verdot",
        "Pinot Blanc",
        "Pinot Gris",
        "Pinot Noir",
        "Riesling",
        "Sangiovese",
        "Sauvignon Blanc",
        "Semillon",
        "Syrah",
        "Tempranillo",
        "Viognier",
        "Zinfandel",
    ]
    lte_all = 0
    print_str = ""
    for r in region:
        if r == "WA":
            continue
        print_str += f"Region: {r}\n"
        lte_region = 0
        for s in station[r]:
            print_str += f"\tStation: {s}\n"
            lte_station = 0
            for si in site[s]:
                print_str += f"\t\tSite: {si}\n"
                lte_site = 0
                for c in cultivars:
                    try:
                        data, _ = ld.load_and_process_ca_data_coldhardiness(r, si, c, s)
                    except FileNotFoundError as e:
                        continue
                    lte_cultivar = 0
                    for d in data:
                        lte_cultivar += d.loc[:, "LTE50"].notna().sum()
                    print_str += f"\t\t\tLTE {c}: {len(data)}, {lte_cultivar}\n"
                    lte_site += lte_cultivar
                print_str += f"\t\tLTE {si}: {lte_site}\n"
                lte_station += lte_site
            print_str += f"\tLTE {s}: {lte_station}\n"
            lte_region += lte_station
        print_str += f"LTE {r}: {lte_region}\n"
        lte_all += lte_region
    print_str += f"LTE All: {lte_all}\n"

    print(print_str)


if __name__ == "__main__":
    main()
