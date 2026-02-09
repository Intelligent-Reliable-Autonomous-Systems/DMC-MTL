import numpy as np

# Available cultivars for simulation
CROP_NAMES = {
    "grape_phenology_": np.array(
        [
            "Aligote",
            "Alvarinho",
            "Auxerrois",
            "Barbera",
            "Cabernet_Franc",
            "Cabernet_Sauvignon",
            "Chardonnay",
            "Chenin_Blanc",
            "Concord",
            "Durif",
            "Gewurztraminer",
            "Green_Veltliner",
            "Grenache",
            "Lemberger",
            "Malbec",
            "Melon",
            "Merlot",
            "Mourvedre",
            "Muscat_Blanc",
            "Nebbiolo",
            "Petit_Verdot",
            "Pinot_Blanc",
            "Pinot_Gris",
            "Pinot_Noir",
            "Riesling",
            "Sangiovese",
            "Sauvignon_Blanc",
            "Semillon",
            "Tempranillo",
            "Viognier",
            "Zinfandel",
        ],
        dtype=str,
    ),
    "grape_coldhardiness_ferg": np.array(
        [
            "Barbera",
            "Cabernet_Franc",
            "Cabernet_Sauvignon",
            "Chardonnay",
            "Chenin_Blanc",
            "Concord",
            "Gewurztraminer",
            "Grenache",
            "Lemberger",
            "Malbec",
            "Merlot",
            "Mourvedre",
            "Nebbiolo",
            "Pinot_Gris",
            "Riesling",
            "Sangiovese",
            "Sauvignon_Blanc",
            "Semillon",
            "Syrah",
            "Viognier",
            "Zinfandel",
        ],
        dtype=str,
    ),
    "grape_coldhardiness_reg": np.array(
        [
            "Cabernet_Franc",  # 0
            "Cabernet_Sauvignon",  # 1
            "Chardonnay",  # 2
            "Gewurztraminer",  # 3
            "Merlot",  # 4
            "Pinot_Blanc",  # 5
            "Pinot_Gris",  # 6
            "Pinot_Noir",  # 7
            "Riesling",  # 8
            "Sauvignon_Blanc",  # 9
            "Syrah",  # 10
            "Zinfandel",  # 11
        ],
        dtype=str,
    ),
    "grape_coldhardiness_all": np.array(
        [
            "Aligote",
            "Alvarinho",
            "Auxerrois",
            "Barbera",
            "Cabernet_Franc",
            "Cabernet_Sauvignon",
            "Chardonnay",
            "Chenin_Blanc",
            "Concord",
            "Gewurztraminer",
            "Grenache",
            "Lemberger",
            "Malbec",
            "Melon",
            "Merlot",
            "Mourvedre",
            "Muscat_Blanc",
            "Nebbiolo",
            "Pinot_Blanc",
            "Pinot_Gris",
            "Pinot_Noir",
            "Riesling",
            "Sangiovese",
            "Sauvignon_Blanc",
            "Semillon",
            "Syrah",
            "Viognier",
            "Zinfandel",
        ],
        dtype=str,
    ),
    "wofost_": np.array(
        [
            "Winter_Wheat_101",
            "Winter_Wheat_102",
            "Winter_Wheat_103",
            "Winter_Wheat_104",
            "Winter_Wheat_105",
            "Winter_Wheat_106",
            "Winter_Wheat_107",
            "Bermude",
            "Apache",
        ],
        dtype=str,
    ),
    "cherry_coldhardiness_ferg": np.array(
        [
            "Attika",
            "Benton",
            "Bing",
            "Black_Gold",
            "Black_Pearl",
            "Black_Tartarian",
            "Chelan",
            "Early_Robin",
            "Early_Star",
            # "Emperor_Francis",
            "Hedelfingen",
            # "Index",
            "Kristen",
            "Lambert",
            "Lapins",
            "Napoleon",
            "Rainier",
            "Regina",
            "Sandra_Rose",
            "Santina",
            "Schmidt",
            "Schneider",
            "Selah",
            "Skeena",
            "Stella",
            "Sunburst",
            "Sweetheart",
            "Tieton",
            "Ulster",
            "Van",
            "Vega",
        ],
        dtype=str,
    ),
}

REGIONS = np.array(["BCOV", "ONNP", "WA"])  # "NY", "NS", "MI"])
STATIONS = np.array(
    ["Kelowna", "Kelowna_UBC", "Osoyoos_CS", "Summerland", "Grimbsy_Mtn", "Vineland_Station", "Roza2"]
)  # "SMREC", "Beaver", "Bejo", "Geneva", "Hudson", "Portland"])
SITES = np.array(
    [
        "WestKelowna",  # 0
        "OKFallsEast",  # 1
        "OKFallsWest",  # 2
        "OliverEast",  # 3
        "OliverSouth",  # 4
        "OliverWest",  # 5
        "BlackSage",  # 6
        "OsoyoosNorth",  # 7
        "OsoyoosNortheast",  # 8
        "OsoyoosSoutheast",  # 9
        "OsoyoosWest",  # 10
        "NaramataBench",  # 11
        "BeamsvilleBench",  # 12
        "LincolnLakeshore",  # 13
        "TwentyMileBench",  # 14
        "VinemountRidge",  # 15
        "CreekShores",  # 16
        "FourMileCreek",  # 17
        "NiagaraLakeshores",  # 18
        "NiagaraRiver",  # 19
        "ShortHillsBench",  # 20
        "StDavidsBench",  # 21
        "Prosser",  # 22
        # "Mich1",
        # "Beaver",
        # "Bejo",
        # "Geneva",
        # "Hudson",
        # "Portland"
    ]
)

REGION_NAMES = {
    "BCOV": {
        "Kelowna": ["WestKelowna"],
        "Kelowna_UBC": ["OKFallsEast", "OKFallsWest", "OliverEast", "OliverSouth", "OliverWest"],
        "Osoyoos_CS": ["BlackSage", "OsoyoosNorth", "OsoyoosNortheast", "OsoyoosSoutheast", "OsoyoosWest"],
        "Summerland": ["NaramataBench"],
    },
    "ONNP": {
        "Grimbsy_Mtn": ["BeamsvilleBench", "LincolnLakeshore", "TwentyMileBench", "VinemountRidge"],
        "Vineland_Station": [
            "CreekShores",
            "FourMileCreek",
            "NiagaraLakeshores",
            "NiagaraRiver",
            "ShortHillsBench",
            "StDavidsBench",
        ],
    },
    "WA": {"Roza2": ["Prosser"]},
}

CULT_STR = np.array(
    [
        "query: Cabernet Franc",  # 0
        "query: Cabernet Sauvignon",  # 1
        "query: Chardonnay",  # 2
        "query: Gewurztraminer",  # 3
        "query: Merlot",  # 4
        "query: Pinot Blanc",  # 5
        "query: Pinot Gris",  # 6
        "query: Pinot Noir",  # 7
        "query: Riesling",  # 8
        "query: Sauvignon Blanc",  # 9
        "query: Syrah",  # 10
        "query: Zinfandel",  # 11
    ]
)

SITE_STR = np.array(
    [
        "WestKelowna",  # 0
        "OKFallsEast",  # 1
        "OKFallsWest",  # 2
        "OliverEast",  # 3
        "OliverSouth",  # 4
        "OliverWest",  # 5
        "BlackSage",  # 6
        "OsoyoosNorth",  # 7
        "OsoyoosNortheast",  # 8
        "OsoyoosSoutheast",  # 9
        "OsoyoosWest",  # 10
        "NaramataBench",  # 11
        "BeamsvilleBench",  # 12
        "LincolnLakeshore",  # 13
        "TwentyMileBench",  # 14
        "VinemountRidge",  # 15
        "CreekShores",  # 16
        "FourMileCreek",  # 17
        "NiagaraLakeshores",  # 18
        "NiagaraRiver",  # 19
        "ShortHillsBench",  # 20
        "StDavidsBench",  # 21
        "Prosser",  # 22
        # "Mich1",
        # "Beaver",
        # "Bejo",
        # "Geneva",
        # "Hudson",
        # "Portland"
    ]
)

REGION_STR = np.array(
    ["query: British Colubmia Okanagan Valley", "query: Ontario Niagara Peninsula", "query: Washington, United States"]
)  # "NY", "NS", "MI"])
STATION_STR = np.array(
    [
        "query: Kelowna",
        "query: Kelowna UBC",
        "query: Osoyoos CS",
        "query: Summerland",
        "query: Grimbsy Mountain",
        "query: Vineland Station",
        "query: Roza2",
    ]
)  # "SMREC", "Beaver", "Bejo", "Geneva", "Hudson", "Portland"])
