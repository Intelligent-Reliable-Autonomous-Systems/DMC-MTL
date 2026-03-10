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

REGIONS = np.array(["BCOV", "ONNP", "WA", "NY", "NS", "MI"])
STATIONS = np.array(
    [
        "Kelowna",
        "Kelowna_UBC",
        "Osoyoos_CS",
        "Summerland",
        "Grimbsy_Mtn",
        "Vineland_Station",
        "Roza2",
        "SWMREC",
        "Beaver",
        "Bejo",
        "Geneva",
        "Hudson",
        "Portland",
    ]
)
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
        "Mich1",
        "Beaver",
        "Bejo",
        "Geneva",
        "Hudson",
        "Portland",
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

CULT_STR_VERB = np.array(
    [
        "query: Cabernet Franc is an early- to mid-budding, mid-ripening red grape cultivar with medium vigor and loose to moderately compact clusters, producing wines with moderate acidity and a lighter, translucent red color. It has good cold-hardiness (tolerating around -15C/5F) and prefers well-drained soils such as gravelly loam or limestone.",
        "query: Cabernet Sauvignon is a late-budding, late-ripening red grape cultivar with small, compact clusters, producing deeply colored wines with firm acidity and strong tannins. It has moderate cold-hardiness (tolerating around -15C/5F) and prefers well-drained gravelly or sandy-loam soils.",
        "query: Chardonnay is an early- to mid-ripening white grape cultivar with moderate vigor and medium-sized, loosely packed clusters, producing wines with bright acidity and a pale to golden color. It has moderate cold-hardiness (tolerating around -15C/5F) and prefers well-drained soils such as limestone, chalk, or clay.",
        "query: Gewurztraminer is an early- to mid-ripening white grape cultivar with low to moderate vigor and tightly packed clusters, producing intensely aromatic wines with low to moderate acidity and a golden to deep yellow color. It has low to moderate cold-hardiness (tolerating around -12C/10F) and prefers well-drained, calcareous or loamy soils.",
        "query: Merlot is a mid-budding, mid- to late-ripening red grape cultivar with medium vigor and medium-sized, loosely packed clusters, producing wines with soft tannins, moderate acidity, and a deep ruby color. It has moderate cold-hardiness (tolerating around -15C/5F) and prefers well-drained clay or gravelly soils.",
        "query: Pinot Blanc is an early- to mid-ripening white grape cultivar with moderate vigor and medium-sized, loosely packed clusters, producing wines with moderate acidity and a pale yellow to straw color. It has moderate cold-hardiness (tolerating around -15C/5F) and prefers well-drained, calcareous or loamy soils.",
        "query: Pinot Gris is an early- to mid-ripening white grape cultivar with moderate vigor and medium-sized, loosely packed clusters, producing wines with medium acidity and a color ranging from pale yellow to coppery gray. It has moderate cold-hardiness (tolerating around -15C/5F) and prefers well-drained, fertile soils such as loam or limestone.",
        "query: Pinot Noir is an early-budding, mid- to late-ripening red grape cultivar with low to moderate vigor and small, loosely packed clusters, producing wines with bright acidity and a light to medium ruby color. It has moderate cold-hardiness (tolerating around -15C/5F) and prefers well-drained, calcareous or loamy soils.",
        "query: Riesling is an early- to mid-budding, late-ripening white grape cultivar with moderate vigor and small to medium, compact clusters, producing wines with high acidity and a pale yellow to golden color. It has good cold-hardiness (tolerating around -20C/-4F) and prefers well-drained soils such as slate, limestone, or loam.",
        "query: Sauvignon Blanc is an early- to mid-ripening white grape cultivar with moderate vigor and medium-sized, loose clusters, producing wines with high acidity and a pale greenish-yellow color. It has moderate cold-hardiness (tolerating around -15C/5F) and prefers well-drained soils such as gravel, limestone, or sandy loam.",
        "query: Syrah is a mid- to late-budding, mid- to late-ripening red grape cultivar with high vigor and medium to large, compact clusters, producing wines with moderate acidity and a deep, dark purple color. It has moderate cold-hardiness (tolerating around -15C/5F) and prefers well-drained, stony or gravelly soils.",
        "query: Zinfandel is a mid- to late-budding, mid- to late-ripening red grape cultivar with high vigor and medium to large, loosely packed clusters, producing wines with moderate acidity and a deep ruby to purple color. It has low to moderate cold-hardiness (tolerating around -12C/10F) and prefers well-drained, sandy or loamy soils.",
    ]
)

SITE_STR_VERB = np.array(
    [
        "query: The West Kelowna site in the Okanagan Valley, British Columbia (approximately 49.86 N, -119.58 W) is a semi-arid, rain-shadow viticultural area with hot, sunny summers and cold winters (low annual precipitation ~350 mm) that support a long, intense growing season and moderate frost risk, and well-drained, mineral-rich soils including volcanic rock, glacial till, sandy loam and clay derived from ancient volcanic and glacial deposits that favor diverse grape varieties and concentrated fruit.",  # 0
        "query: The Okanagan Falls East site in the Okanagan Valley, British Columbia (approximately 49.35 N, -119.57 W) features a semi-arid, lake-moderated climate with warm summer days and cool nights that preserve acidity through a long, steady ripening season and relatively low frost risk, and is underpinned by diverse, well-drained soils including glaciofluvial gravel, sand, sandy loam and mineral-rich pockets derived from ancient glacial deposits that favor balanced vine growth and fruit quality.",  # 1
        "query: The Okanagan Falls West site in the Okanagan Valley, British Columbia (approximately 49.29 N, -119.53 W) sits on the western side of the Skaha Lake area just outside the Okanagan Falls sub-GI, where the climate is warm and semi-arid with hot, sunny days tempered by cool nights and lake-moderated temperatures that extend the growing season, and the soils are diverse, well-drained glaciofluvial deposits of gravel, sand and mineral-rich loams that promote balanced vine growth.",  # 2
        "query: The Oliver East site in the Okanagan Valley, British Columbia (approximately 49.18 N, -119.55 W) experiences a warm, semi-arid continental climate with long, hot, sunny summers and cool nights under a pronounced rain-shadow that supports a long growing season with strong diurnal shifts, and is underlain by well-drained, sandy, gravelly and glaciofluvial soils typical of the southern Okanagan that encourage deep root systems and concentrated fruit for a wide range of vinifera varieties.",  # 3
        "query: The Oliver South viticultural area in the southern Okanagan Valley, British Columbia (approximately 49.18 N, -119.55 W) lies in one of Canada's warmest, semi-arid, high-sun desert-influenced climates with hot, dry summers and cool nights that help maintain acidity in grapes, and is characterized by well-drained, glaciofluvial and sandy-gravel soils with low moisture retention that promote deep rooting and concentration in a broad range of varieties from reds to whites.",  # 4
        "query: The Oliver West site in the Okanagan Valley, British Columbia (approximately 49.18 N, -119.55 W) is part of a warm, semi-arid continental climate with hot, dry summers and cool nights under a pronounced rain-shadow that helps grapes ripen with concentration and preserve natural acidity, and is underlain by well-drained, glaciofluvial gravels, sands, and sandy loams typical of the South Okanagan that promote deep rooting and balanced vine growth for both red and white varieties.",  # 5
        "query: The Black Sage Bench site in the Okanagan Valley, British Columbia (approximately 49.06 N, 119.33 W) features a very warm, semi-arid “pocket desert” climate with hot, sunny summers and cool nights that enhance ripening and acidity retention, and is underlain by deep, well-drained sandy glaciofluvial soils that force vines to root deeply and produce concentrated fruit with minimal foliage.",  # 6
        "query: The Osoyoos North site in the Okanagan Valley, British Columbia (approximately 49.03 N, -119.46 W) lies just north of the town of Osoyoos in one of Canada's warmest, semi-arid, desert-influenced climates with hot, dry summers and cool nights that promote concentrated ripening and acidity retention, and is underlain by deep, well-drained sandy and glaciofluvial soils typical of the southern valley that encourage deep rooting and balanced vine development.",  # 7
        "query: The Osoyoos North East site in the Okanagan Valley, British Columbia (approximately 49.03 N, -119.46 W) lies in the semi-arid, rain-shadow-influenced southern Okanagan with hot, dry summers and cool nights that promote ripeness and acidity retention in grapes, and its soils are generally well-drained glaciofluvial sands, gravels, and mineral-rich loams typical of the broader Osoyoos/Oliver area that encourage deep root development and concentrated fruit character.",  # 8
        "query: The Osoyoos South East Bench site in the Okanagan Valley, British Columbia (approximately 49.03 N, -119.46 W, near the southeast flank of Osoyoos) experiences a hot, semi-arid, rain-shadow-influenced climate with long, dry, sunny summers and cool nights that promote ripening while preserving acidity, and is underlain by well-drained sandy, glaciofluvial soils with interspersed gravel and mineral-rich components typical of the southern valley that encourage deep vine rooting and concentrated fruit.",  # 9
        "query: The Osoyoos West site in the Okanagan Valley, British Columbia (approximately 49.03 N, -119.46 W) lies in a hot, semi-arid continental climate with long, dry, sunny summers and cooler nights that favor ripening and acidity retention and is underlain by well-drained, sandy glaciofluvial soils with traces of gravel and loam typical of the west side of Osoyoos Lake that encourage deep rooting and concentrated fruit.",  # 10
        "query: The Naramata Bench site in the Okanagan Valley, British Columbia (approximately 49.35 N, 119 35 W) has a moderate continental climate with warm, sunny summers and cool nights moderated by nearby Okanagan Lake that extend the growing season and preserve acidity, and is underlain by well-drained, mainly sandy to silt-loam glacial lake and mixed sediments with pockets of gravel and mineral-rich soils that encourage deep rooting and balanced vine growth.",  # 11
        "query: The Beamsville Bench site in the Ontario Niagara Peninsula (approximately 43.17 N, -79.50 W) has a cool-climate, lake-moderated growing season with warm, sunny days and cooling breezes off Lake Ontario that extend ripening and moderate frost risk, and is underlain by deep, well-drained, heterogeneous soils of gravel, sand, silt, clay and fossil-rich dolomitic limestone derived from the Niagara Escarpment that impart notable minerality and balanced moisture for vine growth.",  # 12
        "query: The Lincoln Lakeshore site in the Ontario Niagara Peninsula (approximately 43.11 N, -79.34 W) lies along the southern shore of Lake Ontario with a cool, lake-moderated climate that yields a long, temperate growing season with warm days and cool nights, and is underlain by well-drained, variable soils including light sandy soils and pockets of red clay loam over ancient Lake Iroquois plains that provide balanced moisture and heat for vine growth.",  # 13
        "query: The Twenty Mile Bench site in the Ontario Niagara Peninsula (approximately 43.15 N, -79.39 W) has a cool-climate, moderating influence from Lake Ontario with warm, sunny growing seasons and gentle temperature swings aided by breezes and escarpment shelter, and is underlain by deep, well-drained glacial clay and till soils rich in limestone and shale that retain moisture yet offer good drainage for balanced vine growth and fruit development.",  # 14
        "query: The Vinemount Ridge site in the Ontario Niagara Peninsula (approximately 43.19 N, -79.68 W) lies just above the brow of the Niagara Escarpment where a cool-climate, lake-moderated region with early spring warming, warm summers and moderately cool nights produces a relatively high diurnal range, and is underlain by deep, well-drained clay-loam and silty soils enriched with shale and natural moraine drainage that balance moisture retention with good aeration for balanced vine growth and fruit development.",  # 15
        "query: The Creek Shores site in the Ontario Niagara Peninsula (approximately 43.18 N, -79.33 W) lies on the Lake Iroquois Plain with a cool, moderating Lake Ontario influence that brings gradual spring warming, warm summers with cool nights, and an extended growing season, and is underlain by diverse, well-drained glacial delta sediments and sedimentary soils including lighter sands and porous loams over Queenston Shale that support balanced vine growth and concentrated fruit.",  # 16
        "query: The Four Mile Creek site in the Ontario Niagara Peninsula (approximately 43.25 N, -79.12 W) lies on a relatively flat, inland plain with a cool-climate, lake-moderated growing season featuring warm, sunny days and cool nights that extend ripening and preserve acidity, and is underlain by fertile, well-drained soils of glacial sediments over Queenston shale, including rich clay loams that retain moisture yet provide good drainage for balanced vine growth and fruit development.",  # 17
        "query: The Niagara Lakeshore site in the Ontario Niagara Peninsula (approximately 43.01 N, -79.08 W) sits along the southern shore of Lake Ontario with a cool, lake-moderated climate that features a long, consistent growing season with warm days, cool nights, and extended ripening into late October, and is underlain by well-drained glacial and lake-deposited soils of clay, silt, deltaic sands, and reddish sandy soils near the shore that support deep rooting and balanced moisture for quality vine growth.",  # 18
        "query: The Niagara River site in the Ontario Niagara Peninsula (approximately 43.90 N, -79.80 W) occupies a narrow strip of east-facing, gently sloped vineyards along the Niagara River with a cool-climate, lake- and river-moderated growing season of warm days and cool nights that extends ripening and mitigates frost risk, and is underlain by predominantly well-drained stratified fine sands over red Queenston shale with variable drainage that encourages deep root penetration and balanced vine growth.",  # 19
        "query: The Short Hills Bench site in the Ontario Niagara Peninsula (approximately 43.10 N, -79.28 W) has a cool-climate, lake-moderated growing season with warm, sunny days and cool nights that extend ripening and promote acidity, and is underlain by deep, varied soils of water-stratified clay, brown silty clay and sand/gravels with good drainage and water-holding capacity that support balanced vine growth and concentrated fruit character.",  # 20
        "query: The St. David's Bench site in the Ontario Niagara Peninsula (approximately 43.158 N, -79.124 W) lies on a gently sloping bench south of Lake Ontario with a cool-climate, lake- and escarpment-influenced growing season marked by early spring warming, warm sunny days and cool nights that extend ripening, and is underlain by deep, moisture-retentive silty clay and clay-loam soils over red sandstone that drain moderately and support balanced vine growth and concentrated fruit.",  # 21
        "query: The Prosser viticultural area in Washington, United States (approximately 46.2189 N, -119.7025 W) is situated in the heart of the Yakima Valley AVA where a semi-arid, high-sunshine climate with hot, dry summers, cool nights, and low annual precipitation supports a long growing season under irrigation, and the region's soils are typically well-drained silt-loams and gravelly alluvial deposits over basalt bedrock that favor deep rooting and balanced vine growth for premium wine grapes.",  # 22
        "query: The Michigan site in the Lake Michigan Shore region of the United States (approximately 42.60 N, -86.10 W) lies along the eastern shore of Lake Michigan where a cool, lake-moderated continental climate brings warm summer days, cool nights, and a relatively long growing season, and the vineyards are underlain by well-drained glacial sandy loams and gravelly soils that provide balanced drainage and moderate fertility for steady vine growth and fruit development.",
        "query: The Beaver site in Nova Scotia, Canada (approximately 45.10 N, -64.50 W) lies in a cool maritime region influenced by the Bay of Fundy where mild summers, cool nights, and a gradual autumn create a moderate growing season, and the vineyards are underlain by well-drained glacial and alluvial soils of sandy loams and gravels that support balanced moisture and resilient vine growth with moderate fertility and favorable heat retention",
        "query: The Bejo site in New York State, United States (approximately 42.75 N, -76.10 W) lies within the broader Finger Lakes region where deep glacial lakes moderate a cool continental climate characterized by warm summer days, cool nights, and an extended autumn ripening period, and the vineyards are underlain by well-drained glacial till soils consisting of gravelly loams, silts, and shale-derived sediments that allow deep root penetration and balanced water availability for consistent vine growth and fruit maturation.",
        "query: The Geneva site in New York State, United States (approximately 42.88 N, -76.98 W) sits along the northern shores of Seneca Lake in the Finger Lakes region where deep lake waters moderate winter cold and extend the growing season with warm days and cool nights, and the surrounding vineyards are typically planted on well-drained glacial till soils composed of silty loams, gravel deposits, and shale fragments over sedimentary bedrock that support deep rooting, steady moisture availability, and balanced vine growth.",
        "query: The Hudson site in New York State, United States (approximately 42.25 N, -73.79 W) lies in the Hudson Valley along the Hudson River where a cool continental climate moderated by the river brings warm summer days, cool nights, and a moderate growing season, and the vineyards are underlain by well-drained glacial and alluvial soils of silt loam, sand, and gravel over shale and limestone that support balanced vine growth and steady fruit development.",
        "query: The Portland site in New York State, United States (approximately 42.38 N, -79.47 W) lies along the southern shore of Lake Erie within the Lake Erie viticultural region where a cool, lake-moderated climate brings warm days, cool nights, and a relatively long growing season, and the vineyards are underlain by well-drained glacial silts, clays, and gravelly loams that retain moisture while supporting consistent vine growth and fruit ripening.",
    ]
)

SITE_STR = np.array(
    [
        "query: WestKelowna",  # 0
        "query: OKFallsEast",  # 1
        "query: OKFallsWest",  # 2
        "query: OliverEast",  # 3
        "query: OliverSouth",  # 4
        "query: OliverWest",  # 5
        "query: BlackSage",  # 6
        "query: OsoyoosNorth",  # 7
        "query: OsoyoosNortheast",  # 8
        "query: OsoyoosSoutheast",  # 9
        "query: OsoyoosWest",  # 10
        "query: NaramataBench",  # 11
        "query: BeamsvilleBench",  # 12
        "query: LincolnLakeshore",  # 13
        "query: TwentyMileBench",  # 14
        "query: VinemountRidge",  # 15
        "query: CreekShores",  # 16
        "query: FourMileCreek",  # 17
        "query: NiagaraLakeshores",  # 18
        "query: NiagaraRiver",  # 19
        "query: ShortHillsBench",  # 20
        "query: StDavidsBench",  # 21
        "query: Prosser",  # 22
        "query: Mich1",
        "query: Beaver",
        "query: Bejo",
        "query: Geneva",
        "query: Hudson",
        "query: Portland",
    ]
)

STATION_STR_VERB = np.array(
    [
        "query: The Kelowna weather station in the Okanagan Valley of British Columbia (49.96 N, -119.30 W) records a continental, semi-arid climate with dry, hot, sunny summers and cool, relatively dry winters moderated by nearby Okanagan Lake, yielding long growing seasons with warm days and cool nights and modest annual precipitation.",
        "query: The Kelowna UBCO weather station in the Okanagan Valley of British Columbia (49.9408 N, -119.4002 W) records a semi-arid, continental climate with hot, dry, sunny summers and cool, relatively dry winters moderated locally by the valley and nearby Okanagan Lake, yielding long growing seasons with warm days, cool nights, low precipitation, and notable diurnal temperature variation.",
        "query: The Osoyoos CS weather station in the Okanagan Valley of British Columbia (49.03 N, -119.44 W) records a cold, semi-arid climate with hot, dry summers, cool nights, and relatively mild winters under a strong rain shadow that produces low annual precipitation and long growing seasons",
        "query: The Summerland weather station in the Okanagan Valley of British Columbia (49.57 N, -119.65 W) records a semi-arid continental climate with hot, dry, sunny summers and cool, relatively mild winters moderated by nearby Okanagan Lake, producing one of the warmest and driest growing seasons in Canada with low annual precipitation and strong diurnal temperature swings.",
        "query: The Grimsby Mountain weather station on the Ontario Niagara Peninsula (43.187 N, -79.565 W) records a humid continental climate with warm, sometimes hot summers and cold, snowy winters influenced by nearby Lake Ontario and the Niagara Escarpment that extend the growing season and moderate extreme temperatures.",
        "query: The Vineland Station weather station on the Ontario Niagara Peninsula (43.18 N, -79.40 W) records a humid continental climate with warm, humid summers, cold winters, and significant year-round precipitation moderated by nearby Lake Ontario and local topography.",
        "query: The Roza2 weather station in Washington, United States is associated with the broader Roza/Prosser area in south-central Washington (46.78 N, -120.45 W) and records a semi-arid, continental climate with hot, dry summers, cool nights, and low annual precipitation typical of the Yakima Valley region that supports irrigated agriculture and long growing seasons",
        "query: The SWMREC weather station in southwest Michigan, United States (approximately 42.09 N, -86.36 W) records a humid continental climate with warm, humid summers and cold winters moderated by nearby Lake Michigan, producing relatively long growing seasons with warm days, cool nights, and moderate annual precipitation typical of the Lake Michigan Shore region.",
        "query: The Beaver weather station in Nova Scotia, Canada (approximately 45.10 N, -64.50 W) records a cool maritime climate influenced by the Bay of Fundy and the Atlantic Ocean with mild summers, cool nights, and relatively moderate winters, producing a moderate growing season with frequent moisture and modest annual precipitation.",
        "query: The Bejo weather station in the Finger Lakes region of New York State, United States (approximately 42.75 N, -76.10 W) records a humid continental climate with warm summers, cool nights, and cold winters moderated locally by nearby Finger Lakes, producing a moderate growing season with balanced precipitation and notable seasonal temperature variation.",
        "query: The Geneva weather station in the Finger Lakes region of New York State, United States (approximately 42.87 N, -76.99 W) records a humid continental climate with warm summers, cool nights, and cold winters moderated by nearby Seneca Lake, producing a relatively long growing season with steady precipitation and significant diurnal temperature variation.",
        "query: The Hudson weather station in the Hudson Valley of New York State, United States (approximately 42.25 N, -73.79 W) records a humid continental climate with warm, humid summers and cold winters influenced by the Hudson River valley, producing a moderate growing season with consistent precipitation and seasonal temperature variability.",
        "query: The Portland weather station along the Lake Erie shoreline of New York State, United States (approximately 42.38 N, -79.47 W) records a humid continental climate with warm summers, cool nights, and cold winters moderated by nearby Lake Erie, producing a relatively long growing season with lake-influenced temperatures and moderate annual precipitation.",
    ]
)

REGION_STR_VERB = np.array(
    [
        "query: The Okanagan Valley in British Columbia is a semi-arid, continental-influenced wine region with long, warm, sunny summers, cool nights, and low annual precipitation, where well-drained soils and pronounced diurnal temperature swings create ideal conditions for ripening a wide range of vinifera grapes while preserving acidity and flavor concentration.",
        "query: The Ontario Niagara Peninsula is a cool-climate, lake-moderated wine region with warm, sunny summers, cool nights, and fertile, well-drained soils over limestone and glacial deposits, creating an extended growing season that supports balanced ripening and high-quality vinifera grape production.",
        "query: Washington State is a semi-arid, continental-influenced wine region with hot, sunny days, cool nights, and low annual precipitation in the rain-shadow of the Cascade Mountains, where well-drained alluvial and volcanic soils combined with irrigation allow precise control of vine vigor and high-quality vinifera grape production.",
        "query: New York State in the United States is a cool- to moderate-climate wine region strongly influenced by large lakes and river valleys, with warm summers, cool nights, and cold winters moderated locally by the Finger Lakes, Lake Erie, and Lake Ontario, where glacially derived soils of shale, silt, and gravel support balanced vine growth and high-quality vinifera grape production.",
        "query: Nova Scotia in Canada is a cool maritime wine region influenced by the Atlantic Ocean and the Bay of Fundy, with mild summers, cool nights, and relatively moderate winters, where well-drained glacial and coastal soils and a long, cool growing season support the production of fresh, aromatic vinifera and hybrid grape varieties.",
        "query: Michigan in the United States is a cool-climate, lake-moderated wine region surrounding Lake Michigan where warm summer days, cool nights, and long autumns extend the growing season, and well-drained glacial sandy loams and gravelly soils support balanced vine growth and high-quality vinifera grape production.",
    ]
)

REGION_STR = np.array(
    [
        "query: British Colubmia Okanagan Valley",
        "query: Ontario Niagara Peninsula",
        "query: Washington, United States",
        "query: New York State, United States",
        "query: Nova Scotia, Canada",
        "query: Michigan, United States",
    ]
)
STATION_STR = np.array(
    [
        "query: Kelowna",
        "query: Kelowna UBC",
        "query: Osoyoos CS",
        "query: Summerland",
        "query: Grimbsy Mountain",
        "query: Vineland Station",
        "query: Roza2",
        "query: SWMREC",
        "query: Beaver",
        "query: Bejo",
        "query: Geneva",
        "query: Hudson",
        "query: Portland",
    ]
)
