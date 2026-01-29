"""
util.py

Utility functions for the model_egnine class

Modified by Will Solow, 2024
"""

import yaml
import os
import datetime
import torch
from inspect import getmembers, isclass
import importlib.util
import numpy as np

from model_engine.models.base_model import Model

EPS = 1e-12
PHENOLOGY_INT = {"Ecodorm": 0, "Budbreak": 1, "Bloom": 2, "Veraison": 3, "Ripe": 4}

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


def param_loader(config: dict) -> dict:
    """
    Load the configuration of a model from dictionary
    """
    try:
        model_name, model_num = config["model_parameters"].split(":")
    except:
        raise Exception(f"Incorrectly specified model_parameters file `{config['model_parameters']}`")

    fname = f"{os.getcwd()}/{config['config_fpath']}{model_name}.yaml"
    try:
        model = yaml.safe_load(open(fname))
    except:
        raise Exception(f"Unable to load file: {fname}. Check that file exists")

    try:
        cv = model["ModelParameters"]["Sets"][model_num]
    except:
        raise Exception(
            f"Incorrectly specified parameter file {fname}. Ensure that `{model_name}` contains parameter set `{model_num}`"
        )

    for c in cv.keys():
        cv[c] = cv[c][0]

    return cv


def per_task_param_loader(config: dict, params: list) -> torch.Tensor:
    """
    Load the available configurations of a model from dictionary and put them on tensor
    """

    dtype = config.DataConfig.dtype.rsplit("_", 1)[0]
    fname = f"{os.getcwd()}/{config.PConfig.config_fpath}{dtype}.yaml"
    try:
        model = yaml.safe_load(open(fname))
    except:
        raise Exception(f"Unable to load file: {fname}. Check that file exists")
    init_params = []
    for n in CROP_NAMES[config.DataConfig.dtype]:
        try:
            cv = model["ModelParameters"]["Sets"][n]
        except:
            raise Exception(
                f"Incorrectly specified parameter file {fname}. Ensure that `{config.DataConfig.dtype}` contains parameter set `{n}`"
            )
        task_params = []
        for c in params:
            if c in cv.keys():
                task_params.append(cv[c][0])
        init_params.append(task_params)

    return torch.tensor(init_params)


def get_models(folder_path: str) -> list[type]:
    """
    Get all the models in the /models/ folder
    """
    constructors = {}

    # Loop through all files in the folder
    for filename in os.listdir(folder_path):
        if filename.endswith(".py"):
            file_path = os.path.join(folder_path, filename)

            # Remove the .py extension
            module_name = filename[:-3]

            spec = importlib.util.spec_from_file_location(module_name, file_path)
            module = importlib.util.module_from_spec(spec)
            spec.loader.exec_module(module)

            for name, obj in getmembers(module):
                if isclass(obj) and (issubclass(obj, Model)):
                    constructors[f"{name}"] = obj
        elif os.path.isdir(f"{folder_path}/{filename}"):
            constr = get_models(f"{folder_path}/{filename}")
            constructors = constructors | constr

    return constructors


def int_to_day_of_year(day_number: int) -> datetime.datetime:
    """
    Converts an integer representing the day of the year to a date.
    """
    return datetime.datetime(1900, 1, 1) + datetime.timedelta(days=day_number - 1)


def tensor_appendleft(tensor: torch.Tensor, new_values: torch.Tensor) -> torch.Tensor:
    """
    Insert new_values at the left (index 0), shift tensor right, and drop last elements.
    Supports 1D or 2D tensors.
    """
    # Make sure new_values can broadcast to tensor shape except last dim = 1
    if not isinstance(new_values, torch.Tensor):
        new_values = torch.tensor(new_values).to(tensor.device)
    new_values = new_values.unsqueeze(-1) if new_values.ndim == 0 else new_values
    new_values = new_values.unsqueeze(-1) if new_values.dim() == tensor.dim() - 1 else new_values

    # Shift right by slicing all except last element on last dim
    shifted = torch.cat([new_values, tensor[..., :-1]], dim=-1)
    return shifted


def tensor_pop(tensor: torch.Tensor, fill_value=0) -> tuple[torch.Tensor, torch.Tensor]:
    """
    Remove and return the last element from tensor along last dimension,
    shift everything left, fill last position with fill_value.
    Returns (shifted_tensor, popped_values).
    Supports 1D or 2D tensors.
    """
    popped = tensor[..., -1].clone()
    shifted = torch.cat([tensor[..., 1:], torch.full_like(tensor[..., -1:], fill_value)], dim=-1)

    return shifted, popped
