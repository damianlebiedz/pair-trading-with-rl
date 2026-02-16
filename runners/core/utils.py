import logging
import os
from datetime import datetime
from dateutil.relativedelta import relativedelta
from sb3_contrib import RecurrentPPO
from stable_baselines3 import A2C
from stable_baselines3.common.base_class import BaseAlgorithm

logger = logging.getLogger(__name__)


def generate_date_lists(initial_config, n):
    generated_lists = {}

    for name, start_date_str in initial_config.items():
        start_date = datetime.strptime(start_date_str, "%Y-%m-%d")
        date_list = []

        for m in range(n):
            new_date = start_date + relativedelta(months=m)
            date_list.append(new_date.strftime("%Y-%m-%d"))

        list_name = f"{name}_list"
        generated_lists[list_name] = date_list

    return generated_lists


def load_model(path: str, device: str = "cpu") -> BaseAlgorithm:
    filename = os.path.basename(path).lower()

    if "recurrent_ppo" in filename:
        return RecurrentPPO.load(path, device=device)
    elif "a2c" in filename:
        return A2C.load(path, device=device)
    else:
        raise ValueError(f"Unsupported model: {path}")
