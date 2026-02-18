import logging
import os
from datetime import datetime
from dateutil.relativedelta import relativedelta
from sb3_contrib import RecurrentPPO
from stable_baselines3 import A2C
from stable_baselines3.common.base_class import BaseAlgorithm
from omegaconf import DictConfig, OmegaConf
from hydra.core.hydra_config import HydraConfig

logger = logging.getLogger(__name__)


def save_hydra_config_snapshot(cfg: DictConfig, root_dir: str) -> None:
    hydra_dir = os.path.join(root_dir, ".hydra")
    os.makedirs(hydra_dir, exist_ok=True)
    OmegaConf.save(config=cfg, f=os.path.join(hydra_dir, "config.yaml"))
    try:
        hydra_cfg = HydraConfig.get()
        OmegaConf.save(config=hydra_cfg, f=os.path.join(hydra_dir, "hydra.yaml"))
        OmegaConf.save(
            config=hydra_cfg.overrides.task, f=os.path.join(hydra_dir, "overrides.yaml")
        )

        logger.info(f"Saved .hydra configs to: {hydra_dir}")

    except ValueError:
        logger.warning("HydraConfig not initialized")


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
