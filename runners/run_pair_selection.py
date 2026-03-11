import json
import logging
import os
import hydra
import pandas as pd
from omegaconf import OmegaConf, DictConfig

from modules.core.config import Config
from runners.core.pipelines import (
    execute_pair_selection,
)
from runners.core.utils import (
    generate_date_lists,
    save_hydra_config_snapshot,
)

logger = logging.getLogger(__name__)


@hydra.main(version_base=None, config_path="../config", config_name="run_backtest")
def run_pair_selection(cfg: DictConfig):
    script_dir = os.path.dirname(os.path.abspath(__file__))
    project_root = os.path.abspath(os.path.join(script_dir, ".."))

    config_path = os.path.join(
        project_root, "config/helpers/data_fetching_pipeline.yaml"
    )
    if not os.path.exists(config_path):
        raise FileNotFoundError(f"Config not found: {config_path}")
    helper_cfg = OmegaConf.load(config_path)

    base_output_dir = os.path.join(project_root, "data", "pair_selection")
    os.makedirs(base_output_dir, exist_ok=True)

    save_hydra_config_snapshot(cfg=cfg, root_dir=base_output_dir)

    json_path = os.path.join(
        project_root, "config/schemas/list_of_assets.json"
    )
    with open(json_path, "r", encoding="utf-8") as f:
        universe_data = json.load(f)

    cfg = Config(**OmegaConf.to_container(cfg, resolve=True))

    config_dates = {
        "pair_selection_start": helper_cfg.start,
        "pair_selection_end": helper_cfg.end,
    }

    number_of_iterations = helper_cfg.iterations
    lists = generate_date_lists(config_dates, number_of_iterations)

    for i in range(number_of_iterations):
        ps_start = lists["pair_selection_start_list"][i]
        ps_end = lists["pair_selection_end_list"][i]

        current_selection_date = pd.to_datetime(ps_start)
        month_key = current_selection_date.strftime("%Y-%m")

        if month_key not in universe_data:
            logger.error(f"Universe for month {month_key} not found")
            continue

        current_iteration_tickers = universe_data[month_key]["assets"]

        output_dir = os.path.join(base_output_dir, month_key)
        os.makedirs(output_dir, exist_ok=True)

        logger.info(f"--- Running Pair Selection for {month_key} ---")

        execute_pair_selection(
            tickers=current_iteration_tickers,
            ps_start=ps_start,
            ps_end=ps_end,
            interval=cfg.market.interval,
            output_dir=output_dir,
        )


if __name__ == "__main__":
    run_pair_selection()
