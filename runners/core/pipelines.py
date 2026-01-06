import logging
import os
import sys
from datetime import datetime
from typing import Any
import pandas as pd
from omegaconf import DictConfig

from modules.core.models import StrategyResult
from modules.core.statistical_tests import (
    ssd_cumulative_returns,
    pearson_correlation,
    engle_granger_cointegration,
)
from modules.data_services.data_loaders import load_data
from modules.data_services.data_utils import (
    save_strategy_result,
    load_btc_benchmark,
    merge_by_pair,
    save_dataframe,
)
from modules.performance.strategy import Strategy
from modules.visualization.plots import plot_positions, plot_pnl, plot_zscore

logger = logging.getLogger(__name__)


def setup_run_environment(calling_file: str) -> str:
    script_dir = os.path.dirname(os.path.abspath(calling_file))
    project_root = os.path.abspath(os.path.join(script_dir, ".."))

    timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    output_dir = os.path.join(project_root, "results", timestamp)

    os.makedirs(output_dir, exist_ok=True)

    root_logger = logging.getLogger()
    root_logger.setLevel(logging.INFO)

    formatter = logging.Formatter(
        "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
    )

    file_handler = logging.FileHandler(os.path.join(output_dir, "execution.log"))
    file_handler.setFormatter(formatter)
    root_logger.addHandler(file_handler)

    if not any(isinstance(h, logging.StreamHandler) for h in root_logger.handlers):
        console_handler = logging.StreamHandler(sys.stdout)
        console_handler.setFormatter(formatter)
        root_logger.addHandler(console_handler)

    logger.debug("--- Environment Setup ---")
    logger.debug(f"Output Directory: {output_dir}")

    return output_dir


def execute_optimization(
    cfg: DictConfig,
    bt: Strategy,
    static_params: dict[str, Any],
    param_space: list[Any],
    metric: tuple[str, str],
) -> dict[str, Any]:

    beta_opt_start = cfg.performance.optimization.beta_start
    opt_start = cfg.performance.optimization.start
    opt_end = cfg.performance.optimization.end

    best_params, best_score = bt.run_optimization(
        static_params=static_params,
        param_space=param_space,
        metric=metric,
        opt_start=opt_start,
        opt_end=opt_end,
        beta_opt_start=beta_opt_start,
        n_iter=cfg.performance.optimization.n_iter,
        random_state=cfg.performance.optimization.random_state,
        replicates=cfg.performance.optimization.replicates,
        penalty_bad=cfg.performance.optimization.penalty_bad,
    )

    best_params.update(static_params)
    log = (best_params, best_score)
    logger.info(log)

    return best_params


def execute_testing(
    cfg: DictConfig,
    bt: Strategy,
    best_params: dict[str, Any],
    ticker_x: str,
    ticker_y: str,
    output_dir: str,
    beta_test_start: str,
    test_start: str,
    test_end: str,
) -> StrategyResult:

    window_factor = best_params["window_factor"]
    entry_threshold = best_params["entry_threshold"]
    exit_threshold = best_params["exit_threshold"]
    stop_loss = best_params["stop_loss"]

    result = bt.run_strategy(
        window_factor=window_factor,
        entry_threshold=entry_threshold,
        exit_threshold=exit_threshold,
        stop_loss=stop_loss,
        test_start=test_start,
        test_end=test_end,
        beta_test_start=beta_test_start,
    )

    save_strategy_result(
        result=result,
        file_name=f"test_{ticker_x}_{ticker_y}_{test_start}_{test_end}",
        directory=output_dir,
    )

    plot_positions(result, directory=output_dir, save=True, show=True)
    btc_data = load_btc_benchmark(
        test_start=test_start,
        test_end=test_end,
        interval=cfg.market.interval,
    )
    plot_pnl(result, btc_data, directory=output_dir, save=True, show=True)
    plot_zscore(result, directory=output_dir, save=True, show=True)

    logger.info("Testing completed, returning StrategyResult.")

    return result


def execute_pair_selection(cfg, output_dir) -> pd.DataFrame:
    df = load_data(
        tickers=cfg.tickers,
        start=cfg.pair_selection.start,
        end=cfg.pair_selection.end,
        interval=cfg.market.interval,
    )

    ssd_c_returns_df = ssd_cumulative_returns(df)
    corr_log_returns_df = pearson_correlation(df, source="log_returns")
    eg_log_prices_df = engle_granger_cointegration(df, source="log_prices")

    merged_df = (
        merge_by_pair(
            dfs=[ssd_c_returns_df, corr_log_returns_df, eg_log_prices_df],
            keep_cols=[["ssd"], ["corr_log_returns"], ["eg_p_value"]],
        )
        .sort_values("eg_p_value", ascending=True)
        .reset_index(drop=True)
    )

    save_dataframe(
        df=merged_df,
        file_name=f"pair_selection_{cfg.pair_selection.start}_{cfg.pair_selection.end}",
        directory=output_dir,
    )

    logger.info("Pair Selection completed, returning DataFrame.")

    return merged_df
